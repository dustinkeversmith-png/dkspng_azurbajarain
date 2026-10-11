"""Round-robin slice loop. Each trial asks for three candidates and C++ picks the winner."""

from __future__ import annotations

import os
import re
from typing import Any

import azure.durable_functions as df

from orchestrators.dynamic_injector import collect_tournament, integer_distance
from orchestrators.loop_optimizer import bp
from orchestrators.run_report import append_attempt, publish_report


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


@bp.orchestration_trigger(context_name="context")
def modular_model_orchestrator(context: df.DurableOrchestrationContext):
    incoming = context.get_input() or {}
    max_trials = int(incoming.get("max_trials", 40))
    target_loss = float(incoming.get("target_loss", 0.01))
    checkpoint_blob = incoming.get("checkpoint_blob", "target_model.json")
    journal_blob = incoming.get("journal_blob", "target_journal.json")
    journal_capacity = int(incoming.get("journal_capacity", 10))
    rolling_config = incoming.get("rolling_config") or {
        "visible_history_count": 4,
        "tracked_metrics": ["loss", "delta", "status"],
    }
    training_params = incoming.get("training_params") or {
        "target_sparsity": 0.20,
        "exploration_mode": "tournament_triad",
    }
    meta = yield context.call_activity(
        "activity_gateway_read_meta",
        {"checkpoint_blob": checkpoint_blob, "journal_blob": journal_blob},
    )
    layer_names = list(meta.get("layers") or [])
    initial_loss = float(meta.get("loss", 999.0))
    best_loss = initial_loss
    trial = 0
    trials: list[dict[str, Any]] = []
    while trial < max_trials and best_loss > target_loss and layer_names:
        active_layer = layer_names[trial % len(layer_names)]
        slice_state = yield context.call_activity(
            "activity_gateway_read_slice",
            {
                "checkpoint_blob": checkpoint_blob,
                "journal_blob": journal_blob,
                "layer_name": active_layer,
            },
        )
        current_loss = float(slice_state.get("loss", best_loss))
        synthesized = yield context.call_activity(
            "activity_gateway_synthesize",
            {
                "layer_name": active_layer,
                "in_dim": slice_state["in_dim"],
                "out_dim": slice_state["out_dim"],
                "active_w": slice_state["totr_w"],
                "active_b": slice_state["biases"],
                "current_loss": current_loss,
                "max_residual": slice_state.get("max_residual", 0.0),
                "directives": slice_state.get("directives") or meta.get("directives") or {},
                "telemetry": slice_state.get("telemetry") or {},
                "gradient_hint": slice_state.get("gradient_hint", ""),
                "rolling_config": rolling_config,
                "training_params": training_params,
                "history": trials[-int(rolling_config.get("visible_history_count", 4)) :],
            },
        )
        if isinstance(synthesized, dict) and "candidates" in synthesized:
            candidates = synthesized.get("candidates") or []
            prompt_tokens = int(synthesized.get("prompt_tokens") or 0)
            completion_tokens = int(synthesized.get("completion_tokens") or 0)
            total_tokens = int(synthesized.get("total_tokens") or 0)
            mutation_failed = bool(synthesized.get("mutation_failed"))
        else:
            candidates = synthesized
            prompt_tokens = completion_tokens = total_tokens = 0
            mutation_failed = False
        if mutation_failed or not candidates:
            trial += 1
            trials.append(
                {
                    "trial": trial,
                    "layer": active_layer,
                    "loss": current_loss,
                    "delta": 0.0,
                    "status": "mutation_failed",
                    "winner": "",
                    "winner_step": "",
                    "gradient_hint": slice_state.get("gradient_hint", ""),
                    "reasoning": str(synthesized.get("reasoning") or "") if isinstance(synthesized, dict) else "",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "error": synthesized.get("error", "") if isinstance(synthesized, dict) else "",
                }
            )
            context.set_custom_status(
                {
                    "trial": trial,
                    "layer": active_layer,
                    "best_loss": best_loss,
                    "winner": "",
                    "total_tokens": total_tokens,
                    "status": "mutation_failed",
                }
            )
            continue
        receipt = yield context.call_activity(
            "activity_gateway_commit_tournament",
            {
                "layer_name": active_layer,
                "candidates": candidates,
                "target_blob": checkpoint_blob,
                "journal_blob": journal_blob,
                "journal_capacity": journal_capacity,
            },
        )
        trial += 1
        loss = float(receipt.get("loss", current_loss))
        winner_name = receipt.get("winner", "")
        winner_cand = next((candidate for candidate in candidates if candidate.get("name") == winner_name), None)
        if winner_cand:
            dist = integer_distance(str(winner_cand.get("w", "")), str(slice_state["totr_w"]))
            gamma_match = re.search(r"g:([0-9.]+)", str(winner_cand.get("w", "")))
            gamma_val = gamma_match.group(1) if gamma_match else "?"
            winner_step = f"g={gamma_val} d={dist}"
        else:
            winner_step = ""
        trials.append(
            {
                "trial": trial,
                "layer": active_layer,
                "loss": loss,
                "delta": float(receipt.get("delta", loss - current_loss)),
                "status": receipt.get("status"),
                "winner": winner_name,
                "winner_step": winner_step,
                "gradient_hint": slice_state.get("gradient_hint", ""),
                "reasoning": str(synthesized.get("reasoning") or "") if isinstance(synthesized, dict) else "",
                "confidence": synthesized.get("confidence") if isinstance(synthesized, dict) else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
        if receipt.get("status") == "committed_natively":
            best_loss = min(best_loss, loss)
        context.set_custom_status(
            {
                "trial": trial,
                "layer": active_layer,
                "best_loss": best_loss,
                "winner": receipt.get("winner"),
                "total_tokens": total_tokens,
            }
        )
    token_total = sum(int(row.get("total_tokens") or 0) for row in trials)
    return {
        "status": "completed",
        "trials": trial,
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "token_total": token_total,
        "token_avg": (token_total / trial) if trial else 0.0,
        "history": trials,
    }


@bp.activity_trigger(input_name="payload")
def activity_gateway_read_meta(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    return _plain(tensor_engine.read_architecture_meta(payload["checkpoint_blob"], payload["journal_blob"]))


@bp.activity_trigger(input_name="payload")
def activity_gateway_read_slice(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    return _plain(
        tensor_engine.read_architecture_slice(
            payload["checkpoint_blob"], payload["journal_blob"], payload["layer_name"]
        )
    )


@bp.activity_trigger(input_name="payload")
def activity_gateway_commit_tournament(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    receipt = _plain(
        tensor_engine.commit_tournament(
            payload["layer_name"],
            payload["candidates"],
            payload["target_blob"],
            payload["journal_blob"],
            int(payload.get("journal_capacity", 10)),
        )
    )
    lines = [
        f"Layer: {payload.get('layer_name')}",
        f"Status: {receipt.get('status')}",
        f"Winner: {receipt.get('winner')}",
        f"Loss: {receipt.get('loss')}",
        f"Delta: {receipt.get('delta')}",
        f"Checksum: {receipt.get('checksum')}",
        "",
        "Scored candidates:",
    ]
    for row in receipt.get("candidates") or []:
        lines.append(f"- {row.get('name')} status {row.get('status')} loss {row.get('loss', '-')}")
    publish_report(f"decision {payload.get('layer_name')}", "\n".join(lines))
    return receipt


def _attempt_report(payload: dict, record: dict[str, Any]) -> str:
    lines = [
        f"Layer: {payload.get('layer_name')}",
        f"Attempt: {record.get('attempt')}",
        f"Accepted: {record.get('accepted')}",
        f"Error: {record.get('error') or '-'}",
        f"Tokens: prompt {record.get('prompt_tokens')} completion {record.get('completion_tokens')} total {record.get('total_tokens')}",
        f"Active w: `{payload.get('active_w')}`",
        f"Active b: `{payload.get('active_b')}`",
        f"Gradient: {payload.get('gradient_hint')}",
        "",
        "Candidates:",
    ]
    for candidate in record.get("candidates") or []:
        distance = integer_distance(str(candidate.get("w") or ""), str(payload.get("active_w") or ""))
        lines.append(f"- {candidate.get('name')} distance {distance}")
        lines.append(f"  - w: `{candidate.get('w')}`")
        lines.append(f"  - b: `{candidate.get('b')}`")
    if not record.get("candidates"):
        lines.append("- none")
    lines.extend(["", "Raw completion:", "", "```json", str(record.get("content") or ""), "```"])
    return "\n".join(lines)


@bp.activity_trigger(input_name="payload")
def activity_gateway_synthesize(payload: dict) -> dict[str, Any]:
    """Direct chat completion. Three mutated candidates, no system message, no tools."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

    def complete(prompt: str) -> tuple[str, int, int, int]:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            reasoning_effort="none",
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            int(usage.prompt_tokens) if usage is not None else 0,
            int(usage.completion_tokens) if usage is not None else 0,
            int(usage.total_tokens) if usage is not None else 0,
        )

    def on_attempt(record: dict[str, Any]) -> None:
        logged = {
            "source": "modular_model_orchestrator",
            "layer": payload.get("layer_name"),
            "active_w": payload.get("active_w"),
            "gradient_hint": payload.get("gradient_hint"),
            **record,
        }
        append_attempt(logged)
        publish_report(f"trial {payload.get('layer_name')} attempt {record.get('attempt')}", _attempt_report(payload, record))

    result = collect_tournament(payload, complete, on_attempt)
    print(
        f"[tokens] prompt={result['prompt_tokens']} completion={result['completion_tokens']} "
        f"total={result['total_tokens']} attempts={result['attempts']} "
        f"mutation_failed={result['mutation_failed']}"
    )
    return result
