"""Round-robin slice loop. Each trial asks for three candidates and C++ picks the winner."""

from __future__ import annotations

import os
from typing import Any

import azure.durable_functions as df

from orchestrators.loop_optimizer import bp
from orchestrators.run_report import append_attempt, publish_report
from orchestrators.tournament_evaluator import TournamentParseError, completion_context, extract_tournament


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
    rolling_config = incoming.get("rolling_config") or {}
    training_params = incoming.get("training_params") or {}
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
        reader = completion_context(slice_state)
        note = ""
        synthesized: dict[str, Any] = {}
        receipt: dict[str, Any] = {}
        candidates: list[dict[str, Any]] = []
        prompt_tokens = completion_tokens = total_tokens = 0
        accepted = False
        for _attempt in range(3):
            compiled = yield context.call_activity(
                "activity_gateway_compile_prompt",
                {
                    "checkpoint_blob": checkpoint_blob,
                    "journal_blob": journal_blob,
                    "layer_name": active_layer,
                    "visible_history": int(rolling_config.get("visible_history_count", -1)),
                    "int_bound": int(training_params.get("int_bound", -1)),
                    "retry_note": note,
                },
            )
            synthesized = yield context.call_activity(
                "activity_gateway_synthesize",
                {
                    "prompt": compiled.get("prompt", ""),
                    "attempt": _attempt + 1,
                    "layer_name": active_layer,
                    "context": reader,
                },
            )
            prompt_tokens += int(synthesized.get("prompt_tokens") or 0)
            completion_tokens += int(synthesized.get("completion_tokens") or 0)
            total_tokens += int(synthesized.get("total_tokens") or 0)
            candidates = synthesized.get("candidates") or []
            if synthesized.get("mutation_failed") or not candidates:
                note = str(synthesized.get("error") or "completion did not contain three candidates")
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
            if receipt.get("status") == "rejected_zero_mutation":
                note = "candidates did not mutate the active slice"
                continue
            accepted = True
            break
        if not accepted:
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
                    "probe": reader["probe"],
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
        trial += 1
        loss = float(receipt.get("loss", current_loss))
        winner_name = receipt.get("winner", "")
        winner_step = str(receipt.get("winner_step") or "")
        trials.append(
            {
                "trial": trial,
                "layer": active_layer,
                "loss": loss,
                "delta": float(receipt.get("delta", loss - current_loss)),
                "status": receipt.get("status"),
                "winner": winner_name,
                "winner_step": winner_step,
                "probe": reader["probe"],
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
        "",
        "Parameters:",
    ]
    context = payload.get("context") or {}
    for field in context.get("parameters") or []:
        lines.append(f"- {field.get('key')} ({field.get('role')}, {field.get('scheme', '')}): `{field.get('token')}`")
    probe = context.get("probe") or {}
    lines.append(f"Probe {probe.get('key', '')}: {probe.get('token', '')}")
    lines.extend(["", "Candidates:"])
    for candidate in record.get("candidates") or []:
        lines.append(f"- {candidate.get('name')}")
        fields = candidate.get("fields") or {}
        if not fields:
            fields = {"w": candidate.get("w"), "b": candidate.get("b")}
        for key, token in fields.items():
            lines.append(f"  - {key}: `{token}`")
    if not record.get("candidates"):
        lines.append("- none")
    lines.extend(["", "Raw completion:", "", "```json", str(record.get("content") or ""), "```"])
    return "\n".join(lines)


@bp.activity_trigger(input_name="payload")
def activity_gateway_compile_prompt(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    prompt = tensor_engine.compile_prompt(
        payload["checkpoint_blob"],
        payload["journal_blob"],
        payload["layer_name"],
        int(payload.get("visible_history", -1)),
        int(payload.get("int_bound", -1)),
        str(payload.get("retry_note") or ""),
    )
    return {"prompt": prompt}


@bp.activity_trigger(input_name="payload")
def activity_gateway_synthesize(payload: dict) -> dict[str, Any]:
    """Send one compiled prompt. Candidate text is extracted; C++ scores it."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": str(payload.get("prompt") or "")}],
        response_format={"type": "json_object"},
        reasoning_effort="none",
    )
    usage = response.usage
    text = response.choices[0].message.content or ""
    prompt_tokens = int(usage.prompt_tokens) if usage is not None else 0
    completion_tokens = int(usage.completion_tokens) if usage is not None else 0
    total_tokens = int(usage.total_tokens) if usage is not None else 0
    try:
        candidates, meta = extract_tournament(text, dict(payload.get("context") or {}))
        error = ""
        failed = False
    except TournamentParseError as exc:
        candidates, meta = [], {}
        error = str(exc)
        failed = True
    record = {
        "attempt": int(payload.get("attempt") or 1),
        "accepted": not failed,
        "error": error,
        "content": text,
        "candidates": candidates,
        "reasoning": meta.get("reasoning", ""),
        "confidence": meta.get("confidence", 0.5),
        "meta": meta,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    append_attempt({
        "source": "modular_model_orchestrator",
        "layer": payload.get("layer_name"),
        "parameters": (payload.get("context") or {}).get("parameters"),
        "probe": (payload.get("context") or {}).get("probe"),
        **record,
    })
    publish_report(f"trial {payload.get('layer_name')} attempt {record['attempt']}", _attempt_report(payload, record))
    print(
        f"[tokens] prompt={prompt_tokens} completion={completion_tokens} "
        f"total={total_tokens} mutation_failed={failed}"
    )
    return {
        "candidates": candidates,
        "reasoning": meta.get("reasoning", ""),
        "confidence": meta.get("confidence", 0.5),
        "meta": meta,
        "mutation_failed": failed,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
