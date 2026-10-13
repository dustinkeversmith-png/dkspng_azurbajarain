"""Slice trials. The model compiles the prompt and names the candidate fields."""

from __future__ import annotations

import json
import os
from typing import Any

import azure.durable_functions as df
import azure.functions as func
from starlette.requests import Request
from starlette.responses import Response

from orchestrators.tournament_evaluator import TournamentParseError, completion_context, extract_tournament

bp = df.Blueprint()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    return str(value)


def _slice_loop(context: df.DurableOrchestrationContext, incoming: dict[str, Any], defaults: dict[str, Any]):
    max_trials = int(incoming.get("max_trials", defaults["max_trials"]))
    target_loss = float(incoming.get("target_loss", defaults["target_loss"]))
    checkpoint_blob = incoming.get("checkpoint_blob", defaults["checkpoint_blob"])
    journal_blob = incoming.get("journal_blob", defaults["journal_blob"])
    journal_capacity = int(incoming.get("journal_capacity", defaults["journal_capacity"]))
    visible_history = int((incoming.get("rolling_config") or {}).get("visible_history_count", 3))
    int_bound = int((incoming.get("training_params") or {}).get("int_bound", 25))
    model_meta = yield context.call_activity(
        "activity_read_model_meta",
        {"checkpoint_blob": checkpoint_blob, "journal_blob": journal_blob},
    )
    layer_names = list(model_meta.get("layers") or [])
    best_loss = float(model_meta.get("loss", 999.0))
    trial = 0
    trials: list[dict[str, Any]] = []
    while trial < max_trials and best_loss > target_loss and layer_names:
        active_layer = layer_names[trial % len(layer_names)]
        slice_state = yield context.call_activity(
            "activity_read_layer_slice",
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
        accepted = False
        for _attempt in range(3):
            compiled = yield context.call_activity(
                "activity_compile_prompt",
                {
                    "checkpoint_blob": checkpoint_blob,
                    "journal_blob": journal_blob,
                    "layer_name": active_layer,
                    "visible_history": visible_history,
                    "int_bound": int_bound,
                    "retry_note": note,
                },
            )
            synthesized = yield context.call_activity(
                "activity_complete_slice",
                {
                    "prompt": compiled.get("prompt", ""),
                    "attempt": _attempt + 1,
                    "layer_name": active_layer,
                    "context": reader,
                },
            )
            if synthesized.get("mutation_failed"):
                note = str(synthesized.get("error") or "completion did not match the candidate scheme")
                continue
            receipt = yield context.call_activity(
                "activity_commit_slice",
                {
                    "layer_name": active_layer,
                    "candidates": synthesized.get("candidates") or [],
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
        trial += 1
        loss = float(receipt.get("loss", current_loss)) if accepted else current_loss
        trials.append(
            {
                "trial": trial,
                "layer": active_layer,
                "loss": loss,
                "status": receipt.get("status") if accepted else "mutation_failed",
                "winner": receipt.get("winner", "") if accepted else "",
            }
        )
        if accepted and receipt.get("status") == "committed_natively":
            best_loss = min(best_loss, loss)
        context.set_custom_status({"trial": trial, "layer": active_layer, "best_loss": best_loss})
    return {"status": "completed", "trials": trial, "best_loss": best_loss, "history": trials}


@bp.orchestration_trigger(context_name="context")
def optimize_network_orchestrator(context: df.DurableOrchestrationContext):
    incoming = context.get_input() or {}
    result = yield from _slice_loop(
        context,
        incoming,
        {
            "max_trials": 5,
            "target_loss": 0.05,
            "checkpoint_blob": "sliding_totr.json",
            "journal_blob": "sliding_journal.json",
            "journal_capacity": 5,
        },
    )
    return result


@bp.orchestration_trigger(context_name="context")
def optimize_modular_orchestrator(context: df.DurableOrchestrationContext):
    incoming = context.get_input() or {}
    result = yield from _slice_loop(
        context,
        incoming,
        {
            "max_trials": 30,
            "target_loss": 0.05,
            "checkpoint_blob": "modular_model.json",
            "journal_blob": "modular_journal.json",
            "journal_capacity": 5,
        },
    )
    return result


@bp.activity_trigger(input_name="payload")
def activity_read_model_meta(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    return _plain(tensor_engine.read_model_meta(payload["checkpoint_blob"], payload["journal_blob"]))


@bp.activity_trigger(input_name="payload")
def activity_read_layer_slice(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    return _plain(
        tensor_engine.read_layer_slice(
            payload["checkpoint_blob"], payload["journal_blob"], payload["layer_name"]
        )
    )


@bp.activity_trigger(input_name="payload")
def activity_compile_prompt(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    prompt = tensor_engine.compile_prompt(
        payload["checkpoint_blob"],
        payload["journal_blob"],
        payload["layer_name"],
        int(payload.get("visible_history", 3)),
        int(payload.get("int_bound", 25)),
        str(payload.get("retry_note") or ""),
    )
    return {"prompt": prompt}


@bp.activity_trigger(input_name="payload")
def activity_complete_slice(payload: dict) -> dict[str, Any]:
    """One compiled prompt. The candidate scheme decides which fields are tokens."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
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
    print(
        f"[tokens] prompt={prompt_tokens} completion={completion_tokens} "
        f"total={total_tokens} mutation_failed={failed}"
    )
    return {
        "candidates": candidates,
        "reasoning": meta.get("reasoning", ""),
        "confidence": meta.get("confidence", 0.5),
        "mutation_failed": failed,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


@bp.activity_trigger(input_name="payload")
def activity_commit_slice(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    return _plain(
        tensor_engine.commit_tournament(
            payload["layer_name"],
            payload["candidates"],
            payload["target_blob"],
            payload["journal_blob"],
            int(payload.get("journal_capacity", 5)),
        )
    )


@bp.durable_client_input(client_name="client")
@bp.route(route="api/orchestrators/{name}", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
async def start_optimize_loop(req: Request, client):
    """Start one orchestration. The body is the trial budget."""
    try:
        body = await req.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    instance_id = await client.start_new(req.path_params["name"], client_input=body)
    payload = client.create_http_management_payload(instance_id)
    return Response(json.dumps(payload), status_code=202, media_type="application/json")
