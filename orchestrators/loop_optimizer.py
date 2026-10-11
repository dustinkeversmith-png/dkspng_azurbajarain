"""One HTTP start runs several TOTR trials. The model sees only the journal text."""

from __future__ import annotations

import json
import os
from typing import Any

import azure.durable_functions as df
import azure.functions as func
from starlette.requests import Request
from starlette.responses import Response

bp = df.Blueprint()


def synthesis_prompt(payload: dict[str, Any]) -> str:
    """User message only. Last three receipts, no system envelope."""
    history = (payload.get("history") or [])[-3:]
    if history:
        history_lines = "\n".join(
            f"- T{entry['trial']}: loss={float(entry['loss']):.4f}, st={entry.get('status', 'rec')}"
            for entry in history
        )
    else:
        history_lines = f"- Baseline: loss={float(payload['loss']):.4f}"
    return f"""Tuner 2-4-1 ReLU MLP.
Past:
{history_lines}
Best: loss={float(payload['loss']):.4f}, res={float(payload['max_residual']):.4f}
Active:
l1_w: "{payload['l1_w']}"
l2_w: "{payload['l2_w']}"
l1_b: "{payload['l1_b']}"
l2_b: "{payload['l2_b']}"

Rules:
1. If diverged/rejected: set gamma in [0.015, 0.040], ints in [-20, 20].
2. If improving: fine-tune ints by +/- 2 to 5, gamma in [0.025, 0.055].
Output JSON ONLY:
{{"strategy":"fine"|"recovery","l1_w":"r:1|g:<gamma>|A:<2 ints>|B:<4 ints>","l2_w":"r:1|g:<gamma>|A:<4 ints>|B:<1 int>","l1_b":"<4 floats>","l2_b":"<1 float>"}}"""


def candidate_from_text(text: str, fallback: dict[str, Any] | None = None) -> dict[str, str]:
    """Pull the four token strings out of a model completion."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model did not return a JSON object")
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model JSON must be an object")
    aliases = {
        "l1_w": ("l1_w", "l1w", "layer1_w"),
        "l2_w": ("l2_w", "l2w", "layer2_w"),
        "l1_b": ("l1_b", "l1b", "layer1_b", "l1_bias", "b1"),
        "l2_b": ("l2_b", "l2b", "layer2_b", "l2_bias", "b2"),
    }
    candidate: dict[str, str] = {}
    for key, names in aliases.items():
        value = next((parsed.get(name) for name in names if isinstance(parsed.get(name), str)), None)
        if isinstance(value, str) and value.strip():
            candidate[key] = value.strip()
        elif fallback and isinstance(fallback.get(key), str) and str(fallback[key]).strip():
            candidate[key] = str(fallback[key]).strip()
        else:
            raise ValueError(f"model omitted {key}")
    return candidate


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


@bp.orchestration_trigger(context_name="context")
def optimize_network_orchestrator(context: df.DurableOrchestrationContext):
    incoming = context.get_input() or {}
    max_trials = int(incoming.get("max_trials", 5))
    target_loss = float(incoming.get("target_loss", 0.05))
    checkpoint_blob = incoming.get("checkpoint_blob", "sliding_totr.json")
    journal_blob = incoming.get("journal_blob", "sliding_journal.json")
    best_loss = 999.0
    trial = 0
    trials: list[dict[str, Any]] = []

    while trial < max_trials and best_loss > target_loss:
        state = yield context.call_activity(
            "activity_read_state",
            {"checkpoint_blob": checkpoint_blob, "journal_blob": journal_blob},
        )
        current_loss = float(state.get("loss", 1.0))
        best_loss = min(best_loss, current_loss)
        if best_loss <= target_loss:
            break
        guess = yield context.call_activity(
            "activity_llm_synthesize",
            {
                "loss": current_loss,
                "max_residual": state.get("max_residual", 0.0),
                "l1_w": state["l1_w"],
                "l2_w": state["l2_w"],
                "l1_b": state["l1_b"],
                "l2_b": state["l2_b"],
                "history": trials[-3:],
            },
        )
        receipt = yield context.call_activity(
            "activity_commit_candidate",
            {
                "candidate": guess,
                "target_blob": checkpoint_blob,
                "journal_blob": journal_blob,
            },
        )
        trial += 1
        trials.append(
            {
                "trial": trial,
                "loss": receipt.get("loss"),
                "max_residual": receipt.get("max_residual", 0.0),
                "status": receipt.get("status"),
                "checksum": receipt.get("checksum"),
            }
        )
        if receipt.get("status") == "committed_natively":
            best_loss = min(best_loss, float(receipt.get("loss", current_loss)))
        context.set_custom_status({"trial": trial, "best_loss": best_loss})

    return {"status": "completed", "trials": trial, "best_loss": best_loss, "history": trials}


@bp.activity_trigger(input_name="payload")
def activity_read_state(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    raw = tensor_engine.read_totr_state_native(payload["checkpoint_blob"], payload["journal_blob"])
    return _plain(raw)


@bp.activity_trigger(input_name="payload")
def activity_commit_candidate(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    candidate = payload["candidate"]
    raw = tensor_engine.commit_winner_native(
        candidate["l1_w"],
        candidate["l2_w"],
        candidate["l1_b"],
        candidate["l2_b"],
        payload["target_blob"],
        payload["journal_blob"],
    )
    return _plain(raw)


@bp.activity_trigger(input_name="payload")
def activity_llm_synthesize(payload: dict) -> dict[str, str]:
    """Lean direct completion: no tool schemas, reasoning_effort='none'."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    prompt = synthesis_prompt(payload)
    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        reasoning_effort="none",
    )
    usage = response.usage
    if usage is not None:
        print(
            f"[tokens] prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}"
        )
    return candidate_from_text(response.choices[0].message.content or "", fallback=payload)


def slice_prompt(payload: dict[str, Any]) -> str:
    """One layer slice. The static text stays short enough for a 160-token input."""
    short_status = {
        "rejected_divergent": "div",
        "rejected": "rej",
        "committed_natively": "ok",
    }
    history = (payload.get("history") or [])[-3:]
    if history:
        history_lines = "\n".join(
            f"- T{entry['trial']} {float(entry['loss']):.4f} {short_status.get(entry['status'], entry['status'])}"
            for entry in history
        )
    else:
        history_lines = f"- Base {float(payload['current_loss']):.4f}"
    return (
        f"Slice {payload['layer_name']} {payload['in_dim']}x{payload['out_dim']}\n"
        f"{history_lines}\n"
        f"L {float(payload['current_loss']):.4f}\n"
        f"w:{payload['active_w']}\n"
        f"b:{payload['active_b']}\n"
        "div g .015-.035 else .03-.065 ints -25..25\n"
        "json {\"strategy\":\"recovery\",\"w\":\"r:1|g:.02|A:..|B:..\",\"b\":\"..\"}"
    )


def slice_from_text(text: str, fallback: dict[str, Any]) -> dict[str, str]:
    """Keep the two slice strings. A missing field stays on the active slice."""
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    parsed: dict[str, Any] = {}
    if start >= 0 and end > start:
        try:
            loaded = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            parsed = loaded
    weight = parsed.get("w")
    bias = parsed.get("b")
    return {
        "w": weight.strip() if isinstance(weight, str) and weight.strip() else str(fallback["active_w"]),
        "b": bias.strip() if isinstance(bias, str) and bias.strip() else str(fallback["active_b"]),
    }


@bp.orchestration_trigger(context_name="context")
def optimize_modular_orchestrator(context: df.DurableOrchestrationContext):
    incoming = context.get_input() or {}
    max_trials = int(incoming.get("max_trials", 30))
    target_loss = float(incoming.get("target_loss", 0.05))
    checkpoint_blob = incoming.get("checkpoint_blob", "modular_model.json")
    journal_blob = incoming.get("journal_blob", "modular_journal.json")
    model_meta = yield context.call_activity(
        "activity_read_model_meta",
        {"checkpoint_blob": checkpoint_blob, "journal_blob": journal_blob},
    )
    layer_names = list(model_meta.get("layers") or ["layer1", "layer2"])
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
        guess = yield context.call_activity(
            "activity_llm_synthesize_slice",
            {
                "layer_name": active_layer,
                "in_dim": slice_state["in_dim"],
                "out_dim": slice_state["out_dim"],
                "active_w": slice_state["totr_w"],
                "active_b": slice_state["biases"],
                "current_loss": current_loss,
                "max_residual": slice_state.get("max_residual", 0.0),
                "history": trials[-3:],
            },
        )
        receipt = yield context.call_activity(
            "activity_commit_slice",
            {
                "layer_name": active_layer,
                "candidate": guess,
                "target_blob": checkpoint_blob,
                "journal_blob": journal_blob,
            },
        )
        trial += 1
        trials.append(
            {
                "trial": trial,
                "layer": active_layer,
                "loss": receipt.get("loss"),
                "status": receipt.get("status"),
            }
        )
        if receipt.get("status") == "committed_natively":
            best_loss = min(best_loss, float(receipt.get("loss", current_loss)))
        context.set_custom_status({"trial": trial, "layer": active_layer, "best_loss": best_loss})
    return {"status": "completed", "trials": trial, "best_loss": best_loss, "history": trials}


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
def activity_commit_slice(payload: dict) -> dict[str, Any]:
    from tools import tensor_engine

    candidate = payload["candidate"]
    return _plain(
        tensor_engine.commit_slice_native(
            payload["layer_name"],
            candidate["w"],
            candidate["b"],
            payload["target_blob"],
            payload["journal_blob"],
        )
    )


@bp.activity_trigger(input_name="payload")
def activity_llm_synthesize_slice(payload: dict) -> dict[str, str]:
    """Stateless raw AzureOpenAI completion for a single layer slice."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    prompt = slice_prompt(payload)
    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        reasoning_effort="none",
        max_completion_tokens=110,
    )
    usage = response.usage
    if usage is not None:
        print(
            f"[tokens] prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}"
        )
    return slice_from_text(response.choices[0].message.content or "{}", payload)


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
