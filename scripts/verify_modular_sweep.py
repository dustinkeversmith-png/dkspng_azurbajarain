"""Measure the expanded tournament prompt on a regressor and an autoencoder."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrators.dynamic_injector import collect_tournament, integer_distance
from orchestrators.run_report import append_attempt, publish_report
from tools import tensor_engine

def _no_matrix(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "weights":
                raise SystemExit(f"weight matrix leaked at {path}.{key}")
            _no_matrix(item, f"{path}.{key}")
    elif isinstance(value, list):
        if value and isinstance(value[0], list):
            raise SystemExit(f"nested matrix leaked at {path}")
        for index, item in enumerate(value):
            _no_matrix(item, f"{path}[{index}]")


def _load_settings() -> None:
    settings = json.loads((ROOT / "local.settings.json").read_text(encoding="utf-8"))["Values"]
    for key, value in settings.items():
        os.environ.setdefault(key, value)


def _section(kind: str, payload: dict, attempts: list[dict], synthesized: dict, receipt: dict, seeded: dict) -> str:
    lines = [
        f"## {kind}",
        "",
        f"- Layer: {payload.get('layer_name')}",
        f"- Seed loss: {seeded.get('loss')}",
        f"- Seed checksum: {seeded.get('checksum')}",
        f"- Active w: `{payload.get('active_w')}`",
        f"- Active b: `{payload.get('active_b')}`",
        f"- Gradient: {payload.get('gradient_hint')}",
        f"- Attempts: {synthesized.get('attempts')}",
        f"- Billed tokens: {synthesized.get('total_tokens')}",
        f"- Status: {receipt.get('status')}",
        f"- Winner: {receipt.get('winner')}",
        f"- Loss: {receipt.get('loss')}",
        f"- Delta: {receipt.get('delta')}",
        f"- Checksum: {receipt.get('checksum')}",
        "",
        "| Candidate | Distance | Status | Loss | w | b |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    scored = {str(row.get("name")): row for row in (receipt.get("candidates") or [])}
    for candidate in synthesized.get("candidates") or []:
        name = str(candidate.get("name"))
        row = scored.get(name) or {}
        distance = integer_distance(str(candidate.get("w") or ""), str(payload.get("active_w") or ""))
        lines.append(
            f"| {name} | {distance} | {row.get('status', '-')} | {row.get('loss', '-')} | `{candidate.get('w')}` | `{candidate.get('b')}` |"
        )
    lines.extend(["", "### Raw replies", ""])
    for attempt in attempts:
        lines.append(f"#### Attempt {attempt.get('attempt')} accepted={attempt.get('accepted')}")
        if attempt.get("error"):
            lines.append(f"Error: {attempt.get('error')}")
        lines.extend(["", "```json", str(attempt.get("content") or ""), "```", ""])
    return "\n".join(lines)


def _check(kind: str, factory) -> tuple[int, str, str]:
    from openai import AzureOpenAI

    blob = f"expand_{kind}_{uuid.uuid4().hex[:8]}.json"
    journal = f"expand_{kind}_{uuid.uuid4().hex[:8]}.json"
    seeded = factory().seed_to_storage(blob, journal, 10)
    _no_matrix(dict(seeded))
    layer_name = list(seeded["layers"])[0]
    slice_state = tensor_engine.read_architecture_slice(blob, journal, layer_name)
    _no_matrix(dict(slice_state))
    payload = {
        "layer_name": layer_name,
        "in_dim": int(slice_state["in_dim"]),
        "out_dim": int(slice_state["out_dim"]),
        "active_w": slice_state["totr_w"],
        "active_b": slice_state["biases"],
        "current_loss": float(slice_state["loss"]),
        "max_residual": float(slice_state["max_residual"]),
        "directives": dict(slice_state["directives"]),
        "telemetry": dict(slice_state["telemetry"]),
        "gradient_hint": slice_state["gradient_hint"],
        "rolling_config": {"visible_history_count": 4, "tracked_metrics": ["loss", "delta", "status"]},
        "training_params": {
            "target_sparsity": 0.20,
            "exploration_mode": "tournament_triad",
            "gamma_bounds": {"fine": [0.02, 0.04], "explore": [0.04, 0.07]},
        },
        "history": [
            {"trial": 1, "loss": 2.0, "delta": 0.4, "status": "rejected"},
            {"trial": 2, "loss": 1.4, "delta": -0.2, "status": "committed_natively", "winner": "c2"},
            {"trial": 3, "loss": 4.0, "delta": 1.1, "status": "rejected_divergent", "winner": "c3"},
            {"trial": 4, "loss": 1.2, "delta": -0.1, "status": "committed_natively", "winner": "c1"},
        ],
    }
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

    attempts: list[dict] = []

    def on_attempt(record: dict) -> None:
        attempts.append(record)
        append_attempt({"source": "verify_modular_sweep", "kind": kind, "layer": payload["layer_name"], **record})

    synthesized = collect_tournament(payload, complete, on_attempt)
    if synthesized["mutation_failed"]:
        publish_report(
            f"verify {kind}",
            _section(kind, payload, attempts, synthesized, {"status": "mutation_failed", "winner": "", "loss": payload["current_loss"], "delta": 0, "checksum": seeded.get("checksum"), "candidates": []}, dict(seeded)),
        )
        raise SystemExit(f"{kind} candidates copied the active slice: {synthesized['error']}")
    candidates = synthesized["candidates"]
    for candidate in candidates:
        if integer_distance(candidate["w"], payload["active_w"]) < 2:
            raise SystemExit(f"{kind} {candidate['name']} did not mutate the active slice")
    receipt = tensor_engine.commit_tournament(layer_name, candidates, blob, journal, 10)
    _no_matrix(dict(receipt))
    scored = [dict(row) for row in receipt["candidates"] if row["status"] == "scored"]
    if scored:
        best = min(scored, key=lambda row: float(row["loss"]))
        if receipt["winner"] != best["name"]:
            raise SystemExit(f"{kind} winner {receipt['winner']} is not the lowest scored candidate")
    if receipt["status"] == "committed_natively" and float(receipt["loss"]) >= float(seeded["loss"]):
        raise SystemExit(f"{kind} committed without lowering the loss")
    if receipt["status"] != "committed_natively" and receipt["checksum"] != seeded["checksum"]:
        raise SystemExit(f"{kind} changed the checkpoint without a commit")
    accepted = attempts[-1] if attempts else {}
    total = int(accepted.get("total_tokens") or synthesized["total_tokens"])
    distances = [integer_distance(candidate["w"], payload["active_w"]) for candidate in candidates]
    print(
        f"{kind} prompt={accepted.get('prompt_tokens')} "
        f"completion={accepted.get('completion_tokens')} total={total} "
        f"attempts={synthesized['attempts']} billed={synthesized['total_tokens']} "
        f"distances={distances} winner={receipt['winner']} status={receipt['status']} "
        f"loss={receipt['loss']} checksum={receipt['checksum']} "
        f"hint={slice_state['gradient_hint']}"
    )
    section = _section(kind, payload, attempts, synthesized, dict(receipt), dict(seeded))
    return total, kind, section


def main() -> None:
    _load_settings()
    sections = ["One slice each for MlpRegressor and BottleneckAutoencoder.", ""]
    for kind, factory in (
        ("mlp_regressor", tensor_engine.MlpRegressor),
        ("bottleneck_ae", tensor_engine.BottleneckAutoencoder),
    ):
        _total, _name, section = _check(kind, factory)
        sections.append(section)
    report = publish_report("verify modular sweep", "\n\n".join(sections))
    print(f"[report] latest {report.parent / 'latest_report.md'}")


if __name__ == "__main__":
    main()
