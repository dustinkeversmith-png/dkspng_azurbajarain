"""Build one slice prompt from directives, telemetry, and a rolling trial window."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

PARSE_ATTEMPTS = 3

_WEIGHT_KEYS = ("w", "weight", "weights", "totr", "token")
_BIAS_KEYS = ("b", "bias", "biases")
_SLOT_NAMES = {
    "c1": "c1",
    "c2": "c2",
    "c3": "c3",
    "candidate1": "c1",
    "candidate2": "c2",
    "candidate3": "c3",
    "candidate_1": "c1",
    "candidate_2": "c2",
    "candidate_3": "c3",
}


class TournamentParseError(ValueError):
    """The completion did not yield three mutated slice tokens."""


def _metric(entry: dict[str, Any], name: str) -> str:
    if name == "status":
        raw = str(entry.get("status", ""))
        return {"rejected_divergent": "div", "rejected": "rej", "committed_natively": "ok"}.get(raw, raw)
    if name == "winner":
        return str(entry.get("winner", ""))
    value = entry.get(name)
    if value is None and isinstance(entry.get("metrics"), dict):
        value = entry["metrics"].get(name)
    if isinstance(value, float):
        return f"{value:.2f}"
    if value is None:
        return "-"
    return str(value)


def _field(token: str, key: str) -> str:
    match = re.search(rf"(?:^|\|){re.escape(key)}:([^|]*)", token)
    return "" if match is None else match.group(1).strip()


def factor_ints(token: str, key: str) -> list[int]:
    text = _field(token, key).replace(",", " ")
    if not text:
        return []
    codes: list[int] = []
    for part in text.split():
        try:
            codes.append(int(round(float(part))))
        except ValueError:
            return []
    return codes


def clamp_int(code: int, bound: int = 99) -> int:
    """Keep a TOTR code inside the decoder range. 100 becomes 99."""
    return max(-bound, min(bound, code))


def integer_distance(candidate: str, active: str) -> int:
    """How many A/B codes differ. A different length counts as a full mismatch."""
    distance = 0
    for key in ("A", "B"):
        left = factor_ints(candidate, key)
        right = factor_ints(active, key)
        if not left or not right or len(left) != len(right):
            return max(len(left), len(right), 2)
        distance += sum(item != other for item, other in zip(left, right))
    return distance


def build_parameterized_slice_prompt(payload: dict[str, Any], retry_note: str = "") -> str:
    """Structured user message. Three tournament candidates, no tool schema."""
    directives = payload.get("directives") or {}
    training = payload.get("training_params") or {}
    history = payload.get("history") or []

    # 1. Trajectory formatting with velocity / winning step
    if history:
        history_lines = []
        for e in history[-3:]:
            step = e.get("winner_step", "")
            step_str = f" step={step}" if step else ""
            history_lines.append(
                f"- T{e.get('trial')} loss={float(e.get('loss', 0.0)):.4f} "
                f"delta={float(e.get('delta', 0.0)):+.4f} win={e.get('winner', 'none')}{step_str}"
            )
        history_block = "\n".join(history_lines)
    else:
        history_block = f"Base loss: {float(payload.get('current_loss', 0.0)):.4f}"

    active_w = str(payload.get("active_w") or "")
    int_bound = int(training.get("int_bound", 25))
    retry_str = f"\n[CORRECTION REQUIRED: {retry_note}]" if retry_note else ""
    del directives
    roles = _gamma_roles(history)

    return f"""Target: Minimize loss for slice {payload.get('layer_name')} [{payload.get('in_dim')}x{payload.get('out_dim')}].
History Trajectory:
{history_block}
Current Best Loss: {float(payload.get('current_loss', 0.0)):.4f}
Active Baseline: w:"{active_w}" | b:"{payload.get('active_b')}"

Directional Gradient Probe (comma-separated marks, one per coordinate):
{payload.get('gradient_hint', 'A:[] B:[]')}
Actionable Probe Rules:
- '+2' steep ascent: a positive step raised loss a lot. Decrease that coordinate by 6 to 12.
- '+1' gentle ascent: decrease that coordinate by 2 to 5.
- '-2' steep descent: a positive step lowered loss a lot. Increase that coordinate by 6 to 12.
- '-1' gentle descent: increase that coordinate by 2 to 5. A '-1' is a direction, not an empty field.
- '0' is a dead ReLU. When every mark is 0, keep A and B integers unchanged and raise each bias by 0.2 to 0.5.
Integer bound: every A and B code must be inside [-{int_bound}, {int_bound}]. If an active code is already outside that range, move it inside. Never write a code below -99 or above 99.

MANDATORY CANDIDATE ROLES (Never copy active w except when every probe mark is 0. Otherwise mutate at least 2 ints in A and B):
{roles}

Output valid JSON ONLY with this exact schema:
{{
  "reasoning": "<1-2 sentences diagnosing probe signs and justifying step sizes>",
  "confidence": <float 0.1 to 1.0>,
  "c1": {{"w": "r:1|g:<gamma>|A:<ints>|B:<ints>", "b": "<floats>"}},
  "c2": {{"w": "r:1|g:<gamma>|A:<ints>|B:<ints>", "b": "<floats>"}},
  "c3": {{"w": "r:1|g:<gamma>|A:<ints>|B:<ints>", "b": "<floats>"}}
}}{retry_str}"""


def _gamma_roles(history: list[Any]) -> str:
    """Shrink gamma after a rise or a divergence, and settle after a large drop."""
    if history:
        last = history[-1]
        status = str(last.get("status") or "")
        try:
            delta = float(last.get("delta") or 0.0)
        except (TypeError, ValueError):
            delta = 0.0
        if status == "rejected_divergent" or delta > 0.0:
            return (
                "- Previous step diverged or raised loss. Every candidate uses gamma <= 0.015.\n"
                "- c1 gamma 0.005-0.010. c2 gamma 0.008-0.012. c3 gamma 0.010-0.015."
            )
        if delta < -0.1:
            return (
                "- Previous step dropped loss by more than 0.1. Settle instead of exploring.\n"
                "- c1 and c2 gamma 0.01-0.03. c3 gamma 0.02-0.04."
            )
    return (
        "- c1 (Gradient Follower): move opposite +2/+1/-2/-1. Gamma 0.015-0.035.\n"
        "- c2 (Momentum Step): follow the winning step. Gamma 0.030-0.055.\n"
        "- c3 (Basin Explorer): larger step on steep marks. Gamma 0.050-0.080."
    )


def _probe_is_dead(hint: str) -> bool:
    return "+" not in hint and "-" not in hint


def _bias_raised(updated: str, active: str) -> bool:
    new_values = _numbers(updated)
    old_values = _numbers(active)
    if new_values is None or old_values is None or len(new_values) != len(old_values):
        return False
    return any(new >= old + 0.15 for new, old in zip(new_values, old_values))


def _load_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise TournamentParseError("completion has no JSON object")
    try:
        loaded = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise TournamentParseError("completion JSON is incomplete") from exc
    if not isinstance(loaded, dict):
        raise TournamentParseError("completion JSON must be an object")
    return loaded


def _slot_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    return _SLOT_NAMES.get(re.sub(r"[\s\-]+", "", name.strip().lower()), "")


def _lookup_slot(node: dict[str, Any], slot: str) -> Any:
    for key, value in node.items():
        if _slot_name(key) == slot:
            return value
    return None


def _slots_from_list(items: list[Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    positional: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            positional.append(item)
            continue
        named = _slot_name(item.get("name") or item.get("id") or item.get("candidate") or "")
        if named:
            found.setdefault(named, item)
        else:
            positional.append(item)
    if len(found) < 3 and len(positional) >= 3:
        for slot, item in zip(("c1", "c2", "c3"), positional):
            found.setdefault(slot, item)
    return found


def _collect_slots(parsed: dict[str, Any]) -> dict[str, Any]:
    found = {slot: _lookup_slot(parsed, slot) for slot in ("c1", "c2", "c3")}
    if all(value is not None for value in found.values()):
        return found
    for key in ("candidates", "tournament", "hypotheses", "proposals", "result", "output"):
        block = parsed.get(key)
        if isinstance(block, list):
            found.update({slot: item for slot, item in _slots_from_list(block).items() if found.get(slot) is None})
        elif isinstance(block, dict):
            nested = _collect_slots(block)
            for slot, item in nested.items():
                if found.get(slot) is None:
                    found[slot] = item
    if any(value is None for value in found.values()):
        for value in parsed.values():
            if isinstance(value, dict):
                for slot in ("c1", "c2", "c3"):
                    if found.get(slot) is None:
                        found[slot] = _lookup_slot(value, slot)
            elif isinstance(value, list):
                for slot, item in _slots_from_list(value).items():
                    if found.get(slot) is None:
                        found[slot] = item
    return {slot: item for slot, item in found.items() if item is not None}


def _numbers(value: object) -> list[float] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        if not parts:
            return None
        try:
            return [float(part) for part in parts]
        except ValueError:
            return None
    if isinstance(value, list) and value and not isinstance(value[0], (list, dict)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _codes(value: object, expected: int) -> list[int] | None:
    numbers = _numbers(value)
    if numbers is None or len(numbers) != expected:
        return None
    return [clamp_int(int(round(number))) for number in numbers]


def _gamma_value(value: object) -> float | None:
    numbers = _numbers(value)
    if numbers is None or len(numbers) != 1:
        return None
    gamma = numbers[0]
    if gamma <= 0.0 or gamma > 5.0:
        return None
    return gamma


def _clamp_token(token: str) -> str:
    left = factor_ints(token, "A")
    right = factor_ints(token, "B")
    clamped_left = [clamp_int(code) for code in left]
    clamped_right = [clamp_int(code) for code in right]
    if clamped_left == left and clamped_right == right:
        return token.strip()
    gamma = _gamma_value(_field(token, "g"))
    return _pack(0.02 if gamma is None else gamma, clamped_left, clamped_right)


def _pack(gamma: float, left: list[int], right: list[int]) -> str:
    return (
        f"r:1|g:{gamma:.6g}|A:{' '.join(str(code) for code in left)}"
        f"|B:{' '.join(str(code) for code in right)}"
    )


def _weight_token(item: object, rows: int, cols: int) -> str | None:
    if isinstance(item, str) and "|A:" in item and "|B:" in item:
        if len(factor_ints(item, "A")) == rows and len(factor_ints(item, "B")) == cols:
            return _clamp_token(item)
        return None
    if not isinstance(item, dict):
        return None
    for key in _WEIGHT_KEYS:
        token = _weight_token(item.get(key), rows, cols)
        if token:
            return token
    source = item.get("w") if isinstance(item.get("w"), dict) else item
    if not isinstance(source, dict):
        return None
    left = _codes(source.get("A", source.get("a")), rows)
    right = _codes(source.get("B", source.get("b_vec")), cols)
    gamma = _gamma_value(source.get("g", source.get("gamma")))
    if left is None or right is None or gamma is None:
        return None
    return _pack(gamma, left, right)


def _bias_text(item: object, expected: int, fallback: str) -> str:
    sources: list[object] = [item]
    if isinstance(item, dict):
        sources = [item.get(key) for key in _BIAS_KEYS] + [item.get("w") if isinstance(item.get("w"), dict) else None]
    for source in sources:
        numbers = _numbers(source)
        if numbers is not None and len(numbers) == expected:
            if isinstance(source, str):
                return source.strip()
            return " ".join(f"{number:.6g}" for number in numbers)
    parts = _numbers(fallback)
    if parts is not None and len(parts) == expected:
        return fallback.strip()
    raise TournamentParseError("candidate bias has the wrong length")


def parse_tournament(text: str, fallback: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Pull three mutated c1/c2/c3 tokens plus the reasoning scratchpad."""
    parsed = _load_object(text)
    slots = _collect_slots(parsed)
    active_w = str(fallback.get("active_w") or "")
    active_bias = str(fallback.get("active_b") or "")
    rows = len(factor_ints(active_w, "A"))
    cols = len(factor_ints(active_w, "B"))
    if rows == 0 or cols == 0:
        raise TournamentParseError("active slice token has no A/B factors")
    candidates: list[dict[str, str]] = []
    for name in ("c1", "c2", "c3"):
        item = slots.get(name)
        if item is None:
            raise TournamentParseError(f"{name} is missing from response")
        weight = _weight_token(item, rows, cols)
        if not weight:
            raise TournamentParseError(f"{name} has no valid TOTR string with {rows} A ints and {cols} B ints")
        distance = integer_distance(weight, active_w)
        bias = _bias_text(item, cols, active_bias)
        dead = _probe_is_dead(str(fallback.get("gradient_hint") or ""))
        if distance < 2 and not (dead and _bias_raised(bias, active_bias)):
            raise TournamentParseError(f"{name} changes {distance} ints; need at least 2 and must not copy active w")
        candidates.append({
            "name": name,
            "w": weight,
            "b": bias,
            "gamma": _field(weight, "g"),
        })
    confidence = parsed.get("confidence", 0.5)
    meta = {
        "reasoning": str(parsed.get("reasoning") or ""),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.5,
        "gammas": {row["name"]: row["gamma"] for row in candidates},
    }
    return candidates, meta


def collect_tournament(
    payload: dict[str, Any],
    complete: Callable[[str], tuple[str, int, int, int]],
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Call the model up to three times. Identical active-slice copies are not returned."""
    prompt_tokens = completion_tokens = total_tokens = 0
    last_error = "no completion"
    for attempt in range(1, PARSE_ATTEMPTS + 1):
        note = "" if attempt == 1 else f"Retry: {last_error}. Change at least 2 ints in A and B. Never copy w."
        text, prompt_n, completion_n, total_n = complete(build_parameterized_slice_prompt(payload, note))
        prompt_tokens += prompt_n
        completion_tokens += completion_n
        total_tokens += total_n
        try:
            candidates, meta = parse_tournament(text, payload)
        except TournamentParseError as exc:
            last_error = str(exc)
            if on_attempt is not None:
                on_attempt(
                    {
                        "attempt": attempt,
                        "accepted": False,
                        "error": last_error,
                        "content": text,
                        "reasoning": "",
                        "confidence": 0.5,
                        "meta": {},
                        "prompt_tokens": prompt_n,
                        "completion_tokens": completion_n,
                        "total_tokens": total_n,
                    }
                )
            continue
        if on_attempt is not None:
            on_attempt(
                {
                    "attempt": attempt,
                    "accepted": True,
                    "error": "",
                    "content": text,
                    "candidates": candidates,
                    "reasoning": meta["reasoning"],
                    "confidence": meta["confidence"],
                    "meta": meta,
                    "prompt_tokens": prompt_n,
                    "completion_tokens": completion_n,
                    "total_tokens": total_n,
                }
            )
        return {
            "candidates": candidates,
            "reasoning": meta["reasoning"],
            "confidence": meta["confidence"],
            "meta": meta,
            "mutation_failed": False,
            "error": "",
            "attempts": attempt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    return {
        "candidates": [],
        "reasoning": "",
        "confidence": 0.5,
        "meta": {},
        "mutation_failed": True,
        "error": last_error,
        "attempts": PARSE_ATTEMPTS,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
