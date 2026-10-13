"""Extract tournament candidates with the model's candidate scheme. C++ accepts each field."""

from __future__ import annotations

from typing import Any, Callable

PARSE_ATTEMPTS = 3


class TournamentParseError(ValueError):
    """The completion did not match the candidate scheme."""


def completion_context(slice_state: dict[str, Any]) -> dict[str, Any]:
    """Shape, active tokens, probe, and scheme taken from the slice the model published."""
    parameters = [dict(row) for row in (slice_state.get("parameters") or [])]
    return {
        "architecture": str(slice_state.get("architecture") or "mlp"),
        "rows": int(slice_state.get("in_dim") or 0),
        "cols": int(slice_state.get("out_dim") or 0),
        "active": {str(row["key"]): str(row["token"]) for row in parameters},
        "scheme": dict(slice_state.get("candidate_scheme") or {}),
        "parameters": parameters,
        "probe": dict(slice_state.get("probe") or {}),
    }


def field_token(state: dict[str, Any], role: str) -> str:
    for row in state.get("parameters") or []:
        if row.get("role") == role:
            return str(row["token"])
    raise KeyError(role)


def extract_tournament(text: str, context: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pull the scheme's slots. Each field is accepted by the encoding that role uses."""
    from tools import tensor_engine

    try:
        parsed = tensor_engine.extract_candidates(
            str(context.get("architecture") or "mlp"),
            text,
            int(context.get("rows") or 0),
            int(context.get("cols") or 0),
            dict(context.get("active") or {}),
            dict(context.get("scheme") or {}),
        )
    except ValueError as exc:
        raise TournamentParseError(str(exc)) from exc
    parsed = dict(parsed)
    candidates = [dict(row) for row in parsed["candidates"]]
    confidence = parsed.get("confidence", 0.5)
    meta = {
        "reasoning": str(parsed.get("reasoning") or ""),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.5,
        "gammas": {str(key): str(value) for key, value in dict(parsed.get("gammas") or {}).items()},
    }
    return candidates, meta


def collect_tournament(
    compile_prompt: Callable[[str], str],
    complete: Callable[[str], tuple[str, int, int, int]],
    commit: Callable[[list[dict[str, str]]], dict[str, Any]],
    context: dict[str, Any],
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile in C++, complete, extract with the candidate scheme, and let C++ score the tokens."""
    prompt_tokens = completion_tokens = total_tokens = 0
    last_error = "no completion"
    for attempt in range(1, PARSE_ATTEMPTS + 1):
        note = "" if attempt == 1 else last_error
        prompt = compile_prompt(note)
        text, prompt_n, completion_n, total_n = complete(prompt)
        prompt_tokens += prompt_n
        completion_tokens += completion_n
        total_tokens += total_n
        try:
            candidates, meta = extract_tournament(text, context)
        except TournamentParseError as exc:
            last_error = str(exc)
            if on_attempt is not None:
                on_attempt(_attempt(attempt, False, last_error, text, [], {}, prompt_n, completion_n, total_n))
            continue
        receipt = commit(candidates)
        status = str(receipt.get("status") or "")
        if status == "rejected_zero_mutation":
            last_error = "candidates did not mutate the active slice"
            if on_attempt is not None:
                on_attempt(_attempt(attempt, False, last_error, text, candidates, meta, prompt_n, completion_n, total_n))
            continue
        if on_attempt is not None:
            on_attempt(_attempt(attempt, True, "", text, candidates, meta, prompt_n, completion_n, total_n))
        return {
            "candidates": candidates,
            "reasoning": meta["reasoning"],
            "confidence": meta["confidence"],
            "meta": meta,
            "receipt": receipt,
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
        "receipt": {},
        "mutation_failed": True,
        "error": last_error,
        "attempts": PARSE_ATTEMPTS,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _attempt(attempt, accepted, error, text, candidates, meta, prompt_n, completion_n, total_n) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "accepted": accepted,
        "error": error,
        "content": text,
        "candidates": candidates,
        "reasoning": meta.get("reasoning", ""),
        "confidence": meta.get("confidence", 0.5),
        "meta": meta,
        "prompt_tokens": prompt_n,
        "completion_tokens": completion_n,
        "total_tokens": total_n,
    }
