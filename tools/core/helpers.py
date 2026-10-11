"""Numeric checks. The walks live in the compiled tensor engine."""

from __future__ import annotations

import json
from typing import Any


def _engine():
    from tools import tensor_engine

    return tensor_engine


def matrix_or_empty(value: Any, label: str):
    return _engine().coerce_matrix(value, label, True)


def vector_or_empty(value: Any, label: str):
    return _engine().coerce_vector(value, label, True)


def require_numbers(value: Any, label: str):
    return _engine().coerce_vector(value, label, False)


def require_matrix(value: Any, label: str):
    return _engine().coerce_matrix(value, label, False)


def json_object_from_text(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("expected a JSON object in the specialist text")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("specialist JSON must be an object")
    return parsed
