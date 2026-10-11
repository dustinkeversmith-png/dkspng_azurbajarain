"""Model state shared by the Python tools."""

from __future__ import annotations

from typing import Any, Protocol


class ModelState(Protocol):
    def layers(self) -> dict[str, Any]:
        """Return layer names mapped to weight and bias matrices."""

    def parameter_count(self) -> int:
        """Return the number of stored weights and biases."""
