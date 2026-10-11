"""Decorators that stay optional when the Functions runtime is not installed."""

from __future__ import annotations

try:
    from azure_functions_agents import tool, workflow_tool
except ImportError:

    def tool(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

    def workflow_tool(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

__all__ = ["tool", "workflow_tool"]
