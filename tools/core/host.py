"""Register class methods with the Functions host.

This walk only discovers classes. It does not score, reduce, or transform tensors.
That work stays in the compiled C++ extensions.


The runtime only scans modules that sit directly in tools/. These classes live in
subpackages, so function_app.py asks this module for the workflow tools instead.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil


def argument_object(value):
    """Turn a tool payload into a dict. A bad payload becomes an empty dict."""
    if isinstance(value, dict):
        inner = value.get("args")
        if isinstance(inner, (dict, str)):
            payload = argument_object(inner)
            for key, item in value.items():
                if key != "args":
                    payload[key] = item
            return payload
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return argument_object(parsed)
    body = text[1:-1] if text.startswith("{") and text.endswith("}") else text
    parts = [part.strip().strip("\"'") for part in body.split(",") if part.strip()]
    if len(parts) >= 2 and len(parts) % 2 == 0:
        return {parts[index]: parts[index + 1] for index in range(0, len(parts), 2)}
    return {}


_TOOL_FIELDS = {
    "read_totr_state_native": ("checkpoint_blob", "journal_blob"),
    "commit_winner_native": ("l1_w", "l2_w", "l1_b", "l2_b", "target_blob", "journal_blob"),
}


def project_tools():
    from azure_functions_agents._function_tool import WorkflowTool, tool, workflow_tool
    from azure_functions_agents.discovery.tools import ProjectTools

    import tools

    user_tools = []
    workflow_tools = []
    failed_loads: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for info in pkgutil.walk_packages(tools.__path__, prefix="tools."):
        try:
            module = importlib.import_module(info.name)
        except Exception as exc:
            failed_loads.append((info.name, f"{type(exc).__name__}: {exc}"))
            continue
        for value in vars(module).values():
            if not isinstance(value, type) or value.__module__ != module.__name__:
                continue
            specs = getattr(value, "HOST_TOOLS", None)
            if not specs:
                continue
            instance = value()
            for name, description in specs:
                key = (value.__name__, name)
                if key in seen:
                    continue
                seen.add(key)
                method = getattr(instance, name)

                def adapter(*args, _method=method, **kwargs):
                    if len(args) == 1 and not kwargs:
                        return _method(argument_object(args[0]))
                    payload = argument_object(args[0]) if args else {}
                    payload.update(kwargs)
                    return _method(payload)

                adapter.__name__ = name
                adapter.__doc__ = method.__doc__ or description
                fields = _TOOL_FIELDS.get(name)
                if fields:
                    adapter.__signature__ = inspect.Signature(
                        [
                            inspect.Parameter(
                                field,
                                inspect.Parameter.KEYWORD_ONLY,
                                default="",
                                annotation=str,
                            )
                            for field in fields
                        ]
                    )
                marked = workflow_tool(name=name, description=description)(adapter)
                user_tools.append(tool(name=name, description=description)(marked))
                workflow_tools.append(
                    WorkflowTool(
                        name=name,
                        description=description,
                        handler=adapter,
                        public=True,
                    )
                )

    return ProjectTools(
        user_tools=user_tools,
        workflow_tools=workflow_tools,
        failed_loads=failed_loads,
    )
