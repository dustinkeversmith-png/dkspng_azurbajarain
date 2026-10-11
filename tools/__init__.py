"""Load the compiled native extensions from build/native.

Numeric work and Azurite I/O do not run in Python. Loops over weights,
scores, summaries, signatures, and sockets belong in these extensions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = _ROOT / "build" / "native"
_EXTENSIONS = ("native_engine", "tensor_engine")


def load_extension(name: str):
    """Import one pybind module built by native/CMakeLists.txt."""
    qualified = f"tools.{name}"
    cached = sys.modules.get(qualified)
    if cached is not None:
        return cached
    if not NATIVE_DIR.is_dir():
        raise ImportError(
            f"Native build directory {NATIVE_DIR} does not exist. "
            "Build it with the commands in README.md."
        )
    matches = sorted(NATIVE_DIR.glob(f"{name}*.pyd")) + sorted(NATIVE_DIR.glob(f"{name}*.so"))
    if not matches:
        raise ImportError(
            f"No compiled {name} module in {NATIVE_DIR}. "
            "Build it with the commands in README.md."
        )
    path = matches[-1]
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load native module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def __getattr__(name: str):
    if name in _EXTENSIONS:
        return load_extension(name)
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
