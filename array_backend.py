"""
A minimal backend manager for NumPy / CuPy switching.
"""

import importlib

# --- default backend is NumPy ---
import numpy as _np

_backend_name = "numpy"
_backend_module = _np


def set_array_backend(name: str):
    """
    Switch global backend to 'numpy' or 'cupy'.
    Falls back to numpy if import or initialization fails.
    """
    global _backend_name, _backend_module

    name = name.lower()
    if name not in ("numpy", "cupy"):
        raise ValueError("Backend must be 'numpy' or 'cupy'")

    if name == "cupy":
        try:
            _backend_module = importlib.import_module("cupy")
            _backend_name = "cupy"
        except ImportError:
            print("[array_backend] CuPy not found. Falling back to NumPy.")
            _backend_module = _np
            _backend_name = "numpy"
    else:
        _backend_module = _np
        _backend_name = "numpy"


def get_array_backend():
    """
    Return the current backend module (np or cp).
    Example:
        xp = get_backend()
        x = xp.arange(10)
    """
    return _backend_module


def current_array_backend_name() -> str:
    """Return the current backend name as string ('numpy' or 'cupy')."""
    return _backend_name
