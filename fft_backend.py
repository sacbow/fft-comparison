"""
fft_backend.py
Unified FFT backend interface with array_backend integration
and plan caching for pyFFTW.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from array_backend import get_array_backend, current_array_backend_name


# ---------------------------------------------------------------------------
# Global thread setting for FFTW
# ---------------------------------------------------------------------------

_FFTW_THREADS = 1  # default value

def set_fftw_threads(n: int):
    """Set global thread count for FFTW backend."""
    global _FFTW_THREADS
    _FFTW_THREADS = max(1, int(n))

def get_fftw_threads() -> int:
    """Get global FFTW thread count."""
    return _FFTW_THREADS


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class FFTBackend:
    """Abstract FFT backend interface."""

    def fft2(self, x: Any) -> Any:
        raise NotImplementedError

    def ifft2(self, x: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# SciPy backend (CPU)
# ---------------------------------------------------------------------------

class SciPyFFTBackend(FFTBackend):
    """FFT backend using scipy.fft (CPU)."""

    def __init__(self):
        import scipy.fft as sp_fft
        self.sp_fft = sp_fft
        self.xp = get_array_backend()

    def fft2(self, x):
        return self.sp_fft.fft2(self.xp.asarray(x))

    def ifft2(self, x):
        return self.sp_fft.ifft2(self.xp.asarray(x))


# ---------------------------------------------------------------------------
# CuPy backend (GPU)
# ---------------------------------------------------------------------------

class CuPyFFTBackend(FFTBackend):
    """FFT backend using cupy.fft (GPU)."""

    def __init__(self):
        self.xp = get_array_backend()

    def fft2(self, x):
        return self.xp.fft.fft2(self.xp.asarray(x))

    def ifft2(self, x):
        return self.xp.fft.ifft2(self.xp.asarray(x))


# ---------------------------------------------------------------------------
# FFTW backend (CPU, with global thread count)
# ---------------------------------------------------------------------------

class FFTWBackend(FFTBackend):
    """
    FFT backend using pyFFTW (a Python wrapper for FFTW).
    Provides 2D FFT and IFFT operations with plan caching.
    Thread count is taken from a global setting (set_fftw_threads()).
    """

    def __init__(self, planner_effort: str = "FFTW_MEASURE"):
        import pyfftw
        self.pyfftw = pyfftw
        self.xp = get_array_backend()
        self.threads = get_fftw_threads()
        self.planner_effort = planner_effort
        self._plans: Dict[
            Tuple[Tuple[int, ...], Any, str],
            Tuple[Any, Any, Any],
        ] = {}

    def _get_plan(self, shape: Tuple[int, ...], dtype: Any, direction: str):
        key = (shape, dtype, direction)
        if key not in self._plans:
            a = self.pyfftw.empty_aligned(shape, dtype=dtype)
            b = self.pyfftw.empty_aligned(shape, dtype=dtype)
            fftw_dir = "FFTW_FORWARD" if direction == "fft" else "FFTW_BACKWARD"
            plan = self.pyfftw.FFTW(
                a,
                b,
                axes=(-2, -1),
                direction=fftw_dir,
                threads=self.threads,
                flags=(self.planner_effort,),
            )
            self._plans[key] = (plan, a, b)
        return self._plans[key]

    def fft2(self, x):
        plan, a, b = self._get_plan(x.shape, x.dtype, "fft")
        a[:] = x
        plan()
        return b

    def ifft2(self, x):
        plan, a, b = self._get_plan(x.shape, x.dtype, "ifft")
        a[:] = x
        plan()
        return b


# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------

_fft_backend: FFTBackend | None = None


def set_fft_backend(name: str):
    """
    Set global FFT backend (scipy, cupy, fftw)
    Ensures compatibility with the current array backend.
    """
    global _fft_backend
    name = name.lower()
    array_name = current_array_backend_name()

    if array_name == "numpy" and name == "cupy":
        raise RuntimeError("Cannot use CuPy FFT backend with NumPy array backend.")
    if array_name == "cupy" and name in ("scipy", "fftw"):
        raise RuntimeError("Cannot use CPU FFT backend (scipy/fftw) with CuPy array backend.")

    if name == "scipy":
        _fft_backend = SciPyFFTBackend()
    elif name == "cupy":
        _fft_backend = CuPyFFTBackend()
    elif name == "fftw":
        _fft_backend = FFTWBackend()
    else:
        raise ValueError(f"Unknown FFT backend: {name}")

    print(f"[fft_backend] FFT backend set to: {name} (array: {array_name}, threads={get_fftw_threads()})")


def get_fft_backend() -> FFTBackend:
    """Return the current FFT backend instance."""
    if _fft_backend is None:
        raise RuntimeError("FFT backend not set. Call set_fft_backend() first.")
    return _fft_backend
