#!/usr/bin/env python3
"""
benchmark.py
Benchmark for FFT + elementwise + reduction on CPU/GPU backends.

Usage examples:
    python benchmark.py --array numpy --fft fftw --threads 4 --size 128 --niter 1000
    python benchmark.py --array cupy --fft cupy --size 512 --niter 200 --profile
"""

import argparse
import time
import cProfile
import pstats
import io
import numpy as np

from array_backend import set_array_backend, get_array_backend, current_array_backend_name
from fft_backend import set_fft_backend, get_fft_backend


# ---------------------------------------------------------------------
# Computation kernels
# ---------------------------------------------------------------------

def fft_forward_and_backward(fft_backend, a):
    """Perform fft2 -> ifft2 and return the reconstructed array."""
    b = fft_backend.fft2(a)
    return fft_backend.ifft2(b)


def elementwise_distance(xp, a, b):
    """Compute |a - b|^2 elementwise and return a new array."""
    return xp.abs(a - b) ** 2


def reduce_sum(xp, a):
    """Sum all elements and return scalar."""
    return xp.sum(a)


# ---------------------------------------------------------------------
# Benchmark main function
# ---------------------------------------------------------------------

def run_benchmark(array_name, fft_name, threads, size, niter, enable_profile=False):
    """Run combined FFT + elementwise + reduction benchmark."""

    # --- setup backends ---
    set_array_backend(array_name)
    set_fft_backend(fft_name)
    xp = get_array_backend()
    fft = get_fft_backend()

    H = size
    print(f"\n[benchmark] array={array_name}, fft={fft_name}, size={H}x{H}, n_iter={niter}")

    # --- initialize arrays ---
    a = xp.random.random((H, H)) + 1j * xp.random.random((H, H))
    b = xp.empty_like(a)

    def synchronize():
        if current_array_backend_name() == "cupy":
            xp.cuda.Stream.null.synchronize()

    # --- one iteration of workload ---
    def iteration():
        nonlocal b
        b = fft_forward_and_backward(fft, a)
        c = elementwise_distance(xp, a, b)
        s = reduce_sum(xp, c)
        return s

    # --- profile or plain timing ---
    if enable_profile:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(niter):
            _ = iteration()
        synchronize()
        profiler.disable()

        s_io = io.StringIO()
        ps = pstats.Stats(profiler, stream=s_io).sort_stats("cumtime")
        ps.print_stats(20)
        print("\n--- cProfile summary (top 20 cumulative) ---")
        print(s_io.getvalue())

    else:
        t0 = time.perf_counter()
        for _ in range(niter):
            _ = iteration()
        synchronize()
        total_time = time.perf_counter() - t0
        print("\n[Results]")
        print(f"Total execution time : {total_time:.4f} s  ({total_time / niter * 1e3:.3f} ms/iter)")

    # prevent lazy evaluation (ensure reduction result computed)
    s = iteration()
    _ = float(s.get()) if array_name == "cupy" else float(s)


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FFT benchmark with elementwise & reduction.")
    parser.add_argument("--array", type=str, choices=["numpy", "cupy"], required=True,
                        help="Array backend (numpy or cupy)")
    parser.add_argument("--fft", type=str, choices=["scipy", "cupy", "fftw"], required=True,
                        help="FFT backend (must match array backend type)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads for FFTW")
    parser.add_argument("--size", type=int, default=256,
                        help="Matrix size (HxH)")
    parser.add_argument("--niter", type=int, default=1000,
                        help="Number of iterations")
    parser.add_argument("--profile", action="store_true",
                        help="Enable cProfile profiling for the whole loop")
    args = parser.parse_args()

    run_benchmark(
        array_name=args.array,
        fft_name=args.fft,
        threads=args.threads,
        size=args.size,
        niter=args.niter,
        enable_profile=args.profile,
    )


if __name__ == "__main__":
    main()
