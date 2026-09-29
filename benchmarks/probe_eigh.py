#!/usr/bin/env python
"""Two GPU measurements for EIGENDECOMPOSITION_OPTIONS.md.

A. Where torch.linalg.eigh's memory goes: each device allocation made inside
   the call (the allocator's trace), in units of one n x n matrix.
B. What the top-r eigenpairs cost instead: randomized subspace iteration
   (q passes, block r + 20), then Rayleigh-Ritz. For each r and q: time,
   extra peak memory, the residual rho = ||K V - V T|| and the largest
   eigenvalue error, both relative to the top eigenvalue.

Kernel: Table 2's simulation (as in profile_gpu.py), RBF with the sigest
bandwidth.

Usage:
  python benchmarks/probe_eigh.py --n 20000 --p 100 --dtype float32
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import synthetic_dataset  # noqa: E402

from torchkm import sigest  # noqa: E402
from torchkm.functions import rbf_kernel  # noqa: E402


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def top_r(K, r: int, q: int, extra: int = 20, seed: int = 0):
    """Top-r eigenpairs of K by q passes of subspace iteration + Rayleigh-Ritz."""
    g = torch.Generator(device=K.device).manual_seed(seed)
    n = K.shape[0]
    Q = torch.randn(n, r + extra, generator=g, device=K.device, dtype=K.dtype)
    Q = torch.linalg.qr(K @ Q).Q
    for _ in range(q):
        Q = torch.linalg.qr(K @ Q).Q
    KQ = K @ Q
    T, S = torch.linalg.eigh(Q.T @ KQ)
    T, S = T[-r:], S[:, -r:]
    V = Q @ S
    rho = torch.linalg.matrix_norm(KQ @ S - V * T, 2)  # K V = (K Q) S
    return V, T, rho


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ranks", type=int, nargs="+", default=[100, 400, 1000])
    ap.add_argument("--passes", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = (
        "cuda"
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    n, dtype = args.n, getattr(torch, args.dtype)

    d = synthetic_dataset(n, args.p, args.seed, n_test=max(n // 10, 10))
    X = torch.from_numpy(d["Xtr"])
    torch.manual_seed(args.seed)
    sig = float(sigest(X))
    K = rbf_kernel(X.to(device=dev, dtype=dtype), sig)
    del X
    unit = K.element_size() * n * n
    print(
        f"n={n} p={args.p} {args.dtype} on {dev}"
        + (f" ({torch.cuda.get_device_name()})" if dev == "cuda" else "")
        + f"; one n x n matrix = {unit / 2**30:.2f} GiB"
    )

    # A. the full eigendecomposition and its allocations
    sync(dev)
    if dev == "cuda":
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=100000)
    t = time.perf_counter()
    e, U = torch.linalg.eigh(K)
    sync(dev)
    t_eigh = time.perf_counter() - t
    print(f"\nA. torch.linalg.eigh: {t_eigh:.1f} s")
    if dev == "cuda":
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        peak = (torch.cuda.max_memory_allocated() - base) / unit
        trace = snap["device_traces"][torch.cuda.current_device()]
        sizes = [
            ev["size"] / unit
            for ev in trace
            if ev["action"] == "alloc" and ev["size"] >= 0.01 * unit
        ]
        print(f"   peak: K (1 unit) + {peak:.2f} units allocated inside eigh")
        print(
            "   allocations inside eigh (units):", ", ".join(f"{s:.2f}" for s in sizes)
        )
    e = e.flip(0)
    del U
    if dev == "cuda":
        torch.cuda.empty_cache()

    # B. top-r eigenpairs only
    print(f"\nB. top-r eigenpairs (e1 = {float(e[0]):.4g})")
    for r in args.ranks:
        for q in args.passes:
            sync(dev)
            if dev == "cuda":
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            V, T, rho = top_r(K, r, q)
            sync(dev)
            dt = time.perf_counter() - t
            extra = (
                f"{(torch.cuda.max_memory_allocated() - base) / unit:5.3f}"
                if dev == "cuda"
                else "  n/a"
            )
            err = float((T.flip(0) - e[:r]).abs().max() / e[0])
            print(
                f"   r={r:5d} q={q}: {dt:7.2f} s ({dt / t_eigh:6.1%} of eigh)  extra peak {extra} units  "
                f"rho/e1 {float(rho) / float(e[0]):.1e}  eigenvalue error/e1 {err:.1e}  "
                f"e_(r+1)/e1 {float(e[r] / e[0]):.1e}"
            )
            del V, T, rho


if __name__ == "__main__":
    main()
