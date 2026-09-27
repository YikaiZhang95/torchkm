#!/usr/bin/env python
"""Where an exact TorchKM SVM fit spends GPU time and memory.

One simulated problem (Table 2's Gaussian mixture, n rows, p features), the
Q1 protocol (50 lambdas, 10 folds). Three measurements:

1. Phases, from TorchKMSVC.fit_timing_ (CUDA-synchronised at each boundary):
   kernel build, eigendecomposition, its error check, lambda path, CV fits,
   with the solver iterations (n_passes_).
2. Memory by step, torch.cuda.max_memory_allocated() after resetting the peak
   before each step, in units of one n x n matrix: the kernel, the
   eigendecomposition (input + eigenvectors + cuSOLVER workspace), what stays
   resident for the solver, and the peak of the whole fit.
3. Operators, torch.profiler over one whole fit: device time per aten
   operator, grouped into eigendecomposition, kernel build (addmm_/exp_),
   matrix-vector and matrix-matrix products (mv/mm/dot), and elementwise work,
   plus the number of device-to-host scalar reads (aten::_local_scalar_dense:
   every .item(), float() or Python comparison on a GPU tensor), each of which
   makes the CPU wait until the GPU has finished all queued work.

Usage:
  python benchmarks/profile_gpu.py --n 20000 --p 100 --dtype float32
  python benchmarks/profile_gpu.py --n 20000 --p 100 --dtype float64 --trace t.json
The trace opens in chrome://tracing or https://ui.perfetto.dev: gaps between
GPU kernels are the host-side waits.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import make_folds, synthetic_dataset  # noqa: E402

from torchkm import sigest  # noqa: E402
from torchkm.estimators import TorchKMSVC  # noqa: E402
from torchkm.functions import rbf_kernel  # noqa: E402


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def op_time(e, dev: str) -> float:
    """Self time of an aggregated profiler event in microseconds."""
    names = (
        ("self_device_time_total", "self_cuda_time_total")
        if dev == "cuda"
        else ("self_cpu_time_total",)
    )
    for name in names:
        v = getattr(e, name, None)
        if v:
            return float(v)
    return 0.0


def category(name: str) -> str:
    if "eigh" in name:
        return "eigendecomposition"
    if name in ("aten::addmm_", "aten::exp_"):
        return "kernel build"
    if name in ("aten::mv", "aten::mm", "aten::dot", "aten::addmv", "aten::bmm"):
        return "matrix-vector / matrix-matrix products"
    return "elementwise, reductions, copies"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--nlam", type=int, default=50)
    ap.add_argument("--lam-max", type=float, default=1e3)
    ap.add_argument("--lam-min", type=float, default=1e-3)
    ap.add_argument("--kkt-eps", type=float, default=1e-3)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trace", default=None, help="write a Chrome trace here")
    args = ap.parse_args()
    dev = (
        "cuda"
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    n, dtype = args.n, getattr(torch, args.dtype)
    size = torch.finfo(dtype).bits // 8
    n2 = n * n * size  # bytes of one n x n matrix

    d = synthetic_dataset(n, args.p, args.seed, n_test=max(n // 10, 10))
    X, y = d["Xtr"].astype(args.dtype), d["ytr"].astype(args.dtype)
    torch.manual_seed(args.seed)
    sig = float(sigest(torch.from_numpy(d["Xtr"])))
    lams = np.logspace(np.log10(args.lam_max), np.log10(args.lam_min), args.nlam)
    foldid = make_folds(d["ytr"], args.folds, args.seed)

    def model(rows: int, folds=foldid) -> TorchKMSVC:
        return TorchKMSVC(
            kernel="rbf",
            rbf_sigma=sig,
            Cs=1.0 / (2.0 * rows * lams),
            nC=len(lams),
            cv=args.folds,
            foldid=folds,
            device=dev,
            tol=args.tol,
            max_iter=args.max_iter,
            KKTeps=args.kkt_eps,
            dtype=args.dtype,
            random_state=args.seed,
        )

    print(
        f"n={n} p={args.p} {args.dtype} on {dev}"
        + (f" ({torch.cuda.get_device_name()})" if dev == "cuda" else "")
        + f"; {args.nlam} lambdas, {args.folds} folds; one n x n matrix = {n2 / 2**30:.2f} GiB"
    )
    # start-up (CUDA context, cuBLAS/cuSOLVER handles), not measured
    w = np.r_[np.flatnonzero(y > 0)[:64], np.flatnonzero(y < 0)[:64]]
    model(len(w), make_folds(d["ytr"][w], args.folds, args.seed)).fit(X[w], y[w])
    sync(dev)

    # 1. phases of one fit
    clf = model(n)
    t0 = time.perf_counter()
    clf.fit(X, y)
    sync(dev)
    wall = time.perf_counter() - t0
    t = clf.fit_timing_
    print(f"\n1. phases (s), whole fit {wall:.2f} s")
    for k in (
        "kernel",
        "eigendecomposition",
        "factorization_error",
        "path",
        "cross_validation",
    ):
        print(f"   {k:20s} {t[k]:9.3f}  {t[k] / wall:6.1%}")
    rest = wall - sum(t[k] for k in t if k != "total")
    print(f"   {'rest':20s} {rest:9.3f}  {rest / wall:6.1%}")
    passes = clf.n_passes_
    print(
        f"   solver iterations: path {passes['path']:,}, CV {passes['cross_validation']:,} "
        f"(fold-iterations); {t['cross_validation'] / (args.folds * args.nlam) * 1e3:.1f} ms "
        "per fold fit"
    )

    # 2. memory by step
    if dev == "cuda":
        peak_fit = clf.peak_gpu_memory_bytes_
        del clf
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        Xt = torch.as_tensor(X, device=dev)
        K = rbf_kernel(Xt, sig)
        sync(dev)
        k_peak = torch.cuda.max_memory_allocated() - base
        torch.cuda.reset_peak_memory_stats()
        e, U = torch.linalg.eigh(K)
        sync(dev)
        e_peak = torch.cuda.max_memory_allocated() - base
        resident = torch.cuda.memory_allocated() - base
        print("\n2. memory (units of one n x n matrix)")
        print(f"   kernel build, peak          {k_peak / n2:5.2f}")
        print(
            f"   eigendecomposition, peak    {e_peak / n2:5.2f}  (kernel + eigenvectors + workspace)"
        )
        print(f"   resident after it (K and U) {resident / n2:5.2f}")
        print(
            f"   whole fit, peak             {peak_fit / n2:5.2f}  = {peak_fit / 2**30:.2f} GiB"
        )
        print(
            f"   reserved by the allocator   {torch.cuda.memory_reserved() / n2:5.2f}"
        )
        del K, U, e, Xt
        torch.cuda.empty_cache()
    else:
        print("\n2. memory: CUDA only")

    # 3. operators
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
    with profile(activities=acts) as prof:
        model(n).fit(X, y)
        sync(dev)
    if args.trace:
        prof.export_chrome_trace(args.trace)
    events = prof.key_averages()
    total = sum(op_time(e, dev) for e in events)
    groups: dict = {}
    for e in events:
        groups[category(e.key)] = groups.get(category(e.key), 0.0) + op_time(e, dev)
    clock = "GPU" if dev == "cuda" else "CPU"
    print(
        f"\n3. {clock} time by operator group (profiler, one fit: {total / 1e6:.2f} s of {clock} time)"
    )
    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"   {g:40s} {v / 1e6:9.3f} s  {v / total:6.1%}")
    print("   top operators:")
    for e in sorted(events, key=lambda e: -op_time(e, dev))[:10]:
        print(f"     {e.key:32s} calls {e.count:9,d}  {op_time(e, dev) / 1e6:9.3f} s")
    reads = sum(e.count for e in events if e.key == "aten::_local_scalar_dense")
    print(f"   device-to-host scalar reads (each waits for the GPU): {reads:,}")


if __name__ == "__main__":
    main()
