"""Memory envelope of TorchKM's exact mode, and where Nyström takes over.

Exact mode eigendecomposes the full n x n kernel matrix, so its peak device
memory grows with n^2 and is the binding constraint on the largest problem a
card can handle. This script measures that envelope directly:

* For each ``n`` in ``--sizes`` it fits the estimator in exact mode on
  synthetic Gaussian-mixture data (``p`` features), recording wall-clock time,
  the PyTorch allocator peak, the NVML process peak, host RSS, and the
  prediction from :func:`torchkm.memory.exact_mode_memory_estimate`. The sweep
  stops at the first CUDA out-of-memory error (recorded as its own row).
* It then fits the Nyström path (``low_rank=True``) at every ``n`` in
  ``--nystrom-sizes``, which continues past the exact-mode ceiling.
* At the end it reports the empirical number of n x n float64 matrices
  resident at the peak, ``(peak - path) / (8 n^2)``, so
  ``torchkm.memory.EXACT_MODE_COPIES`` can be checked against the hardware.

Every estimator shares the eigendecomposition, so ``--estimators svm dwd logit
kqr`` should give the same curve; the default runs SVM only.

Examples
--------
Paper-scale sweep on a GPU (stops when exact mode runs out of memory)::

    python benchmarks/bench_memory_envelope.py --device cuda \\
        --sizes 5000 10000 15000 20000 25000 30000 35000 40000 50000 \\
        --nystrom-sizes 10000 50000 100000 250000 500000 1000000 \\
        --out benchmarks/results/envelope.json

CPU smoke test::

    python benchmarks/bench_memory_envelope.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import numpy as np
import torch

from _common import (
    PeakMemory,
    ResultWriter,
    add_common_args,
    banner,
    c_grid,
    classification_metrics,
    fmt_bytes,
    free_cuda,
    get_device,
    protocol_dict,
    quantile_metrics,
    smoke_settings,
    synthetic_dataset,
    synthetic_regression,
    timed,
    warmup,
)

DEFAULT_SIZES = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000]
DEFAULT_NYSTROM_SIZES = [10000, 50000, 100000, 250000, 500000, 1000000]


def make_estimator(name: str, Cs, args, dev, seed, *, low_rank: bool):
    from torchkm.estimators import TorchKMDWD, TorchKMKQR, TorchKMLogit, TorchKMSVC

    common: Dict[str, Any] = dict(
        kernel="rbf",
        Cs=Cs,
        nC=len(Cs),
        cv=int(args.folds),
        device=dev,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        random_state=int(seed),
    )
    if getattr(args, "kkt_eps", None) is not None:
        common["KKTeps"] = float(args.kkt_eps)
    if getattr(args, "kkt_scaled", False):
        common["kkt_scaled"] = True
    if not low_rank:
        common["eigh_backend"] = getattr(args, "eigh_backend", "auto")
    if low_rank:
        common.update(
            low_rank=True, num_landmarks=int(args.landmarks), nys_k=int(args.rank)
        )
    if name == "kqr":
        return TorchKMKQR(tau=0.5, **common)
    return {"svm": TorchKMSVC, "dwd": TorchKMDWD, "logit": TorchKMLogit}[name](**common)


def one_fit(name: str, n: int, args, dev, seed, *, low_rank: bool) -> Dict[str, Any]:
    from torchkm.memory import exact_mode_memory_estimate

    Cs = c_grid(args.grid_size, args.c_max, args.c_min)
    if name == "kqr":
        data = synthetic_regression(n, args.p, seed)
    else:
        data = synthetic_dataset(n, args.p, seed)
    est = make_estimator(name, Cs, args, dev, seed, low_rank=low_rank)
    rec: Dict[str, Any] = dict(
        estimator=name,
        mode="nystrom" if low_rank else "exact",
        n=int(n),
        p=int(args.p),
        seed=int(seed),
        predicted_exact_bytes=int(
            exact_mode_memory_estimate(
                n, nlam=len(Cs), backend=getattr(args, "eigh_backend", "auto")
            )
        ),
        params=(
            dict(num_landmarks=int(args.landmarks), nys_k=int(args.rank))
            if low_rank
            else {}
        ),
    )
    free_cuda(dev)
    try:
        with PeakMemory(dev) as pm, timed(dev) as t:
            est.fit(data["Xtr"], data["ytr"])
    except torch.cuda.OutOfMemoryError as err:
        rec.update(status="oom", error=str(err)[:600])
        free_cuda(dev)
        return rec
    rec.update(
        status="ok",
        time_s=t.dt,
        memory=pm.result,
        torch_peak_bytes=est.peak_gpu_memory_bytes_,
        eigh_backend_used=getattr(est, "eigh_backend_", None),
        eigh_seconds=getattr(est, "eigh_seconds_", None),
    )
    peak = est.peak_gpu_memory_bytes_
    if peak is not None and n > 0:
        path_bytes = 2 * n * len(Cs) * 8
        rec["empirical_copies"] = (peak - path_bytes) / (8.0 * n * n)
    pred = est.predict(data["Xte"])
    if name == "kqr":
        rec.update(quantile_metrics(data["yte"], pred, 0.5))
    else:
        scores = est.decision_function(data["Xte"])
        rec.update(classification_metrics(data["yte"], scores))
    del est
    free_cuda(dev)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, repeats=1)
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="exact-mode n values",
    )
    ap.add_argument(
        "--nystrom-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_NYSTROM_SIZES,
        help="Nyström n values",
    )
    ap.add_argument("--p", type=int, default=100, help="number of features")
    ap.add_argument(
        "--estimators",
        nargs="+",
        default=["svm"],
        choices=["svm", "dwd", "logit", "kqr"],
    )
    ap.add_argument("--landmarks", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=300)
    ap.add_argument(
        "--continue-after-oom",
        action="store_true",
        help="keep trying larger exact-mode sizes after the first OOM",
    )
    ap.add_argument("--skip-nystrom", action="store_true")
    args = smoke_settings(ap.parse_args())
    if args.smoke:
        args.sizes = [200, 400]
        args.nystrom_sizes = [400, 800]
        args.landmarks, args.rank, args.p = 50, 20, 10

    dev = get_device(args.device)
    banner(
        "TorchKM memory envelope",
        device=dev,
        exact_sizes=args.sizes,
        nystrom_sizes=[] if args.skip_nystrom else args.nystrom_sizes,
        estimators=args.estimators,
        grid=f"{args.grid_size} C values in [{args.c_min}, {args.c_max}]",
        folds=args.folds,
    )
    writer = ResultWriter(
        args.out,
        script="bench_memory_envelope.py",
        args=args,
        protocol=protocol_dict(
            args, data="torchkm.data_gen Gaussian mixture", p=args.p
        ),
    )
    warmup(dev)

    print(
        f"{'est':>5} {'mode':>8} {'n':>8} {'status':>7} {'time':>9} "
        f"{'torch peak':>11} {'nvml proc':>10} {'host rss':>9} {'predicted':>10} {'copies':>7}"
    )
    for name in args.estimators:
        oom_seen = False
        for n in args.sizes:
            for r in range(args.repeats):
                seed = args.seed + r
                if oom_seen and not args.continue_after_oom:
                    rec = dict(
                        estimator=name,
                        mode="exact",
                        n=int(n),
                        seed=seed,
                        status="skipped_after_oom",
                    )
                    writer.add(rec)
                    print(f"{name:>5} {'exact':>8} {n:>8} {'skip':>7}")
                    continue
                rec = one_fit(name, n, args, dev, seed, low_rank=False)
                writer.add(rec)
                _print(rec)
                if rec["status"] == "oom":
                    oom_seen = True
        if args.skip_nystrom:
            continue
        for n in args.nystrom_sizes:
            for r in range(args.repeats):
                rec = one_fit(name, n, args, dev, args.seed + r, low_rank=True)
                writer.add(rec)
                _print(rec)

    # Summary: the empirical copies constant and the envelope per estimator.
    from torchkm.memory import EXACT_MODE_COPIES

    print()
    for name in args.estimators:
        ok = [
            r
            for r in writer.doc["records"]
            if r.get("estimator") == name
            and r.get("mode") == "exact"
            and r.get("status") == "ok"
        ]
        oom = [
            r
            for r in writer.doc["records"]
            if r.get("estimator") == name
            and r.get("mode") == "exact"
            and r.get("status") == "oom"
        ]
        copies = [
            r["empirical_copies"] for r in ok if r.get("empirical_copies") is not None
        ]
        largest = max((r["n"] for r in ok), default=None)
        first_oom = min((r["n"] for r in oom), default=None)
        msg = f"{name}: largest exact-mode n that fit = {largest}; first OOM at n = {first_oom}"
        if copies:
            msg += (
                f"; empirical n x n copies at peak = {np.median(copies):.2f} "
                f"(torchkm.memory.EXACT_MODE_COPIES = {EXACT_MODE_COPIES:g})"
            )
        print(msg)
    if args.out:
        print(f"\nresults written to {args.out}")


def _print(rec: Dict[str, Any]) -> None:
    mem = rec.get("memory") or {}
    copies = rec.get("empirical_copies")
    print(
        f"{rec['estimator']:>5} {rec['mode']:>8} {rec['n']:>8} {rec['status']:>7} "
        f"{rec.get('time_s', float('nan')):>8.1f}s "
        f"{fmt_bytes(rec.get('torch_peak_bytes')):>11} "
        f"{fmt_bytes(mem.get('nvml_process_peak')):>10} "
        f"{fmt_bytes(mem.get('host_rss_peak')):>9} "
        f"{fmt_bytes(rec.get('predicted_exact_bytes')):>10} "
        f"{'-' if copies is None else f'{copies:.2f}':>7}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
