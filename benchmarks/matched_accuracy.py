#!/usr/bin/env python
"""Full versus truncated spectrum at matched, certified accuracy.

This is the experiment in section 7 of EIGENDECOMPOSITION_OPTIONS.md.

Both spectra run torchkm.experimental.SpectralSVMPath, and every path lambda
and every cross-validation fold stops at the same certified duality gap. Their
time and memory therefore compare at equal accuracy. The shipped cvksvm runs
as a reference at its usual tolerances; its path gaps are certified afterwards
with the same certificate.

Each run reports:
- wall time by phase;
- peak GPU memory, from the allocator and from NVML;
- products with an n x n matrix;
- fallback steps;
- the largest certified gaps;
- the selected lambda, its CV error and the test accuracy.
A run too large for the card's memory, or for cuSOLVER's eigh, becomes a row
with the error.

The data is Table 2's simulation (as in profile_gpu.py) or, with --dataset,
one of Q1's real data sets, loaded and split as in q1_full_kernel.py. The
kernel is RBF at the sigest bandwidth. Test predictions are built in row
blocks, so the peak memory counts K and the solver only.

Usage:
  python benchmarks/matched_accuracy.py --n 20000 --gaps 1e-3 1e-4 --out m20k.json
  python benchmarks/matched_accuracy.py --n 20000 --grid hard --gaps 1e-3
  python benchmarks/matched_accuracy.py --n 60000 --solvers truncated --gaps 1e-3
  python benchmarks/matched_accuracy.py --n 20000 --no-cv --gaps 1e-3  # smallest
  python benchmarks/matched_accuracy.py --dataset a9a --data-dir ~/libsvm_data \
      --seed 52 --gaps 1e-3
  python benchmarks/matched_accuracy.py --n 100000 --solvers matrix_free --gaps 1e-3

--solvers matrix_free runs the truncated spectrum on
torchkm.experimental.RBFKernelOperator: K is never stored, and every product
recomputes it in blocks of rows. Alone, it builds no n x n matrix at all, so
its peak memory is its own.

--solvers largen largen_wide run the external large-n proposal (the folder
torchkm_final_product, given by --largen-path) on the same stored float32
kernel, with the shipped solver's KKT tolerance: largen as delivered (one C
value at a time), largen_wide with its wide scheduler (blocks of 16 C values
and their folds share each pass over K). Like the shipped solver, it stops at
its own tolerances, so its path gaps are certified afterwards.
  python benchmarks/matched_accuracy.py --n 20000 --gaps 1e-3 \
      --solvers shipped truncated largen largen_wide \
      --largen-path ~/torchkm_final_product

The lambda grid follows Q1. --grid q1 (the default for the simulation, as in
Q1_sim) runs over 1e3 .. 1e-3, with the shipped solver at KKTeps 1e-3.
--grid hard (the default for a real data set, as in Q1_real) runs over
1e-2 .. 2e-5, with the shipped solver at KKTeps 1e-6.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PeakMemory, env_snapshot, make_folds  # noqa: E402
from _common import load_dataset, synthetic_dataset  # noqa: E402

from torchkm import sigest  # noqa: E402
from torchkm.cvksvm import cvksvm  # noqa: E402
from torchkm.experimental import (  # noqa: E402
    RBFKernelOperator,
    SpectralSVMPath,
    hinge_duality_gap,
)
from torchkm.functions import kernelMult, rbf_kernel  # noqa: E402


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def free(dev):
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


def top_eigenvalue(K, iters=30):
    v = torch.ones(K.shape[0], dtype=K.dtype, device=K.device)
    for _ in range(iters):
        v = K @ v
        v = v / v.norm()
    return 1.05 * float(v @ (K @ v))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset",
        default="sim",
        help="sim (Table 2's simulation at --n, --p) or a Q1 real data set, e.g. a9a",
    )
    ap.add_argument("--data-dir", default=None, help="directory of LIBSVM files")
    ap.add_argument("--n", type=int, default=20000, help="simulation rows")
    ap.add_argument("--p", type=int, default=100, help="simulation features")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--grid",
        choices=["q1", "hard"],
        default=None,
        help="default: q1 for the simulation, hard for a real data set",
    )
    ap.add_argument("--nlam", type=int, default=50)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--no-cv", action="store_true", help="path only")
    ap.add_argument("--gaps", type=float, nargs="+", default=[1e-3, 1e-4])
    ap.add_argument(
        "--solvers",
        nargs="+",
        choices=[
            "shipped",
            "full",
            "truncated",
            "matrix_free",
            "largen",
            "largen_wide",
        ],
        default=["shipped", "full", "truncated"],
        help="matrix_free: the truncated spectrum on a kernel that is never "
        "stored; run it alone to measure its memory (no n x n matrix is built). "
        "largen, largen_wide: the external large-n proposal (--largen-path), "
        "with the shipped solver's KKT tolerance",
    )
    ap.add_argument(
        "--largen-path",
        default=None,
        help="folder of the large-n proposal (torchkm_final_product)",
    )
    ap.add_argument("--rank", type=int, default=400)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()
    largen = [s for s in args.solvers if s.startswith("largen")]
    if largen and (args.largen_path is None or args.no_cv):
        ap.error("--solvers largen needs --largen-path and cross-validation")
    if largen:
        sys.path.insert(0, args.largen_path)
        from largen import DenseKernel, LargeNSVM, SolverOptions
    dev = (
        "cuda"
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    dtype = getattr(torch, args.dtype)

    if args.dataset == "sim":
        name = f"sim_{args.n}x{args.p}"
        d = synthetic_dataset(
            args.n, args.p, args.seed, name=name, n_test=max(args.n // 10, 10)
        )
    else:
        d = load_dataset(args.dataset, args.data_dir, seed=args.seed)
    args.grid = args.grid or ("q1" if args.dataset == "sim" else "hard")
    n = d["n_train"]
    X = torch.from_numpy(d["Xtr"])
    torch.manual_seed(args.seed)
    sig = float(sigest(X))
    y = torch.from_numpy(d["ytr"]).to(dev, dtype)
    Xd = X.to(dev, dtype)
    # the stored kernel, unless every solver is matrix-free
    stored = any(s != "matrix_free" for s in args.solvers)
    t = time.perf_counter()
    K = rbf_kernel(Xd, sig) if stored else None
    sync(dev)
    kernel_seconds = time.perf_counter() - t
    Xte = torch.from_numpy(d["Xte"]).to(dev, dtype)
    yte = torch.from_numpy(d["yte"]).to(dev, dtype)
    unit = Xd.element_size() * n * n
    if args.grid == "q1":
        lams, kkt = np.logspace(3, -3, args.nlam), 1e-3
    else:
        lams, kkt = np.logspace(-2, np.log10(2e-5), args.nlam), 1e-6
    foldid = (
        None
        if args.no_cv
        else torch.from_numpy(make_folds(d["ytr"], args.folds, args.seed))
    )
    # bounds K's top eigenvalue, for certifying the paths of the solvers that
    # stop at their own tolerances
    lmax = top_eigenvalue(K) if "shipped" in args.solvers or largen else None
    print(
        f"{d['name']}: n={n} p={d['p']} n_test={d['n_test']} {args.dtype} on {dev}"
        + (f" ({torch.cuda.get_device_name()})" if dev == "cuda" else "")
        + f"; grid {args.grid}, {args.nlam} lambdas, "
        + ("no CV" if foldid is None else f"{args.folds} folds")
        + f"; one n x n matrix = {unit / 2**30:.2f} GiB; kernel {kernel_seconds:.2f} s",
        flush=True,
    )

    def test_accuracy(alphas, j):
        # the test kernel in row blocks of at most 2**28 entries (1 GiB in
        # float32): never resident during a fit
        rows = max(1, 2**28 // n)
        f = torch.cat(
            [
                kernelMult(Xte[i : i + rows], Xd, sig) @ alphas[1:, j]
                for i in range(0, len(Xte), rows)
            ]
        )
        f = f + alphas[0, j]
        return float((torch.where(f > 0, 1.0, -1.0) == yte).double().mean())

    solvers = list(args.solvers)
    if foldid is None and "shipped" in solvers:
        # cvksvm always cross-validates; a path-only reference would not match
        solvers.remove("shipped")
        print("--no-cv: the shipped solver is left out (it always runs its CV)")
    runs = []
    for solver in solvers:
        uncertified = solver == "shipped" or solver.startswith("largen")
        for gap_tol in [None] if uncertified else args.gaps:
            free(dev)
            row = dict(solver=solver, gap_tol=gap_tol)
            try:
                with PeakMemory(dev) as mem:
                    t = time.perf_counter()
                    if solver == "shipped":
                        m = cvksvm(
                            Kmat=K,
                            y=y,
                            nlam=len(lams),
                            ulam=torch.from_numpy(lams),
                            foldid=(
                                foldid
                                if foldid is not None
                                else torch.ones(n, dtype=torch.int64)
                            ),
                            nfolds=args.folds if foldid is not None else 1,
                            eps=1e-5,
                            maxit=1_000_000,
                            gamma=1e-8,
                            KKTeps=kkt,
                            device=dev,
                            dtype=dtype,
                            # factorize K in place and rebuild it, as the
                            # estimators do (one n x n copy less at the peak)
                            rebuild_kmat=lambda: rbf_kernel(Xd, sig),
                        )
                        with contextlib.redirect_stdout(io.StringIO()):
                            m.fit()
                        K = m.Kmat  # the rebuilt kernel: K's old storage holds U
                    elif solver.startswith("largen"):
                        opts = SolverOptions(
                            maxit=200_000,  # the proposal's pilot setting
                            KKTeps=kkt,
                            KKTeps2=kkt,
                            scheduler=(
                                "wide" if solver == "largen_wide" else "dependency"
                            ),
                        )
                        with warnings.catch_warnings():  # its incomplete-path warning
                            warnings.simplefilter("ignore")
                            model = LargeNSVM(
                                DenseKernel(K, take_ownership=True), options=opts
                            )
                            m = model.fit(
                                y, torch.from_numpy(1.0 / (2.0 * n * lams)), foldid
                            )
                        del model
                    else:
                        m = SpectralSVMPath(
                            (
                                RBFKernelOperator(Xd, sig)
                                if solver == "matrix_free"
                                else K
                            ),
                            y,
                            lams,
                            foldid,
                            spectrum="full" if solver == "full" else "truncated",
                            rank=args.rank,
                            passes=args.passes,
                            gap_tol=gap_tol,
                            seed=args.seed,
                        ).fit()
                    sync(dev)
                    seconds = time.perf_counter() - t
            except (torch.cuda.OutOfMemoryError, torch.linalg.LinAlgError) as err:
                # Too large for the card's memory, or for cuSOLVER's eigh: at
                # n = 60,000 (float32) its workspace-size query fails before
                # anything is allocated (see probe_eigh_size.py).
                msg = str(err).splitlines()[0]
                row.update(error=msg.split(", when calling")[0].split(". Tried")[0])
                runs.append(row)
                print(f"{solver} gap {gap_tol}: {msg}", flush=True)
                if solver == "shipped":  # an in-place attempt may have overwritten K
                    m = None
                    del K
                    K = rbf_kernel(Xd, sig)
                continue
            peak = mem.result
            row.update(
                seconds=seconds,
                peak_allocated_units=(
                    None
                    if peak["torch_max_allocated"] is None
                    else peak["torch_max_allocated"] / unit
                ),
                peak_nvml_gib=(
                    None
                    if (peak["nvml_process_peak"] or peak["nvml_device_peak"]) is None
                    else (peak["nvml_process_peak"] or peak["nvml_device_peak"]) / 2**30
                ),
            )
            if solver == "shipped":
                alphas = m.alpmat
                # CV misclassification from the held-out scores (cvksvm.cv
                # expects labels on the CPU; this stays on the fit's device)
                cverr = (
                    (torch.where(m.pred > 0, 1.0, -1.0) != y[:, None])
                    .double()
                    .mean(0)
                    .cpu()
                    .numpy()
                    if foldid is not None
                    else np.zeros(len(lams))
                )
                gaps = []
                for j, lam in enumerate(lams):
                    g, _, _ = hinge_duality_gap(
                        K,
                        y,
                        alphas[1:, j],
                        float(alphas[0, j]),
                        float(lam),
                        refine=50,
                        lmax=lmax,
                    )
                    gaps.append(float(g))
                row.update(
                    passes=int(m.npass.sum()),
                    cv_passes=int(m.cvnpass.sum()) if foldid is not None else 0,
                    path_gap_max=max(gaps),
                    path_gap_median=float(np.median(gaps)),
                    phases=getattr(m, "timing", None),
                )
            elif solver.startswith("largen"):
                alphas = m.alpmat.to(dev, dtype)
                # NaN marks a lambda whose folds did not all finish
                cverr = np.nan_to_num(m.cv_error.numpy(), nan=np.inf)
                gaps = [
                    float(
                        hinge_duality_gap(
                            K,
                            y,
                            alphas[1:, j],
                            float(alphas[0, j]),
                            float(lam),
                            refine=50,
                            lmax=lmax,
                        )[0]
                    )
                    for j, lam in enumerate(lams)
                ]
                kc = m.kernel_counts
                row.update(
                    reads=dict(total=kc["passes"]),
                    average_width=kc["average_logical_width"],
                    iterations=int(m.npass.sum() + m.cvnpass.sum()),
                    complete=bool(m.complete),
                    path_gap_max=max(gaps),
                    path_gap_median=float(np.median(gaps)),
                )
            else:
                alphas = m.alphas
                cverr = (
                    m.cv_error.numpy() if foldid is not None else np.zeros(len(lams))
                )
                row.update(
                    phases=m.timing,
                    reads=m.counts["reads"],
                    column_iterations=m.counts["column_iterations"],
                    fallbacks=m.counts["fallbacks"],
                    path_iterations=sum(m.path_iterations),
                    cv_iterations=sum(m.cv_iterations) if foldid is not None else 0,
                    path_gap_max=float(m.gaps.max()),
                    path_gap_median=float(m.gaps.median()),
                    fold_gap_max=(
                        float(m.fold_gaps.max()) if foldid is not None else None
                    ),
                    all_certified=bool(
                        m.converged.all() and (foldid is None or m.fold_converged.all())
                    ),
                    spectrum=m.spectrum_info,
                )
            j = int(np.argmin(cverr)) if foldid is not None else len(lams) - 1
            row.update(
                selected_lambda=float(lams[j]),
                cv_error=float(cverr[j]) if foldid is not None else None,
                test_accuracy=test_accuracy(alphas, j),
            )
            runs.append(row)
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k != "spectrum"}, default=float
                ),
                flush=True,
            )
            del m, alphas
            free(dev)

    def fmt(value, spec):
        return "" if value is None else format(value, spec)

    print(
        "\n| solver | gap target | time (s) | peak (n x n units) | peak NVML (GiB) | "
        "n x n reads | path gap max | fold gap max | fallbacks | selected lambda | "
        "CV error | test acc |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        if "error" in r:
            print(f"| {r['solver']} | {r['gap_tol']} | {r['error']} |" + " |" * 9)
            continue
        reads = r.get("reads")
        cells = [
            r["solver"],
            "shipped tolerances" if r["gap_tol"] is None else format(r["gap_tol"], "g"),
            fmt(r["seconds"], ".1f"),
            fmt(r["peak_allocated_units"], ".2f"),
            fmt(r["peak_nvml_gib"], ".1f"),
            "" if reads is None else format(sum(reads.values()), ","),
            fmt(r["path_gap_max"], ".1e"),
            fmt(r.get("fold_gap_max"), ".1e"),
            str(r.get("fallbacks", "")),
            fmt(r["selected_lambda"], ".3g"),
            fmt(r["cv_error"], ".4f"),
            fmt(r["test_accuracy"], ".4f"),
        ]
        print("| " + " | ".join(cells) + " |")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                dict(
                    args=vars(args),
                    kernel_seconds=kernel_seconds,
                    env=env_snapshot(),
                    runs=runs,
                ),
                fh,
                indent=2,
                default=float,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
