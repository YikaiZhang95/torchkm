#!/usr/bin/env python
"""10-fold CV over 50 lambdas on the whole covtype.binary training set:
truncated-spectrum TorchKM (matrix-free kernel) against cuML SVC.

  data     covtype.libsvm.binary.scale, the stratified 80/20 split of
           _common.load_dataset (seed 52): 464,809 train, 116,203 test, float32
  folds    10 stratified folds (_common.make_folds, seed 52), shared
  kernel   RBF exp(-gamma d^2), gamma 32 (the best of the single-fit sweep)
  grid     50 log-spaced lambdas from --lam-max down to 1 / (2 n C*), n the
           training rows and C* = --c-star (100, cuML's best single fit), so
           that value is the last grid point. Both methods use
           C = 1 / (2 n lambda) with n the rows of each fit
  torchkm  SpectralSVMPath(spectrum="truncated") on RBFKernelOperator(fused=True)
           (the 864 GB kernel is never stored), wide blocks of --block lambdas
           x (10 folds + whole data), at most --fit-cap iterations per lambda
           of a block (fit_cap x block per block); the whole-data fit at the
           selected lambda is part of the path. Fits that reach the cap are
           reported with their certified gaps, not dropped
  cuml     cuml.svm.SVC: one fit per (lambda, fold) on the other nine folds,
           scored on the held-out fold, then one fit on all training rows at
           the selected lambda. Resumable: each finished lambda is saved
  select   the lambda with the highest CV accuracy (the first, i.e. largest,
           on ties), for both
  timing   everything from the kernel setup to the test scores, CUDA
           synchronised; cuML's is the sum over its saved lambdas plus the
           final fit
  memory   NVML peak (the device's when the per-process reading is missing)

Run (one method per call; results in revision_results/covtype_full_cv_<method>.json):
  python benchmarks/covtype_full_cv.py --method torchkm --data-dir ~/libsvm_data
  python benchmarks/covtype_full_cv.py --method cuml --data-dir ~/libsvm_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    PeakMemory,
    classification_metrics,
    load_dataset,
    make_folds,
)
from covtype_full_trunc import LoggedOperator, test_scores  # noqa: E402


def peak(pm):
    return pm.result["nvml_process_peak"] or pm.result["nvml_device_peak"]


def run_torchkm(args, data, Xtr, Xte, ytr, foldid, lams):
    from torchkm.experimental import RBFKernelOperator, SpectralSVMPath

    sigma = args.gamma / 2.0  # exp(-2 sigma d^2) = exp(-gamma d^2)
    dev = Xtr.device
    with PeakMemory(str(dev)) as pm:
        t0 = time.perf_counter()
        K = LoggedOperator(RBFKernelOperator(Xtr, sigma, fused=True), args.log_every, t0)
        m = SpectralSVMPath(
            K,
            ytr,
            lams,
            torch.as_tensor(foldid, device=dev),
            spectrum="truncated",
            rank=args.rank,
            gap_tol=args.gap_tol,
            fit_cap=args.fit_cap,
            block=args.block,
            seed=args.seed,
        ).fit()
        torch.cuda.synchronize()
        t_fit = time.perf_counter() - t0
        cv_acc = 1.0 - m.cv_error.numpy()
        best = int(np.argmax(cv_acc))
        b, alpha = m.alphas[0, best], m.alphas[1:, best]
        scores = test_scores(K.K, Xte, alpha, b).cpu().numpy()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    rec = classification_metrics(data["yte"], scores)
    rec.update(
        method="torchkm_trunc",
        time_s=dt,
        fit_s=t_fit,
        predict_s=dt - t_fit,
        timing=dict(m.timing),
        products=K.products,
        cv_curve=cv_acc.tolist(),
        selected=float(lams[best]),
        best_index=best,
        cv_accuracy=float(cv_acc[best]),
        gaps=m.gaps.tolist(),
        converged=m.converged.tolist(),
        fold_converged_frac=m.fold_converged.double().mean(0).tolist(),
        fold_gap_max=np.nanmax(m.fold_gaps.numpy(), axis=0).tolist(),
        selected_certified=bool(m.converged[best] and m.fold_converged[:, best].all()),
        spectrum_info=m.spectrum_info,
        params=dict(rank=args.rank, gap_tol=args.gap_tol, fit_cap=args.fit_cap,
                    block=args.block, fused=True),
        gpu_bytes=peak(pm),
    )
    return rec


def run_cuml(args, data, Xtr, Xte, ytr, foldid, lams, progress_path):
    from cuml.svm import SVC

    done = {}
    if os.path.exists(progress_path):
        with open(progress_path) as fh:
            for line in fh:
                r = json.loads(line)
                done[r["index"]] = r
        print(f"resuming: {len(done)} lambdas done", flush=True)
    SVC(kernel="rbf", C=1.0, gamma=args.gamma).fit(Xtr[:256], ytr[:256])  # warm-up
    gpu = []
    for j, lam in enumerate(lams):
        if j in done:
            continue
        with PeakMemory("cuda") as pm:
            t0 = time.perf_counter()
            correct = 0
            for k in np.unique(foldid):
                tr, va = foldid != k, foldid == k
                svc = SVC(
                    kernel="rbf",
                    C=1.0 / (2.0 * int(tr.sum()) * lam),
                    gamma=args.gamma,
                    cache_size=args.cache_mb,
                )
                svc.fit(Xtr[tr], ytr[tr])
                pred = np.asarray(svc.predict(Xtr[va])).reshape(-1)
                correct += int((np.where(pred > 0, 1.0, -1.0) == ytr[va]).sum())
            dt = time.perf_counter() - t0
        r = dict(index=j, lam=float(lam), cv_accuracy=correct / len(ytr), seconds=dt,
                 gpu_bytes=peak(pm))
        print(f"  lambda {j + 1}/{len(lams)} = {lam:.3g}: CV acc {r['cv_accuracy']:.4f}, "
              f"{dt:.0f}s", flush=True)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        done[j] = r
    cv_acc = np.array([done[j]["cv_accuracy"] for j in range(len(lams))])
    best = int(np.argmax(cv_acc))
    with PeakMemory("cuda") as pm:
        t0 = time.perf_counter()
        svc = SVC(kernel="rbf", C=1.0 / (2.0 * len(ytr) * lams[best]),
                  gamma=args.gamma, cache_size=args.cache_mb)
        svc.fit(Xtr, ytr)
        scores = np.asarray(svc.decision_function(Xte)).reshape(-1)
        t_final = time.perf_counter() - t0
    cv_time = sum(done[j]["seconds"] for j in range(len(lams)))
    rec = classification_metrics(data["yte"], scores)
    rec.update(
        method="cuml",
        time_s=cv_time + t_final,
        cv_s=cv_time,
        final_s=t_final,
        cv_curve=cv_acc.tolist(),
        selected=float(lams[best]),
        best_index=best,
        cv_accuracy=float(cv_acc[best]),
        params=dict(cache_mb=args.cache_mb),
        gpu_bytes=max([peak(pm)] + [done[j]["gpu_bytes"] or 0 for j in done]),
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["torchkm", "cuml"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gamma", type=float, default=32.0)
    ap.add_argument("--lam-max", type=float, default=1e-3)
    ap.add_argument("--c-star", type=float, default=100.0)
    ap.add_argument("--grid-size", type=int, default=50)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--rank", type=int, default=400)
    ap.add_argument("--gap-tol", type=float, default=1e-3)
    ap.add_argument("--fit-cap", type=int, default=40, help="iterations per lambda")
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--cache-mb", type=float, default=2000.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out-dir", default="revision_results")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    args = ap.parse_args()

    data = load_dataset("covtype", args.data_dir, seed=args.seed)
    n = data["Xtr"].shape[0]
    lams = np.logspace(
        np.log10(args.lam_max), np.log10(1.0 / (2.0 * n * args.c_star)), args.grid_size
    ).tolist()
    foldid = make_folds(data["ytr"], args.folds, args.seed)
    print(
        f"covtype: n_train={n:,} n_test={data['Xte'].shape[0]:,} gamma={args.gamma:g} "
        f"lambda {lams[0]:.3g} .. {lams[-1]:.4g} ({len(lams)}), {args.folds} folds, "
        f"method={args.method}",
        flush=True,
    )
    out = os.path.join(args.out_dir, f"covtype_full_cv_{args.method}{args.tag}.json")
    if args.method == "torchkm":
        Xtr = torch.from_numpy(data["Xtr"]).float().cuda()
        Xte = torch.from_numpy(data["Xte"]).float().cuda()
        ytr = torch.as_tensor(data["ytr"], dtype=torch.float32, device="cuda")
        torch.linalg.qr(torch.randn(64, 8, device="cuda"))  # CUDA context, untimed
        rec = run_torchkm(args, data, Xtr, Xte, ytr, foldid, lams)
    else:
        Xtr = np.ascontiguousarray(data["Xtr"], dtype=np.float32)
        Xte = np.ascontiguousarray(data["Xte"], dtype=np.float32)
        ytr = data["ytr"].astype(np.float32)
        rec = run_cuml(args, data, Xtr, Xte, ytr, foldid, lams,
                       out.replace(".json", "_progress.jsonl"))
    rec.update(grid_lambda=lams, gamma=args.gamma, n_train=n, folds=args.folds,
               seed=args.seed)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(
        f"done: {rec['method']} time={rec['time_s'] / 60:.1f} min "
        f"selected lambda={rec['selected']:.3g} CV acc={rec['cv_accuracy']:.4f} "
        f"test acc={rec['accuracy']:.4f} GPU peak={(rec['gpu_bytes'] or 0) / 1e9:.2f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
