#!/usr/bin/env python
"""cuML SVC on the whole covtype.binary training set (464,809 rows), on the GPU.

  data     covtype.libsvm.binary.scale (581,012 x 54), labels 1 -> -1 and
           2 -> +1, the stratified 80/20 split of _common.load_dataset (seed 52):
           464,809 train and 116,203 test rows, float32
  kernel   RBF exp(-gamma d^2); by default gamma = 2 sig, sig from sigest on
           the training rows (as Q1), or the values of --gamma
  fits     one fit per --C on all training rows, then test predictions; no CV
  timing   fit and prediction, CUDA synchronised; the CUDA context is warmed up
  memory   NVML peak of the process; the whole device's when the per-process
           reading is unavailable (the GPU otherwise idle)

Each C appends one JSON line to --out as soon as it finishes.

Run:
  python benchmarks/covtype_full_cuml.py --data-dir ~/libsvm_data --C 1 \
      --out revision_results/covtype_full_cuml.jsonl
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
from _common import PeakMemory, classification_metrics, load_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--C", type=float, nargs="+", default=[1.0])
    ap.add_argument(
        "--gamma",
        type=float,
        nargs="+",
        default=None,
        help="RBF gamma values (default: 2 sig from sigest); every C runs at each",
    )
    ap.add_argument("--cache-mb", type=float, default=2000.0, help="cuML kernel cache")
    ap.add_argument("--max-iter", type=int, default=-1, help="cuML SMO cap (-1: none)")
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--out", default="revision_results/covtype_full_cuml.jsonl")
    args = ap.parse_args()

    from cuml.svm import SVC
    from torchkm import sigest

    t = time.perf_counter()
    data = load_dataset("covtype", args.data_dir, seed=args.seed)
    Xtr = np.ascontiguousarray(data["Xtr"], dtype=np.float32)
    Xte = np.ascontiguousarray(data["Xte"], dtype=np.float32)
    ytr, yte = data["ytr"].astype(np.float32), data["yte"]
    torch.manual_seed(args.seed)
    sig = float(sigest(torch.from_numpy(data["Xtr"]).double()))
    n = Xtr.shape[0]
    print(
        f"covtype: n_train={n:,} n_test={Xte.shape[0]:,} p={Xtr.shape[1]} "
        f"sig={sig:.4g} gamma={2 * sig:.4g} (loaded in {time.perf_counter() - t:.0f}s)",
        flush=True,
    )

    # warm-up: CUDA context and cuML start-up, untimed
    SVC(kernel="rbf", C=1.0, gamma=2 * sig).fit(Xtr[:256], ytr[:256])

    gammas = args.gamma or [2 * sig]
    for gamma, C in [(g, c) for g in gammas for c in args.C]:
        svc = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            cache_size=args.cache_mb,
            max_iter=args.max_iter,
        )
        with PeakMemory("cuda") as pm:
            t0 = time.perf_counter()
            svc.fit(Xtr, ytr)
            torch.cuda.synchronize()
            t_fit = time.perf_counter() - t0
            t1 = time.perf_counter()
            scores = np.asarray(svc.decision_function(Xte)).reshape(-1)
            torch.cuda.synchronize()
            t_pred = time.perf_counter() - t1
        print(f"gamma={gamma:g} C={C:g}: fit {t_fit:.1f}s, predict {t_pred:.1f}s", flush=True)
        rec = classification_metrics(yte, scores)
        rec.update(
            C=C,
            lam=1.0 / (2.0 * n * C),
            gamma=gamma,
            n_train=n,
            n_test=int(Xte.shape[0]),
            fit_s=t_fit,
            predict_s=t_pred,
            n_support=int(np.asarray(svc.n_support_).sum()),
            gpu_bytes=pm.result["nvml_process_peak"] or pm.result["nvml_device_peak"],
            nvml_process_peak=pm.result["nvml_process_peak"],
            nvml_device_peak=pm.result["nvml_device_peak"],
            cache_mb=args.cache_mb,
            max_iter=args.max_iter,
        )
        print(
            f"gamma={gamma:g} C={C:g} (lambda={rec['lam']:.3g}): acc={rec['accuracy']:.4f} "
            f"fit={t_fit:.1f}s predict={t_pred:.1f}s SVs={rec['n_support']:,} "
            f"GPU peak={(rec['gpu_bytes'] or 0) / 1e9:.2f} GB",
            flush=True,
        )
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
