#!/usr/bin/env python
"""Truncated-spectrum TorchKM on the whole covtype.binary training set, one fit.

The setting of covtype_full_cuml.py, for a side-by-side with one cuML SVC fit:

  data     covtype.libsvm.binary.scale, the stratified 80/20 split of
           _common.load_dataset (seed 52): 464,809 train, 116,203 test, float32
  kernel   RBF exp(-gamma d^2), never stored: RBFKernelOperator recomputes it
           in row blocks on every product (the stored kernel would be 864 GB)
  fit      SpectralSVMPath(spectrum="truncated") at lambda = 1 / (2 n C), no
           cross-validation, stopped at the certified relative duality gap
           --gap-tol or after --fit-cap iterations. With --path L > 1 the
           solver first fits L - 1 larger lambdas, log-spaced from
           --lam-start, each warm-starting the next; only the last (the
           target) is scored
  timing   the fit (spectrum construction included) and the test scores
  memory   NVML peak of the process (the device's when that is unavailable)

Progress goes to stdout every --log-every products with K. The result is one
JSON line appended to --out.

Run:
  python benchmarks/covtype_full_trunc.py --data-dir ~/libsvm_data --gamma 32 --C 100
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


class LoggedOperator:
    """K for the solver: counts products and prints progress."""

    def __init__(self, K, every, t0):
        self.K, self.every, self.t0, self.products = K, every, t0, 0
        self.shape, self.dtype, self.device = K.shape, K.dtype, K.device

    def __matmul__(self, B):
        out = self.K @ B
        self.products += 1
        if self.products % self.every == 0:
            torch.cuda.synchronize()
            dt = time.perf_counter() - self.t0
            print(
                f"  {self.products} products, {dt / 60:.1f} min, "
                f"{dt / self.products:.1f} s/product",
                flush=True,
            )
        return out


def test_scores(K, Xte, alpha, b):
    """b + K(Xte, Xtr) alpha, with the training kernel's operator ``K``
    (fused when it is: no block of the test kernel is formed)."""
    return K.cross(Xte, alpha) + b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gamma", type=float, default=32.0)
    ap.add_argument("--C", type=float, default=100.0)
    ap.add_argument("--rank", type=int, default=400)
    ap.add_argument("--gap-tol", type=float, default=1e-3)
    ap.add_argument("--fit-cap", type=int, default=600, help="iteration cap")
    ap.add_argument("--block-gb", type=float, default=1.0, help="kernel row block")
    ap.add_argument(
        "--fused",
        action="store_true",
        help="fused kernel products (RBFKernelOperator(fused=True)) up to 128 columns",
    )
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--path", type=int, default=1, help="lambdas down to the target")
    ap.add_argument("--lam-start", type=float, default=1e-5, help="first path lambda")
    ap.add_argument("--passes", type=int, default=4, help="subspace iterations")
    ap.add_argument("--lanczos-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--out", default="revision_results/covtype_full_trunc.jsonl")
    args = ap.parse_args()

    from torchkm.experimental import RBFKernelOperator, SpectralSVMPath

    data = load_dataset("covtype", args.data_dir, seed=args.seed)
    dev = "cuda"
    Xtr = torch.from_numpy(data["Xtr"]).float().to(dev)
    Xte = torch.from_numpy(data["Xte"]).float().to(dev)
    ytr = torch.as_tensor(data["ytr"], dtype=torch.float32, device=dev)
    n = Xtr.shape[0]
    lam = 1.0 / (2.0 * n * args.C)
    lams = (
        np.logspace(np.log10(args.lam_start), np.log10(lam), args.path).tolist()
        if args.path > 1
        else [lam]
    )
    sigma = args.gamma / 2.0  # exp(-2 sigma d^2) = exp(-gamma d^2)
    print(
        f"covtype: n_train={n:,} n_test={Xte.shape[0]:,} gamma={args.gamma:g} "
        f"C={args.C:g} lambda={lam:.3g} rank={args.rank} gap_tol={args.gap_tol:g} "
        f"fit_cap={args.fit_cap} fused={args.fused} path={args.path} "
        f"passes={args.passes} lanczos_steps={args.lanczos_steps}",
        flush=True,
    )
    torch.linalg.qr(torch.randn(64, 8, device=dev))  # CUDA context, untimed
    torch.cuda.synchronize()

    with PeakMemory(dev) as pm:
        t0 = time.perf_counter()
        K = LoggedOperator(
            RBFKernelOperator(
                Xtr, sigma, block_bytes=int(args.block_gb * 2**30), fused=args.fused
            ),
            args.log_every,
            t0,
        )
        m = SpectralSVMPath(
            K,
            ytr,
            lams,
            spectrum="truncated",
            rank=args.rank,
            gap_tol=args.gap_tol,
            fit_cap=args.fit_cap,
            passes=args.passes,
            lanczos_steps=args.lanczos_steps,
            seed=args.seed,
        ).fit()
        torch.cuda.synchronize()
        t_fit = time.perf_counter() - t0
        t1 = time.perf_counter()
        b, alpha = m.alphas[0, -1], m.alphas[1:, -1]
        scores = test_scores(K.K, Xte, alpha, b).cpu().numpy()
        torch.cuda.synchronize()
        t_pred = time.perf_counter() - t1

    gap = float(m.gaps[-1])
    rec = classification_metrics(data["yte"], scores)
    rec.update(
        method="torchkm_trunc",
        fused=args.fused,
        gamma=args.gamma,
        C=args.C,
        lam=lam,
        n_train=n,
        n_test=int(Xte.shape[0]),
        rank=args.rank,
        gap_tol=args.gap_tol,
        fit_cap=args.fit_cap,
        gap=gap,
        converged=bool(m.converged[-1]),
        iterations=int(sum(m.path_iterations)),
        path=args.path,
        path_lambdas=lams,
        path_iterations=list(m.path_iterations),
        path_gaps=m.gaps.tolist(),
        passes=args.passes,
        lanczos_steps=args.lanczos_steps,
        products=K.products,
        fit_s=t_fit,
        predict_s=t_pred,
        timing=dict(m.timing),
        spectrum_info=m.spectrum_info,
        n_nonzero=int((alpha != 0).sum()),
        gpu_bytes=pm.result["nvml_process_peak"] or pm.result["nvml_device_peak"],
    )
    print(
        f"done: acc={rec['accuracy']:.4f} gap={gap:.2e} converged={rec['converged']} "
        f"iterations={rec['iterations']} products={K.products} "
        f"fit={t_fit / 60:.1f} min predict={t_pred:.1f}s "
        f"GPU peak={(rec['gpu_bytes'] or 0) / 1e9:.2f} GB",
        flush=True,
    )
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    main()
