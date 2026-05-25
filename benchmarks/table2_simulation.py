"""Reproduce Table 2 on a GPU: scikit-learn vs ThunderSVM vs TorchKM.

Table 2 compares the SVM objective value (equation 1) and end-to-end run time on
Gaussian-cluster synthetic data (``torchkm.data_gen``) across sizes (n, p),
averaged over 50 independent runs.

This script follows the source result notebook exactly for the objective value
and kernel:

  * One RBF bandwidth ``sig = sigest(X_train)`` is drawn per run, and the common
    kernel is ``Kmat = rbf_kernel(X_train, sig)`` = exp(-2*sig*||.||^2).
  * The tuning grid is 50 log-uniform *lambda* values over [1e-3, 1e3], each
    transferred to the LIBSVM C parameterization via C = 1/(2*n*lambda).
  * Baselines are RBF-SVMs fit with ``gamma = sig`` (the notebook's setting; note
    this is half the bandwidth of Kmat) and tuned by 10-fold CV; TorchKM uses the
    CV solver with ``is_exact=0`` and ``rbf_sigma=sig``. Reported **time** is the
    full train-and-tune pipeline -- for the baselines this is the 10-fold CV
    over the whole grid plus the final model fit, all inside the timed region.
  * The **objective** is ``objfun`` = lam*aKa + sum(hinge)/n evaluated on Kmat.
    TorchKM uses its selected lambda* = ulam[best_ind]. The baselines reproduce
    the notebook's regularization weight, which is the *leftover loop variable*
    ``lam`` -- i.e. ulam[-1] = the smallest grid lambda (1e-3) -- not lam_best.
    Set ``--baseline-lambda best`` to instead use each baseline's own lam_best.

ThunderSVM is optional. Install it first (https://github.com/Xtra-Computing/thundersvm),
then either run this script from its python directory (``cd thundersvm/python/``)
or pass ``--thundersvm-path /path/to/thundersvm/python``. If it cannot be
imported the ThunderSVM column is skipped.

Example (paper scale, GPU):
    python benchmarks/table2_simulation.py --repeats 50 --device cuda \
        --thundersvm-path /path/to/thundersvm/python
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from torchkm import data_gen, rbf_kernel, sigest, standardize
from torchkm.estimators import TorchKMSVC

from _common import get_device, lam_grid, mean_se, svm_objective, timed, warmup

# Synthetic-data parameters (paper / source notebook).
NM, MU, RO, NFOLDS = 5, 2.0, 3.0, 10
# (n, p) cells of Table 2.
SIZES = [(10000, 10), (10000, 100), (10000, 1000), (20000, 10), (20000, 100), (20000, 1000)]


def make_split(n: int, p: int, seed: int):
    Xtr, ytr, _ = data_gen(n, NM, p, p // 2, p // 2, MU, RO, seed)
    Xte, yte, _ = data_gen(n // 10, NM, p, p // 2, p // 2, MU, RO, seed)
    return standardize(Xtr), ytr, standardize(Xte), yte


def run_torchkm(Xtr, ytr, sig, Kmat, y_t, device, seed, max_iter, Cs):
    """Fit TorchKM (timed) and return (objfun at its lambda*, time)."""
    clf = TorchKMSVC(
        kernel="rbf", rbf_sigma=sig, Cs=Cs, nC=len(Cs), cv=NFOLDS,
        device=device, random_state=seed, max_iter=max_iter, is_exact=0,
    )
    with timed(device) as t:
        clf.fit(Xtr.numpy(), ytr.numpy())
    lam = 1.0 / (2.0 * Xtr.shape[0] * clf.best_C_)  # = ulam[best_ind]
    alpha = torch.as_tensor(clf.alpha_, dtype=torch.double, device=device)
    return svm_objective(Kmat, y_t, alpha, clf.intercept_, lam), t.dt


def libsvm_obj_time(SVC, Xtr_np, ytr_np, sig, Kmat, y_t, n, ulam, device, baseline_lambda):
    """Tune + fit an RBF-SVM (sklearn or thundersvm) exactly as the notebook does.

    Returns (objfun, time). gamma=sig; lam_best chosen by 10-fold CV accuracy.
    The objective regularization weight follows ``baseline_lambda``:
    'notebook' uses ulam[-1] (the leftover loop variable), 'best' uses lam_best.
    """
    from sklearn.model_selection import cross_val_score

    with timed(device) as t:
        cv = [
            cross_val_score(SVC(kernel="rbf", C=float(1.0 / (2 * n * float(l))), gamma=sig), Xtr_np, ytr_np, cv=NFOLDS).mean()
            for l in ulam
        ]
        lam_best = float(ulam[int(np.argmax(cv))])
        model = SVC(kernel="rbf", C=float(1.0 / (2 * n * lam_best)), gamma=sig).fit(Xtr_np, ytr_np)

    alpha_full = np.zeros(n)
    alpha_full[np.asarray(model.support_)] = np.asarray(model.dual_coef_).ravel()
    alpha = torch.as_tensor(alpha_full, dtype=torch.double, device=device)
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    lam_reg = float(ulam[-1]) if baseline_lambda == "notebook" else lam_best
    return svm_objective(Kmat, y_t, alpha, intercept, lam_reg), t.dt


def load_thundersvm(path):
    if path:
        sys.path.insert(0, path)
    from thundersvm import SVC

    return SVC


def fmt(obj_se, time_mean):
    if obj_se is None:
        return f"{'skipped':>16} {'-':>8}"
    m, se = obj_se
    return f"{m:>7.3f} ({se:>6.3f}) {time_mean:>8.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3, help="runs per cell (paper: 50)")
    ap.add_argument(
        "--sizes", nargs="+", default=None, metavar="N,P",
        help="one or more cells to run, e.g. --sizes 10000,10 (default: full grid)",
    )
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument(
        "--max-iter", type=int, default=100000,
        help="TorchKM solver iterations; the default 1000 under-converges (paper used 1e6)",
    )
    ap.add_argument(
        "--baseline-lambda", choices=["notebook", "best"], default="notebook",
        help="objective reg weight for baselines: 'notebook'=ulam[-1] (exact), 'best'=lam_best",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sklearn", action="store_true", help="scikit-learn is very slow at scale")
    ap.add_argument("--skip-thunder", action="store_true")
    ap.add_argument("--thundersvm-path", default=None, help="path to thundersvm/python")
    args = ap.parse_args()

    sizes = SIZES
    if args.sizes is not None:
        sizes = [tuple(int(v) for v in s.split(",")) for s in args.sizes]

    device = get_device(args.device)
    print(f"device={device}  repeats={args.repeats}  folds={NFOLDS}  grid=50 lambda in [1e-3,1e3]")

    from sklearn.svm import SVC as SkSVC

    ThunderSVC = None
    if not args.skip_thunder:
        try:
            ThunderSVC = load_thundersvm(args.thundersvm_path)
        except ImportError:
            print(
                "ThunderSVM not importable -> skipping that column. Install it "
                "(https://github.com/Xtra-Computing/thundersvm), then `cd thundersvm/python/` "
                "or pass --thundersvm-path /path/to/thundersvm/python."
            )
    print()
    warmup(device)

    header = (
        f"{'n':>7} {'p':>5} | {'scikit-learn obj / t(s)':>25} | "
        f"{'ThunderSVM obj / t(s)':>25} | {'TorchKM obj / t(s)':>25}"
    )
    print(header)
    print("-" * len(header))

    ulam = lam_grid()  # 50 lambda values over [1e-3, 1e3]
    for n, p in sizes:
        Cs = 1.0 / (2.0 * n * ulam)  # transfer lambda grid to the LIBSVM C sequence
        sk, th, tk = ([], []), ([], []), ([], [])  # (objs, times)
        for i in range(args.repeats):
            Xtr, ytr, _, _ = make_split(n, p, args.seed + i)
            sig = sigest(Xtr)
            Kmat = rbf_kernel(Xtr.to(torch.double).to(device), sig)
            y_t = ytr.to(torch.double).to(device)
            Xtr_np, ytr_np = Xtr.numpy(), ytr.numpy()

            o, dt = run_torchkm(Xtr, ytr, sig, Kmat, y_t, device, args.seed + i, args.max_iter, Cs)
            tk[0].append(o)
            tk[1].append(dt)
            if not args.skip_sklearn:
                o, dt = libsvm_obj_time(SkSVC, Xtr_np, ytr_np, sig, Kmat, y_t, n, ulam, device, args.baseline_lambda)
                sk[0].append(o)
                sk[1].append(dt)
            if ThunderSVC is not None:
                o, dt = libsvm_obj_time(ThunderSVC, Xtr_np, ytr_np, sig, Kmat, y_t, n, ulam, device, args.baseline_lambda)
                th[0].append(o)
                th[1].append(dt)

        def cell(pair):
            if not pair[0]:
                return fmt(None, None)
            return fmt(mean_se(pair[0]), mean_se(pair[1])[0])

        print(f"{n:>7} {p:>5} | {cell(sk)} | {cell(th)} | {cell(tk)}")


if __name__ == "__main__":
    main()
