"""Reproduce Table 2 on a GPU: scikit-learn vs ThunderSVM vs TorchKM.

Table 2 compares the SVM objective value (equation 1) and end-to-end run time on
Gaussian-cluster synthetic data (``torchkm.data_gen``) across sizes (n, p),
averaged over 50 independent runs.

Protocol (matches the paper):
  * One RBF bandwidth ``sig = sigest(X_train)`` is drawn per run, defining the
    common kernel ``K = rbf_kernel(X_train, sig)`` = exp(-2*sig*||.||^2).
  * Each library is tuned by 10-fold CV over a 50-point C grid (log-uniform over
    [1e-3, 1e3]); the reported **time** is that full train-and-tune pipeline.
    Baselines use ``gamma = 2*sig`` so their RBF kernel equals K (scikit-learn /
    LIBSVM use exp(-gamma*||.||^2)); TorchKM is fit with ``rbf_sigma=sig``.
  * The reported **objective** is equation (1) evaluated for every method at the
    *same* tuning parameter -- the CV-selected lambda* -- so the three solvers
    are compared on the same optimization problem (paper Section 3). Because C
    and lambda are in one-to-one correspondence via C = 1/(2*n*lambda), the
    baselines are evaluated at C* = 1/(2*n*lambda*). At a fixed lambda and kernel
    the SVM objective is a unique convex minimum, so a converged TorchKM matches
    the scikit-learn / ThunderSVM objective; TorchKM's advantage is the time
    column (integrated CV). The objective lands around 0.5 because, once
    converged, CV selects a regularized lambda* (the generalizing operating
    point) rather than the interpolating end of the grid.

Convergence note: TorchKM needs a large iteration budget (``--max-iter``,
default 100000; the paper used 1e6). With the library default of 1000 the path
under-converges, returns near-zero solutions at moderate C, and CV collapses to
the unregularized end (objective ~ 0). Use a GPU for the paper-scale budget.

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

from _common import c_grid, get_device, mean_se, svm_objective, timed, warmup

# Synthetic-data parameters (paper / source notebook).
NM, MU, RO, NFOLDS = 5, 2.0, 3.0, 10
# (n, p) cells of Table 2.
SIZES = [(10000, 10), (10000, 100), (10000, 1000), (20000, 10), (20000, 100), (20000, 1000)]


def make_split(n: int, p: int, seed: int):
    Xtr, ytr, _ = data_gen(n, NM, p, p // 2, p // 2, MU, RO, seed)
    Xte, yte, _ = data_gen(n // 10, NM, p, p // 2, p // 2, MU, RO, seed)
    return standardize(Xtr), ytr, standardize(Xte), yte


def run_torchkm(Xtr, ytr, sig, K, y_t, device, seed, max_iter):
    """Fit TorchKM (timed). Returns (objective at its lambda*, time, C*)."""
    clf = TorchKMSVC(
        kernel="rbf", rbf_sigma=sig, Cs=c_grid(), nC=50,
        cv=NFOLDS, device=device, random_state=seed, max_iter=max_iter,
    )
    with timed(device) as t:
        clf.fit(Xtr.numpy(), ytr.numpy())
    Cstar = clf.best_C_
    lam = 1.0 / (2.0 * Xtr.shape[0] * Cstar)
    alpha = torch.as_tensor(clf.alpha_, dtype=torch.double, device=device)
    return svm_objective(K, y_t, alpha, clf.intercept_, lam), t.dt, Cstar


def libsvm_time(SVC, Xtr_np, ytr_np, gamma, device):
    """Time the full 10-fold CV grid search + final fit for a libsvm-style SVC."""
    from sklearn.model_selection import cross_val_score

    Cs = c_grid()
    with timed(device) as t:
        scores = [
            cross_val_score(SVC(kernel="rbf", C=float(c), gamma=gamma), Xtr_np, ytr_np, cv=NFOLDS).mean()
            for c in Cs
        ]
        SVC(kernel="rbf", C=float(Cs[int(np.argmax(scores))]), gamma=gamma).fit(Xtr_np, ytr_np)
    return t.dt


def libsvm_obj(SVC, Xtr_np, ytr_np, gamma, Cstar, K, y_t, n, device):
    """Objective (equation 1) of a libsvm-style SVC fit at the common C*."""
    model = SVC(kernel="rbf", C=Cstar, gamma=gamma).fit(Xtr_np, ytr_np)
    alpha_full = np.zeros(n)
    alpha_full[np.asarray(model.support_)] = np.asarray(model.dual_coef_).ravel()
    alpha = torch.as_tensor(alpha_full, dtype=torch.double, device=device)
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    lam = 1.0 / (2.0 * n * Cstar)
    return svm_objective(K, y_t, alpha, intercept, lam)


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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sklearn", action="store_true", help="scikit-learn is very slow at scale")
    ap.add_argument("--skip-thunder", action="store_true")
    ap.add_argument("--thundersvm-path", default=None, help="path to thundersvm/python")
    args = ap.parse_args()

    sizes = SIZES
    if args.sizes is not None:
        sizes = [tuple(int(v) for v in s.split(",")) for s in args.sizes]

    device = get_device(args.device)
    print(f"device={device}  repeats={args.repeats}  folds={NFOLDS}  grid=50 C in [1e-3,1e3]")

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

    for n, p in sizes:
        sk, th, tk = ([], []), ([], []), ([], [])  # (objs, times)
        for i in range(args.repeats):
            Xtr, ytr, _, _ = make_split(n, p, args.seed + i)
            sig = sigest(Xtr)
            gamma = 2.0 * sig  # match torchkm's exp(-2*sig*||.||^2) kernel
            K = rbf_kernel(Xtr.to(torch.double).to(device), sig)
            y_t = ytr.to(torch.double).to(device)
            Xtr_np, ytr_np = Xtr.numpy(), ytr.numpy()

            # TorchKM defines the common tuning parameter C* (= its CV selection).
            o, dt, Cstar = run_torchkm(Xtr, ytr, sig, K, y_t, device, args.seed + i, args.max_iter)
            tk[0].append(o)
            tk[1].append(dt)
            if not args.skip_sklearn:
                sk[1].append(libsvm_time(SkSVC, Xtr_np, ytr_np, gamma, device))
                sk[0].append(libsvm_obj(SkSVC, Xtr_np, ytr_np, gamma, Cstar, K, y_t, n, device))
            if ThunderSVC is not None:
                th[1].append(libsvm_time(ThunderSVC, Xtr_np, ytr_np, gamma, device))
                th[0].append(libsvm_obj(ThunderSVC, Xtr_np, ytr_np, gamma, Cstar, K, y_t, n, device))

        def cell(pair):
            if not pair[0]:
                return fmt(None, None)
            return fmt(mean_se(pair[0]), mean_se(pair[1])[0])

        print(f"{n:>7} {p:>5} | {cell(sk)} | {cell(th)} | {cell(tk)}")


if __name__ == "__main__":
    main()
