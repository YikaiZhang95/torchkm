"""Reproduce Table 2 on a GPU: scikit-learn vs ThunderSVM vs TorchKM.

Table 2 compares the SVM objective value (equation 1) and end-to-end run time on
Gaussian-cluster synthetic data (``torchkm.data_gen``) across sizes (n, p),
averaged over 50 independent runs.

Protocol (matches the paper / source notebook):
  * One RBF bandwidth ``sig = sigest(X_train)`` is drawn per run.
  * The competing libraries are trained as RBF-SVMs with ``gamma = sig`` and a
    50-point C grid (log-uniform over [1e-3, 1e3]), tuned by 10-fold CV.
  * All three solutions are scored with the *same* objective (equation 1) on the
    common kernel ``K = rbf_kernel(X_train, sig)`` so the values are comparable.
    Note ``torchkm.rbf_kernel`` uses exp(-2*sig*||.||^2); to make TorchKM use the
    identical kernel it is fit with ``rbf_sigma=sig``.
  * Reported time is the full train-and-tune pipeline.

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


def torchkm_obj_time(Xtr, ytr, sig, K, y_t, device, seed):
    Cs = c_grid()
    clf = TorchKMSVC(
        kernel="rbf", rbf_sigma=sig, Cs=Cs, nC=len(Cs),
        cv=NFOLDS, device=device, random_state=seed,
    )
    with timed(device) as t:
        clf.fit(Xtr.numpy(), ytr.numpy())
    alpha = torch.as_tensor(clf.alpha_, dtype=torch.double, device=device)
    lam = 1.0 / (2.0 * Xtr.shape[0] * clf.best_C_)
    return svm_objective(K, y_t, alpha, clf.intercept_, lam), t.dt


def libsvm_obj_time(SVC, Xtr_np, ytr_np, sig, K, y_t, n, device):
    """Tune + fit an RBF-SVM via a scikit-learn-style ``SVC`` (sklearn or thundersvm)."""
    from sklearn.model_selection import cross_val_score

    Cs = c_grid()
    with timed(device) as t:
        scores = [
            cross_val_score(SVC(kernel="rbf", C=float(c), gamma=sig), Xtr_np, ytr_np, cv=NFOLDS).mean()
            for c in Cs
        ]
        best_c = float(Cs[int(np.argmax(scores))])
        model = SVC(kernel="rbf", C=best_c, gamma=sig).fit(Xtr_np, ytr_np)

    alpha_full = np.zeros(n)
    alpha_full[np.asarray(model.support_)] = np.asarray(model.dual_coef_).ravel()
    alpha = torch.as_tensor(alpha_full, dtype=torch.double, device=device)
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    lam = 1.0 / (2.0 * n * best_c)
    return svm_objective(K, y_t, alpha, intercept, lam), t.dt


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
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sklearn", action="store_true", help="scikit-learn is very slow at scale")
    ap.add_argument("--skip-thunder", action="store_true")
    ap.add_argument("--thundersvm-path", default=None, help="path to thundersvm/python")
    args = ap.parse_args()

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

    for n, p in SIZES:
        sk, th, tk = ([], []), ([], []), ([], [])  # (objs, times)
        for i in range(args.repeats):
            Xtr, ytr, _, _ = make_split(n, p, args.seed + i)
            sig = sigest(Xtr)
            K = rbf_kernel(Xtr.to(torch.double).to(device), sig)
            y_t = ytr.to(torch.double).to(device)
            Xtr_np, ytr_np = Xtr.numpy(), ytr.numpy()

            o, dt = torchkm_obj_time(Xtr, ytr, sig, K, y_t, device, args.seed + i)
            tk[0].append(o)
            tk[1].append(dt)
            if not args.skip_sklearn:
                o, dt = libsvm_obj_time(SkSVC, Xtr_np, ytr_np, sig, K, y_t, n, device)
                sk[0].append(o)
                sk[1].append(dt)
            if ThunderSVC is not None:
                o, dt = libsvm_obj_time(ThunderSVC, Xtr_np, ytr_np, sig, K, y_t, n, device)
                th[0].append(o)
                th[1].append(dt)

        def cell(pair):
            if not pair[0]:
                return fmt(None, None)
            return fmt(mean_se(pair[0]), mean_se(pair[1])[0])

        print(f"{n:>7} {p:>5} | {cell(sk)} | {cell(th)} | {cell(tk)}")


if __name__ == "__main__":
    main()
