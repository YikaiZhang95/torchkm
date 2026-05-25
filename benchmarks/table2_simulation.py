"""Reproduce Table 2: scikit-learn vs TorchKM on synthetic data.

Table 2 compares objective values and end-to-end run time on Gaussian-cluster
synthetic data (``torchkm.data_gen``) across sizes (n, p). The paper averages
over 50 independent runs and also reports a ThunderSVM column; ThunderSVM needs
a from-source CUDA build, so this lean script reports TorchKM and the
scikit-learn SVC baseline only.

Objective values use the same functional, equation (1), evaluated at each
method's selected lambda, so they are comparable (see ``_common.svm_objective``).

Example:
    python benchmarks/table2_simulation.py --repeats 3
    python benchmarks/table2_simulation.py --repeats 50 --device cuda   # paper scale
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from torchkm import data_gen, rbf_kernel, sigest, standardize
from torchkm.estimators import TorchKMSVC

from _common import c_grid, get_device, mean_se, svm_objective, timed, warmup

# Synthetic-data parameters (paper / example notebook).
NM, MU, RO, NFOLDS = 5, 2.0, 3.0, 10
# (n, p) cells of Table 2.
SIZES = [(10000, 10), (10000, 100), (10000, 1000), (20000, 10), (20000, 100), (20000, 1000)]


def make_split(n: int, p: int, seed: int):
    Xtr, ytr, _ = data_gen(n, NM, p, p // 2, p // 2, MU, RO, seed)
    Xte, yte, _ = data_gen(n // 10, NM, p, p // 2, p // 2, MU, RO, seed)
    return standardize(Xtr), ytr, standardize(Xte), yte


def run_torchkm(Xtr, ytr, device, seed):
    Cs = c_grid()
    clf = TorchKMSVC(kernel="rbf", Cs=Cs, nC=len(Cs), cv=NFOLDS, device=device, random_state=seed)
    with timed(device) as t:
        clf.fit(Xtr.numpy(), ytr.numpy())

    # Objective on the common training kernel at the selected lambda.
    sigma = clf.kernel_state_["sigma"]
    K = rbf_kernel(Xtr.to(torch.double).to(device), sigma)
    alpha = torch.as_tensor(clf.alpha_, dtype=torch.double, device=device)
    y = ytr.to(torch.double).to(device)
    lam = 1.0 / (2.0 * Xtr.shape[0] * clf.best_C_)
    return svm_objective(K, y, alpha, clf.intercept_, lam), t.dt


def run_sklearn(Xtr, ytr, device):
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC

    Xtr_np, ytr_np = Xtr.numpy(), ytr.numpy()
    n = Xtr.shape[0]
    sigma = sigest(Xtr)
    # torchkm's rbf_kernel uses exp(-2*sigma*||.||^2); scikit-learn uses
    # exp(-gamma*||.||^2). gamma = 2*sigma makes both solve the same kernel SVM.
    gamma = 2.0 * sigma

    Cs = c_grid()
    with timed(device) as t:
        scores = [
            cross_val_score(SVC(kernel="rbf", C=float(c), gamma=gamma), Xtr_np, ytr_np, cv=NFOLDS).mean()
            for c in Cs
        ]
        best_c = float(Cs[int(np.argmax(scores))])
        model = SVC(kernel="rbf", C=best_c, gamma=gamma).fit(Xtr_np, ytr_np)

    alpha_full = np.zeros(n)
    alpha_full[model.support_] = model.dual_coef_.ravel()
    K = rbf_kernel(Xtr.to(torch.double).to(device), sigma)
    alpha = torch.as_tensor(alpha_full, dtype=torch.double, device=device)
    y = ytr.to(torch.double).to(device)
    lam = 1.0 / (2.0 * n * best_c)
    return svm_objective(K, y, alpha, float(model.intercept_[0]), lam), t.dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3, help="runs per cell (paper: 50)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sklearn", action="store_true", help="scikit-learn is very slow at scale")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device={device}  repeats={args.repeats}  folds={NFOLDS}  grid=50 C in [1e-3,1e3]\n")
    warmup(device)

    header = f"{'n':>7} {'p':>5} | {'TorchKM obj':>22} {'time(s)':>9} | {'sklearn obj':>22} {'time(s)':>9}"
    print(header)
    print("-" * len(header))

    for n, p in SIZES:
        tk_obj, tk_t, sk_obj, sk_t = [], [], [], []
        for i in range(args.repeats):
            Xtr, ytr, _, _ = make_split(n, p, args.seed + i)
            o, dt = run_torchkm(Xtr, ytr, device, args.seed + i)
            tk_obj.append(o)
            tk_t.append(dt)
            if not args.skip_sklearn:
                o, dt = run_sklearn(Xtr, ytr, device)
                sk_obj.append(o)
                sk_t.append(dt)

        tk_om, tk_os = mean_se(tk_obj)
        tk_tm, _ = mean_se(tk_t)
        if sk_obj:
            sk_om, sk_os = mean_se(sk_obj)
            sk_tm, _ = mean_se(sk_t)
            sk_cell = f"{sk_om:>9.3f} ({sk_os:>6.3f}) {sk_tm:>9.1f}"
        else:
            sk_cell = f"{'skipped':>22} {'-':>9}"
        print(f"{n:>7} {p:>5} | {tk_om:>9.3f} ({tk_os:>6.3f}) {tk_tm:>9.1f} | {sk_cell}")


if __name__ == "__main__":
    main()
