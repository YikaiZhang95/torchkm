"""Reproduce Table 2 on a GPU: scikit-learn vs ThunderSVM vs TorchKM.

Table 2 of the submitted paper compares the SVM objective value (equation 1)
and end-to-end run time on Gaussian-cluster synthetic data (``torchkm.data_gen``)
across sizes (n, p), averaged over 50 independent runs. The revision reports
**test accuracy and AUC** alongside the time (the reviewer's point: users pick
a library by its predictions, not its training objective) and **peak GPU
memory**; the objective comparison moves to ``bench_solver_quality.py``, which
evaluates every solver at the same fixed lambda.

This script keeps the source notebook's protocol for the objective column:

  * One RBF bandwidth ``sig = sigest(X_train)`` is drawn per run, and the common
    kernel is ``Kmat = rbf_kernel(X_train, sig)`` = exp(-2*sig*||.||^2).
  * The tuning grid is 50 log-uniform *lambda* values over [1e-3, 1e3], each
    transferred to the LIBSVM C parameterization via C = 1/(2*n*lambda).
  * Baselines are RBF-SVMs fit with ``gamma = sig`` (the notebook's setting; note
    this is half the bandwidth of Kmat; pass ``--matched-kernel`` for
    ``gamma = 2*sig``, the same kernel as TorchKM) and tuned by 10-fold CV;
    TorchKM uses ``rbf_sigma=sig`` with ``is_exact=0`` by default (``--exact`` for
    is_exact=1). Reported **time** is the full train-and-tune pipeline: for the
    baselines the 10-fold CV over the whole grid plus the final model fit.
  * The **objective** is ``objfun`` = lam*aKa + sum(hinge)/n evaluated on Kmat.
    TorchKM uses its selected lambda* = ulam[best_ind]. The baselines reproduce
    the notebook's regularization weight, which is the *leftover loop variable*
    ``lam`` -- i.e. ulam[-1] = the smallest grid lambda (1e-3) -- not lam_best.
    ``--baseline-lambda best`` uses each baseline's own lam_best instead. Neither
    is a like-for-like solver comparison; use ``bench_solver_quality.py`` for
    that.

ThunderSVM is optional. Install it first (https://github.com/Xtra-Computing/thundersvm),
then either run this script from its python directory (``cd thundersvm/python/``)
or pass ``--thundersvm-path /path/to/thundersvm/python``. If it cannot be
imported the ThunderSVM column is skipped.

Example (paper scale, GPU):
    python benchmarks/table2_simulation.py --repeats 50 --device cuda \\
        --thundersvm-path /path/to/thundersvm/python --out benchmarks/results/table2.json
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from torchkm import data_gen, rbf_kernel, sigest, standardize
from torchkm.estimators import TorchKMSVC

from _common import (
    PeakMemory,
    ResultWriter,
    classification_metrics,
    fmt_bytes,
    get_device,
    lam_grid,
    mean_se,
    svm_objective,
    timed,
    warmup,
)

# Synthetic-data parameters (paper / source notebook).
NM, MU, RO, NFOLDS = 5, 2.0, 3.0, 10
# (n, p) cells of Table 2.
SIZES = [
    (10000, 10),
    (10000, 100),
    (10000, 1000),
    (20000, 10),
    (20000, 100),
    (20000, 1000),
]


def make_split(n: int, p: int, seed: int):
    Xtr, ytr, _ = data_gen(n, NM, p, p // 2, p // 2, MU, RO, seed)
    Xte, yte, _ = data_gen(n // 10, NM, p, p // 2, p // 2, MU, RO, seed)
    return standardize(Xtr), ytr, standardize(Xte), yte


def run_torchkm(
    Xtr, ytr, Xte, yte, sig, Kmat, y_t, device, seed, max_iter, Cs, is_exact
):
    """Fit TorchKM (timed) and return a record: objective at lambda*, time, metrics."""
    clf = TorchKMSVC(
        kernel="rbf",
        rbf_sigma=sig,
        Cs=Cs,
        nC=len(Cs),
        cv=NFOLDS,
        device=device,
        random_state=seed,
        max_iter=max_iter,
        is_exact=is_exact,
    )
    with PeakMemory(device) as pm, timed(device) as t:
        clf.fit(Xtr.numpy(), ytr.numpy())
    lam = 1.0 / (2.0 * Xtr.shape[0] * clf.best_C_)  # = ulam[best_ind]
    alpha = torch.as_tensor(clf.alpha_, dtype=torch.double, device=device)
    rec = dict(
        library="torchkm",
        objective=svm_objective(Kmat, y_t, alpha, clf.intercept_, lam),
        time_s=t.dt,
        memory=pm.result,
        torch_peak_bytes=clf.peak_gpu_memory_bytes_,
        best_C=float(clf.best_C_),
        lam_star=float(lam),
    )
    rec.update(classification_metrics(yte.numpy(), clf.decision_function(Xte.numpy())))
    return rec


def libsvm_obj_time(
    SVC,
    name,
    Xtr_np,
    ytr_np,
    Xte_np,
    yte_np,
    sig,
    gamma,
    Kmat,
    y_t,
    n,
    ulam,
    device,
    baseline_lambda,
):
    """Tune + fit an RBF-SVM (sklearn or thundersvm) exactly as the notebook does.

    gamma is the baseline's RBF parameter; lam_best is chosen by 10-fold CV
    accuracy. The objective regularization weight follows ``baseline_lambda``:
    'notebook' uses ulam[-1] (the leftover loop variable), 'best' uses lam_best.
    """
    from sklearn.model_selection import cross_val_score

    with PeakMemory(device) as pm, timed(device) as t:
        cv = [
            cross_val_score(
                SVC(kernel="rbf", C=float(1.0 / (2 * n * float(l))), gamma=gamma),
                Xtr_np,
                ytr_np,
                cv=NFOLDS,
            ).mean()
            for l in ulam
        ]
        lam_best = float(ulam[int(np.argmax(cv))])
        model = SVC(kernel="rbf", C=float(1.0 / (2 * n * lam_best)), gamma=gamma).fit(
            Xtr_np, ytr_np
        )

    alpha_full = np.zeros(n)
    alpha_full[np.asarray(model.support_)] = np.asarray(model.dual_coef_).ravel()
    alpha = torch.as_tensor(alpha_full, dtype=torch.double, device=device)
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    lam_reg = float(ulam[-1]) if baseline_lambda == "notebook" else lam_best
    rec = dict(
        library=name,
        objective=svm_objective(Kmat, y_t, alpha, intercept, lam_reg),
        time_s=t.dt,
        memory=pm.result,
        best_C=float(1.0 / (2 * n * lam_best)),
        lam_star=lam_best,
        objective_lambda=lam_reg,
    )
    rec.update(
        classification_metrics(
            yte_np, np.asarray(model.decision_function(Xte_np)).ravel()
        )
    )
    return rec


def load_thundersvm(path):
    if path:
        sys.path.insert(0, path)
    from thundersvm import SVC

    return SVC


def fmt(recs):
    if not recs:
        return f"{'skipped':>32}"
    obj = mean_se([r["objective"] for r in recs])
    acc = mean_se([r["accuracy"] for r in recs])
    t = mean_se([r["time_s"] for r in recs])[0]
    return f"{obj[0]:>6.3f}({obj[1]:.3f}) {acc[0]:>6.4f}({acc[1]:.4f}) {t:>8.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repeats", type=int, default=3, help="runs per cell (paper: 50)")
    ap.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        metavar="N,P",
        help="one or more cells to run, e.g. --sizes 10000,10 (default: full grid)",
    )
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument(
        "--max-iter",
        type=int,
        default=100000,
        help="TorchKM solver iterations; the default 1000 under-converges (paper used 1e6)",
    )
    ap.add_argument(
        "--baseline-lambda",
        choices=["notebook", "best"],
        default="notebook",
        help="objective reg weight for baselines: 'notebook'=ulam[-1] (exact), 'best'=lam_best",
    )
    ap.add_argument(
        "--matched-kernel",
        action="store_true",
        help="give the baselines gamma=2*sig (TorchKM's kernel) instead of the notebook's gamma=sig",
    )
    ap.add_argument(
        "--exact",
        action="store_true",
        help="use TorchKM exact cross-validation (is_exact=1; default is 0)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-sklearn", action="store_true", help="scikit-learn is very slow at scale"
    )
    ap.add_argument("--skip-thunder", action="store_true")
    ap.add_argument("--thundersvm-path", default=None, help="path to thundersvm/python")
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--smoke", action="store_true", help="one tiny cell on CPU")
    args = ap.parse_args()

    sizes = SIZES
    if args.sizes is not None:
        sizes = [tuple(int(v) for v in s.split(",")) for s in args.sizes]
    if args.smoke:
        sizes, args.repeats, args.max_iter = [(300, 10)], 1, 200

    device = get_device(args.device)
    print(
        f"device={device}  repeats={args.repeats}  folds={NFOLDS}  grid=50 lambda in [1e-3,1e3]"
        f"  baseline kernel={'matched (gamma=2 sig)' if args.matched_kernel else 'notebook (gamma=sig)'}"
    )

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
    writer = ResultWriter(
        args.out,
        script="table2_simulation.py",
        args=args,
        protocol=dict(
            data="torchkm.data_gen Gaussian mixture (fast spectral decay; favourable regime)",
            folds=NFOLDS,
            grid="50 lambda in [1e-3, 1e3], C = 1/(2 n lambda)",
            objective_note="baseline objective weight per --baseline-lambda; see bench_solver_quality.py",
        ),
    )
    warmup(device)

    header = (
        f"{'n':>7} {'p':>5} | {'scikit-learn obj / acc / t(s)':>32} | "
        f"{'ThunderSVM obj / acc / t(s)':>32} | {'TorchKM obj / acc / t(s)':>32} | {'TorchKM peak':>12}"
    )
    print(header)
    print("-" * len(header))

    ulam = lam_grid()  # 50 lambda values over [1e-3, 1e3]
    for n, p in sizes:
        Cs = 1.0 / (2.0 * n * ulam)  # transfer lambda grid to the LIBSVM C sequence
        sk, th, tk = [], [], []
        for i in range(args.repeats):
            Xtr, ytr, Xte, yte = make_split(n, p, args.seed + i)
            sig = sigest(Xtr)
            gamma = 2.0 * sig if args.matched_kernel else sig
            Kmat = rbf_kernel(Xtr.to(torch.double).to(device), sig)
            y_t = ytr.to(torch.double).to(device)
            Xtr_np, ytr_np = Xtr.numpy(), ytr.numpy()
            Xte_np, yte_np = Xte.numpy(), yte.numpy()
            info = dict(
                n=n, p=p, repeat=i, seed=args.seed + i, bandwidth_sigest=float(sig)
            )

            rec = run_torchkm(
                Xtr,
                ytr,
                Xte,
                yte,
                sig,
                Kmat,
                y_t,
                device,
                args.seed + i,
                args.max_iter,
                Cs,
                int(args.exact),
            )
            rec.update(info)
            writer.add(rec)
            tk.append(rec)
            if not args.skip_sklearn:
                rec = libsvm_obj_time(
                    SkSVC,
                    "sklearn_svc",
                    Xtr_np,
                    ytr_np,
                    Xte_np,
                    yte_np,
                    sig,
                    gamma,
                    Kmat,
                    y_t,
                    n,
                    ulam,
                    device,
                    args.baseline_lambda,
                )
                rec.update(info)
                writer.add(rec)
                sk.append(rec)
            if ThunderSVC is not None:
                rec = libsvm_obj_time(
                    ThunderSVC,
                    "thundersvm",
                    Xtr_np,
                    ytr_np,
                    Xte_np,
                    yte_np,
                    sig,
                    gamma,
                    Kmat,
                    y_t,
                    n,
                    ulam,
                    device,
                    args.baseline_lambda,
                )
                rec.update(info)
                writer.add(rec)
                th.append(rec)
            del Kmat, y_t

        peak = mean_se(
            [r["torch_peak_bytes"] for r in tk if r.get("torch_peak_bytes") is not None]
        )[0]
        print(
            f"{n:>7} {p:>5} | {fmt(sk)} | {fmt(th)} | {fmt(tk)} | {fmt_bytes(peak):>12}"
        )
    if args.out:
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
