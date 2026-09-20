"""Solver-quality check: the kernel-SVM objective at a fixed lambda, solver by solver.

The submitted paper's Table 2 compared objective values across libraries that
each selected a different lambda by cross-validation. That mixes model
selection with solver accuracy. The clean check fixes lambda and asks every
solver for the minimiser of the same objective (equation (1) of the
supplement) on the same kernel matrix::

    (1/n) sum_i (1 - y_i f_i)_+  +  lambda * alpha^T K alpha,   f = K alpha + b

TorchKM's finite-smoothing solver converges to the exact optimum, so its
objective should match or beat every SMO solver at the same lambda up to
solver tolerance. Any gap is a finding, not a table entry. Results belong in
the documentation (solver quality page), not in the paper's headline table.

For each (n, p) in ``--sizes`` and each lambda in ``--lambdas``:

* data from ``torchkm.data_gen`` (Gaussian mixture), one bandwidth from
  ``sigest``, ``K = rbf_kernel(X, sig)`` shared by every solver;
* TorchKM fit at that single lambda (``Cs=[C]``; the cross-validation part of
  the fit is irrelevant here and kept to two folds);
* scikit-learn ``SVC`` (and ThunderSVM / cuML when importable) at
  ``C = 1 / (2 n lambda)`` with ``gamma = 2 sig``, the same kernel;
* the objective of each solution, evaluated with the same code, and the gap
  to the best value.

Examples
--------
::

    python benchmarks/bench_solver_quality.py --device cuda --sizes 10000,10 10000,100 \\
        --lambdas 1e-1 1e-2 1e-3 --repeats 5 --out benchmarks/results/solver_quality.json

    python benchmarks/bench_solver_quality.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Dict, List

import numpy as np
import torch

from _common import (
    ResultWriter,
    add_common_args,
    banner,
    free_cuda,
    gamma_from_sigest,
    get_device,
    smoke_settings,
    svm_objective,
    synthetic_dataset,
    timed,
    warmup,
)
from _libraries import library_availability

DEFAULT_SIZES = ["10000,10", "10000,100", "10000,1000"]


def torchkm_solution(data, sig, lam, args, dev, seed):
    from torchkm.estimators import TorchKMSVC

    n = data["Xtr"].shape[0]
    C = 1.0 / (2.0 * n * lam)
    clf = TorchKMSVC(
        kernel="rbf",
        rbf_sigma=float(sig),
        Cs=[C],
        nC=1,
        cv=2,
        device=dev,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        is_exact=int(args.is_exact),
        random_state=seed,
        **({} if args.kkt_eps is None else {"KKTeps": float(args.kkt_eps)}),
    )
    with timed(dev) as t:
        clf.fit(data["Xtr"], data["ytr"])
    conv = clf.converged_
    return (
        np.asarray(clf.alpha_, dtype=float),
        float(clf.intercept_),
        t.dt,
        None if conv is None else bool(np.all(conv)),
    )


def smo_solution(make_svc, data, sig, lam, dev):
    """Fit an sklearn-style SVC at C = 1/(2 n lambda) and return (alpha, b, time)."""
    n = data["Xtr"].shape[0]
    C = 1.0 / (2.0 * n * lam)
    model = make_svc(C=float(C), gamma=gamma_from_sigest(sig))
    with timed(dev) as t:
        model.fit(data["Xtr"], data["ytr"])
    alpha = np.zeros(n)
    support = np.asarray(model.support_).reshape(-1).astype(int)
    alpha[support] = np.asarray(model.dual_coef_).reshape(-1)
    b = float(np.asarray(model.intercept_).reshape(-1)[0])
    return alpha, b, t.dt


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, repeats=1)
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, metavar="N,P")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3])
    ap.add_argument("--is-exact", type=int, default=0, help="TorchKM is_exact flag")
    ap.add_argument(
        "--solvers",
        nargs="+",
        default=["torchkm", "sklearn_svc", "thundersvm", "cuml_svc"],
        choices=["torchkm", "sklearn_svc", "thundersvm", "cuml_svc"],
    )
    ap.add_argument("--thundersvm-path", default=None)
    args = smoke_settings(ap.parse_args())
    if args.smoke:
        args.sizes, args.lambdas = ["300,10"], [1e-1, 1e-2]
    if args.thundersvm_path:
        sys.path.insert(0, args.thundersvm_path)

    dev = get_device(args.device)
    sizes = [tuple(int(v) for v in s.split(",")) for s in args.sizes]
    availability = library_availability(args.solvers)
    solvers = [s for s in args.solvers if availability[s] is None]
    for s, err in availability.items():
        if err is not None:
            print(f"[skip] {s}: {err}")
    banner(
        "Kernel-SVM objective at fixed lambda",
        device=dev,
        sizes=sizes,
        lambdas=args.lambdas,
        solvers=solvers,
    )
    writer = ResultWriter(
        args.out,
        script="bench_solver_quality.py",
        args=args,
        protocol={
            "objective": "(1/n) sum (1 - y f)_+ + lambda alpha^T K alpha on the shared kernel",
            "C": "1 / (2 n lambda)",
            "gamma": "2 * sigest (same kernel as TorchKM)",
            "sizes": sizes,
            "lambdas": args.lambdas,
        },
    )
    warmup(dev)

    from torchkm import rbf_kernel, sigest

    makers: Dict[str, Any] = {}
    if "sklearn_svc" in solvers:
        from sklearn.svm import SVC as SkSVC

        makers["sklearn_svc"] = lambda C, gamma: SkSVC(
            kernel="rbf", C=C, gamma=gamma, cache_size=2000, tol=1e-6
        )
    if "thundersvm" in solvers:
        from thundersvm import SVC as ThSVC

        makers["thundersvm"] = lambda C, gamma: ThSVC(
            kernel="rbf", C=C, gamma=gamma, tol=1e-6
        )
    if "cuml_svc" in solvers:
        from cuml.svm import SVC as CuSVC

        makers["cuml_svc"] = lambda C, gamma: CuSVC(
            kernel="rbf", C=C, gamma=gamma, tol=1e-6, output_type="numpy"
        )

    print(
        f"{'n':>7} {'p':>5} {'lambda':>8} {'solver':>12} {'objective':>12} {'gap':>10} {'time':>8}"
    )
    for n, p in sizes:
        for r in range(args.repeats):
            seed = args.seed + r
            data = synthetic_dataset(n, p, seed)
            torch.manual_seed(seed)
            Xt = torch.from_numpy(data["Xtr"])
            sig = float(sigest(Xt))
            K = rbf_kernel(Xt.to(dev), sig)
            y_t = torch.from_numpy(data["ytr"]).to(dev)
            for lam in args.lambdas:
                rows: List[Dict[str, Any]] = []
                for solver in solvers:
                    rec: Dict[str, Any] = dict(
                        n=n, p=p, seed=seed, repeat=r, lam=float(lam), solver=solver
                    )
                    try:
                        if solver == "torchkm":
                            alpha, b, dt, converged = torchkm_solution(
                                data, sig, lam, args, dev, seed
                            )
                            rec["converged"] = converged
                        else:
                            alpha, b, dt = smo_solution(
                                makers[solver], data, sig, lam, dev
                            )
                        a_t = torch.as_tensor(alpha, dtype=torch.double, device=dev)
                        rec.update(
                            objective=svm_objective(K, y_t, a_t, b, float(lam)),
                            time_s=dt,
                            status="ok",
                        )
                    except Exception as err:
                        traceback.print_exc()
                        rec.update(
                            status="failed", error=f"{type(err).__name__}: {err}"[:800]
                        )
                    rows.append(rec)
                best = min(
                    (x["objective"] for x in rows if x.get("objective") is not None),
                    default=None,
                )
                for rec in rows:
                    if best is not None and rec.get("objective") is not None:
                        rec["gap_to_best"] = rec["objective"] - best
                        rec["relative_gap"] = (rec["objective"] - best) / max(
                            abs(best), 1e-12
                        )
                    writer.add(rec)
                    obj = rec.get("objective")
                    gap = rec.get("relative_gap")
                    print(
                        f"{n:>7} {p:>5} {lam:>8.0e} {rec['solver']:>12} "
                        f"{'-' if obj is None else f'{obj:.6f}':>12} "
                        f"{'-' if gap is None else f'{gap:.2e}':>10} "
                        f"{rec.get('time_s', float('nan')):>7.1f}s",
                        flush=True,
                    )
            del K, y_t
            free_cuda(dev)
    if args.out:
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
