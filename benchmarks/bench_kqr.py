"""Kernel quantile regression: TorchKMKQR against the CPU implementations.

GPU-accelerated kernel quantile regression with exact cross-validation does not
exist outside TorchKM, so the comparison is against the CPU packages the
constituency uses today. This script runs the Python side and exports the
exact splits and folds so the R side (``benchmarks/r/bench_kqr.R``: ``fastkqr``
and ``kernlab::kqr``) sees identical data.

Per dataset, repeat and quantile level ``tau``:

* ``torchkm_kqr`` (exact) and ``torchkm_kqr_nystrom``: integrated CV over the
  lambda path, end-to-end time, peak memory;
* ``linear_qr``: scikit-learn ``QuantileRegressor`` (linear, ``highs`` solver)
  tuned over the same lambda grid with the same folds, so the value of the
  kernel is visible;
* metrics: pinball (check) loss on the test set, empirical coverage
  ``P(y <= q_hat)`` against the target ``tau``, and for the synthetic set the
  RMSE to the true conditional quantile.

Datasets (LIBSVM regression files in ``--data-dir``; ``.bz2`` accepted):
``cadata`` (California housing, 20,640), ``abalone`` (4,177), ``cpusmall``
(8,192), ``space_ga`` (3,107) for exact mode; ``YearPredictionMSD`` (463,715)
for the Nyström path. ``synthetic`` is a heteroscedastic model with known
quantiles and needs no files. Features are standardised on the training split;
targets are centred and scaled by the training standard deviation for the
solvers and mapped back before scoring.

Examples
--------
::

    python benchmarks/bench_kqr.py --data-dir ~/libsvm --datasets synthetic cadata abalone cpusmall \\
        --taus 0.1 0.5 0.9 --repeats 5 --device cuda --export-splits benchmarks/results/kqr_splits \\
        --out benchmarks/results/kqr.json
    Rscript benchmarks/r/bench_kqr.R benchmarks/results/kqr_splits benchmarks/results/kqr_r.csv

    python benchmarks/bench_kqr.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any, Dict

import numpy as np
import torch

from _common import (
    PeakMemory,
    ResultWriter,
    add_common_args,
    banner,
    c_grid,
    cv_sweep,
    free_cuda,
    get_device,
    lam_from_c,
    make_folds,
    open_libsvm,
    protocol_dict,
    quantile_metrics,
    smoke_settings,
    synthetic_regression,
    timed,
    warmup,
)

REG_DATASETS: Dict[str, Dict[str, Any]] = {
    "cadata": dict(train="cadata"),
    "abalone": dict(train="abalone"),
    "cpusmall": dict(train="cpusmall"),
    "space_ga": dict(train="space_ga"),
    "YearPredictionMSD": dict(train="YearPredictionMSD", test="YearPredictionMSD.t"),
}


def load_regression(
    name: str, data_dir: str, seed: int, n_synthetic: int
) -> Dict[str, Any]:
    from sklearn.datasets import load_svmlight_file, load_svmlight_files
    from sklearn.model_selection import train_test_split

    if name == "synthetic":
        return synthetic_regression(n_synthetic, 5, seed)
    cfg = REG_DATASETS[name]
    train_path = os.path.join(data_dir, cfg["train"])
    if cfg.get("test"):
        with open_libsvm(train_path) as ftr, open_libsvm(
            os.path.join(data_dir, cfg["test"])
        ) as fte:
            Xtr, ytr, Xte, yte = load_svmlight_files((ftr, fte), dtype=np.float64)
        Xtr, Xte = Xtr.toarray(), Xte.toarray()
    else:
        with open_libsvm(train_path) as f:
            X, y = load_svmlight_file(f, dtype=np.float64)
        Xtr, Xte, ytr, yte = train_test_split(
            X.toarray(), y, test_size=0.2, random_state=seed
        )
    return dict(
        name=name,
        Xtr=Xtr,
        ytr=np.asarray(ytr, dtype=float),
        Xte=Xte,
        yte=np.asarray(yte, dtype=float),
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        p=int(Xtr.shape[1]),
    )


def standardize_split(data: Dict[str, Any]) -> Dict[str, Any]:
    """Standardise X on the training split; centre/scale y for the solvers."""
    mu, sd = data["Xtr"].mean(axis=0), data["Xtr"].std(axis=0)
    sd[sd == 0] = 1.0
    ym, ys = float(data["ytr"].mean()), float(data["ytr"].std() or 1.0)
    out = dict(data)
    out["Xtr"] = (data["Xtr"] - mu) / sd
    out["Xte"] = (data["Xte"] - mu) / sd
    out["ytr_s"] = (data["ytr"] - ym) / ys
    out["y_mean"], out["y_scale"] = ym, ys
    return out


def export_split(
    directory: str, name: str, r: int, data: Dict[str, Any], foldid: np.ndarray
) -> None:
    os.makedirs(directory, exist_ok=True)
    p = data["Xtr"].shape[1]
    header = ",".join([f"x{j}" for j in range(p)] + ["y", "fold"])
    np.savetxt(
        os.path.join(directory, f"{name}_rep{r}_train.csv"),
        np.column_stack([data["Xtr"], data["ytr"], foldid]),
        delimiter=",",
        header=header,
        comments="",
    )
    np.savetxt(
        os.path.join(directory, f"{name}_rep{r}_test.csv"),
        np.column_stack([data["Xte"], data["yte"]]),
        delimiter=",",
        header=",".join([f"x{j}" for j in range(p)] + ["y"]),
        comments="",
    )


def score(data: Dict[str, Any], q_scaled: np.ndarray, tau: float) -> Dict[str, Any]:
    q = np.asarray(q_scaled, dtype=float) * data["y_scale"] + data["y_mean"]
    out = quantile_metrics(data["yte"], q, tau)
    if "true_quantile" in data:
        truth = data["true_quantile"](data["Xte"] * 0 + data["Xte_raw"], tau)
        out["rmse_to_true_quantile"] = float(np.sqrt(np.mean((q - truth) ** 2)))
    return out


def run_torchkm_kqr(data, sig, Cs, foldid, tau, args, dev, seed, *, low_rank: bool):
    from torchkm.estimators import TorchKMKQR

    kw: Dict[str, Any] = dict(
        kernel="rbf",
        rbf_sigma=float(sig),
        Cs=Cs,
        nC=len(Cs),
        cv=int(args.folds),
        foldid=foldid,
        tau=float(tau),
        device=dev,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        random_state=int(seed),
    )
    if low_rank:
        kw.update(
            low_rank=True, num_landmarks=int(args.landmarks), nys_k=int(args.rank)
        )
    reg = TorchKMKQR(**kw)
    with PeakMemory(dev) as pm, timed(dev) as t:
        reg.fit(data["Xtr"], data["ytr_s"])
    q = np.concatenate(
        [
            reg.predict(data["Xte"][i : i + 4096])
            for i in range(0, data["Xte"].shape[0], 4096)
        ]
    )
    rec = dict(
        library="torchkm_kqr_nystrom" if low_rank else "torchkm_kqr",
        mode="nystrom" if low_rank else "exact",
        device=dev,
        status="ok",
        time_s=t.dt,
        memory=pm.result,
        torch_peak_bytes=reg.peak_gpu_memory_bytes_,
        best_C=float(reg.best_C_),
        params=dict(num_landmarks=args.landmarks, nys_k=args.rank) if low_rank else {},
    )
    rec.update(score(data, q, tau))
    return rec


def run_linear_qr(data, Cs, foldid, tau, args):
    from sklearn.linear_model import QuantileRegressor

    n = data["Xtr"].shape[0]
    alphas = list(map(float, lam_from_c(Cs, n)))

    def fit_predict(alpha, Xa, ya, Xb):
        return (
            QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
            .fit(Xa, ya)
            .predict(Xb)
        )

    with PeakMemory("cpu") as pm, timed("cpu") as t:
        t0 = time.perf_counter()
        res = cv_sweep(
            fit_predict,
            data["Xtr"],
            data["ytr_s"],
            foldid,
            alphas,
            score=f"pinball:{tau}",
            time_cap_s=args.time_cap,
            t0=t0,
        )
        q = fit_predict(res["best_param"], data["Xtr"], data["ytr_s"], data["Xte"])
    rec = dict(
        library="linear_qr",
        mode="linear",
        device="cpu",
        status="capped" if res["capped"] else "ok",
        time_s=t.dt,
        memory=pm.result,
        best_lambda=float(res["best_param"]),
        cv=dict(grid_completed=res["grid_completed"], grid_size=res["grid_size"]),
    )
    rec.update(score(data, q, tau))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, repeats=1)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["synthetic"],
        choices=["synthetic"] + sorted(REG_DATASETS),
    )
    ap.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["torchkm_kqr", "torchkm_kqr_nystrom", "linear_qr"],
        choices=["torchkm_kqr", "torchkm_kqr_nystrom", "linear_qr"],
    )
    ap.add_argument("--landmarks", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=300)
    ap.add_argument("--synthetic-n", type=int, default=5000)
    ap.add_argument(
        "--export-splits",
        default=None,
        help="directory for the CSV splits the R script reads",
    )
    args = smoke_settings(ap.parse_args())
    if args.smoke:
        args.datasets, args.taus, args.synthetic_n = ["synthetic"], [0.5], 300
        args.landmarks, args.rank = 40, 20
    if any(d != "synthetic" for d in args.datasets) and not args.data_dir:
        ap.error("--data-dir is required for LIBSVM regression datasets")

    dev = get_device(args.device)
    banner(
        "Kernel quantile regression",
        device=dev,
        datasets=args.datasets,
        taus=args.taus,
        methods=args.methods,
        folds=args.folds,
        grid=f"{args.grid_size} C values in [{args.c_min}, {args.c_max}]",
    )
    writer = ResultWriter(
        args.out,
        script="bench_kqr.py",
        args=args,
        protocol=protocol_dict(
            args,
            taus=args.taus,
            targets="centred and scaled by the training std; metrics in original units",
            r_baselines="benchmarks/r/bench_kqr.R on the exported splits (fastkqr, kernlab::kqr)",
        ),
    )
    warmup(dev)
    from torchkm import sigest

    Cs = c_grid(args.grid_size, args.c_max, args.c_min)
    print(
        f"{'dataset':>10} {'tau':>4} {'method':>20} {'status':>7} {'pinball':>9} {'coverage':>9} {'time':>9}"
    )
    for name in args.datasets:
        for r in range(args.repeats):
            seed = args.seed + r
            raw = load_regression(name, args.data_dir, seed, args.synthetic_n)
            data = standardize_split(raw)
            data["Xte_raw"] = raw["Xte"]
            torch.manual_seed(seed)
            sig = float(sigest(torch.from_numpy(data["Xtr"])))
            foldid = make_folds(np.zeros(data["n_train"]), args.folds, seed)
            if args.export_splits:
                export_split(args.export_splits, name, r, data, foldid)
            info = dict(
                dataset=name,
                n_train=data["n_train"],
                n_test=data["n_test"],
                p=data["p"],
                repeat=r,
                seed=seed,
                bandwidth_sigest=sig,
            )
            for tau in args.taus:
                for method in args.methods:
                    try:
                        if method == "torchkm_kqr":
                            rec = run_torchkm_kqr(
                                data,
                                sig,
                                Cs,
                                foldid,
                                tau,
                                args,
                                dev,
                                seed,
                                low_rank=False,
                            )
                        elif method == "torchkm_kqr_nystrom":
                            rec = run_torchkm_kqr(
                                data,
                                sig,
                                Cs,
                                foldid,
                                tau,
                                args,
                                dev,
                                seed,
                                low_rank=True,
                            )
                        else:
                            rec = run_linear_qr(data, Cs, foldid, tau, args)
                    except Exception as err:
                        traceback.print_exc()
                        rec = dict(
                            library=method,
                            status="failed",
                            error=f"{type(err).__name__}: {err}"[:800],
                        )
                    rec.update(info, tau=float(tau))
                    writer.add(rec)
                    print(
                        f"{name:>10} {tau:>4.2f} {rec['library']:>20} {rec['status']:>7} "
                        f"{rec.get('pinball_loss', float('nan')):>9.4f} "
                        f"{rec.get('coverage', float('nan')):>9.3f} "
                        f"{rec.get('time_s', float('nan')):>8.1f}s",
                        flush=True,
                    )
                    free_cuda(dev)
    if args.out:
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
