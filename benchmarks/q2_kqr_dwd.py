#!/usr/bin/env python
"""Q2: kernel quantile regression and kernel DWD on real benchmarks.

The letter asks for at least one regression benchmark exercising kernel
quantile regression (KQR) and one exercising distance-weighted discrimination
(DWD). One script, two tables. TorchKM runs on the GPU, on the full kernel.
Each competitor runs on the GPU if it can; one that cannot runs on the CPU,
and that is part of the comparison.

KQR (cpusmall 8,192 x 12 and cadata 20,640 x 8; random 80/20 train/test split
per repeat; features and target standardised on the training split, losses
reported in the target's original units; tau = 0.1, 0.5, 0.9)
  torchkm_kqr  TorchKMKQR, exact mode, integrated 10-fold CV over the lambda
               path by pinball loss; GPU
  xgb_qr       XGBoost gradient-boosted quantile regression
               (objective "reg:quantileerror", tree_method "hist"), default
               settings, one fit; GPU (XGBoost computes in float32)
  Metrics: test pinball loss, coverage P(y <= q_hat) against tau, time, memory.

DWD (gisette 6,000 x 5,000 with its 1,000-row test file; MNIST 3-vs-8 from
mnist.scale)
  torchkm_dwd  TorchKMDWD, exact mode, integrated 10-fold CV over the path; GPU
  dwd_pkg      KernGDWD from the pip package ``dwd`` (Carmichael), the Python
               kernel DWD: the MM algorithm of Wang and Zou with a per-fit cap
               of 100 iterations, on the same precomputed RBF kernel. CPU only:
               the package is numpy code with no GPU implementation. Both
               minimise (1/n) sum V(y f) + lambda a'Ka, so they share the lambda
               grid. The package's own KernGDWDCV is not used: it swaps the
               train and test folds, zips its parameter lists instead of
               crossing them, and refits with default parameters. This script
               tunes KernGDWD itself on the shared folds, reusing one
               eigendecomposition per fold across lambda as the package
               intends, and visits the grid coarse to fine so that a sweep
               stopped by --time-cap still spans the whole range
  Metrics: test accuracy, balanced accuracy, AUC, time, memory.

Protocol for both tables
  kernel     RBF exp(-2 sig d^2), sig from sigest on the training features per
             repeat, shared by every kernel method (gamma = 2 sig for the dwd
             package)
  grid       50 log-uniform lambda from --lam-max down to --lam-min (default
             1e-1 to 1e-7), large to small, the order TorchKM's path runs in
  CV         10 folds, identical for every tuned method (stratified for DWD)
  TorchKM    tol 1e-5, KKTeps 1e-3 (the defaults), max_iter 100000, float64,
             on the GPU
  devices    TorchKM and XGBoost on the GPU; the dwd package on the CPU in
             float64. The table has a device column
  timing     tuning + final fit + test predictions, the cost of a tuned model;
             each GPU library is warmed up before its timed block
  memory     GPU methods: NVML peak of the process on the GPU; the dwd package:
             host memory added during the fit
  repeats    3 (seeds 52, 53, 54): new split (KQR), new folds and bandwidth

Data files in --data-dir: cpusmall, cadata (LIBSVM regression), gisette_scale
and gisette_scale.t (.bz2 is fine), mnist.scale and mnist.scale.t.

Run (re-running with the same --out keeps the finished cells computed with the
same settings and runs the rest):
  pip install xgboost dwd
  python benchmarks/q2_kqr_dwd.py --data-dir ~/libsvm_data --out results/q2.json
  python benchmarks/q2_kqr_dwd.py --smoke        # CPU check on synthetic data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    PeakMemory,
    _jsonable,
    classification_metrics,
    env_snapshot,
    fmt_bytes,
    host_rss_bytes,
    load_dataset,
    make_folds,
    mean_se,
    quantile_metrics,
    synthetic_dataset,
    synthetic_regression,
)
from bench_kqr import load_regression, standardize_split  # noqa: E402

KQR_SETS = ["cpusmall", "cadata"]
DWD_SETS = ["gisette", "mnist_3v8"]
KQR_METHODS = ["torchkm_kqr", "xgb_qr"]
DWD_METHODS = ["torchkm_dwd", "dwd_pkg"]
TASK = {**{d: "kqr" for d in KQR_SETS}, **{d: "dwd" for d in DWD_SETS}}


def sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def device_label(dev: str) -> str:
    return "GPU" if dev.startswith("cuda") else "CPU"


class Measured:
    """Wall-clock time and peak memory of a block, on the GPU or the host."""

    def __init__(self, dev: str):
        self.dev = dev

    def __enter__(self) -> "Measured":
        sync(self.dev)
        self.rss0 = host_rss_bytes() or 0
        self.pm = PeakMemory(self.dev).__enter__()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        sync(self.dev)
        self.seconds = time.perf_counter() - self.t0
        self.pm.__exit__(*exc)
        mem = self.pm.result
        if self.dev.startswith("cuda"):
            self.memory = dict(
                where="gpu",
                bytes=mem.get("nvml_process_peak") or mem.get("nvml_device_peak"),
                torch_bytes=mem.get("torch_max_allocated"),
            )
        else:
            peak = mem.get("host_rss_peak")
            self.memory = dict(
                where="host", bytes=None if peak is None else max(0, peak - self.rss0)
            )


def coarse_to_fine(n: int) -> List[int]:
    """0, n/2, n/4, 3n/4, ...: any prefix spans the whole index range."""
    order, seen, step = [], set(), 1 << max(0, (n - 1).bit_length())
    while step >= 1:
        for i in range(0, n, step):
            if i not in seen:
                order.append(i)
                seen.add(i)
        step //= 2
    return order


# ---------------------------------------------------------------------------
# Kernel quantile regression
# ---------------------------------------------------------------------------


def to_original(data, q_scaled) -> np.ndarray:
    return (
        np.asarray(q_scaled, dtype=float).reshape(-1) * data["y_scale"] + data["y_mean"]
    )


def run_torchkm_kqr(data, sig, lams, foldid, tau, dev, args, seed):
    from torchkm.estimators import TorchKMKQR

    n = data["Xtr"].shape[0]
    reg = TorchKMKQR(
        kernel="rbf",
        rbf_sigma=sig,
        Cs=1.0 / (2.0 * n * lams),  # lambda large to small
        nC=len(lams),
        cv=args.folds,
        foldid=foldid,
        tau=tau,
        device=dev,
        tol=args.tol,
        max_iter=args.max_iter,
        KKTeps=args.kkt_eps,
        random_state=seed,
    )
    torch.linalg.eigh(torch.eye(64, dtype=torch.float64, device=dev))  # start-up
    with Measured(dev) as m:
        reg.fit(data["Xtr"], data["ytr_s"])
        q = reg.predict(data["Xte"])
    lam = 1.0 / (2.0 * n * reg.best_C_)
    return dict(
        status="ok",
        device=device_label(dev),
        time_s=m.seconds,
        memory=m.memory,
        selected=float(lam),
        cv_curve=np.asarray(reg.cv_loss_, dtype=float).tolist(),
        params=dict(tol=args.tol, max_iter=args.max_iter, KKTeps=args.kkt_eps),
        **quantile_metrics(data["yte"], to_original(data, q), tau),
    )


def run_xgb_qr(data, sig, lams, foldid, tau, dev, args, seed):
    import xgboost as xgb

    def model(**kw):
        return xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=tau,
            tree_method="hist",
            device=dev,
            random_state=seed,
            **kw,
        )

    model(n_estimators=1).fit(np.zeros((8, 2)), np.arange(8.0))  # start-up
    with Measured(dev) as m:
        reg = model()
        reg.fit(data["Xtr"], data["ytr_s"])
        # a DMatrix is predicted on the model's device, host input included
        q = reg.get_booster().predict(xgb.DMatrix(data["Xte"]))
    config = json.loads(reg.get_booster().save_config())["learner"]["generic_param"]
    return dict(
        status="ok",
        device=device_label(config.get("device", dev)),  # as XGBoost ran it
        time_s=m.seconds,
        memory=m.memory,
        params=dict(settings="XGBoost defaults", xgboost=xgb.__version__),
        **quantile_metrics(data["yte"], to_original(data, q), tau),
    )


# ---------------------------------------------------------------------------
# Kernel DWD
# ---------------------------------------------------------------------------


def run_torchkm_dwd(data, sig, lams, foldid, tau, dev, args, seed):
    from torchkm.estimators import TorchKMDWD

    n = data["Xtr"].shape[0]
    clf = TorchKMDWD(
        kernel="rbf",
        rbf_sigma=sig,
        Cs=1.0 / (2.0 * n * lams),
        nC=len(lams),
        cv=args.folds,
        foldid=foldid,
        device=dev,
        tol=args.tol,
        max_iter=args.max_iter,
        KKTeps=args.kkt_eps,
        random_state=seed,
    )
    torch.linalg.eigh(torch.eye(64, dtype=torch.float64, device=dev))  # start-up
    with Measured(dev) as m:
        clf.fit(data["Xtr"], data["ytr"])
        scores = np.concatenate(
            [
                clf.decision_function(data["Xte"][i : i + 4096])
                for i in range(0, len(data["Xte"]), 4096)
            ]
        )
    return dict(
        status="ok",
        device=device_label(dev),
        time_s=m.seconds,
        memory=m.memory,
        selected=float(1.0 / (2.0 * n * clf.best_C_)),
        cv_curve=(1.0 - np.asarray(clf.cv_mis_, dtype=float)).tolist(),
        params=dict(tol=args.tol, max_iter=args.max_iter, KKTeps=args.kkt_eps),
        **classification_metrics(data["yte"], scores),
    )


def run_dwd_pkg(data, sig, lams, foldid, tau, dev, args, seed):
    from dwd.gen_kern_dwd import KernGDWD, get_K_eig
    from sklearn.metrics.pairwise import rbf_kernel

    gamma = 2.0 * sig  # the same kernel as TorchKM's exp(-2 sig d^2)
    Xtr, ytr = data["Xtr"], data["ytr"]
    with Measured("cpu") as m:
        K = rbf_kernel(Xtr, gamma=gamma)
        folds = []
        for k in np.unique(foldid):
            va = foldid == k
            tr = ~va
            # one eigendecomposition per fold, reused for every lambda
            folds.append((tr, va, get_K_eig(K[np.ix_(tr, tr)])))
        cv_acc: Dict[int, float] = {}
        fits, t_fits = 0, 0.0
        for i in coarse_to_fine(len(lams)):
            accs = []
            for tr, va, eig in folds:
                model = KernGDWD(lambd=float(lams[i]), q=1.0, kernel="precomputed")
                model._K_eig = eig
                np.random.seed(seed)  # the solver starts from a random point
                t = time.perf_counter()
                model.fit(K[np.ix_(tr, tr)], ytr[tr])
                t_fits += time.perf_counter() - t
                fits += 1
                pred = np.sign(model.decision_function(K[np.ix_(tr, va)]))
                accs.append(float(np.mean(pred == ytr[va])))
            cv_acc[i] = float(np.mean(accs))
            if args.time_cap and time.perf_counter() - m.t0 > args.time_cap:
                break
        best = min(cv_acc, key=lambda i: (-cv_acc[i], i))  # ties: larger lambda
        final = KernGDWD(lambd=float(lams[best]), q=1.0, kernel="precomputed")
        np.random.seed(seed)
        final.fit(K, ytr)
        del folds
        scores = np.concatenate(
            [
                final.decision_function(
                    rbf_kernel(Xtr, data["Xte"][i : i + 4096], gamma=gamma)
                )
                for i in range(0, len(data["Xte"]), 4096)
            ]
        )
    curve = [cv_acc.get(i) for i in range(len(lams))]
    return dict(
        status="capped" if len(cv_acc) < len(lams) else "ok",
        device="CPU",  # numpy only: the package has no GPU implementation
        time_s=m.seconds,
        memory=m.memory,
        selected=float(lams[best]),
        cv_curve=curve,
        grid_completed=len(cv_acc),
        seconds_per_fit=t_fits / max(fits, 1),
        params=dict(q=1.0, mm_max_iter=100, mm_obj_tol=1e-5, time_cap=args.time_cap),
        **classification_metrics(data["yte"], scores),
    )


RUN: Dict[str, Callable] = dict(
    torchkm_kqr=run_torchkm_kqr,
    xgb_qr=run_xgb_qr,
    torchkm_dwd=run_torchkm_dwd,
    dwd_pkg=run_dwd_pkg,
)


def cell_settings(method: str, args) -> Dict[str, Any]:
    if method in ("torchkm_kqr", "torchkm_dwd"):
        return dict(tol=args.tol, max_iter=args.max_iter, KKTeps=args.kkt_eps)
    if method == "dwd_pkg":
        return dict(time_cap=args.time_cap)
    return {}


def reusable(rec: Dict[str, Any], args) -> bool:
    if rec.get("status") not in ("ok", "capped"):
        return False
    have = rec.get("params") or {}
    return all(have.get(k) == v for k, v in cell_settings(rec["method"], args).items())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load(name: str, args, seed: int) -> Dict[str, Any]:
    if name == "synthetic_kqr":
        data = synthetic_regression(400, 5, seed)
        data.pop("true_quantile", None)
        return standardize_split(data)
    if name == "synthetic_dwd":
        return synthetic_dataset(400, 10, seed)
    if TASK[name] == "kqr":  # single-file sets: a new 80/20 split per repeat
        return standardize_split(load_regression(name, args.data_dir, seed, 0))
    return load_dataset(name, args.data_dir, seed=args.seed)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def fmt_pm(recs, key, digits=4) -> str:
    vals = [r.get(key) for r in recs if r.get(key) is not None]
    if not vals:
        return "-"
    m, se = mean_se(vals)
    return f"{m:.{digits}f} +- {se:.{digits}f}"


def fmt_mem(recs) -> str:
    vals = [r["memory"]["bytes"] for r in recs if (r.get("memory") or {}).get("bytes")]
    if not vals:
        return "-"
    return f"{fmt_bytes(float(np.mean(vals)))} {recs[0]['memory']['where']}"


def fmt_dev(recs) -> str:
    return recs[-1].get("device") or "-"


def fmt_sel(recs) -> str:
    vals = [r["selected"] for r in recs if r.get("selected") is not None]
    return ", ".join(f"{v:.3g}" for v in vals) if vals else "-"


def notes(recs, lams) -> str:
    out = []
    capped = [r for r in recs if r["status"] == "capped"]
    if capped:
        out.append(
            "capped at --time-cap: "
            + ", ".join(f"{r['grid_completed']}/{len(lams)}" for r in capped)
            + f" lambda values; {np.mean([r['seconds_per_fit'] for r in capped]):.1f} s per fit"
        )
    edge = [
        r
        for r in recs
        if r.get("selected") is not None
        and (
            np.isclose(r["selected"], lams.max())
            or np.isclose(r["selected"], lams.min())
        )
    ]
    if edge:
        out.append(f"{len(edge)} selected at a grid edge: widen the range")
    return "; ".join(out)


def write_markdown(doc: Dict[str, Any], path: str) -> str:
    env, a = doc["environment"], doc["args"]
    lams = np.asarray(doc["grid_lambda"])
    gpu = (env.get("gpu") or {}).get("name") or "no GPU (CPU run)"
    lines = [
        "# Q2: kernel quantile regression and kernel DWD",
        "",
        f"{gpu}; torch {env.get('torch')}; torchkm {env.get('torchkm')} "
        f"({str(env.get('torchkm_commit'))[:10]}).",
        f"{a['folds']}-fold CV on shared folds; {len(lams)} lambda values from "
        f"{lams.max():g} to {lams.min():g}; TorchKM tol {a['tol']:g}, KKTeps {a['kkt_eps']:g}; "
        f"seed {a['seed']} + repeat. Time = tuning + final fit + test predictions. "
        "Cells are mean +- SE over repeats.",
    ]
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in doc["records"]:
        groups.setdefault(
            (r["task"], r["dataset"], r.get("tau"), r["method"]), []
        ).append(r)

    def head(recs):
        r = recs[0]
        return f"| {r['dataset']} | {r['n_train']:,} | {r['n_test']:,} | {r['p']} |"

    kqr = [k for k in groups if k[0] == "kqr"]
    if kqr:
        lines += [
            "",
            "## Kernel quantile regression",
            "",
            "| dataset | n_train | n_test | p | tau | method | device | runs "
            "| test pinball loss | coverage | time (s) | memory | selected lambda | note |",
            "|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|",
        ]
        for key in kqr:
            recs = groups[key]
            ok = [r for r in recs if r["status"] in ("ok", "capped")]
            if not ok:
                err = recs[-1].get("error", recs[-1]["status"])[:100]
                lines.append(
                    f"{head(recs)} {key[2]} | {key[3]} | {fmt_dev(recs)} | 0 | - | - | - | - "
                    f"| - | {err} |"
                )
                continue
            lines.append(
                f"{head(ok)} {key[2]} | {key[3]} | {fmt_dev(ok)} | {len(ok)} "
                f"| {fmt_pm(ok, 'pinball_loss')} "
                f"| {fmt_pm(ok, 'coverage', 3)} | {fmt_pm(ok, 'time_s', 1)} | {fmt_mem(ok)} "
                f"| {fmt_sel(ok)} | {notes(ok, lams)} |"
            )
    dwd = [k for k in groups if k[0] == "dwd"]
    if dwd:
        lines += [
            "",
            "## Kernel DWD",
            "",
            "| dataset | n_train | n_test | p | method | device | runs | test accuracy "
            "| balanced accuracy | AUC | time (s) | memory | selected lambda | note |",
            "|---|---:|---:|---:|---|---|---:|---|---|---|---|---|---|---|",
        ]
        for key in dwd:
            recs = groups[key]
            ok = [r for r in recs if r["status"] in ("ok", "capped")]
            if not ok:
                err = recs[-1].get("error", recs[-1]["status"])[:100]
                lines.append(
                    f"{head(recs)} {key[3]} | {fmt_dev(recs)} | 0 | - | - | - | - | - "
                    f"| - | {err} |"
                )
                continue
            lines.append(
                f"{head(ok)} {key[3]} | {fmt_dev(ok)} | {len(ok)} | {fmt_pm(ok, 'accuracy')} "
                f"| {fmt_pm(ok, 'balanced_accuracy')} | {fmt_pm(ok, 'auc')} "
                f"| {fmt_pm(ok, 'time_s', 1)} | {fmt_mem(ok)} | {fmt_sel(ok)} | {notes(ok, lams)} |"
            )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return text


def save_json(doc: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(_jsonable(doc), fh, indent=1)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default=None, help="directory of LIBSVM files")
    ap.add_argument("--datasets", nargs="+", default=KQR_SETS + DWD_SETS)
    ap.add_argument("--methods", nargs="+", default=KQR_METHODS + DWD_METHODS)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--grid-size", type=int, default=50)
    ap.add_argument("--lam-max", type=float, default=1e-1)
    ap.add_argument("--lam-min", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument(
        "--tol", type=float, default=1e-5, help="TorchKM step tolerance (eps)"
    )
    ap.add_argument("--kkt-eps", type=float, default=1e-3, help="TorchKM KKT tolerance")
    ap.add_argument("--max-iter", type=int, default=100_000)
    ap.add_argument(
        "--time-cap",
        type=float,
        default=7200,
        help="seconds per dwd_pkg sweep (dataset x repeat) before it stops; 0: none",
    )
    ap.add_argument("--out", default="benchmarks/results/q2_kqr_dwd.json")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic CPU check")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.smoke:
        args.datasets, args.folds, args.grid_size, args.repeats = (
            ["synthetic_kqr", "synthetic_dwd"],
            3,
            5,
            1,
        )
        args.taus = [0.5]
        TASK.update(synthetic_kqr="kqr", synthetic_dwd="dwd")
    if not args.smoke and not args.data_dir:
        ap.error("--data-dir is required unless --smoke")
    dev = (
        "cuda"
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    if dev != args.device:
        print(f"[warn] {args.device} not available, running on cpu")

    from torchkm import sigest

    lams = np.logspace(np.log10(args.lam_max), np.log10(args.lam_min), args.grid_size)
    doc: Dict[str, Any] = dict(
        script="q2_kqr_dwd.py",
        args=vars(args),
        grid_lambda=lams.tolist(),
        environment=env_snapshot(),
        records=[],
    )
    if os.path.exists(args.out):  # resume
        with open(args.out) as fh:
            old = json.load(fh)
        for k in ("folds", "grid_size", "lam_max", "lam_min", "seed"):
            if old.get("args", {}).get(k) != vars(args)[k]:
                sys.exit(
                    f"{args.out} used a different {k}: use a new --out or delete it"
                )
        kept = [
            r
            for r in old.get("records", [])
            if r.get("method") in RUN and reusable(r, args)
        ]
        doc["records"] = kept
        print(
            f"resuming {args.out}: {len(kept)} finished cells kept, "
            f"{len(old.get('records', [])) - len(kept)} to redo"
        )
    done = {
        (r["dataset"], r.get("tau"), r["method"], r["repeat"]) for r in doc["records"]
    }
    md_path = os.path.splitext(args.out)[0] + ".md"

    for ds in args.datasets:
        task = TASK[ds]
        methods = [
            m
            for m in args.methods
            if m in (KQR_METHODS if task == "kqr" else DWD_METHODS)
        ]
        taus = args.taus if task == "kqr" else [None]
        for r in range(args.repeats):
            seed = args.seed + r
            data = load(ds, args, seed)
            torch.manual_seed(seed)
            sig = float(sigest(torch.from_numpy(np.ascontiguousarray(data["Xtr"]))))
            y_for_folds = data["ytr"]
            foldid = make_folds(y_for_folds, args.folds, seed)
            if r == 0:
                print(
                    f"\n== {ds} ({task}): n_train={data['Xtr'].shape[0]:,} "
                    f"n_test={data['Xte'].shape[0]:,} p={data['Xtr'].shape[1]}"
                )
            for tau in taus:
                for m in methods:
                    if (ds, tau, m, r) in done:
                        continue
                    info = dict(
                        dataset=ds,
                        task=task,
                        tau=tau,
                        method=m,
                        repeat=r,
                        seed=seed,
                        n_train=int(data["Xtr"].shape[0]),
                        n_test=int(data["Xte"].shape[0]),
                        p=int(data["Xtr"].shape[1]),
                        bandwidth_sigest=sig,
                    )
                    try:
                        rec = RUN[m](data, sig, lams, foldid, tau, dev, args, seed)
                    except Exception as err:  # the next cell still runs
                        traceback.print_exc()
                        rec = dict(
                            status="failed", error=f"{type(err).__name__}: {err}"[:800]
                        )
                    rec.update(info)
                    doc["records"].append(rec)
                    save_json(doc, args.out)
                    main_metric = (
                        f"pinball={rec.get('pinball_loss', float('nan')):.4f} "
                        f"cover={rec.get('coverage', float('nan')):.3f}"
                        if task == "kqr"
                        else f"acc={rec.get('accuracy', float('nan')):.4f}"
                    )
                    print(
                        f"   {m:12s} r{r}"
                        + (f" tau={tau}" if tau is not None else "")
                        + f" {rec['status']:7s} {main_metric} t={rec.get('time_s', float('nan')):.1f}s"
                    )
                    if dev.startswith("cuda"):
                        torch.cuda.empty_cache()
            write_markdown(doc, md_path)
    print("\n" + write_markdown(doc, md_path))
    print(f"results: {args.out}  table: {md_path}")


if __name__ == "__main__":
    main()
