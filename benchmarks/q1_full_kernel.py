#!/usr/bin/env python
"""Q1: TorchKM against the GPU kernel libraries, every method on the full kernel.

One table: rows = datasets, columns = test accuracy, wall-clock time and peak
GPU memory of each method, mean +- SE over repeats. Nothing is approximated:
no Nystrom centres, no random features; every method sees the full n x n
kernel.

Protocol (identical for every method)
  kernel      RBF exp(-2 sig d^2); one bandwidth per repeat from sigest on the
              training features, shared by every method (gamma = 2 sig for
              cuML/KeOps, bandwidth 1/(2 sqrt(sig)) for Falkon/EigenPro)
  grid        50 log-uniform lambda from 1e-2 down to 2e-5, the same for every
              dataset (--lam-max, --lam-min), swept from large to small lambda
              (TorchKM's path warm-starts each lambda from the previous one).
              The paper's grid was 1e-1..1e-5; at KKTeps=1e-6 TorchKM's KKT
              test fails for lambda >= 0.027 and at 1e-5 (a7a), so the default
              stays inside that band. The SVM solvers (TorchKM, cuML) are given
              C = 1/(2 n lambda), n = rows of the fit: TorchKM's own mapping
              between its penalised mean loss and the libsvm objective
              C * sum(loss) + ||w||^2 / 2. Falkon and KeOps take lambda as the
              ridge penalty. EigenPro has no lambda: its 50-value grid is the
              number of epochs, 1..50. A dataset whose selected value lies on
              an edge of the grid is flagged in the table: widen the range then
  selection   5-fold stratified CV on the same folds for every method, then
              one fit on the full training set at the selected value
  precision   float64 everywhere
  timing      wall clock for everything a user pays for a tuned model: kernel
              construction, the whole CV sweep, the final fit and the test
              predictions; CUDA synchronised; one-off library start-up (JIT
              compilation, context creation) is done before the clock starts
  memory      peak GPU memory of the process from NVML (comparable across
              libraries; falls back to whole-device peak where the driver hides
              per-process figures) and the PyTorch allocator peak where the
              library allocates through PyTorch

Methods
  torchkm     TorchKMSVC, hinge loss: one eigendecomposition of the kernel,
              the whole lambda path and the exact CV formula; is_exact=0 (the
              default), KKTeps from --kkt-eps. "Exceeded maximum delta
              iterations for lambda i" in the log means the KKT test was still
              unmet after --delta-len smoothing rounds for the i-th lambda of
              the path (small to large): the last iterate is kept and the
              table's note column shows the converged fraction. --delta-len 16
              gives the solver more rounds
  cuml        cuml.svm.SVC, hinge loss, SMO on the full kernel: one fit per
              (C, fold), 5 x 50 + 1 fits
  falkon      falkon.Falkon, squared loss, M = n centres (every training row,
              so no Nystrom approximation), preconditioned conjugate gradient:
              one fit per (lambda, fold)
  keops       kernel ridge regression, squared loss, matrix-free: the kernel is
              a pykeops LazyTensor that is never materialised and
              (K + n lambda I) alpha = y is solved by conjugate gradient. KeOps'
              own LazyTensor.solve is this loop without an iteration cap, so
              the 12-line CG below adds --keops-maxiter and a relative
              tolerance --keops-tol; one solve per (lambda, fold)
  eigenpro    eigenpro2.KernelModel (EigenPro 2, Ma and Belkin 2019), squared
              loss, every training row a centre, preconditioned SGD. One run of
              50 epochs per fold records the validation accuracy after each
              epoch; the best epoch count is refit on all rows. The package's
              LOBPCG eigensolver for the preconditioner returned NaN eigenpairs
              in float64, so this script gives it torch.linalg.eigh on the same
              2,000-row subsample kernel (the algorithm is otherwise untouched)

Install on the GPU machine (TorchKM's own environment plus):
  pip install pykeops
  pip install --no-build-isolation git+https://github.com/EigenPro/EigenPro-pytorch.git
      # (its setup.py imports torch, so the build must see the installed torch)
  pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
  # Falkon is not on PyPI: wheels for torch 2.4-2.7 / CUDA 11.8-12.8 on its own
  # index, keyed by the exact torch and CUDA version; otherwise build from source
  # (needs nvcc matching torch's CUDA): pip install --no-build-isolation
  # git+https://github.com/FalkonML/falkon.git
  TAG=$(python -c "import torch; print('torch-%s_cu%s' % (torch.__version__.split('+')[0],
                                                         torch.version.cuda.replace('.', '')))")
  pip install falkon -f https://falkon.dibris.unige.it/$TAG.html

Run (re-running with the same --out resumes; finished cells are skipped):
  python benchmarks/q1_full_kernel.py --data-dir ~/libsvm_data --out results/q1.json
  python benchmarks/q1_full_kernel.py --smoke    # CPU check on synthetic data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    PeakMemory,
    _jsonable,
    env_snapshot,
    fmt_bytes,
    load_dataset,
    make_folds,
    mean_se,
    synthetic_dataset,
)

# The paper's sets that fit the full kernel on one 48 GB GPU (n_train <= 33k).
DATASETS = [
    "a7a",
    "a8a",
    "a9a",
    "w7a",
    "mnist_3v8",
    "mnist_4v9",
    "ijcnn1_30k",
    "covtype_30k",
]
METHODS = ["torchkm", "cuml", "falkon", "keops", "eigenpro"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def sign_pm1(scores) -> np.ndarray:
    return np.where(np.asarray(scores, dtype=float).reshape(-1) > 0, 1.0, -1.0)


def in_chunks(fn: Callable, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Apply ``fn`` to row blocks so K_test never exceeds chunk x n_train."""
    parts = [
        np.asarray(fn(X[i : i + chunk])).reshape(-1) for i in range(0, len(X), chunk)
    ]
    return np.concatenate(parts)


def warm_rows(y: np.ndarray, per_class: int = 32) -> np.ndarray:
    """Row indices with both classes present, for the untimed warm-up fit."""
    return np.r_[np.flatnonzero(y > 0)[:per_class], np.flatnonzero(y < 0)[:per_class]]


def cv_select(fit_predict, X, y, foldid, grid, time_cap, t0) -> List[float]:
    """Mean validation accuracy of each grid value over the folds.

    Stops early once ``time_cap`` seconds have passed since ``t0``; the values
    completed so far are then used for selection.
    """
    cv_acc: List[float] = []
    for value in grid:
        accs = []
        for k in np.unique(foldid):
            va = foldid == k
            pred = fit_predict(value, X[~va], y[~va], X[va])
            accs.append(float(np.mean(sign_pm1(pred) == y[va])))
        cv_acc.append(float(np.mean(accs)))
        if time_cap and time.perf_counter() - t0 > time_cap:
            break
    return cv_acc


def result(pm: PeakMemory, dt: float, pred, yte, **fields) -> Dict[str, Any]:
    mem = pm.result
    return dict(
        status="ok",
        accuracy=float(np.mean(sign_pm1(pred) == yte)),
        time_s=float(dt),
        gpu_bytes=mem.get("nvml_process_peak") or mem.get("nvml_device_peak"),
        torch_bytes=mem.get("torch_max_allocated"),
        host_rss_bytes=mem.get("host_rss_peak"),
        **fields,
    )


def sweep(fit_predict, data, foldid, grid, dev, args, params, label="lambda"):
    """CV sweep, fit at the selected value, test predictions: one timed region."""
    Xtr, ytr, Xte, yte = data["Xtr"], data["ytr"], data["Xte"], data["yte"]
    w = warm_rows(ytr)
    fit_predict(grid[0], Xtr[w], ytr[w], Xte[:8])  # start-up costs, untimed
    sync(dev)
    with PeakMemory(dev) as pm:
        t0 = time.perf_counter()
        cv_acc = cv_select(fit_predict, Xtr, ytr, foldid, grid, args.time_cap, t0)
        best = int(np.argmax(cv_acc))
        pred = fit_predict(grid[best], Xtr, ytr, Xte)
        sync(dev)
        dt = time.perf_counter() - t0
    rec = result(
        pm,
        dt,
        pred,
        yte,
        selected=float(grid[best]),
        selected_label=label,
        cv_accuracy=cv_acc[best],
        cv_curve=cv_acc,
        grid_completed=len(cv_acc),
        grid_size=len(grid),
        params=params,
    )
    if len(cv_acc) < len(grid):
        rec["status"] = "capped"
    return rec


# ---------------------------------------------------------------------------
# The five methods
# ---------------------------------------------------------------------------


def run_torchkm(data, sig, lams, foldid, dev, args, seed):
    from torchkm.estimators import TorchKMSVC

    # The estimator takes C and forms lambda = 1/(2 n C) itself, n = training
    # rows. The path must run from large to small lambda (each solution warm-starts
    # the next), so the grid is handed over in that order: C ascending.
    n = data["Xtr"].shape[0]
    Cs = 1.0 / (2.0 * n * lams)
    clf = TorchKMSVC(
        kernel="rbf",
        rbf_sigma=sig,
        Cs=Cs,
        nC=len(Cs),
        cv=args.folds,
        foldid=foldid,
        device=dev,
        tol=args.tol,
        max_iter=args.max_iter,
        KKTeps=args.kkt_eps,
        delta_len=args.delta_len,
        is_exact=0,
        random_state=seed,
    )
    # start-up: CUDA context, cuSOLVER/cuBLAS handles
    torch.linalg.eigh(torch.eye(64, dtype=torch.float64, device=dev))
    sync(dev)
    with PeakMemory(dev) as pm:
        t0 = time.perf_counter()
        clf.fit(data["Xtr"], data["ytr"])
        pred = in_chunks(clf.decision_function, data["Xte"])
        sync(dev)
        dt = time.perf_counter() - t0
    conv = clf.converged_
    return result(
        pm,
        dt,
        pred,
        data["yte"],
        selected=float(1.0 / (2.0 * n * clf.best_C_)),
        selected_label="lambda",
        cv_accuracy=1.0 - float(clf.cv_mis_[clf.best_ind_]),
        cv_curve=(1.0 - np.asarray(clf.cv_mis_, dtype=float)).tolist(),
        grid_completed=len(lams),
        grid_size=len(lams),
        converged_frac=None if conv is None else float(np.mean(conv)),
        params=dict(
            loss="hinge",
            solver="eigendecomposition + lambda path + exact CV",
            is_exact=0,
            tol=args.tol,
            max_iter=args.max_iter,
            KKTeps=args.kkt_eps,
            delta_len=args.delta_len,
            C="1/(2 n lambda), n = training rows",
            dtype="float64",
        ),
    )


def run_cuml(data, sig, lams, foldid, dev, args, seed):
    from cuml.svm import SVC

    gamma = 2.0 * sig

    def fit_predict(lam, Xa, ya, Xb):
        model = SVC(
            kernel="rbf",
            C=1.0 / (2.0 * Xa.shape[0] * lam),
            gamma=gamma,
            cache_size=args.svc_cache_mb,
            output_type="numpy",
        )
        model.fit(Xa, ya)
        return model.decision_function(Xb)

    params = dict(
        loss="hinge", solver="SMO", gamma=gamma, cache_size_mb=args.svc_cache_mb
    )
    params.update(C="1/(2 n lambda), n = rows of the fit", dtype="float64")
    return sweep(fit_predict, data, foldid, list(map(float, lams)), dev, args, params)


def run_falkon(data, sig, lams, foldid, dev, args, seed):
    import falkon
    from falkon.kernels import GaussianKernel

    kernel = GaussianKernel(sigma=1.0 / (2.0 * math.sqrt(sig)))
    options = falkon.FalkonOptions(
        use_cpu=not dev.startswith("cuda"), keops_active="no", debug=False
    )

    def fit_predict(lam, Xa, ya, Xb):
        model = falkon.Falkon(
            kernel=kernel,
            penalty=float(lam),
            M=Xa.shape[0],  # every training row is a centre: full kernel, no Nystrom
            maxiter=args.falkon_maxiter,
            seed=seed,
            options=options,
        )
        model.fit(torch.from_numpy(Xa), torch.from_numpy(ya).reshape(-1, 1))
        return model.predict(torch.from_numpy(Xb)).reshape(-1).cpu().numpy()

    params = dict(
        loss="squared",
        solver="preconditioned CG, M = n",
        maxiter=args.falkon_maxiter,
        dtype="float64",
    )
    return sweep(fit_predict, data, foldid, list(map(float, lams)), dev, args, params)


def conjugate_gradient(matvec, b, ridge, tol, maxiter):
    """Solve (A + ridge I) x = b for symmetric positive definite A given as ``matvec``."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = float(r.pow(2).sum())
    stop = tol**2 * float(b.pow(2).sum())
    for it in range(1, maxiter + 1):
        Ap = matvec(p) + ridge * p
        a = rr / float((p * Ap).sum())
        x += a * p
        r -= a * Ap
        rr_new = float(r.pow(2).sum())
        if rr_new < stop:
            return x, it, False
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, maxiter, True


def run_keops(data, sig, lams, foldid, dev, args, seed):
    from pykeops.torch import LazyTensor

    gamma = 2.0 * sig
    hits = [0]  # solves that stopped at --keops-maxiter

    def kernel_matvec(rows, cols):
        """v -> K(rows, cols) v with K = exp(-gamma |x - x'|^2), never materialised."""
        x_i, x_j = LazyTensor(rows[:, None, :]), LazyTensor(cols[None, :, :])
        K = (-gamma * ((x_i - x_j) ** 2).sum(-1)).exp()
        return lambda v: K @ v

    def fit_predict(lam, Xa, ya, Xb):
        xa, xb = torch.from_numpy(Xa).to(dev), torch.from_numpy(Xb).to(dev)
        y = torch.from_numpy(ya).to(dev).reshape(-1, 1)
        # (K + n lambda I) alpha = y: the normal equations of mean squared loss + lambda penalty
        alpha, _, capped = conjugate_gradient(
            kernel_matvec(xa, xa),
            y,
            float(xa.shape[0] * lam),
            args.keops_tol,
            args.keops_maxiter,
        )
        hits[0] += int(capped)
        return kernel_matvec(xb, xa)(alpha).reshape(-1).cpu().numpy()

    params = dict(
        loss="squared",
        solver="matrix-free CG on a LazyTensor kernel",
        cg_tol=args.keops_tol,
        cg_maxiter=args.keops_maxiter,
        dtype="float64",
    )
    rec = sweep(fit_predict, data, foldid, list(map(float, lams)), dev, args, params)
    rec["cg_maxiter_hits"] = hits[0]
    return rec


def exact_subsample_eigh(samples, kernel_fn, top_q):
    """Top eigenpairs of the subsample kernel by torch.linalg.eigh.

    Drop-in for ``eigenpro2.utils.eigh.nystrom_kernel_eigh`` (same return
    convention), which uses LOBPCG for q = 666 of 2,000 and gave NaNs.
    """
    kmat = kernel_fn(samples, samples)
    n_s = samples.shape[0]
    vals, vecs = torch.linalg.eigh(kmat / n_s)
    k = min(top_q + 1, n_s // 3)
    return vals.flip(0)[:k], vecs.flip(1)[:, :k] / math.sqrt(n_s), kmat.diag().max()


def run_eigenpro(data, sig, lams, foldid, dev, args, seed):  # lams unused: epochs grid
    import eigenpro2.models as epm
    from eigenpro2.kernels import gaussian

    epm.nystrom_kernel_eigh = exact_subsample_eigh
    bandwidth = 1.0 / (2.0 * math.sqrt(sig))
    epochs = args.grid_size
    mem_gb = (
        torch.cuda.get_device_properties(dev).total_memory / 2**30 / 2  # float64
        if dev.startswith("cuda")
        else 8
    )
    Xtr, ytr, Xte, yte = data["Xtr"], data["ytr"], data["Xte"], data["yte"]

    def kernel_fn(a, b):
        return gaussian(a, b, bandwidth=bandwidth)

    def onehot(v):  # column 0 = class -1, column 1 = class +1
        return torch.from_numpy(np.stack([v < 0, v > 0], 1).astype(np.float64)).to(dev)

    def train(Xa, ya, n_epochs, Xb=None, yb=None):
        torch.manual_seed(seed)
        xa = torch.from_numpy(Xa).to(dev)
        model = epm.KernelModel(kernel_fn, xa, 2, device=dev)
        res = model.fit(
            xa,
            onehot(ya),
            None if Xb is None else torch.from_numpy(Xb).to(dev),
            None if yb is None else onehot(yb),
            epochs=n_epochs,
            mem_gb=mem_gb,
            print_every=1,
            run_epoch_eval=Xb is not None,
        )
        return model, res

    def predict(model, X):
        def block(Z):
            z = torch.from_numpy(np.ascontiguousarray(Z)).to(dev)
            return (model.forward(z).argmax(1) * 2 - 1).cpu().numpy()

        return in_chunks(block, X)

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # the package's weights follow the default
    try:
        train(Xtr[:256], ytr[:256], 1)  # start-up, untimed
        sync(dev)
        with PeakMemory(dev) as pm:
            t0 = time.perf_counter()
            curves = []
            for k in np.unique(foldid):
                va = foldid == k
                model, res = train(Xtr[~va], ytr[~va], epochs, Xtr[va], ytr[va])
                curves.append(
                    [float(res[e][1]["multiclass-acc"]) for e in range(epochs)]
                )
                del model
            cv_acc = np.mean(curves, axis=0).tolist()
            best = int(np.argmax(cv_acc))
            model, _ = train(Xtr, ytr, best + 1)
            pred = predict(model, Xte)
            sync(dev)
            dt = time.perf_counter() - t0
    finally:
        torch.set_default_dtype(default_dtype)
    return result(
        pm,
        dt,
        pred,
        yte,
        selected=float(best + 1),
        selected_label="epochs",
        cv_accuracy=cv_acc[best],
        cv_curve=cv_acc,
        grid_completed=epochs,
        grid_size=epochs,
        params=dict(
            loss="squared",
            solver="EigenPro 2 preconditioned SGD, all rows as centres",
            preconditioner="top eigenvectors of a 2,000-row subsample kernel (torch.linalg.eigh)",
            dtype="float64",
        ),
    )


RUN = dict(
    torchkm=run_torchkm,
    cuml=run_cuml,
    falkon=run_falkon,
    keops=run_keops,
    eigenpro=run_eigenpro,
)


def unavailable(method: str, dev: str) -> Optional[str]:
    """None when the method can run here, otherwise the reason it cannot."""
    try:
        if method == "cuml":
            if not dev.startswith("cuda"):
                return "cuML needs a CUDA device"
            import cuml.svm  # noqa: F401
        elif method == "falkon":
            import falkon  # noqa: F401
        elif method == "keops":
            import pykeops.torch  # noqa: F401
        elif method == "eigenpro":
            import eigenpro2  # noqa: F401
        else:
            import torchkm  # noqa: F401
    except Exception as err:  # ImportError, or a broken CUDA build
        return f"{type(err).__name__}: {err}"[:300]
    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_json(doc: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(_jsonable(doc), fh, indent=1)
    os.replace(path + ".tmp", path)


def at_grid_edge(rec: Dict[str, Any], doc: Dict[str, Any]) -> bool:
    """True when the selected value is the first or last grid value."""
    if rec.get("selected_label") == "epochs":
        return rec["selected"] >= rec["grid_size"]
    lo, hi = min(doc["grid_lambda"]), max(doc["grid_lambda"])
    return bool(np.isclose(rec["selected"], lo) or np.isclose(rec["selected"], hi))


def write_markdown(doc: Dict[str, Any], path: str) -> str:
    env, a = doc["environment"], doc["args"]
    gpu = (env.get("gpu") or {}).get("name") or "no GPU (CPU run)"
    lines = [
        "# Q1: TorchKM vs GPU kernel libraries, full kernel",
        "",
        f"{gpu}; torch {env.get('torch')}; torchkm {env.get('torchkm')} "
        f"({str(env.get('torchkm_commit'))[:10]}); float64 everywhere.",
        f"{a['folds']}-fold CV on shared stratified folds, {a['grid_size']} grid values "
        f"(lambda in [{a['lam_min']:g}, {a['lam_max']:g}]; epochs 1..{a['grid_size']} for EigenPro), "
        f"{a['repeats']} repeats (seeds {a['seed']}..{a['seed'] + a['repeats'] - 1}).",
        "Time = CV sweep + final fit + test predictions. Memory = NVML peak of the process.",
        "Cells are mean +- SE over repeats; 'selected' lists the chosen value per repeat.",
        "",
        "| dataset | n_train | n_test | p | method | test accuracy | time (s) | GPU memory | selected | note |",
        "|---|---:|---:|---:|---|---|---|---|---|---|",
    ]
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in doc["records"]:
        groups.setdefault((r["dataset"], r["method"]), []).append(r)
    for (ds, m), recs in groups.items():
        ok = [r for r in recs if r["status"] in ("ok", "capped")]
        head = f"| {ds} | {recs[0]['n_train']:,} | {recs[0]['n_test']:,} | {recs[0]['p']} | {m} |"
        if not ok:
            note = "; ".join(
                sorted({r.get("note") or r.get("error") or r["status"] for r in recs})
            )
            lines.append(f"{head} - | - | - | - | {note[:120]} |")
            continue
        acc = mean_se([r["accuracy"] for r in ok])
        t = mean_se([r["time_s"] for r in ok])
        mem = mean_se([r["gpu_bytes"] for r in ok])
        label = ok[0]["selected_label"]
        sel = ", ".join(
            f"{r['selected']:.0f}" if label == "epochs" else f"{r['selected']:.3g}"
            for r in ok
        )
        notes = []
        capped = [r for r in ok if r["status"] == "capped"]
        if capped:
            notes.append(
                "capped: "
                + ", ".join(f"{r['grid_completed']}/{r['grid_size']}" for r in capped)
            )
        hits = sum(r.get("cg_maxiter_hits") or 0 for r in ok)
        if hits:
            notes.append(f"CG cap hit in {hits} solves")
        conv = [r["converged_frac"] for r in ok if r.get("converged_frac") is not None]
        if conv and min(conv) < 1:
            notes.append(
                "solver converged on "
                + ", ".join(f"{c:.0%}" for c in conv)
                + " of the grid"
            )
        if len(ok) < len(recs):
            notes.append(f"{len(recs) - len(ok)} repeat(s) failed")
        edge = [f"r{r['repeat']}" for r in ok if at_grid_edge(r, doc)]
        if edge:
            notes.append(
                "selected at grid edge (" + ", ".join(edge) + "): widen the range"
            )
        lines.append(
            f"{head} {acc[0]:.4f} +- {acc[1]:.4f} | {t[0]:.1f} +- {t[1]:.1f} | "
            f"{fmt_bytes(mem[0])} | {label} = {sel} | {'; '.join(notes)} |"
        )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default=None, help="directory of LIBSVM files")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    ap.add_argument("--repeats", type=int, default=3, help="seeds per dataset")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--grid-size", type=int, default=50, help="lambda values (and epochs)"
    )
    ap.add_argument("--lam-max", type=float, default=1e-2, help="largest lambda")
    ap.add_argument("--lam-min", type=float, default=2e-5, help="smallest lambda")
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument(
        "--time-cap",
        type=float,
        default=7200,
        help="seconds per method x dataset x repeat before a CV sweep stops (0: none)",
    )
    ap.add_argument("--kkt-eps", type=float, default=1e-6, help="TorchKM KKT tolerance")
    ap.add_argument("--delta-len", type=int, default=8, help="TorchKM smoothing rounds")
    ap.add_argument("--tol", type=float, default=1e-5, help="TorchKM step tolerance")
    ap.add_argument(
        "--max-iter", type=int, default=100_000, help="TorchKM iteration cap"
    )
    ap.add_argument(
        "--svc-cache-mb", type=float, default=2000, help="cuML kernel cache"
    )
    ap.add_argument(
        "--falkon-maxiter", type=int, default=20, help="Falkon CG iterations"
    )
    ap.add_argument(
        "--keops-maxiter", type=int, default=500, help="KeOps CG iteration cap"
    )
    ap.add_argument(
        "--keops-tol", type=float, default=1e-6, help="KeOps CG relative tolerance"
    )
    ap.add_argument("--out", default="benchmarks/results/q1_full_kernel.json")
    ap.add_argument(
        "--smoke", action="store_true", help="tiny synthetic CPU-sized check"
    )
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress lines reach the log at once

    if args.smoke:
        args.datasets, args.folds, args.grid_size, args.repeats = ["synthetic"], 3, 4, 1
        args.keops_maxiter, args.time_cap = 50, 0
    dev = (
        "cuda"
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    if dev != args.device:
        print(f"[warn] {args.device} not available, running on cpu")
    reasons = {m: unavailable(m, dev) for m in args.methods}
    for m, why in reasons.items():
        if why:
            print(f"[skip] {m}: {why}")

    from torchkm import sigest

    # large to small: the order TorchKM's path and every sweep below run in
    lams = np.logspace(np.log10(args.lam_max), np.log10(args.lam_min), args.grid_size)
    doc: Dict[str, Any] = dict(
        script="q1_full_kernel.py",
        args=vars(args),
        grid_lambda=lams.tolist(),
        environment=env_snapshot(),
        records=[],
    )
    if os.path.exists(args.out):  # resume: keep finished cells, redo the rest
        with open(args.out) as fh:
            old = json.load(fh)
        protocol = ("folds", "grid_size", "lam_max", "lam_min", "seed")
        if any(old.get("args", {}).get(k) != vars(args)[k] for k in protocol):
            sys.exit(
                f"{args.out} was written with a different protocol (folds, grid or seed): "
                "use a new --out or delete it"
            )
        doc["records"] = old.get("records", [])
    done = {
        (r["dataset"], r["method"], r["repeat"])
        for r in doc["records"]
        if r["status"] in ("ok", "capped")
    }
    md_path = os.path.splitext(args.out)[0] + ".md"
    print(
        f"device={dev} datasets={args.datasets} methods={args.methods} out={args.out}"
    )

    for ds in args.datasets:
        if ds == "synthetic":
            data = synthetic_dataset(600, 10, args.seed)
        else:
            data = load_dataset(ds, args.data_dir, seed=args.seed)
        print(
            f"\n== {ds}: n_train={data['n_train']:,} n_test={data['n_test']:,} "
            f"p={data['p']} positive fraction={data['pos_frac']:.3f}"
        )
        for r in range(args.repeats):
            seed = args.seed + r
            torch.manual_seed(seed)
            sig = float(sigest(torch.from_numpy(data["Xtr"])))
            foldid = make_folds(data["ytr"], args.folds, seed)
            for m in args.methods:
                if (ds, m, r) in done:
                    continue
                info = dict(
                    dataset=ds,
                    n_train=data["n_train"],
                    n_test=data["n_test"],
                    p=data["p"],
                    repeat=r,
                    seed=seed,
                    method=m,
                    bandwidth_sigest=sig,
                )
                if reasons[m]:
                    rec = dict(status="unavailable", note=reasons[m])
                else:
                    try:
                        rec = RUN[m](data, sig, lams, foldid, dev, args, seed)
                    except Exception as err:  # the next cell still runs
                        traceback.print_exc()
                        rec = dict(
                            status="failed", error=f"{type(err).__name__}: {err}"[:800]
                        )
                rec.update(info)
                doc["records"].append(rec)
                save_json(doc, args.out)
                print(
                    f"   {m:9s} r{r} {rec['status']:11s} acc={rec.get('accuracy', float('nan')):.4f} "
                    f"t={rec.get('time_s', float('nan')):8.1f}s mem={fmt_bytes(rec.get('gpu_bytes'))}"
                    + (
                        f" {rec.get('selected_label')}={rec.get('selected'):.3g}"
                        + (" (grid edge)" if at_grid_edge(rec, doc) else "")
                        if "selected" in rec
                        else ""
                    )
                )
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
        write_markdown(doc, md_path)
    print("\n" + write_markdown(doc, md_path))
    print(f"results: {args.out}  table: {md_path}")


if __name__ == "__main__":
    main()
