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
  selection   10-fold stratified CV (the paper's protocol; --folds) on the same
              folds for every method, then
              one fit on the full training set at the selected value
  precision   float32 everywhere by default (--dtype float64 for double): the
              features go to every method in that precision and each computes
              in it; the bandwidth is estimated in float64 either way
  timing      wall clock for everything a user pays for a tuned model: kernel
              construction, the whole CV sweep, the final fit and the test
              predictions; CUDA synchronised; one-off library start-up (JIT
              compilation, context creation) is done before the clock starts
  memory      peak GPU memory of the process from NVML (comparable across
              libraries; falls back to whole-device peak where the driver hides
              per-process figures) and the PyTorch allocator peak where the
              library allocates through PyTorch
  objective   Table 2's column for the two hinge-loss solvers: the SVM
              objective, equation (1), (1/n) sum (1 - y f)_+ + lambda a'Ka, of
              the fit on all training rows, in float64 on the training kernel
              and outside the timed region. cuML's is at the lambda it
              selected; TorchKM's is read off its path at the same lambda, so
              the two compare as solvers (at TorchKM's own lambda when cuML
              has no result). Falkon, KeOps and EigenPro minimise squared
              loss, so the column is empty for them

Datasets
  real        a7a, a8a, a9a, w7a, MNIST 3v8, MNIST 4v9, ijcnn1 (30k stratified
              subsample), covtype (30k subsample, 20k test): the paper's sets at
              sizes where the full kernel fits one 48 GB GPU (--data-dir)
  simulation  Table 2's Gaussian mixture (torchkm.data_gen: 5 centres per class,
              shift 2, noise 3, standardised) at Table 2's cells, named
              sim_<n>x<p>: n = 10,000 and 20,000 rows, p = 10, 100, 1000. The
              data are redrawn for every repeat (seed 52 + repeat); the test set
              is n/10 rows from the same mixture. Any sim_<n>x<p> works

Methods (default: all six; --methods picks a subset)
  torchkm     TorchKMSVC, hinge loss: one eigendecomposition of the kernel,
              the whole lambda path and the exact CV formula; is_exact=0 (the
              default), KKTeps from --kkt-eps, dtype from --dtype. "Exceeded maximum delta
              iterations for lambda i" in the log means the KKT test was still
              unmet after --delta-len smoothing rounds for the i-th lambda of
              the path (small to large): the last iterate is kept and the
              table's note column shows the converged fraction. --delta-len 16
              gives the solver more rounds. The eigendecomposition overwrites
              the kernel and the kernel is rebuilt, so the peak is 5 n x n
              matrices (kernel's storage, cuSOLVER's 4-matrix workspace)
  torchkm_trunc
              TorchKMSVC(spectrum="truncated"), experimental: the same kernel,
              but only its top --trunc-rank eigenpairs, so no full
              eigendecomposition (peak about 1.2 n x n matrices); every lambda
              and every fold stops at the certified relative duality gap
              --gap-tol. --trunc-block lambdas are fitted together with all
              their folds, so each product with K serves block x (folds + 1)
              fits. --tol, --kkt-eps and --delta-len do not apply
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
              2,000-row subsample kernel (the algorithm is otherwise untouched;
              exact top eigenpairs in either precision)

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

Run (re-running with the same --out resumes: a finished cell is kept when it was
computed with the same settings for its method, e.g. TorchKM's --tol and --kkt-eps,
and computed again otherwise):
  python benchmarks/q1_full_kernel.py --data-dir ~/libsvm_data --out results/q1.json
  python benchmarks/q1_full_kernel.py --datasets sim_10000x100 --repeats 1   # one cell
  python benchmarks/q1_full_kernel.py --smoke    # CPU check on a tiny simulation
"""

from __future__ import annotations

import argparse
import csv
import inspect
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
    classification_metrics,
    _jsonable,
    env_snapshot,
    fmt_bytes,
    load_dataset,
    make_folds,
    mean_se,
    synthetic_dataset,
)

# The paper's sets that fit the full kernel on one 48 GB GPU (n_train <= 32,768,
# the largest n cuSOLVER's eigendecomposition accepts; a9a has 32,561).
DATASETS = [
    "a7a",
    "a8a",
    "a9a",
    "w7a",
    "mnist_3v8",
    "mnist_4v9",
    "ijcnn1_30k",
    "covtype_30k",
    # Table 2's simulation cells (Gaussian mixture), redrawn per repeat
    "sim_10000x10",
    "sim_10000x100",
    "sim_10000x1000",
    "sim_20000x10",
    "sim_20000x100",
    "sim_20000x1000",
]
METHODS = ["torchkm", "torchkm_trunc", "cuml", "falkon", "keops", "eigenpro"]
TORCHKM = ("torchkm", "torchkm_trunc")


def parse_sim(name: str) -> Optional[tuple]:
    """``sim_<n>x<p>`` -> (n, p); None for a real dataset."""
    if not name.startswith("sim_"):
        return None
    n, p = name[4:].split("x")
    return int(n), int(p)


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


def cast(data: Dict[str, Any], dtype: str) -> Dict[str, Any]:
    """The dataset with its feature and label arrays in ``dtype``."""
    arrays = {k: data[k].astype(dtype) for k in ("Xtr", "ytr", "Xte", "yte")}
    return {**data, **arrays}


def svm_objectives(
    X, y, sig, alphas, lams, dev, *, intercepts=None, scores=None, chunk=4096
) -> List[float]:
    """Equation (1), (1/n) sum_i (1 - y_i f_i)_+ + lambda a'Ka, for each column
    of ``alphas`` at the matching ``lams``: float64, training kernel
    exp(-2 sig |x - x'|^2) built in row blocks. f = K a + b, or the method's own
    training ``scores`` (cuML: they carry the sign convention of its dual
    coefficients, which the penalty a'Ka does not see)."""
    from torchkm.functions import kernelMult

    n = X.shape[0]
    Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
    A = torch.as_tensor(np.asarray(alphas, np.float64).reshape(n, -1), device=dev)
    KA = torch.cat(
        [kernelMult(Xt[i : i + chunk], Xt, sig) @ A for i in range(0, n, chunk)]
    )
    if scores is None:
        F = KA + torch.as_tensor(np.asarray(intercepts, np.float64), device=dev)
    else:
        F = torch.as_tensor(np.asarray(scores, np.float64).reshape(n, -1), device=dev)
    yt = torch.as_tensor(np.asarray(y, np.float64), device=dev).reshape(-1, 1)
    hinge = torch.clamp(1.0 - yt * F, min=0.0).mean(0)
    lam = torch.as_tensor(np.asarray(lams, np.float64), device=dev)
    return (hinge + lam * (A * KA).sum(0)).cpu().tolist()


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
    """A finished cell: test accuracy, balanced accuracy and AUC from the decision
    scores, wall-clock time and peak memory."""
    mem = pm.result
    metrics = classification_metrics(yte, pred)
    return dict(
        status="ok",
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        auc=metrics["auc"],
        test_pos_frac=metrics["test_pos_frac"],
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


def run_torchkm(data, sig, lams, foldid, dev, args, seed, truncated=False):
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
        dtype=args.dtype,
        random_state=seed,
        store_path=True,  # the whole-data solution at every lambda, for the objective
        spectrum="truncated" if truncated else "full",
        spectrum_rank=args.trunc_rank,
        gap_tol=args.gap_tol,
        spectrum_block=args.trunc_block,
    )
    # start-up: CUDA context, cuSOLVER/cuBLAS handles
    torch.linalg.eigh(torch.eye(64, dtype=getattr(torch, args.dtype), device=dev))
    sync(dev)
    with PeakMemory(dev) as pm:
        t0 = time.perf_counter()
        clf.fit(data["Xtr"], data["ytr"])
        sync(dev)
        t_fit = time.perf_counter()
        pred = in_chunks(clf.decision_function, data["Xte"])
        sync(dev)
        dt = time.perf_counter() - t0
    conv = clf.converged_
    profile = dict(clf.fit_timing_, prediction=dt - (t_fit - t0))
    path = clf.alpmat_path_.double().numpy()  # row 0 the intercepts, then alpha
    objective = svm_objectives(
        data["Xtr"], data["ytr"], sig, path[1:], lams, dev, intercepts=path[0]
    )
    return result(
        pm,
        dt,
        pred,
        data["yte"],
        selected=float(1.0 / (2.0 * n * clf.best_C_)),
        selected_label="lambda",
        objective=objective[clf.best_ind_],
        objective_path=objective,
        time_profile=profile,
        passes=clf.n_passes_,
        fit_profile=clf.fit_profile_,
        cv_accuracy=1.0 - float(clf.cv_mis_[clf.best_ind_]),
        cv_curve=(1.0 - np.asarray(clf.cv_mis_, dtype=float)).tolist(),
        grid_completed=len(lams),
        grid_size=len(lams),
        converged_frac=None if conv is None else float(np.mean(conv)),
        params=dict(
            cell_settings("torchkm_trunc" if truncated else "torchkm", args),
            loss="hinge",
            solver=(
                "top eigenpairs + lambda path + CV fits, each at a certified gap"
                if truncated
                else "eigendecomposition + lambda path + exact CV"
            ),
            C="1/(2 n lambda), n = training rows",
        ),
    )


def run_torchkm_trunc(data, sig, lams, foldid, dev, args, seed):
    """TorchKMSVC(spectrum="truncated"): the same kernel, no full eigendecomposition."""
    return run_torchkm(data, sig, lams, foldid, dev, args, seed, truncated=True)


def run_cuml(data, sig, lams, foldid, dev, args, seed):
    from cuml.svm import SVC

    gamma = 2.0 * sig
    last = {}

    def fit_predict(lam, Xa, ya, Xb):
        model = SVC(
            kernel="rbf",
            C=1.0 / (2.0 * Xa.shape[0] * lam),
            gamma=gamma,
            cache_size=args.svc_cache_mb,
            output_type="numpy",
        )
        model.fit(Xa, ya)
        last["model"] = model
        return model.decision_function(Xb)

    params = dict(
        loss="hinge", solver="SMO", gamma=gamma, cache_size_mb=args.svc_cache_mb
    )
    params.update(C="1/(2 n lambda), n = rows of the fit", dtype=args.dtype)
    rec = sweep(fit_predict, data, foldid, list(map(float, lams)), dev, args, params)
    model = last["model"]  # sweep's last fit: all training rows, selected lambda
    alpha = np.zeros(data["Xtr"].shape[0])
    alpha[np.asarray(model.support_)] = np.asarray(model.dual_coef_).ravel()
    scores = model.decision_function(data["Xtr"])
    rec["objective"] = svm_objectives(
        data["Xtr"], data["ytr"], sig, alpha, [rec["selected"]], dev, scores=scores
    )[0]
    return rec


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
        dtype=args.dtype,
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
        dtype=args.dtype,
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
    dtype = getattr(torch, args.dtype)
    # the package sizes its batches for 4-byte entries: halve the budget in float64
    mem_gb = (
        torch.cuda.get_device_properties(dev).total_memory
        / 2**30
        / (torch.finfo(dtype).bits // 32)
        if dev.startswith("cuda")
        else 8
    )
    Xtr, ytr, Xte, yte = data["Xtr"], data["ytr"], data["Xte"], data["yte"]

    def kernel_fn(a, b):
        return gaussian(a, b, bandwidth=bandwidth)

    def onehot(v):  # column 0 = class -1, column 1 = class +1
        return torch.from_numpy(np.stack([v < 0, v > 0], 1).astype(v.dtype)).to(dev)

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
            out = model.forward(z)  # score of class +1 minus score of class -1
            return (out[:, 1] - out[:, 0]).cpu().numpy()

        return in_chunks(block, X)

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # the package's weights follow the default
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
            dtype=args.dtype,
        ),
    )


RUN = dict(
    torchkm=run_torchkm,
    torchkm_trunc=run_torchkm_trunc,
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


def cell_settings(method: str, args: argparse.Namespace) -> Dict[str, Any]:
    """The solver settings a method's result depends on, as stored in its record."""
    if method == "torchkm":
        own = dict(
            is_exact=0,
            tol=args.tol,
            max_iter=args.max_iter,
            KKTeps=args.kkt_eps,
            delta_len=args.delta_len,
        )
    elif method == "torchkm_trunc":
        from torchkm.experimental import SpectralSVMPath

        # bias: the solver's default, recorded so that cells computed before it
        # changed (from 0.5 to 4) are computed again
        bias = inspect.signature(SpectralSVMPath).parameters["bias"].default
        own = dict(
            spectrum_rank=args.trunc_rank,
            gap_tol=args.gap_tol,
            spectrum_block=args.trunc_block,
            certify_bias=bias,
        )
    elif method == "cuml":
        own = dict(cache_size_mb=args.svc_cache_mb)
    elif method == "falkon":
        own = dict(maxiter=args.falkon_maxiter)
    elif method == "keops":
        own = dict(cg_tol=args.keops_tol, cg_maxiter=args.keops_maxiter)
    else:
        own = {}
    return dict(own, dtype=args.dtype)


def reusable(rec: Dict[str, Any], args: argparse.Namespace) -> bool:
    """A finished cell computed with the same settings as this run."""
    if rec.get("status") not in ("ok", "capped"):
        return False
    recorded = {"objective_path", "time_profile", "fit_profile"}
    if rec["method"] in TORCHKM and not recorded <= set(rec):
        return False  # computed before the objective and profile were recorded
    if rec["method"] == "cuml" and "objective" not in rec:
        return False
    have = rec.get("params") or {}
    return all(have.get(k) == v for k, v in cell_settings(rec["method"], args).items())


def at_grid_edge(rec: Dict[str, Any], doc: Dict[str, Any]) -> bool:
    """True when the selected value is the first or last grid value."""
    if rec.get("selected_label") == "epochs":
        return rec["selected"] >= rec["grid_size"]
    lo, hi = min(doc["grid_lambda"]), max(doc["grid_lambda"])
    return bool(np.isclose(rec["selected"], lo) or np.isclose(rec["selected"], hi))


def shown_objective(rec: Dict[str, Any], doc: Dict[str, Any]) -> Optional[float]:
    """The SVM objective the table shows: cuML's at its lambda, TorchKM's at the
    lambda cuML selected in the same repeat (read off its path), else its own."""
    if rec["method"] in TORCHKM and rec.get("objective_path"):
        for c in doc["records"]:
            if (
                c["method"] == "cuml"
                and (c["dataset"], c["repeat"]) == (rec["dataset"], rec["repeat"])
                and c.get("objective") is not None
            ):
                grid = np.log(np.asarray(doc["grid_lambda"]))
                j = int(np.argmin(np.abs(grid - np.log(c["selected"]))))
                return rec["objective_path"][j]
    return rec.get("objective")


def write_markdown(doc: Dict[str, Any], path: str) -> str:
    env, a = doc["environment"], doc["args"]
    gpu = (env.get("gpu") or {}).get("name") or "no GPU (CPU run)"
    lines = [
        "# Q1: TorchKM vs GPU kernel libraries, full kernel",
        "",
        f"{gpu}; torch {env.get('torch')}; torchkm {env.get('torchkm')} "
        f"({str(env.get('torchkm_commit'))[:10]}); {a.get('dtype', 'float64')} everywhere.",
        f"{a['folds']}-fold CV on shared stratified folds, {a['grid_size']} grid values "
        f"(lambda in [{a['lam_min']:g}, {a['lam_max']:g}]; epochs 1..{a['grid_size']} for EigenPro), "
        f"seed {a['seed']} + repeat index; 'runs' = repeats in that row.",
        "Time = CV sweep + final fit + test predictions. Memory = NVML peak of the process.",
        "SVM objective = equation (1), (1/n) sum (1 - y f)_+ + lambda a'Ka, of the fit on all "
        "training rows, in float64 on the training kernel (Table 2's column), for the "
        "hinge-loss solvers; TorchKM's is read off its path at the lambda cuML selected in the "
        "same repeat, so the two compare at the same lambda.",
        "Cells are mean +- SE over repeats; 'selected' lists the chosen value per repeat "
        "(median and range beyond five repeats).",
        "",
        "| dataset | n_train | n_test | p | positive | method | runs | test accuracy "
        "| balanced accuracy | AUC | SVM objective | time (s) | GPU memory | selected | note |",
        "|---|---:|---:|---:|---:|---|---:|---|---|---|---|---|---|---|---|",
    ]
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in doc["records"]:
        groups.setdefault((r["dataset"], r["method"]), []).append(r)
    for (ds, m), recs in groups.items():
        ok = [r for r in recs if r["status"] in ("ok", "capped")]
        prior = recs[0].get("train_pos_frac")
        prior_s = "-" if prior is None else f"{prior:.1%}"
        head = (
            f"| {ds} | {recs[0]['n_train']:,} | {recs[0]['n_test']:,} | {recs[0]['p']} "
            f"| {prior_s} | {m} |"
        )
        if not ok:
            note = "; ".join(
                sorted({r.get("note") or r.get("error") or r["status"] for r in recs})
            )
            lines.append(f"{head} 0 | - | - | - | - | - | - | - | {note[:120]} |")
            continue
        acc = mean_se([r["accuracy"] for r in ok])

        def pm_se(vals) -> str:
            vals = [v for v in vals if v is not None]
            if not vals:
                return "-"
            m_, se_ = mean_se(vals)
            return f"{m_:.4f} +- {se_:.4f}"

        t = mean_se([r["time_s"] for r in ok])
        mem = mean_se([r["gpu_bytes"] for r in ok])
        label = ok[0]["selected_label"]
        values = [r["selected"] for r in ok]
        fmt_v = (lambda v: f"{v:.0f}") if label == "epochs" else (lambda v: f"{v:.3g}")
        if len(values) <= 5:
            sel = ", ".join(fmt_v(v) for v in values)
        else:
            sel = (
                f"median {fmt_v(float(np.median(values)))}, "
                f"range {fmt_v(min(values))} to {fmt_v(max(values))}"
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
            f"{head} {len(ok)} | {acc[0]:.4f} +- {acc[1]:.4f} "
            f"| {pm_se(r.get('balanced_accuracy') for r in ok)} "
            f"| {pm_se(r.get('auc') for r in ok)} "
            f"| {pm_se(shown_objective(r, doc) for r in ok)} | {t[0]:.1f} +- {t[1]:.1f} | "
            f"{fmt_bytes(mem[0])} | {label} = {sel} | {'; '.join(notes)} |"
        )
    lines += profile_table(doc)
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    write_cv_profile(doc, os.path.splitext(path)[0] + "_cv_profile.csv")
    return text


def write_cv_profile(doc: Dict[str, Any], path: str) -> None:
    """TorchKM per lambda: seconds of the whole-data fit and of the fold fits
    (fitted together), and the solver iterations of each fold, one CSV row per
    (dataset, repeat, lambda)."""
    rows = [
        r
        for r in doc["records"]
        if r["method"] == "torchkm" and (r.get("fit_profile") or {}).get("cv_passes")
    ]
    if not rows:
        return
    folds = len(rows[0]["fit_profile"]["cv_passes"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["dataset", "repeat", "dtype", "lambda_index", "lambda", "path_seconds"]
            + ["cv_seconds", "path_passes", "cv_passes"]
            + [f"cv_passes_fold{k + 1}" for k in range(folds)]
        )
        for r in rows:
            fp = r["fit_profile"]
            for j, lam in enumerate(fp["lambdas"]):
                per_fold = [f[j] for f in fp["cv_passes"]]
                w.writerow(
                    [r["dataset"], r["repeat"], r["params"]["dtype"], j, f"{lam:.6g}"]
                    + [f"{fp['path_seconds'][j]:.6f}", f"{fp['cv_seconds'][j]:.6f}"]
                    + [fp["path_passes"][j], sum(per_fold)]
                    + per_fold
                )


PHASES = [
    ("kernel", "kernel"),
    ("eigendecomposition", "eigendecomposition"),
    ("factorization_error", "error check"),
    ("path", "lambda path"),
    ("cross_validation", "CV fits"),
    ("prediction", "test prediction"),
]


def profile_table(doc: Dict[str, Any]) -> List[str]:
    """Where TorchKM's time goes, per dataset: mean seconds over repeats."""
    recs: Dict[str, List[Dict[str, Any]]] = {}
    for r in doc["records"]:
        if r["method"] == "torchkm" and r.get("time_profile"):
            recs.setdefault(r["dataset"], []).append(r)
    if not recs:
        return []
    lines = [
        "",
        "## TorchKM time profile",
        "",
        "Mean seconds over repeats. Kernel, eigendecomposition, its error check, "
        "lambda path (whole-data fits) and CV fits make up the fit; 'other' is the rest "
        "of the fit (data conversion, selection, copies). Passes = solver iterations, "
        "path / CV (each a few matrix-vector products with the n x n kernel). "
        "Per CV fit = CV fits / (folds x lambdas); the 10 folds of a lambda are "
        "fitted together, and <out>_cv_profile.csv has the seconds per lambda and the "
        "iterations per fold and lambda.",
        "",
        "| dataset | runs | "
        + " | ".join(label for _, label in PHASES)
        + " | other | total (s) | passes | per CV fit (ms) |",
        "|---|---:|" + "---:|" * (len(PHASES) + 4),
    ]
    n_fits = doc["args"]["folds"] * doc["args"]["grid_size"]
    for ds, rs in recs.items():
        mean = {
            k: float(np.mean([r["time_profile"].get(k, 0.0) for r in rs]))
            for k, _ in PHASES
        }
        fit_total = float(np.mean([r["time_profile"]["total"] for r in rs]))
        other = fit_total - sum(v for k, v in mean.items() if k != "prediction")
        total = float(np.mean([r["time_s"] for r in rs]))
        share = lambda v: f"{v:.2f} ({v / total:.0%})"  # noqa: E731
        passes = [r["passes"] for r in rs if r.get("passes")]
        pas = (
            f"{np.mean([q['path'] for q in passes]):,.0f} / "
            f"{np.mean([q['cross_validation'] for q in passes]):,.0f}"
            if passes
            else "-"
        )
        lines.append(
            f"| {ds} | {len(rs)} | "
            + " | ".join(share(mean[k]) for k, _ in PHASES)
            + f" | {share(other)} | {total:.1f} | {pas} "
            + f"| {1e3 * mean['cross_validation'] / n_fits:.1f} |"
        )
    return lines


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
    ap.add_argument(
        "--redo",
        nargs="+",
        choices=METHODS,
        default=[],
        help="on resume, compute these methods' cells again even if finished",
    )
    ap.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="precision of every method (default float32)",
    )
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    ap.add_argument("--repeats", type=int, default=3, help="seeds per dataset")
    ap.add_argument("--folds", type=int, default=10, help="CV folds (paper: 10)")
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
        "--trunc-rank", type=int, default=400, help="torchkm_trunc: eigenpairs kept"
    )
    ap.add_argument(
        "--trunc-block",
        type=int,
        default=10,
        help="torchkm_trunc: lambdas fitted together with their folds (1 = serial)",
    )
    ap.add_argument(
        "--gap-tol",
        type=float,
        default=1e-3,
        help="torchkm_trunc: certified relative duality gap per lambda and fold",
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
        args.datasets, args.folds, args.grid_size, args.repeats = (
            ["sim_600x10"],
            3,
            4,
            1,
        )
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
        # Keep a finished cell only if it was computed with this run's settings
        # for its method (e.g. TorchKM's tol and KKTeps); everything else is
        # dropped and computed again, so one table never mixes settings.
        kept = [
            r
            for r in old.get("records", [])
            if reusable(r, args) and r["method"] not in args.redo
        ]
        dropped = len(old.get("records", [])) - len(kept)
        doc["records"] = kept
        print(
            f"resuming {args.out}: {len(kept)} finished cells kept, "
            f"{dropped} failed or computed with other settings will be redone"
        )
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
        sim = parse_sim(ds)
        data = (
            None
            if sim
            else cast(load_dataset(ds, args.data_dir, seed=args.seed), args.dtype)
        )
        for r in range(args.repeats):
            seed = args.seed + r
            if sim:  # Table 2 protocol: fresh data for every repeat
                data = cast(
                    synthetic_dataset(
                        sim[0], sim[1], seed, name=ds, n_test=sim[0] // 10
                    ),
                    args.dtype,
                )
            if r == 0:
                print(
                    f"\n== {ds}: n_train={data['n_train']:,} n_test={data['n_test']:,} "
                    f"p={data['p']} positive fraction={data['pos_frac']:.3f}"
                )
            torch.manual_seed(seed)
            sig = float(sigest(torch.from_numpy(data["Xtr"]).double()))
            foldid = make_folds(data["ytr"], args.folds, seed)
            for m in args.methods:
                if (ds, m, r) in done:
                    continue
                info = dict(
                    dataset=ds,
                    n_train=data["n_train"],
                    n_test=data["n_test"],
                    train_pos_frac=data["pos_frac"],
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
                    + (
                        f" objective={rec['objective']:.4f}"
                        if rec.get("objective") is not None
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
