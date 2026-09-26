#!/usr/bin/env python
"""Q3: an accounting for the covtype result in Table 4.

The letter: an accuracy of 0.807 on covtype.binary is well below what kernel
methods on that data are usually reported to achieve, which raises the
possibility that the Nystrom configuration in Table 4 is under-powered and that
the speedups there are partly bought with approximation quality.

What the code shows: table4_nystrom.py ran TorchKM on covtype at Nystrom rank
30 on 2,000 landmarks, while every other dataset, and the scikit-learn baseline
on covtype, used rank 300. This script reruns that row, then raises the budget,
with the baseline at the same budget every time. One script, one table.

  budget     landmarks m in {2000, 5000, 10000} x rank k in {30, 300, 1000},
             k <= m. (2000, 30) is the submitted TorchKM configuration and
             (2000, 300) the one every other dataset and the baseline used
  torchkm    cvknyssvm called as table4_nystrom.py calls it: 50 lambda from
             1e3 down to 1e-3, 10 folds, maxit 1e6, gamma 1e-8, landmarks
             drawn and bandwidth estimated (sigest on the landmarks) from the
             repeat's seed; on the GPU. One change: eps 1e-5 instead of Table
             4's 1e-3, for tighter solves (--eps 1e-3 gives Table 4's)
  sklearn    the paper's baseline: the same Nystrom feature map built by hand
             (RBF kernel on m landmarks, rank-k truncation, Z = C M), LinearSVC
             with its default settings tuned by 10-fold cross_val_score over 50
             lambda from 1e5 down to 1e-5 (C = 1/(2 n lambda)), then a final
             refit on fresh landmarks, as the notebook did. The CV features use
             the landmarks and bandwidth TorchKM draws for the same repeat. CPU
             only; a sweep stops after --time-cap seconds and visits the grid
             coarse to fine, so a capped sweep still spans the range
  data       covtype.libsvm.binary.scale (581,012 x 54), labels 1 -> -1 and
             2 -> +1, one stratified 80/20 split (seed 52): 464,809 train and
             116,203 test rows
  repeats    3 (seeds 52, 53, 54): new landmarks, bandwidth draw and folds
  timing     the whole tuning run: feature map, CV sweep, final fit and test
             predictions; the GPU is warmed up first
  memory     TorchKM: NVML peak of the process on the GPU; sklearn: host memory
             added during the run

The table header repeats the submitted Table 4 row (TorchKM 0.807 in 31 s at
rank 30, scikit-learn 0.786 in 2,269 s at rank 300) and the exact RBF-SVM
accuracy reported for covtype.binary (96.15%, LIBSVM on the full training set;
Hsieh, Si and Dhillon, ICML 2014).

Run (re-running with the same --out keeps the finished cells computed with the
same settings and runs the rest):
  python benchmarks/q3_covtype.py --data-dir ~/libsvm_data --out results/q3.json
  python benchmarks/q3_covtype.py --smoke        # CPU check on synthetic data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from typing import Any, Dict, List

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
    mean_se,
    synthetic_dataset,
)

METHODS = ["torchkm", "sklearn"]
SUBMITTED = {
    (2000, 30): "TorchKM's submitted covtype setting",
    (2000, 300): "setting of the other Table 4 rows and of the baseline",
}


def sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


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


def draw_landmarks(Xtr: torch.Tensor, m: int, g: torch.Generator):
    """m landmarks and their sigest bandwidth, drawn as cvknyssvm draws them:
    a generator seeded with ``random_state`` gives the same pair."""
    from torchkm import sigest

    lm = Xtr[torch.randperm(Xtr.shape[0], generator=g)[:m]]
    return lm, float(sigest(lm, generator=g))


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def run_torchkm(data, m, k, lams, sk_lams, foldid, seed, dev, args):
    from torchkm.cvknyssvm import cvknyssvm

    ytr = torch.from_numpy(data["ytr"])
    torch.linalg.eigh(torch.eye(64, dtype=torch.float64, device=dev))  # start-up
    with Measured(dev) as meas:
        model = cvknyssvm(
            Xmat=torch.from_numpy(data["Xtr"]),
            X_test=torch.from_numpy(data["Xte"]),
            y=ytr,
            nlam=len(lams),
            ulam=torch.as_tensor(lams, dtype=torch.float64),
            foldid=torch.from_numpy(foldid),
            nfolds=args.folds,
            eps=args.eps,
            maxit=args.max_iter,
            gamma=1e-8,
            num_landmarks=m,
            k=k,
            device=dev,
            random_state=seed,
            is_exact=args.is_exact,
        )
        model.fit()
        cv_mis = model.cv(model.pred, ytr).numpy()
        best = int(np.argmin(cv_mis))  # first minimum: the larger lambda on ties
        alp = model.alpmat[:, best].double().cpu()
        scores = (model.Z_test.double().cpu() @ alp[1:] + alp[0]).numpy()
    rec = dict(
        status="ok",
        device="GPU" if dev.startswith("cuda") else "CPU",
        time_s=meas.seconds,
        memory=meas.memory,
        selected=float(lams[best]),
        cv_curve=(1.0 - cv_mis).tolist(),
        rank_used=int(model.k_eff_),
        bandwidth=float(model.sig_w_),
        params=dict(eps=args.eps, max_iter=args.max_iter, is_exact=args.is_exact),
        **classification_metrics(data["yte"], scores),
    )
    del model
    return rec


def nystrom_features(X, lm, sig, k, chunk=65536):
    """Z = K(X, landmarks) M with the rank-k map M = U_k S_k^{-1/2} of the
    landmark kernel, as the notebook builds it; computed in row chunks so the
    host never holds the full n x m kernel block."""
    from torchkm import kernelMult, rbf_kernel

    U, S, _ = torch.linalg.svd(rbf_kernel(lm, sig), full_matrices=False)
    k = min(k, S.numel())
    M = U[:, :k] * (1.0 / torch.sqrt(S[:k]))
    return np.concatenate(
        [
            (kernelMult(X[i : i + chunk], lm, sig) @ M).double().numpy()
            for i in range(0, X.shape[0], chunk)
        ]
    )


def run_sklearn(data, m, k, lams, sk_lams, foldid, seed, dev, args):
    from sklearn import svm
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import cross_val_score

    Xtr = torch.from_numpy(data["Xtr"]).float()
    Xte = torch.from_numpy(data["Xte"]).float()
    ytr = data["ytr"]
    n = Xtr.shape[0]
    with Measured("cpu") as meas, warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        lm, sig = draw_landmarks(Xtr, m, g)  # TorchKM's landmarks for this seed
        Z = nystrom_features(Xtr, lm, sig, k)
        cv_acc: Dict[int, float] = {}
        fits, t_fits = 0, 0.0
        for i in coarse_to_fine(len(sk_lams)):
            t = time.perf_counter()
            cv_acc[i] = float(
                cross_val_score(
                    svm.LinearSVC(C=1.0 / (2.0 * n * sk_lams[i])),
                    Z,
                    ytr,
                    cv=args.folds,
                    n_jobs=args.sklearn_jobs,
                ).mean()
            )
            t_fits += time.perf_counter() - t
            fits += args.folds
            if args.time_cap and time.perf_counter() - meas.t0 > args.time_cap:
                break
        best = max(cv_acc, key=lambda i: (cv_acc[i], -i))  # ties: larger lambda
        # final refit on fresh landmarks, as the notebook did
        lm, sig = draw_landmarks(Xtr, m, g)
        del Z
        clf = svm.LinearSVC(C=1.0 / (2.0 * n * sk_lams[best]))
        clf.fit(nystrom_features(Xtr, lm, sig, k), ytr)
        scores = clf.decision_function(nystrom_features(Xte, lm, sig, k))
    return dict(
        status="capped" if len(cv_acc) < len(sk_lams) else "ok",
        device="CPU",
        time_s=meas.seconds,
        memory=meas.memory,
        selected=float(sk_lams[best]),
        cv_curve=[cv_acc.get(i) for i in range(len(sk_lams))],
        grid_completed=len(cv_acc),
        seconds_per_fit=t_fits / max(fits, 1),
        params=dict(time_cap=args.time_cap, sklearn_jobs=args.sklearn_jobs),
        **classification_metrics(data["yte"], scores),
    )


RUN = dict(torchkm=run_torchkm, sklearn=run_sklearn)


def cell_settings(method: str, args) -> Dict[str, Any]:
    if method == "torchkm":
        return dict(eps=args.eps, max_iter=args.max_iter, is_exact=args.is_exact)
    return dict(time_cap=args.time_cap, sklearn_jobs=args.sklearn_jobs)


def reusable(rec: Dict[str, Any], args) -> bool:
    if rec.get("method") not in RUN or rec.get("status") not in ("ok", "capped"):
        return False
    have = rec.get("params") or {}
    return all(have.get(k) == v for k, v in cell_settings(rec["method"], args).items())


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


def notes(recs, grid) -> str:
    out = []
    capped = [r for r in recs if r["status"] == "capped"]
    if capped:
        out.append(
            "capped at --time-cap: "
            + ", ".join(f"{r['grid_completed']}/{len(grid)}" for r in capped)
            + f" lambda values; {np.mean([r['seconds_per_fit'] for r in capped]):.1f} s per fit"
        )
    edge = [
        r
        for r in recs
        if np.isclose(r["selected"], max(grid)) or np.isclose(r["selected"], min(grid))
    ]
    if edge:
        out.append(f"{len(edge)} selected at a grid edge")
    return "; ".join(out)


def write_markdown(doc: Dict[str, Any], path: str) -> str:
    env, a, info = doc["environment"], doc["args"], doc["data"]
    grids = dict(torchkm=doc["grid_lambda"], sklearn=doc["grid_lambda_sklearn"])
    gpu = (env.get("gpu") or {}).get("name") or "no GPU (CPU run)"
    lines = [
        "# Q3: covtype, Table 4 rerun and Nystrom budget",
        "",
        f"{gpu}; torch {env.get('torch')}; torchkm {env.get('torchkm')} "
        f"({str(env.get('torchkm_commit'))[:10]}).",
        f"{info['name']}: {info['n_train']:,} train / {info['n_test']:,} test rows, "
        f"p = {info['p']}, share of +1 in training {info['pos_frac']:.3f}.",
        f"Table 4 protocol except eps: {a['folds']}-fold CV; TorchKM "
        f"{len(grids['torchkm'])} lambda "
        f"from {max(grids['torchkm']):g} to {min(grids['torchkm']):g}, eps {a['eps']:g} "
        "(Table 4: 1e-3); "
        f"scikit-learn {len(grids['sklearn'])} lambda from {max(grids['sklearn']):g} to "
        f"{min(grids['sklearn']):g}. Time = feature map + CV sweep + final fit + test "
        "predictions. Cells are mean +- SE over repeats.",
        "Submitted Table 4 row: TorchKM 0.807 in 31.1 s (2,000 landmarks, rank 30); "
        "scikit-learn Nystrom 0.786 in 2,269 s (rank 300).",
        "Exact RBF SVM on covtype.binary: 96.15% (LIBSVM, full training set; Hsieh, Si "
        "and Dhillon, ICML 2014).",
        "",
        "| landmarks | rank | method | device | runs | test accuracy | AUC | time (s) "
        "| memory | selected lambda | note |",
        "|---:|---:|---|---|---:|---|---|---|---|---|---|",
    ]
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in doc["records"]:
        groups.setdefault((r["landmarks"], r["rank"], r["method"]), []).append(r)
    for key in sorted(groups, key=lambda t: (t[0], t[1], METHODS.index(t[2]))):
        m, k, method = key
        recs = groups[key]
        ok = [r for r in recs if r["status"] in ("ok", "capped")]
        tag = SUBMITTED.get((m, k), "")
        if not ok:
            err = recs[-1].get("error", recs[-1]["status"])[:100]
            lines.append(
                f"| {m:,} | {k:,} | {method} | {recs[-1].get('device', '-')} | 0 | - | - "
                f"| - | - | - | {'; '.join(x for x in (tag, err) if x)} |"
            )
            continue
        sel = ", ".join(f"{r['selected']:.3g}" for r in ok)
        note = "; ".join(x for x in (tag, notes(ok, grids[method])) if x)
        lines.append(
            f"| {m:,} | {k:,} | {method} | {ok[-1]['device']} | {len(ok)} "
            f"| {fmt_pm(ok, 'accuracy')} | {fmt_pm(ok, 'auc')} | {fmt_pm(ok, 'time_s', 1)} "
            f"| {fmt_mem(ok)} | {sel} | {note} |"
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
    ap.add_argument("--landmarks", type=int, nargs="+", default=[2000, 5000, 10000])
    ap.add_argument("--ranks", type=int, nargs="+", default=[30, 300, 1000])
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--grid-size", type=int, default=50)
    ap.add_argument("--lam-max", type=float, default=1e3, help="TorchKM grid")
    ap.add_argument("--lam-min", type=float, default=1e-3)
    ap.add_argument("--sk-lam-max", type=float, default=1e5, help="sklearn grid")
    ap.add_argument("--sk-lam-min", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument(
        "--eps", type=float, default=1e-5, help="TorchKM tolerance (Table 4: 1e-3)"
    )
    ap.add_argument("--max-iter", type=int, default=1_000_000)
    ap.add_argument(
        "--is-exact",
        type=int,
        default=0,
        choices=[0, 1],
        help="TorchKM: 1 puts the smoothing band on the margin after each fit",
    )
    ap.add_argument(
        "--time-cap",
        type=float,
        default=7200,
        help="seconds per sklearn sweep (budget x repeat) before it stops; 0: none",
    )
    ap.add_argument(
        "--sklearn-jobs",
        type=int,
        default=1,
        help="parallel CV folds for sklearn (1, as in the paper; -1: all cores)",
    )
    ap.add_argument("--out", default="benchmarks/results/q3_covtype.json")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic CPU check")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.smoke:
        args.landmarks, args.ranks, args.folds, args.grid_size, args.repeats = (
            [40, 80],
            [10, 30],
            3,
            5,
            1,
        )
    elif not args.data_dir:
        ap.error("--data-dir is required unless --smoke")
    dev = (
        "cuda"
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    if dev != args.device:
        print(f"[warn] {args.device} not available, running on cpu")

    data = (
        synthetic_dataset(3000, 10, args.seed)
        if args.smoke
        else load_dataset("covtype", args.data_dir, seed=args.seed)
    )
    info = dict(
        name="synthetic" if args.smoke else "covtype.binary",
        n_train=int(data["Xtr"].shape[0]),
        n_test=int(data["Xte"].shape[0]),
        p=int(data["Xtr"].shape[1]),
        pos_frac=float(np.mean(data["ytr"] > 0)),
    )
    print(
        f"== {info['name']}: n_train={info['n_train']:,} n_test={info['n_test']:,} "
        f"p={info['p']}"
    )
    lams = np.logspace(np.log10(args.lam_max), np.log10(args.lam_min), args.grid_size)
    sk_lams = np.logspace(
        np.log10(args.sk_lam_max), np.log10(args.sk_lam_min), args.grid_size
    )
    doc: Dict[str, Any] = dict(
        script="q3_covtype.py",
        args=vars(args),
        data=info,
        grid_lambda=lams.tolist(),
        grid_lambda_sklearn=sk_lams.tolist(),
        environment=env_snapshot(),
        records=[],
    )
    if os.path.exists(args.out):  # resume
        with open(args.out) as fh:
            old = json.load(fh)
        protocol = ("folds", "grid_size", "lam_max", "lam_min", "sk_lam_max")
        for key in protocol + ("sk_lam_min", "seed"):
            if old.get("args", {}).get(key) != vars(args)[key]:
                sys.exit(
                    f"{args.out} used a different {key}: use a new --out or delete it"
                )
        kept = [r for r in old.get("records", []) if reusable(r, args)]
        doc["records"] = kept
        print(
            f"resuming {args.out}: {len(kept)} finished cells kept, "
            f"{len(old.get('records', [])) - len(kept)} to redo"
        )
    done = {
        (r["landmarks"], r["rank"], r["method"], r["repeat"]) for r in doc["records"]
    }
    md_path = os.path.splitext(args.out)[0] + ".md"

    budgets = [(m, k) for m in args.landmarks for k in args.ranks if k <= m]
    for r in range(args.repeats):
        seed = args.seed + r
        # folds from their own stream, independent of the landmark draw
        perm = np.random.default_rng(seed).permutation(info["n_train"])
        foldid = perm % args.folds + 1
        for m, k in budgets:
            for method in args.methods:
                if (m, k, method, r) in done:
                    continue
                cell = dict(landmarks=m, rank=k, method=method, repeat=r, seed=seed)
                try:
                    rec = RUN[method](
                        data, m, k, lams, sk_lams, foldid, seed, dev, args
                    )
                except Exception as err:  # the next cell still runs
                    traceback.print_exc()
                    rec = dict(
                        status="failed", error=f"{type(err).__name__}: {err}"[:800]
                    )
                rec.update(cell)
                doc["records"].append(rec)
                save_json(doc, args.out)
                print(
                    f"   m={m:<6} k={k:<5} {method:8s} r{r} {rec['status']:7s} "
                    f"acc={rec.get('accuracy', float('nan')):.4f} "
                    f"t={rec.get('time_s', float('nan')):.1f}s"
                )
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
            write_markdown(doc, md_path)
    print("\n" + write_markdown(doc, md_path))
    print(f"results: {args.out}  table: {md_path}")


if __name__ == "__main__":
    main()
