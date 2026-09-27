#!/usr/bin/env python
"""Q3b: covtype at the published RBF-SVM setting, in TorchKM's exact mode.

Hsieh, Si and Dhillon (ICML 2014) report 96.15% test accuracy on covtype.binary
with an exact RBF SVM: LIBSVM with gamma 32 and C 32 (their run scripts use
``-c 32 -g 32``), features scaled to [0, 1], all ~465k training rows. Exact
mode holds the n x n kernel on the GPU, so this script fits stratified training
subsamples of growing size and shows how close each gets:

  sizes      n in {5000, 10000, 20000, 30000} rows of the training split
  torchkm    TorchKMSVC in exact mode with the published kernel, exp(-32 d^2)
             (sigma 16 in TorchKM's exp(-2 sigma d^2)), and with TorchKM's
             default sigest width; C over 50 values from 0.1 to 1000 plus 32,
             chosen by 10-fold CV; the table also gives the accuracy at C 32
  cuml       cuML SVC at the published setting (gamma 32, C 32), one fit: the
             LIBSVM algorithm on the GPU, the reference for the same n
  data       covtype.libsvm.binary.scale, labels 1 -> -1 and 2 -> +1, one
             stratified 80/20 split (seed 52); test on all 116,203 test rows
  repeats    3 (seeds 52, 53, 54): new subsample and folds
  timing     fit (tuning included) + test predictions, GPU warmed up first
  memory     NVML peak of the process on the GPU

Run (re-running with the same --out keeps finished cells):
  python benchmarks/q3_covtype_exact.py --data-dir ~/libsvm_data --out results/q3b.json
  python benchmarks/q3_covtype_exact.py --smoke        # CPU check on synthetic data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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
    load_dataset,
    mean_se,
    stratified_subsample,
    synthetic_dataset,
)

GAMMA, C_PAPER = 32.0, 32.0


class Measured:
    """Wall-clock time and NVML peak GPU memory of a block."""

    def __init__(self, dev: str):
        self.dev = dev

    def __enter__(self) -> "Measured":
        if self.dev.startswith("cuda"):
            torch.cuda.synchronize()
        self.pm = PeakMemory(self.dev).__enter__()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.dev.startswith("cuda"):
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0
        self.pm.__exit__(*exc)
        mem = self.pm.result
        self.memory = mem.get("nvml_process_peak") or mem.get("nvml_device_peak")


def run_torchkm(data, sigma, Cs, foldid, dev, args):
    from torchkm.estimators import TorchKMSVC

    torch.linalg.eigh(torch.eye(64, dtype=torch.float64, device=dev))  # start-up
    with Measured(dev) as m:
        clf = TorchKMSVC(
            kernel="rbf",
            rbf_sigma=sigma,
            Cs=Cs,
            nC=len(Cs),
            cv=args.folds,
            foldid=foldid,
            device=dev,
            tol=args.tol,
            max_iter=args.max_iter,
            store_path=True,
        )
        clf.fit(data["Xtr"], data["ytr"])
        scores = np.concatenate(
            [
                clf.decision_function(data["Xte"][i : i + 4096])
                for i in range(0, len(data["Xte"]), 4096)
            ]
        )
    # accuracy at the published C, from the stored path (Cs ascend, so the
    # path's columns follow them)
    j = int(np.argmin(np.abs(np.asarray(Cs) - C_PAPER)))
    alp = clf.alpmat_path_[:, j].double()
    X_train_t = torch.as_tensor(clf.X_fit_, dtype=torch.double)
    f = np.concatenate(
        [
            (
                clf._compute_K_test(
                    torch.as_tensor(data["Xte"][i : i + 4096], dtype=torch.double),
                    X_train_t,
                    clf.kernel_state_,
                )
                @ alp[1:]
                + alp[0]
            ).numpy()
            for i in range(0, len(data["Xte"]), 4096)
        ]
    )
    at_paper = float(np.mean(np.sign(f) == data["yte"]))
    return dict(
        status="ok",
        time_s=m.seconds,
        memory=m.memory,
        selected_C=float(clf.best_C_),
        acc_at_C32=at_paper,
        params=dict(tol=args.tol, max_iter=args.max_iter),
        **classification_metrics(data["yte"], scores),
    )


def run_cuml(data, sigma, Cs, foldid, dev, args):
    from cuml.svm import SVC

    SVC(C=1.0, gamma=1.0).fit(data["Xtr"][:64], data["ytr"][:64])  # start-up
    with Measured(dev) as m:
        svc = SVC(C=C_PAPER, gamma=GAMMA, kernel="rbf", cache_size=args.svc_cache_mb)
        svc.fit(data["Xtr"], data["ytr"])
        scores = np.asarray(svc.decision_function(data["Xte"])).reshape(-1)
    return dict(
        status="ok",
        time_s=m.seconds,
        memory=m.memory,
        selected_C=C_PAPER,
        acc_at_C32=float(np.mean(np.sign(scores) == data["yte"])),
        params=dict(cache_size_mb=args.svc_cache_mb),
        **classification_metrics(data["yte"], scores),
    )


def cells(args) -> List[tuple]:
    out = []
    for n in args.sizes:
        if "torchkm" in args.methods:
            out += [(n, "torchkm", "gamma 32"), (n, "torchkm", "sigest")]
        if "cuml" in args.methods:
            out.append((n, "cuml", "gamma 32"))
    return out


def write_markdown(doc: Dict[str, Any], path: str) -> str:
    env = doc["environment"]
    gpu = (env.get("gpu") or {}).get("name") or "no GPU (CPU run)"
    lines = [
        "# Q3b: covtype at the published RBF setting (exact mode)",
        "",
        f"{gpu}; torch {env.get('torch')}; torchkm {env.get('torchkm')} "
        f"({str(env.get('torchkm_commit'))[:10]}). Test on {doc['n_test']:,} rows.",
        "Published: exact RBF SVM, gamma 32, C 32, all ~465k training rows: 96.15% "
        "(LIBSVM; Hsieh, Si and Dhillon, ICML 2014).",
        "",
        "| n_train | method | kernel | runs | test accuracy (CV-chosen C) | accuracy at C 32 "
        "| chosen C | time (s) | GPU memory |",
        "|---:|---|---|---:|---|---|---|---|---|",
    ]
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in doc["records"]:
        groups.setdefault((r["n_train"], r["method"], r["kernel"]), []).append(r)
    for key in sorted(groups, key=lambda t: (t[0], t[1] != "cuml", t[2])):
        ok = [r for r in groups[key] if r["status"] == "ok"]
        if not ok:
            err = groups[key][-1].get("error", "failed")[:100]
            lines.append(f"| {key[0]:,} | {key[1]} | {key[2]} | 0 | {err} | | | | |")
            continue

        def pm(k, d=4):
            m, se = mean_se([r[k] for r in ok])
            return f"{m:.{d}f} +- {se:.{d}f}"

        mem = [r["memory"] for r in ok if r.get("memory")]
        chosen = ", ".join(f"{r['selected_C']:.3g}" for r in ok)
        memory = fmt_bytes(float(np.mean(mem))) if mem else "-"
        lines.append(
            f"| {key[0]:,} | {key[1]} | {key[2]} | {len(ok)} | {pm('accuracy')} "
            f"| {pm('acc_at_C32')} | {chosen} | {pm('time_s', 1)} | {memory} |"
        )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000, 10000, 20000, 30000])
    ap.add_argument("--methods", nargs="+", default=["torchkm", "cuml"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument(
        "--grid-size", type=int, default=49, help="C values from 0.1 to 1e3"
    )
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=100_000)
    ap.add_argument("--svc-cache-mb", type=float, default=2000)
    ap.add_argument("--out", default="benchmarks/results/q3b_covtype_exact.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if args.smoke:
        args.sizes, args.repeats, args.folds, args.methods = (
            [300, 600],
            1,
            3,
            ["torchkm"],
        )
        args.grid_size, args.max_iter = 4, 5000
    elif not args.data_dir:
        ap.error("--data-dir is required unless --smoke")
    dev = (
        "cuda"
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    from torchkm import sigest

    full = (
        synthetic_dataset(2000, 10, args.seed)
        if args.smoke
        else load_dataset("covtype", args.data_dir, seed=args.seed)
    )
    Cs = np.unique(np.concatenate([np.logspace(-1, 3, args.grid_size), [C_PAPER]]))
    doc: Dict[str, Any] = dict(
        script="q3_covtype_exact.py",
        args=vars(args),
        grid_C=Cs.tolist(),
        n_test=int(len(full["yte"])),
        environment=env_snapshot(),
        records=[],
    )
    if os.path.exists(args.out):
        with open(args.out) as fh:
            old = json.load(fh)
        doc["records"] = [r for r in old.get("records", []) if r.get("status") == "ok"]
        print(f"resuming {args.out}: {len(doc['records'])} finished cells kept")
    done = {
        (r["n_train"], r["method"], r["kernel"], r["repeat"]) for r in doc["records"]
    }
    md_path = os.path.splitext(args.out)[0] + ".md"
    run = dict(torchkm=run_torchkm, cuml=run_cuml)

    for r in range(args.repeats):
        seed = args.seed + r
        for n, method, kernel in cells(args):
            if (n, method, kernel, r) in done:
                continue
            Xtr, ytr = stratified_subsample(full["Xtr"], full["ytr"], n, seed)
            data = dict(Xtr=Xtr, ytr=ytr, Xte=full["Xte"], yte=full["yte"])
            foldid = np.random.default_rng(seed).permutation(n) % args.folds + 1
            if kernel == "sigest":
                torch.manual_seed(seed)
                sigma = float(sigest(torch.from_numpy(np.ascontiguousarray(Xtr))))
            else:
                sigma = GAMMA / 2.0
            try:
                rec = run[method](data, sigma, Cs, foldid, dev, args)
            except Exception as err:  # the next cell still runs
                traceback.print_exc()
                rec = dict(status="failed", error=f"{type(err).__name__}: {err}"[:800])
            rec.update(n_train=n, method=method, kernel=kernel, repeat=r, sigma=sigma)
            doc["records"].append(rec)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(_jsonable(doc), fh, indent=1)
            print(
                f"   n={n:<6} {method:8s} {kernel:9s} r{r} {rec['status']:6s} "
                f"acc={rec.get('accuracy', float('nan')):.4f} "
                f"acc@C32={rec.get('acc_at_C32', float('nan')):.4f} "
                f"t={rec.get('time_s', float('nan')):.1f}s"
            )
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
            write_markdown(doc, md_path)
    print("\n" + write_markdown(doc, md_path))
    print(f"results: {args.out}  table: {md_path}")


if __name__ == "__main__":
    main()
