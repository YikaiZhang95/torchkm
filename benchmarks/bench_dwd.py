"""Kernel distance-weighted discrimination: TorchKMDWD against the CPU implementation.

GPU-accelerated kernel DWD with exact cross-validation does not exist outside
TorchKM; the reference implementation is the R package ``kerndwd`` (Wang and
Zou, 2018). This script runs the Python side (TorchKMDWD on the exact and
Nyström paths, with TorchKMSVC as the in-package reference) and exports the
exact splits and folds so ``benchmarks/r/bench_dwd.R`` runs ``kerndwd`` on
identical data.

Datasets: ``gisette`` (6,000 x 5,000; the high-dimension low-sample-size
regime DWD was designed for), ``ijcnn1_30k`` and ``mnist_3v8`` (mid-size
problems from the SVM suite, to show the speed of exact CV on the DWD path),
or any other key from ``_common.DATASETS``. Metrics: accuracy, balanced
accuracy, AUC, end-to-end time, peak memory.

Examples
--------
::

    python benchmarks/bench_dwd.py --data-dir ~/libsvm --datasets gisette ijcnn1_30k mnist_3v8 \\
        --repeats 5 --device cuda --export-splits benchmarks/results/dwd_splits \\
        --out benchmarks/results/dwd.json
    Rscript benchmarks/r/bench_dwd.R benchmarks/results/dwd_splits benchmarks/results/dwd_r.csv

    python benchmarks/bench_dwd.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import torch

from _common import (
    DATASETS,
    ResultWriter,
    add_common_args,
    banner,
    c_grid,
    free_cuda,
    get_device,
    load_dataset,
    make_folds,
    protocol_dict,
    smoke_settings,
    synthetic_dataset,
    warmup,
)
from _libraries import print_record, run_torchkm

METHODS = {
    "torchkm_dwd": dict(estimator="dwd", low_rank=False),
    "torchkm_dwd_nystrom": dict(estimator="dwd", low_rank=True),
    "torchkm_svm": dict(estimator="svm", low_rank=False),
}


def export_split(directory, name, r, data, foldid):
    os.makedirs(directory, exist_ok=True)
    p = data["Xtr"].shape[1]
    np.savetxt(
        os.path.join(directory, f"{name}_rep{r}_train.csv"),
        np.column_stack([data["Xtr"], data["ytr"], foldid]),
        delimiter=",",
        header=",".join([f"x{j}" for j in range(p)] + ["y", "fold"]),
        comments="",
    )
    np.savetxt(
        os.path.join(directory, f"{name}_rep{r}_test.csv"),
        np.column_stack([data["Xte"], data["yte"]]),
        delimiter=",",
        header=",".join([f"x{j}" for j in range(p)] + ["y"]),
        comments="",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, repeats=1)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["gisette", "ijcnn1_30k", "mnist_3v8"],
        choices=sorted(DATASETS),
    )
    ap.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=list(METHODS)
    )
    ap.add_argument("--landmarks", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=300)
    ap.add_argument(
        "--export-splits",
        default=None,
        help="directory for the CSV splits the R script reads",
    )
    args = smoke_settings(ap.parse_args())
    if args.smoke:
        args.landmarks, args.rank = 40, 20
    elif not args.data_dir:
        ap.error("--data-dir is required unless --smoke")

    dev = get_device(args.device)
    datasets = ["synthetic"] if args.smoke else args.datasets
    banner(
        "Kernel DWD",
        device=dev,
        datasets=datasets,
        methods=args.methods,
        folds=args.folds,
        grid=f"{args.grid_size} C values in [{args.c_min}, {args.c_max}]",
    )
    writer = ResultWriter(
        args.out,
        script="bench_dwd.py",
        args=args,
        protocol=protocol_dict(
            args,
            nystrom=dict(landmarks=args.landmarks, rank=args.rank),
            r_baselines="benchmarks/r/bench_dwd.R on the exported splits (kerndwd)",
        ),
    )
    warmup(dev)
    from torchkm import sigest

    Cs = c_grid(args.grid_size, args.c_max, args.c_min)
    for name in datasets:
        data = (
            synthetic_dataset(300, 10, args.seed)
            if name == "synthetic"
            else load_dataset(name, args.data_dir, seed=args.seed)
        )
        info = dict(
            dataset=name,
            n_train=data["n_train"],
            n_test=data["n_test"],
            p=data["p"],
            train_pos_frac=data["pos_frac"],
        )
        print(
            f"\n== {name}: n_train={data['n_train']:,} n_test={data['n_test']:,} p={data['p']}"
        )
        for r in range(args.repeats):
            seed = args.seed + r
            torch.manual_seed(seed)
            sig = float(sigest(torch.from_numpy(data["Xtr"])))
            foldid = make_folds(data["ytr"], args.folds, seed)
            if args.export_splits:
                export_split(args.export_splits, name, r, data, foldid)
            for method in args.methods:
                spec = METHODS[method]
                try:
                    rec = run_torchkm(
                        data,
                        sig,
                        Cs,
                        foldid,
                        args,
                        dev,
                        seed,
                        estimator=spec["estimator"],
                        low_rank=spec["low_rank"],
                        landmarks=args.landmarks,
                        rank=args.rank,
                    )
                    rec["library"] = method
                except Exception as err:
                    traceback.print_exc()
                    rec = dict(
                        library=method,
                        status="failed",
                        error=f"{type(err).__name__}: {err}"[:800],
                    )
                rec.update(info, repeat=r, seed=seed, bandwidth_sigest=sig)
                writer.add(rec)
                print_record(name, rec)
                free_cuda(dev)
    if args.out:
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
