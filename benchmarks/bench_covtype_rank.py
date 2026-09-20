"""Accuracy versus Nyström budget on covtype: the accounting for Table 4.

The submitted paper's covtype row used ``k=30`` (rank) on 2,000 landmarks
while every other dataset used ``k=300``; the resulting 0.807 is a property of
that configuration, not of the kernel method. This script sweeps landmarks and
rank for TorchKM's Nyström path (and, when installed, Falkon at the same
number of centres) and records accuracy, AUC, time and peak memory for each
budget, so the paper can show the curve instead of one point.

The exact-mode reference on a 30k stratified subsample of covtype is produced
by ``bench_gpu_libraries.py --datasets covtype_30k``.

Examples
--------
::

    python benchmarks/bench_covtype_rank.py --data-dir ~/libsvm --device cuda \\
        --landmarks 2000 5000 10000 20000 --ranks 30 300 1000 full \\
        --repeats 3 --out benchmarks/results/covtype_rank.json

    python benchmarks/bench_covtype_rank.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import List

import torch

from _common import (
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
from _libraries import library_availability, print_record, run_falkon, run_torchkm


def parse_ranks(values: List[str], landmarks: int) -> List[int]:
    out = []
    for v in values:
        k = landmarks if v == "full" else int(v)
        if k <= landmarks and k not in out:
            out.append(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, repeats=1)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--dataset", default="covtype", help="any dataset key; covtype by default"
    )
    ap.add_argument(
        "--landmarks", type=int, nargs="+", default=[2000, 5000, 10000, 20000]
    )
    ap.add_argument("--ranks", nargs="+", default=["30", "300", "1000", "full"])
    ap.add_argument(
        "--with-falkon",
        action="store_true",
        help="also run Falkon at each landmark count",
    )
    ap.add_argument("--falkon-maxiter", type=int, default=20)
    args = smoke_settings(ap.parse_args())
    if args.smoke:
        args.landmarks, args.ranks = [40, 80], ["10", "full"]
    elif not args.data_dir:
        ap.error("--data-dir is required unless --smoke")

    dev = get_device(args.device)
    banner(
        "TorchKM Nyström budget sweep",
        device=dev,
        dataset="synthetic" if args.smoke else args.dataset,
        landmarks=args.landmarks,
        ranks=args.ranks,
        folds=args.folds,
        grid=f"{args.grid_size} C values in [{args.c_min}, {args.c_max}]",
    )
    writer = ResultWriter(
        args.out,
        script="bench_covtype_rank.py",
        args=args,
        protocol=protocol_dict(args, landmarks=args.landmarks, ranks=args.ranks),
    )
    warmup(dev)
    use_falkon = args.with_falkon and library_availability(["falkon"])["falkon"] is None
    if args.with_falkon and not use_falkon:
        print("[skip] falkon: not importable")

    from torchkm import sigest

    data = (
        synthetic_dataset(400, 10, args.seed)
        if args.smoke
        else load_dataset(args.dataset, args.data_dir, seed=args.seed)
    )
    info = dict(
        dataset="synthetic" if args.smoke else args.dataset,
        n_train=data["n_train"],
        n_test=data["n_test"],
        p=data["p"],
        train_pos_frac=data["pos_frac"],
    )
    Cs = c_grid(args.grid_size, args.c_max, args.c_min)
    for r in range(args.repeats):
        seed = args.seed + r
        torch.manual_seed(seed)
        sig = float(sigest(torch.from_numpy(data["Xtr"])))
        foldid = make_folds(data["ytr"], args.folds, seed)
        for m in args.landmarks:
            for k in parse_ranks(args.ranks, m):
                try:
                    rec = run_torchkm(
                        data,
                        sig,
                        Cs,
                        foldid,
                        args,
                        dev,
                        seed,
                        low_rank=True,
                        landmarks=m,
                        rank=k,
                    )
                except Exception as err:
                    traceback.print_exc()
                    rec = dict(
                        library="torchkm_nystrom",
                        status="failed",
                        params=dict(num_landmarks=m, nys_k=k),
                        error=f"{type(err).__name__}: {err}"[:800],
                    )
                rec.update(info, repeat=r, seed=seed, bandwidth_sigest=sig)
                writer.add(rec)
                print_record(info["dataset"], rec)
                free_cuda(dev)
            if use_falkon:
                try:
                    rec = run_falkon(data, sig, Cs, foldid, args, dev, seed, centers=m)
                except Exception as err:
                    traceback.print_exc()
                    rec = dict(
                        library="falkon",
                        status="failed",
                        params=dict(centers=m),
                        error=str(err)[:800],
                    )
                rec.update(info, repeat=r, seed=seed, bandwidth_sigest=sig)
                writer.add(rec)
                print_record(info["dataset"], rec)
                free_cuda(dev)
    if args.out:
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
