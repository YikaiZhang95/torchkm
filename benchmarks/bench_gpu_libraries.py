"""Kernel SVM comparison against the GPU libraries a reader would weigh TorchKM against.

One protocol for every library: the same RBF bandwidth (``sigest`` on the
training features), the same log-uniform grid of ``C`` values, identical
stratified folds, end-to-end timing (kernel or feature construction, the
cross-validation sweep over the whole grid, and the final refit), peak device
memory from the PyTorch allocator and from NVML, and test accuracy, balanced
accuracy and AUC.

Libraries (``--libraries``):

================  ============================================================
torchkm           TorchKM exact path (integrated CV along the lambda path)
torchkm_nystrom   TorchKM Nyström path (``--landmarks``, ``--rank``)
sklearn_svc       scikit-learn ``SVC`` on CPU, the reference libsvm solver
thundersvm        ThunderSVM 0.3.4 (GPU SMO); ``cd thundersvm/python`` or
                  ``--thundersvm-path``
cuml_svc          cuML ``SVC`` (RAPIDS), the direct GPU SMO competitor
falkon            Falkon (GPU Nyström KRR) at each ``--falkon-centers``
linear            tuned logistic regression and LinearSVC, so the value of the
                  kernel is visible on every row
================  ============================================================

Datasets (``--suite`` or ``--datasets``; LIBSVM files in ``--data-dir``):

scaling     Adult at a1a, a3a, a5a, a7a, a8a, a9a: one scaling study, not six
            benchmarks. Exact mode runs while the predicted memory fits.
exact       Problems where the kernel matters, sized for exact mode:
            ijcnn1 (30k stratified subsample), MNIST 3-vs-8 and 4-vs-9,
            covtype (30k subsample), and w7a (imbalanced, report AUC).
imbalanced  w8a and ijcnn1 at full size on the Nyström path.
scale       covtype (581k) and MNIST8m 4-vs-6 (1.27M) on the Nyström path.

Exact-mode libraries are skipped automatically on a dataset whose predicted
exact-mode memory exceeds the device (``--force-exact`` overrides). Libraries
without integrated tuning stop their sweep at ``--time-cap`` seconds and are
reported as ``capped`` with the number of grid values completed.

Examples
--------
::

    python benchmarks/bench_gpu_libraries.py --data-dir ~/libsvm --suite exact \\
        --libraries torchkm sklearn_svc thundersvm cuml_svc linear \\
        --repeats 10 --device cuda --time-cap 14400 --out benchmarks/results/exact.json

    python benchmarks/bench_gpu_libraries.py --data-dir ~/libsvm --suite scale \\
        --libraries torchkm_nystrom falkon linear --falkon-centers 2000 10000 20000 \\
        --repeats 10 --device cuda --out benchmarks/results/scale.json

    python benchmarks/bench_gpu_libraries.py --smoke     # CPU check
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Dict, List

import numpy as np
import torch

from _common import (
    DATASETS,
    SUITES,
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
    stratified_subsample,
    synthetic_dataset,
    warmup,
)
from _libraries import (
    ALL_LIBRARIES,
    library_availability,
    print_record,
    run_cuml_svc,
    run_falkon,
    run_linear,
    run_sklearn_svc,
    run_thundersvm,
    run_torchkm,
)

EXACT_LIBRARIES = {"torchkm", "sklearn_svc", "thundersvm", "cuml_svc"}


def exact_mode_feasible(
    n: int, dev: str, grid_size: int, ram_bytes
) -> tuple[bool, str]:
    from torchkm.memory import device_total_memory, exact_mode_memory_estimate

    need = exact_mode_memory_estimate(n, nlam=grid_size)
    total = device_total_memory(dev) if str(dev).startswith("cuda") else ram_bytes
    if total is None:
        return True, ""
    if need > 0.95 * total:
        return (
            False,
            f"predicted exact-mode memory {need / 1e9:.1f} GB > 95% of {total / 1e9:.1f} GB",
        )
    return True, ""


def run_library(
    lib: str, data, sig, Cs, foldid, args, dev, seed
) -> List[Dict[str, Any]]:
    if lib == "torchkm":
        return [run_torchkm(data, sig, Cs, foldid, args, dev, seed)]
    if lib == "torchkm_nystrom":
        return [
            run_torchkm(
                data,
                sig,
                Cs,
                foldid,
                args,
                dev,
                seed,
                low_rank=True,
                landmarks=m,
                rank=args.rank,
            )
            for m in args.landmarks
        ]
    if lib == "sklearn_svc":
        return [run_sklearn_svc(data, sig, Cs, foldid, args, dev, seed)]
    if lib == "thundersvm":
        return [run_thundersvm(data, sig, Cs, foldid, args, dev, seed)]
    if lib == "cuml_svc":
        return [run_cuml_svc(data, sig, Cs, foldid, args, dev, seed)]
    if lib == "falkon":
        return [
            run_falkon(data, sig, Cs, foldid, args, dev, seed, centers=m)
            for m in args.falkon_centers
        ]
    if lib == "linear":
        return [
            run_linear(data, sig, Cs, foldid, args, dev, seed, model="logreg"),
            run_linear(data, sig, Cs, foldid, args, dev, seed, model="linearsvc"),
        ]
    raise ValueError(lib)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, repeats=1)
    ap.add_argument("--data-dir", default=None, help="directory of LIBSVM files")
    ap.add_argument("--suite", choices=sorted(SUITES), default=None)
    ap.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=None)
    ap.add_argument(
        "--libraries", nargs="+", choices=ALL_LIBRARIES, default=["torchkm", "linear"]
    )
    ap.add_argument(
        "--landmarks",
        type=int,
        nargs="+",
        default=[2000],
        help="TorchKM Nyström landmarks",
    )
    ap.add_argument("--rank", type=int, default=300, help="TorchKM Nyström rank")
    ap.add_argument("--falkon-centers", type=int, nargs="+", default=[2000])
    ap.add_argument("--falkon-maxiter", type=int, default=20)
    ap.add_argument("--svc-cache-mb", type=float, default=2000)
    ap.add_argument(
        "--float32-baselines",
        action="store_true",
        help="feed float32 inputs to ThunderSVM/cuML (their native precision)",
    )
    ap.add_argument("--thundersvm-path", default=None, help="path to thundersvm/python")
    ap.add_argument(
        "--force-exact", action="store_true", help="ignore the memory-envelope check"
    )
    ap.add_argument("--synthetic-n", type=int, default=300, help="--smoke dataset size")
    ap.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="stratified-subsample every training set larger than this (e.g. 20000 to stay at the paper's exact-mode sizes)",
    )
    args = smoke_settings(ap.parse_args())

    if args.thundersvm_path:
        sys.path.insert(0, args.thundersvm_path)

    dev = get_device(args.device)
    if args.smoke:
        datasets = ["synthetic"]
    elif args.datasets:
        datasets = args.datasets
    elif args.suite:
        datasets = SUITES[args.suite]
    else:
        ap.error("pass --suite, --datasets, or --smoke")
    if not args.smoke and not args.data_dir:
        ap.error("--data-dir is required unless --smoke")

    availability = library_availability(args.libraries)
    libraries = [lib for lib in args.libraries if availability[lib] is None]
    for lib, err in availability.items():
        if err is not None:
            print(f"[skip] {lib}: {err}")
    banner(
        "TorchKM vs GPU kernel libraries",
        device=dev,
        datasets=datasets,
        libraries=libraries,
        folds=args.folds,
        grid=f"{args.grid_size} C values in [{args.c_min}, {args.c_max}]",
        repeats=args.repeats,
        time_cap_s=args.time_cap,
    )
    writer = ResultWriter(
        args.out,
        script="bench_gpu_libraries.py",
        args=args,
        protocol=protocol_dict(
            args,
            nystrom=dict(landmarks=args.landmarks, rank=args.rank),
            falkon_centers=args.falkon_centers,
        ),
    )
    ram = writer.doc["environment"].get("ram_bytes")
    warmup(dev)

    from torchkm import sigest

    Cs = c_grid(args.grid_size, args.c_max, args.c_min)
    for ds in datasets:
        if ds == "synthetic":
            data = synthetic_dataset(args.synthetic_n, 10, args.seed)
        else:
            data = load_dataset(ds, args.data_dir, seed=args.seed)
        if args.max_train and data["n_train"] > args.max_train:
            data["Xtr"], data["ytr"] = stratified_subsample(
                data["Xtr"], data["ytr"], args.max_train, args.seed
            )
            data["n_train"] = int(data["Xtr"].shape[0])
            data["pos_frac"] = float(np.mean(data["ytr"] > 0))
        info = dict(
            dataset=ds,
            group=DATASETS.get(ds, {}).get("group", "synthetic"),
            n_train=data["n_train"],
            n_test=data["n_test"],
            p=data["p"],
            train_pos_frac=data["pos_frac"],
            max_train=args.max_train,
        )
        print(
            f"\n== {ds}: n_train={data['n_train']:,} n_test={data['n_test']:,} "
            f"p={data['p']} positive fraction={data['pos_frac']:.3f}"
        )
        feasible, why = exact_mode_feasible(data["n_train"], dev, args.grid_size, ram)
        for r in range(args.repeats):
            seed = args.seed + r
            torch.manual_seed(seed)
            sig = float(sigest(torch.from_numpy(data["Xtr"])))
            foldid = make_folds(data["ytr"], args.folds, seed)
            for lib in libraries:
                if lib in EXACT_LIBRARIES and not feasible and not args.force_exact:
                    rec = dict(
                        info,
                        repeat=r,
                        seed=seed,
                        library=lib,
                        mode="exact",
                        status="exceeds_envelope",
                        note=why,
                    )
                    writer.add(rec)
                    print_record(ds, rec)
                    continue
                try:
                    recs = run_library(lib, data, sig, Cs, foldid, args, dev, seed)
                except Exception as err:  # keep the sweep going; the record says why
                    traceback.print_exc()
                    recs = [
                        dict(
                            library=lib,
                            status="failed",
                            error=f"{type(err).__name__}: {err}"[:800],
                        )
                    ]
                for rec in recs:
                    rec.update(info, repeat=r, seed=seed, bandwidth_sigest=sig)
                    writer.add(rec)
                    print_record(ds, rec)
                free_cuda(dev)
        del data
        free_cuda(dev)
    if args.out:
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
