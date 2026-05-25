"""Reproduce Table 3: TorchKM on the a7a, a8a, w7a LIBSVM benchmarks.

Table 3 reports test accuracy and end-to-end run time on three mid-size LIBSVM
classification datasets, averaged over 10 runs. The paper also reports a 1D-CNN
baseline and ThunderSVM; both need extra setup (a custom training loop / a
from-source CUDA build), so this lean script reports the TorchKM column only.

Data: download the LIBSVM files into --data-dir (train + ".t" test file):
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a7a   (+ a7a.t)
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a   (+ a8a.t)
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a   (+ w7a.t)

Example:
    python benchmarks/table3_benchmarks.py --data-dir /path/to/libsvm --repeats 10
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from torchkm.estimators import TorchKMSVC

from _common import c_grid, get_device, load_train_test, mean_se, timed, warmup

NFOLDS = 10
# name -> (train_file, test_file)
DATASETS = [
    ("a7a", "a7a", "a7a.t"),
    ("a8a", "a8a", "a8a.t"),
    ("w7a", "w7a", "w7a.t"),
]


def run_torchkm(Xtr, ytr, Xte, yte, device, seed, low_rank, num_landmarks, nys_k):
    Cs = c_grid()
    clf = TorchKMSVC(
        kernel="rbf", Cs=Cs, nC=len(Cs), cv=NFOLDS, device=device, random_state=seed,
        low_rank=low_rank, num_landmarks=num_landmarks, nys_k=nys_k,
    )
    with timed(device) as t:
        clf.fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    return acc, t.dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="dir holding the LIBSVM files")
    ap.add_argument("--repeats", type=int, default=3, help="runs per dataset (paper: 10)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low-rank", action="store_true", help="use the Nystrom solver")
    ap.add_argument("--num-landmarks", type=int, default=2000)
    ap.add_argument("--nys-k", type=int, default=1000)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device={device}  repeats={args.repeats}  folds={NFOLDS}  low_rank={args.low_rank}\n")
    warmup(device)

    header = f"{'data':>6} {'n_train':>8} {'p':>5} | {'TorchKM acc':>18} {'time(s)':>9}"
    print(header)
    print("-" * len(header))

    for name, train_f, test_f in DATASETS:
        Xtr, ytr, Xte, yte = load_train_test(
            os.path.join(args.data_dir, train_f), os.path.join(args.data_dir, test_f)
        )
        accs, times = [], []
        for i in range(args.repeats):
            a, dt = run_torchkm(
                Xtr, ytr, Xte, yte, device, args.seed + i,
                args.low_rank, args.num_landmarks, args.nys_k,
            )
            accs.append(a)
            times.append(dt)
        am, ase = mean_se(accs)
        tm, _ = mean_se(times)
        print(f"{name:>6} {Xtr.shape[0]:>8} {Xtr.shape[1]:>5} | {am:>9.3f} ({ase:>6.3f}) {tm:>9.1f}")


if __name__ == "__main__":
    main()
