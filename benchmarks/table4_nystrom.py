"""Reproduce Table 4: Nystrom TorchKM vs scikit-learn Nystrom on large benchmarks.

Table 4 reports test accuracy and end-to-end run time on five larger LIBSVM
datasets (a9a, w8a, ijcnn1, covtype, MNIST8m 4-vs-6), averaged over 10 runs,
using TorchKM's Nystrom solver (``cvknyssvm``, ``low_rank=True``). The paper's
third column is a 1D-CNN baseline (extra training code, omitted here); this lean
script reports TorchKM and a scikit-learn Nystrom + LinearSVC baseline.

Data: download into --data-dir (decompress .bz2/.xz first). a9a/w8a/ijcnn1 ship
with a ".t" test file; covtype and mnist8m are single files and get a split.
    .../binary/a9a (+ a9a.t),  .../binary/w8a (+ w8a.t),  .../binary/ijcnn1 (+ ijcnn1.t)
    .../binary/covtype.libsvm.binary.scale
    .../multiclass/mnist8m.scale   (4-vs-6 subset extracted at load time)

Example:
    python benchmarks/table4_nystrom.py --data-dir /path/to/libsvm \
        --datasets a9a w8a ijcnn1 --repeats 10
"""

from __future__ import annotations

import argparse
import os

import torch

from torchkm import sigest
from torchkm.estimators import TorchKMSVC

from _common import c_grid, get_device, load_train_test, mean_se, timed, warmup

NFOLDS = 10
DATASETS = {
    "a9a": dict(train="a9a", test="a9a.t", classes=None),
    "w8a": dict(train="w8a", test="w8a.t", classes=None),
    "ijcnn1": dict(train="ijcnn1", test="ijcnn1.t", classes=None),
    "covtype": dict(train="covtype.libsvm.binary.scale", test=None, classes=None),
    "mnist8m": dict(train="mnist8m.scale", test=None, classes=(4, 6)),
}


def run_torchkm(Xtr, ytr, Xte, yte, device, seed, num_landmarks, nys_k):
    Cs = c_grid()
    clf = TorchKMSVC(
        kernel="rbf", Cs=Cs, nC=len(Cs), cv=NFOLDS, device=device, random_state=seed,
        low_rank=True, num_landmarks=num_landmarks, nys_k=nys_k,
    )
    with timed(device) as t:
        clf.fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    return acc, t.dt


def run_sklearn_nystrom(Xtr, ytr, Xte, yte, device, seed, num_landmarks):
    from sklearn.kernel_approximation import Nystroem
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    sigma = sigest(torch.as_tensor(Xtr, dtype=torch.double))
    # torchkm rbf_kernel = exp(-2*sigma*||.||^2); Nystroem uses exp(-gamma*||.||^2).
    pipe = make_pipeline(
        Nystroem(kernel="rbf", gamma=2.0 * sigma, n_components=num_landmarks, random_state=seed),
        LinearSVC(),
    )
    grid = GridSearchCV(pipe, {"linearsvc__C": c_grid()}, cv=NFOLDS)
    with timed(device) as t:
        grid.fit(Xtr, ytr)
        pred = grid.predict(Xte)
    return float((pred == yte).mean()), t.dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="dir holding the LIBSVM files")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--repeats", type=int, default=3, help="runs per dataset (paper: 10)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-landmarks", type=int, default=2000)
    ap.add_argument("--nys-k", type=int, default=1000)
    ap.add_argument("--skip-sklearn", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device={device}  repeats={args.repeats}  folds={NFOLDS}  landmarks={args.num_landmarks}\n")
    warmup(device)

    header = (
        f"{'data':>8} {'n_train':>9} {'p':>5} | {'TorchKM acc':>18} {'time(s)':>9} "
        f"| {'sklearn-Nys acc':>18} {'time(s)':>9}"
    )
    print(header)
    print("-" * len(header))

    for name in args.datasets:
        cfg = DATASETS[name]
        test_path = os.path.join(args.data_dir, cfg["test"]) if cfg["test"] else None
        Xtr, ytr, Xte, yte = load_train_test(
            os.path.join(args.data_dir, cfg["train"]), test_path,
            classes=cfg["classes"], seed=args.seed,
        )
        tk_acc, tk_t, sk_acc, sk_t = [], [], [], []
        for i in range(args.repeats):
            a, dt = run_torchkm(
                Xtr, ytr, Xte, yte, device, args.seed + i, args.num_landmarks, args.nys_k
            )
            tk_acc.append(a)
            tk_t.append(dt)
            if not args.skip_sklearn:
                a, dt = run_sklearn_nystrom(
                    Xtr, ytr, Xte, yte, device, args.seed + i, args.num_landmarks
                )
                sk_acc.append(a)
                sk_t.append(dt)

        am, ase = mean_se(tk_acc)
        tm, _ = mean_se(tk_t)
        if sk_acc:
            sam, sase = mean_se(sk_acc)
            stm, _ = mean_se(sk_t)
            sk_cell = f"{sam:>9.3f} ({sase:>6.3f}) {stm:>9.1f}"
        else:
            sk_cell = f"{'skipped':>18} {'-':>9}"
        print(
            f"{name:>8} {Xtr.shape[0]:>9} {Xtr.shape[1]:>5} | "
            f"{am:>9.3f} ({ase:>6.3f}) {tm:>9.1f} | {sk_cell}"
        )


if __name__ == "__main__":
    main()
