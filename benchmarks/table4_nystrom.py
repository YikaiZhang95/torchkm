"""Reproduce Table 4: TorchKM Nystrom kernel SVM vs scikit-learn Nystrom.

Reproduces the source notebook protocol exactly. Datasets: a9a, w8a, ijcnn1,
covtype, mnist8m (4-vs-6 subset). Per-dataset settings follow the notebook
cells (a9a: cell 19-22, rcv1/covtype-like: cell 36-38, MNIST: cell 54-56,
sklearn Nystrom: cell 65-69).

Methods
-------
TorchKM Nystrom: ``cvknyssvm`` invoked directly with
    nlam=50, nfolds=10, gamma=1e-8, num_landmarks=2000, device='cuda'
and per-dataset overrides for ``ulam`` (the lambda grid), ``eps``, ``maxit``,
and ``k`` (Nystrom rank). Time covers ``cvknyssvm(...).fit()``; the kernel
construction happens inside fit. Accuracy is held-out test accuracy at the
CV-selected lambda, computed via the stored ``Z_test`` features.

scikit-learn Nystrom: the same Nystrom feature map the notebook builds
manually (rbf_kernel on 2000 landmarks, SVD truncated to k=300, Z = C @ M),
then ``LinearSVC`` tuned by 10-fold ``cross_val_score`` over ``logspace(5,-5)``.
The final refit follows the notebook and uses a *fresh* ``randperm`` for the
landmarks. Time covers the full pipeline -- transform, CV sweep, final fit.

Data
----
``--data-dir`` should contain the LIBSVM files (decompressed). Pair-mode
datasets need ``<name>`` and ``<name>.t``; single-file datasets are split
deterministically. MNIST8m must be the 4-vs-6 subset (or the multiclass file
``mnist8m.scale``, in which case ``classes=(4,6)`` filters at load time).

    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{a9a,a9a.t,w8a,w8a.t,ijcnn1,ijcnn1.t}
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.scale

Example:
    python benchmarks/table4_nystrom.py --data-dir ./libsvm_data --datasets a9a w8a ijcnn1 \
        --repeats 10 --device cuda
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from torchkm.cvknyssvm import cvknyssvm
from torchkm.functions import kernelMult, rbf_kernel, sigest

from _common import free_cuda, get_device, mean_se, timed, warmup

NLAM = 50
NFOLDS = 10
NUM_LANDMARKS = 2000

# Per-dataset settings -- match the source notebook cells exactly.
# ulam_log = (start, end) for torch.logspace; maxit, k, eps follow the notebook.
DATASETS = {
    "a9a": dict(
        train="a9a",
        test="a9a.t",
        classes=None,
        ulam_log=(-1, -7),
        maxit=10_000_000,
        k=300,
        eps=1e-3,
    ),
    "w8a": dict(
        train="w8a",
        test="w8a.t",
        classes=None,
        ulam_log=(3, -3),
        maxit=1_000_000,
        k=300,
        eps=1e-3,
    ),
    "ijcnn1": dict(
        train="ijcnn1",
        test="ijcnn1.t",
        classes=None,
        ulam_log=(3, -3),
        maxit=1_000_000,
        k=300,
        eps=1e-3,
    ),
    "covtype": dict(
        train="covtype.libsvm.binary.scale",
        test=None,
        classes=None,
        ulam_log=(3, -3),
        maxit=1_000_000,
        k=30,
        eps=1e-3,
    ),
    "mnist8m": dict(
        train="mnist8m.scale",
        test=None,
        classes=(4, 6),
        ulam_log=(3, -3),
        maxit=1_000_000,
        k=300,
        eps=1e-8,
    ),
}


def _open_libsvm(path):
    """Open a LIBSVM file as a binary stream, handling .bz2 / .xz / plain.

    LIBSVM ships ijcnn1, covtype, mnist8m only compressed; this lets the script
    read them straight from --data-dir without a separate decompress step.
    """
    if os.path.exists(path):
        return open(path, "rb")
    if os.path.exists(path + ".bz2"):
        import bz2

        return bz2.open(path + ".bz2", "rb")
    if os.path.exists(path + ".xz"):
        import lzma

        return lzma.open(path + ".xz", "rb")
    raise FileNotFoundError(f"none of {path}, {path}.bz2, {path}.xz exists")


def load_dataset(data_dir, cfg, seed):
    """Load LIBSVM data per the notebook recipe: load_svmlight_files for pairs,
    train_test_split for single files; class filter for MNIST; densify; to torch."""
    from sklearn.datasets import load_svmlight_file, load_svmlight_files

    train_path = os.path.join(data_dir, cfg["train"])
    if cfg["test"] is not None:
        test_path = os.path.join(data_dir, cfg["test"])
        with _open_libsvm(train_path) as ftr, _open_libsvm(test_path) as fte:
            Xtr, ytr, Xte, yte = load_svmlight_files((ftr, fte), dtype=np.float64)
    else:
        from sklearn.model_selection import train_test_split

        with _open_libsvm(train_path) as f:
            X, y = load_svmlight_file(f, dtype=np.float64)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

    if cfg["classes"] is not None:
        a, b = cfg["classes"]
        tr_mask = (ytr == a) | (ytr == b)
        te_mask = (yte == a) | (yte == b)
        Xtr, ytr = Xtr[tr_mask], np.where(ytr[tr_mask] == b, 1.0, -1.0)
        Xte, yte = Xte[te_mask], np.where(yte[te_mask] == b, 1.0, -1.0)
    else:
        ytr = np.where(ytr > 0, 1.0, -1.0)
        yte = np.where(yte > 0, 1.0, -1.0)

    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
        Xte = Xte.toarray()
    Xtr = torch.from_numpy(np.asarray(Xtr)).float()
    Xte = torch.from_numpy(np.asarray(Xte)).float()
    ytr = torch.from_numpy(ytr).float()
    yte = torch.from_numpy(yte).float()
    return Xtr, ytr, Xte, yte


def run_torchkm(Xtr, ytr, Xte, yte, cfg, device, seed):
    """cvknyssvm directly, exactly as in the notebook. Returns (acc, time)."""
    torch.manual_seed(seed)
    nn_obs = Xtr.shape[0]
    ulam = torch.logspace(float(cfg["ulam_log"][0]), float(cfg["ulam_log"][1]), NLAM)
    foldid = torch.randperm(nn_obs) % NFOLDS + 1

    with timed(device) as t:
        model = cvknyssvm(
            Xmat=Xtr,
            X_test=Xte,
            y=ytr,
            nlam=NLAM,
            ulam=ulam,
            foldid=foldid,
            nfolds=NFOLDS,
            eps=cfg["eps"],
            maxit=cfg["maxit"],
            gamma=1e-8,
            num_landmarks=NUM_LANDMARKS,
            k=cfg["k"],
            device=device,
        )
        model.fit()

    cv_mis = model.cv(model.pred, ytr).numpy()
    best_ind = int(np.argmin(cv_mis))
    alpvec = model.alpmat.to("cpu")[:, best_ind]
    result = torch.mv(model.Z_test.double().cpu(), alpvec[1:]) + alpvec[0]
    ypred = torch.where(result > 0, 1.0, -1.0)
    acc = float((ypred == yte).float().mean())
    del model
    return acc, t.dt


def _nystrom_transform(X, landmarks, sig_w, k):
    """Manual Nystrom feature map Z = C @ M, exactly as the notebook builds it."""
    W = rbf_kernel(landmarks, sig_w)
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    k_eff = min(k, len(S))
    U = U[:, :k_eff]
    S = S[:k_eff]
    M = U * (1.0 / torch.sqrt(S))
    C = kernelMult(X, landmarks, sig_w)
    return torch.mm(C, M)


def run_sklearn(Xtr, ytr, Xte, yte, device):
    """Notebook sklearn baseline (cells 65-69): manual Nystrom + LinearSVC tuned
    by 10-fold CV over logspace(5,-5). Returns (acc, time)."""
    from sklearn import svm
    from sklearn.model_selection import cross_val_score

    nn_obs = Xtr.shape[0]
    ulam = torch.logspace(5.0, -5.0, NLAM)
    k = 300
    ytr_np = ytr.numpy()
    yte_np = yte.numpy()

    with timed(device) as t:
        # CV transform: seeded landmarks (notebook cell 68)
        torch.manual_seed(0)
        indices = torch.randperm(nn_obs)[:NUM_LANDMARKS]
        landmarks = Xtr[indices]
        sig_w = sigest(landmarks)
        Z_train = _nystrom_transform(Xtr, landmarks, sig_w, k).numpy()
        cv_results = [
            cross_val_score(
                svm.LinearSVC(C=float(1.0 / (2 * nn_obs * float(lam)))),
                Z_train,
                ytr_np,
                cv=NFOLDS,
            ).mean()
            for lam in ulam
        ]
        lam_best = float(ulam[int(np.argmax(cv_results))])

        # Final refit: fresh landmarks (notebook cell 69)
        indices = torch.randperm(nn_obs)[:NUM_LANDMARKS]
        landmarks = Xtr[indices]
        sig_w = sigest(landmarks)
        Z_train = _nystrom_transform(Xtr, landmarks, sig_w, k).numpy()
        Z_test = _nystrom_transform(Xte, landmarks, sig_w, k).numpy()
        model = svm.LinearSVC(C=float(1.0 / (2 * nn_obs * lam_best))).fit(
            Z_train, ytr_np
        )

    acc = float((model.predict(Z_test) == yte_np).mean())
    return acc, t.dt


def fmt(acc_se, time_mean):
    if acc_se is None:
        return f"{'skipped':>18} {'-':>8}"
    m, se = acc_se
    return f"{m:>7.3f} ({se:>6.3f}) {time_mean:>9.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="dir holding the LIBSVM files")
    ap.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    ap.add_argument(
        "--repeats", type=int, default=1, help="runs per dataset (paper: 10)"
    )
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--skip-sklearn", action="store_true")
    ap.add_argument(
        "--max-iter", type=int, default=None, help="override per-dataset maxit"
    )
    args = ap.parse_args()

    device = get_device(args.device)
    print(
        f"device={device}  repeats={args.repeats}  folds={NFOLDS}  landmarks={NUM_LANDMARKS}"
    )
    print()
    free_cuda(device)
    warmup(device)
    free_cuda(device)

    header = (
        f"{'data':>8} {'n_train':>9} | "
        f"{'TorchKM acc / t(s)':>22} | {'sklearn-Nys acc / t(s)':>22}"
    )
    print(header)
    print("-" * len(header))

    for name in args.datasets:
        cfg = dict(DATASETS[name])
        if args.max_iter is not None:
            cfg["maxit"] = args.max_iter
        Xtr, ytr, Xte, yte = load_dataset(args.data_dir, cfg, args.seed)
        tk, sk = ([], []), ([], [])
        for i in range(args.repeats):
            a, dt = run_torchkm(Xtr, ytr, Xte, yte, cfg, device, args.seed + i)
            tk[0].append(a)
            tk[1].append(dt)
            free_cuda(device)
            if not args.skip_sklearn:
                a, dt = run_sklearn(Xtr, ytr, Xte, yte, device)
                sk[0].append(a)
                sk[1].append(dt)
                free_cuda(device)

        def cell(pair):
            if not pair[0]:
                return fmt(None, None)
            return fmt(mean_se(pair[0]), mean_se(pair[1])[0])

        print(f"{name:>8} {Xtr.shape[0]:>9} | {cell(tk)} | {cell(sk)}")
        del Xtr, ytr, Xte, yte
        free_cuda(device)


if __name__ == "__main__":
    main()
