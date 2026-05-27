"""Reproduce Table 3: neural network vs TorchKM vs ThunderSVM on a7a/a8a/w7a.

Table 3 reports test accuracy and end-to-end run time on three LIBSVM
classification datasets, averaged over 10 runs (the paper). This script follows
the source notebook exactly for each method.

  * Data: ``load_svmlight_files((train, test))`` so features align; labels mapped
    to {-1, +1}; densified to torch float tensors.
  * Lambda grid: ``torch.logspace(-1, -5, 50)`` (the notebook's grid for these
    datasets), transferred to C = 1/(2*n*lambda) for the libsvm-style baselines.
  * TorchKM: exact RBF kernel SVM via ``cvksvm`` directly -- sig = sigest(X),
    Kmat = rbf_kernel(X, sig), then cvksvm(eps=1e-3, maxit=--max-iter, gamma=1e-8,
    is_exact=0) with 10-fold CV. Time covers the solver fit (kernel build is
    outside the timed region, as in the notebook). Accuracy is the held-out test
    accuracy at the CV-selected lambda.
  * ThunderSVM: RBF SVC with gamma=sig, tol=1e-8, tuned by 10-fold cross_val_score
    over the grid. Time covers the CV sweep plus the final fit.
  * Neural network: a 1D-CNN (two conv layers + ReLU + adaptive avg pool) with an
    8-point grid over (conv_channels, fc_units, lr), 10 epochs each. Reports the
    best test accuracy and the total tuning time, exactly as the notebook does.

Data: download the LIBSVM files into --data-dir (train + ".t" test file each):
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{a7a,a8a,w7a}

ThunderSVM is optional (install it, then `cd thundersvm/python/` or pass
--thundersvm-path). Use --device cuda on a GPU.

Example:
    python benchmarks/table3_benchmarks.py --data-dir DATA_DIR --repeats 10 \
        --device cuda --thundersvm-path /path/to/thundersvm/python
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchkm.cvksvm import cvksvm
from torchkm.functions import kernelMult, rbf_kernel, sigest

from _common import get_device, mean_se, timed, warmup

NFOLDS = 10
NLAM = 50
DATASETS = [("a7a", "a7a", "a7a.t"), ("a8a", "a8a", "a8a.t"), ("w7a", "w7a", "w7a.t")]


def load_pair(data_dir, train_f, test_f):
    from sklearn.datasets import load_svmlight_files

    Xtr, ytr, Xte, yte = load_svmlight_files(
        (os.path.join(data_dir, train_f), os.path.join(data_dir, test_f)), dtype=np.float64
    )
    ytr = np.where(ytr > 0, 1.0, -1.0)
    yte = np.where(yte > 0, 1.0, -1.0)
    Xtr = torch.from_numpy(Xtr.toarray()).float()
    Xte = torch.from_numpy(Xte.toarray()).float()
    ytr = torch.from_numpy(ytr).float()
    yte = torch.from_numpy(yte).float()
    return Xtr, ytr, Xte, yte


def run_torchkm(Xtr, ytr, Xte, yte, sig, Kmat, ulam, device, max_iter, is_exact):
    """Exact RBF kernel SVM via cvksvm. Returns (test accuracy, fit time)."""
    with timed(device) as t:
        model = cvksvm(
            Kmat=Kmat, y=ytr, nlam=NLAM, ulam=ulam, nfolds=NFOLDS,
            eps=1e-3, maxit=max_iter, gamma=1e-8, is_exact=is_exact, device=device,
        )
        model.fit()

    cv_mis = model.cv(model.pred, ytr).numpy()
    best_ind = int(np.argmin(cv_mis))
    alpmat = model.alpmat.to("cpu")
    Kmat_new = kernelMult(Xte, Xtr, sig).double()
    result = torch.mv(Kmat_new, alpmat[1:, best_ind]) + alpmat[0, best_ind]
    ypred = torch.where(result > 0, 1.0, -1.0)
    acc = float((ypred == yte).float().mean())
    return acc, t.dt


def run_thunder(SVC, Xtr_np, ytr_np, Xte_np, yte_np, sig, ulam, device):
    """RBF ThunderSVM tuned by 10-fold CV. Returns (test accuracy, time)."""
    from sklearn.model_selection import cross_val_score

    nn_obs = Xtr_np.shape[0]
    with timed(device) as t:
        cv = [
            cross_val_score(SVC(kernel="rbf", C=float(1.0 / (2 * nn_obs * float(l))), gamma=sig, tol=1e-8), Xtr_np, ytr_np, cv=NFOLDS).mean()
            for l in ulam
        ]
        lam_best = float(ulam[int(np.argmax(cv))])
        model = SVC(kernel="rbf", C=float(1.0 / (2 * nn_obs * lam_best)), gamma=sig, tol=1e-8).fit(Xtr_np, ytr_np)
    acc = float((model.predict(Xte_np) == yte_np).mean())
    return acc, t.dt


class CNN1DModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=32, fc_units=64, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * 2, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def run_nn(Xtr, ytr, Xte, yte, device, seed):
    """1D-CNN baseline: grid search over 8 configs, best test accuracy + tuning time."""
    torch.manual_seed(seed)
    ytr01 = (ytr == 1).long().to(device)
    yte01 = (yte == 1).long().to(device)
    Xtr_t = Xtr.unsqueeze(1).to(device)  # (N, 1, F)
    Xte_t = Xte.unsqueeze(1).to(device)
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr01), batch_size=64, shuffle=True)

    best_acc = 0.0
    with timed(device) as t:
        for conv_channels in (16, 32):
            for fc_units in (64, 128):
                for lr in (0.001, 0.01):
                    model = CNN1DModel(conv_channels=conv_channels, fc_units=fc_units).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    model.train()
                    for _ in range(10):
                        for bx, by in train_loader:
                            optimizer.zero_grad()
                            loss = criterion(model(bx), by)
                            loss.backward()
                            optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        pred = model(Xte_t).max(1).indices
                        acc = float((pred == yte01).float().mean())
                    best_acc = max(best_acc, acc)
    return best_acc, t.dt


def load_thundersvm(path):
    if path:
        sys.path.insert(0, path)
    from thundersvm import SVC

    return SVC


def fmt(acc_se, time_mean):
    if acc_se is None:
        return f"{'skipped':>16} {'-':>8}"
    m, se = acc_se
    return f"{m:>7.3f} ({se:>6.3f}) {time_mean:>8.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="dir holding the LIBSVM files")
    ap.add_argument("--repeats", type=int, default=1, help="runs per dataset (paper: 10)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--max-iter", type=int, default=100000, help="cvksvm maxit (notebook: 100000)")
    ap.add_argument("--exact", action="store_true", help="TorchKM exact CV (is_exact=1; default 0)")
    ap.add_argument("--skip-thunder", action="store_true")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--thundersvm-path", default=None, help="path to thundersvm/python")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device={device}  repeats={args.repeats}  folds={NFOLDS}  grid=50 lambda in [1e-5,1e-1]")

    ThunderSVC = None
    if not args.skip_thunder:
        try:
            ThunderSVC = load_thundersvm(args.thundersvm_path)
        except ImportError:
            print(
                "ThunderSVM not importable -> skipping that column. Install it "
                "(https://github.com/Xtra-Computing/thundersvm), then `cd thundersvm/python/` "
                "or pass --thundersvm-path /path/to/thundersvm/python."
            )
    print()
    warmup(device)

    ulam = torch.logspace(-1.0, -5.0, NLAM)
    header = (
        f"{'data':>6} {'n_train':>8} | {'NN acc / t(s)':>18} | "
        f"{'TorchKM acc / t(s)':>18} | {'ThunderSVM acc / t(s)':>21}"
    )
    print(header)
    print("-" * len(header))

    for name, train_f, test_f in DATASETS:
        Xtr, ytr, Xte, yte = load_pair(args.data_dir, train_f, test_f)
        Xtr_np, ytr_np = Xtr.numpy(), ytr.numpy()
        Xte_np, yte_np = Xte.numpy(), yte.numpy()
        nn_acc, tk, th = ([], []), ([], []), ([], [])
        for i in range(args.repeats):
            torch.manual_seed(args.seed + i)
            sig = sigest(Xtr)
            Kmat = rbf_kernel(Xtr, sig)
            a, dt = run_torchkm(Xtr, ytr, Xte, yte, sig, Kmat, ulam, device, args.max_iter, int(args.exact))
            tk[0].append(a)
            tk[1].append(dt)
            if not args.skip_nn:
                a, dt = run_nn(Xtr, ytr, Xte, yte, device, args.seed + i)
                nn_acc[0].append(a)
                nn_acc[1].append(dt)
            if ThunderSVC is not None:
                a, dt = run_thunder(ThunderSVC, Xtr_np, ytr_np, Xte_np, yte_np, sig, ulam, device)
                th[0].append(a)
                th[1].append(dt)

        def cell(pair):
            if not pair[0]:
                return fmt(None, None)
            return fmt(mean_se(pair[0]), mean_se(pair[1])[0])

        print(f"{name:>6} {Xtr.shape[0]:>8} | {cell(nn_acc)} | {cell(tk)} | {cell(th)}")


if __name__ == "__main__":
    main()
