# TorchKM(Coming Soon!)
![GitHub Release](https://img.shields.io/github/v/release/YikaiZhang95/torchkm)

**TorchKM** is a PyTorch-based library for fast **kernel machines** with a focus on:

- **Large-margin classification**: kernel SVM, kernel DWD, kernel logistic regression  
- **Fast model selection**: pathwise solutions over a grid of regularization values (λ) and efficient cross-validation
- **GPU acceleration** via PyTorch/CUDA (with safe CPU fallback)
- A **scikit-learn–compatible wrapper API** for easy integration with sklearn pipelines

## Introduction

`torchkm`, a PyTorch-based library that trains kernel SVMs and other large-margin classifiers with exact leave-one-out cross-validation (LOOCV) error computation. Conventional SVM solvers often face scalability and efficiency challenges, especially on large datasets or when multiple cross-validation runs are required. torchkm computes LOOCV at the same cost as training a single SVM while boosting speed and scalability via CUDA-accelerated matrix operations. Benchmark experiments indicate that TorchKSVM outperforms existing kernel SVM solvers in efficiency and speed. This document shows how to use the `torchkm` package to fit kernel SVM.

When dealing with low-dimensional problems or more complex scenarios, such as requiring non-linear decision boundaries or higher accuracy, kernel SVMs can be formulated using the kernel method within a reproducing kernel Hilbert space (RKHS). For consistency, we adopt the same notation introduced in the high-dimensional case in Chapter One.

Given a random sample $\\{y_i, x_i\\}_{i=1}^n$, the kernel SVM can be formulated as a function estimation problem:

![kernel SVM formulation](https://latex.codecogs.com/svg.image?\dpi{130}&space;\min_{f&space;\in&space;\mathcal{H}_K}&space;\left[&space;\frac{1}{n}&space;\sum_{i=1}^n&space;\left(&space;1&space;-&space;y_i&space;f(\mathbf{x}_i)&space;\right)_{+}&space;&plus;&space;\lambda&space;\|f\|_{\mathcal{H}_K}^2&space;\right])

where ![norm](https://latex.codecogs.com/svg.image?\dpi{120}&space;\left\|f\right\|^2_{\mathcal{H}_K}) is the RKHS norm that acts as a regularizer, and $\lambda > 0$ is a tuning parameter.

According to the representer theorem for reproducing kernels (Wahba, 1990), the solution to our problem takes the form:

![f(x) formula](https://latex.codecogs.com/svg.image?\dpi{130}&space;f(\mathbf{x})&space;=&space;\sum_{i=1}^n&space;\alpha_i^{\mathrm{SVM}}&space;K\left(\mathbf{x}_i,&space;\mathbf{x}\right))

The coefficients $\alpha^{SVM}$ are obtained by solving the optimization problem:

![alpha optimization](https://latex.codecogs.com/svg.image?\dpi{130}&space;\boldsymbol{\alpha}^{\mathrm{SVM}}&space;=&space;\arg\min_{\boldsymbol{\alpha}&space;\in&space;\mathbb{R}^n}&space;\left[&space;\frac{1}{n}&space;\sum_{i=1}^n&space;\left(1&space;-&space;y_i&space;\mathbf{K}_i^{\top}&space;\boldsymbol{\alpha}&space;\right)_{+}&space;&plus;&space;\lambda&space;\boldsymbol{\alpha}^\top&space;\mathbf{K}&space;\boldsymbol{\alpha}&space;\right])

where $\mathbf{K}$ is the kernel matrix.


## Installation

### Minimal install (core solvers)
```bash
pip install torchkm
```

This installs the **runtime** dependencies only (typically `torch`, `numpy`).

### Install the scikit-learn wrapper
```bash
pip install "torchkm[sklearn]"
```

### Development install (tests, lint, build tools)
```bash
pip install -e ".[dev,sklearn]"
pytest -q
```

### Reproducing paper experiments (optional)
We recommend keeping pinned environments in a separate file, e.g.:
```bash
pip install -r requirements-paper.txt
pip install -e .
```

---

## Quickstart: “Happy path” (sklearn-style wrapper)

This example is designed to run **unchanged on CPU or GPU**, and uses the same simulation block you’ve been using:

```python
import numpy as np
import torch

from torchkm.sklearn_wrapper import TorchKMSVC

# -----------------------------
# Robust device selection
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device)

# -----------------------------
# Example data generator + standardize
# If your package already provides these, import them instead.
# -----------------------------
def standardize(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return (X - mu) / sd

def data_gen(nn, nm, pp, p1, p2, mu, ro, sdn):
    g = torch.Generator().manual_seed(int(sdn))

    n_pos = nn // 2
    n_neg = nn - n_pos

    means_pos = ro * torch.randn((nm, pp), generator=g, dtype=torch.float64)
    means_neg = ro * torch.randn((nm, pp), generator=g, dtype=torch.float64)

    means_pos[:, :p1] += mu
    means_neg[:, -p2:] -= mu

    cid_pos = torch.randint(0, nm, (n_pos,), generator=g)
    cid_neg = torch.randint(0, nm, (n_neg,), generator=g)

    X_pos = means_pos[cid_pos] + ro * torch.randn((n_pos, pp), generator=g, dtype=torch.float64)
    X_neg = means_neg[cid_neg] + ro * torch.randn((n_neg, pp), generator=g, dtype=torch.float64)

    X = torch.vstack([X_neg, X_pos])
    # Wrapper accepts either {0,1} or {-1,+1}; here we use {-1,+1}
    y = torch.hstack([
        -torch.ones(n_neg, dtype=torch.float64),
        +torch.ones(n_pos, dtype=torch.float64),
    ])

    perm = torch.randperm(nn, generator=g)
    return X[perm], y[perm], {"neg": means_neg, "pos": means_pos}


# -----------------------------
# Your example block (unchanged)
# -----------------------------
nn = 200 # Number of samples
nm = 5   # Number of clusters per class
pp = 10  # Number of features
p1 = p2 = pp // 2
mu = 2.0
ro = 3
sdn = 42

nlam = 10
torch.manual_seed(sdn)
ulam = torch.logspace(3, -3, steps=nlam, dtype=torch.float64)

X_train, y_train, _ = data_gen(nn, nm, pp, p1, p2, mu, ro, sdn)
X_test,  y_test,  _ = data_gen(nn // 10, nm, pp, p1, p2, mu, ro, sdn)
X_train = standardize(X_train)
X_test  = standardize(X_test)

# -----------------------------
# Wrapper expects numpy arrays (sklearn-style)
# -----------------------------
Xtr = X_train.detach().cpu().numpy().astype(np.float64)
Xte = X_test.detach().cpu().numpy().astype(np.float64)
ytr = y_train.detach().cpu().numpy()
yte = y_test.detach().cpu().numpy()

# Deterministic folds for reproducibility (fold IDs in {1,...,nfolds})
nfolds = 5
foldid = (np.arange(Xtr.shape[0]) % nfolds) + 1

# -----------------------------
# Train / CV-select λ / predict
# -----------------------------
clf = TorchKMSVC(
    kernel="rbf",
    nlam=nlam,
    ulam=ulam.detach().cpu().numpy(),
    nfolds=nfolds,
    foldid=foldid,
    device=device,          # "cuda" or "cpu"
    probability=True,       # enables predict_proba via Platt scaling
    platt_device=device,
    maxit=200,
)

clf.fit(Xtr, ytr)

print("best_ind_   =", clf.best_ind_)
print("best_lambda_=", clf.best_lambda_)

yhat = clf.predict(Xte)
acc = (yhat == yte).mean()
print("test accuracy =", acc)

proba = clf.predict_proba(Xte)
print("proba shape =", proba.shape)
print("first 5 proba rows:\n", proba[:5])
```

---

## Low-level API (direct `cvksvm`)

Use this when you want explicit control of kernel matrices, or when you already have a precomputed kernel.

```python
import torch
from torchkm.cvksvm import cvksvm
from torchkm.functions import sigest, rbf_kernel, kernelMult

device = "cuda" if torch.cuda.is_available() else "cpu"

# X_train, y_train in {-1,+1}, X_test constructed however you like
# Assume X_train/X_test are torch.float64 tensors.
sigma = float(sigest(X_train.detach().cpu(), frac=0.5))

K_train = rbf_kernel(X_train.to(device), sigma)
K_test  = kernelMult(X_test.to(device), X_train.to(device), sigma)

nlam = 10
ulam = torch.logspace(3, -3, steps=nlam, dtype=torch.float64, device=device)

nfolds = 5
foldid = (torch.arange(X_train.shape[0], device=device) % nfolds + 1).to(torch.int64)

model = cvksvm(
    Kmat=K_train,
    y=y_train.to(device),
    nlam=nlam,
    ulam=ulam,
    nfolds=nfolds,
    foldid=foldid,
    eps=1e-5,
    maxit=200,
    gamma=1e-8,
    is_exact=0,
    device=device,
)
model.fit()

cv_err = model.cv(model.pred, y_train.to(device))
best_ind = int(torch.argmin(cv_err).item())

alpvec = model.alpmat[:, best_ind]  # [b, alpha_1..alpha_n]
scores = (K_test @ alpvec[1:] + alpvec[0])
```

---

## Why TorchKM is fast

TorchKM is built around two ideas:

1. **Pathwise training across λ**: reuse expensive linear-algebra structure across many regularization values.
2. **Exact CV/LOOCV trick for large-margin classifiers**: compute CV errors without fully refitting models from scratch.

This makes “train + tune” much closer to the cost of training a single model, especially on GPUs.

---

## Models and APIs

### sklearn-style wrappers (recommended for most users)
- `TorchKMSVC` — kernel SVM classifier
- `TorchKMDWD` — kernel DWD classifier
- `TorchKMLogit` — kernel logistic regression

Common methods:
- `fit(X, y)`
- `decision_function(X)`
- `predict(X)`
- `predict_proba(X)` (if `probability=True`)

### Low-level solvers
- `cvksvm` — kernel SVM with CV over λ
- `cvkdwd` — kernel DWD with CV over λ
- `cvklogit` — kernel logistic with CV over λ

### Utilities
- `rbf_kernel`, `kernelMult`, `sigest` — kernel helpers
- `PlattScalerTorch` — probability calibration (Platt scaling)

---

## Device behavior (CPU/GPU)

- If you pass `device="cuda"` but CUDA is not available, use `device="cpu"`.
- For maximum portability, use:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

If you add `device=None` defaults in the library, the recommended pattern becomes:
```python
clf = TorchKMSVC(device=None)  # auto-selects cuda if available else cpu
```

---

## Benchmarking: wrapper vs direct solver

Since the wrapper calls the same backend solvers, **training time is usually dominated by the solver**.  
The wrapper adds a small overhead for input validation/conversion and convenience features (like optional Platt scaling).

We recommend benchmarking both modes in your environment:

- `direct cvksvm` with precomputed kernels (isolates solver cost)
- wrapper end-to-end (realistic user workflow)

You can use a script like `bench/compare_wrapper_vs_cvksvm.py` (see repository).

---

## Testing

Run unit tests (CPU):

```bash
pip install -e ".[dev,sklearn]"
pytest -q
```

If you have CUDA available, you can optionally add GPU tests to CI later (recommended but not required).

---

## Contributing

Contributions are welcome (issues, PRs, benchmarks, docs).  
A good first contribution is adding:

- more kernels (e.g., Laplacian, Matérn),
- multiclass wrappers (One-vs-Rest / One-vs-One),
- CI + coverage badges,
- additional examples and reproducibility scripts.

---

## Citation

If you use TorchKM in academic work, please cite:

```bibtex
@article{zhang2026torchkm,
  title   = {TorchKM: GPU-Accelerated Kernel Machines with Fast Model Selection in PyTorch},
  author  = {Zhang, Yikai and Jia, Gaoxiang and Wang, Boxiang},
  journal = {Journal of Machine Learning Research (MLOSS track)},
  year    = {2026},
  note    = {Software paper submission}
}
```

---

## License

MIT License. See `LICENSE`.


## Getting help

Any questions or suggestions please contact: <yikai-zhang@uiowa.edu>


