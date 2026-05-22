# TorchKM: Fast Kernel Machines in PyTorch

![PyPI](https://img.shields.io/pypi/v/torchkm)
![Python](https://img.shields.io/pypi/pyversions/torchkm)
![License](https://img.shields.io/github/license/YikaiZhang95/torchkm)
![GitHub Release](https://img.shields.io/github/v/release/YikaiZhang95/torchkm)
[![tests](https://github.com/YikaiZhang95/torchkm/actions/workflows/tests.yml/badge.svg)](https://github.com/YikaiZhang95/torchkm/actions/workflows/tests.yml)
[![docs](https://github.com/YikaiZhang95/torchkm/actions/workflows/docs.yml/badge.svg)](https://github.com/YikaiZhang95/torchkm/actions/workflows/docs.yml)

**TorchKM** is a GPU-accelerated PyTorch-based library for **kernel machines** including kernel SVM with a focus on fast and integrated **train + tune** workflows.

It currently provides:

- **Kernel classification:** kernel SVM, kernel DWD, and kernel logistic regression
- **Kernel regression:** kernel quantile regression
- **Fast model selection:** pathwise solutions over a grid of regularization values (`λ`)
- **Exact cross-validation for kernel machines**
- **GPU acceleration** via PyTorch/CUDA, with safe CPU fallback
- A **scikit-learn–style API** for easy integration into existing Python workflows

## Why TorchKM?

Kernel methods are competitive supervised learning methods on tabular data. In practice, the dominant cost often arises not from a single model fit alone, but from repeated kernel-matrix computations and linear solves across cross-validation folds and tuning parameters.

TorchKM is built for that an integrated training and tuning pipeline. Benchmarks show competitive predictive performance together with substantial speedups over standard baselines.

## Documentation

Full documentation, examples, API reference, benchmark-reproduction notes, and developer guides are available at:

https://yikaizhang95.github.io/torchkm/

## Installation

### Standard install

```bash
pip install torchkm
```

The default installation includes the high-level scikit-learn-style estimator
API used in the examples.

### Development install

```bash
git clone https://github.com/YikaiZhang95/torchkm.git
cd torchkm
pip install -e ".[dev,examples,viz]"
python -m pytest -q
```

## Quickstart

### sklearn-style estimators

```python
import numpy as np
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMSVC

# Toy nonlinear classification task
X, y = make_circles(n_samples=120, factor=0.4, noise=0.08, random_state=0)
X = StandardScaler().fit_transform(X)
y = np.where(y == 0, -1, 1)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

Cs = np.logspace(2, -2, num=4)
device = "cuda" if torch.cuda.is_available() else "cpu"

clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    cv=5,
    device=device,
    probability=True,
    max_iter=40,
)

clf.fit(Xtr, ytr)

print("best C:", clf.best_C_)
print("test accuracy:", (clf.predict(Xte) == yte).mean())
print("first 3 probabilities:\n", clf.predict_proba(Xte[:3]))
```

Set `probability=True` to enable Platt scaling and `predict_proba`.

### Low-rank approximation

Use `low_rank=True` when you want to handle a larger data set. The recommended
scikit-learn-style API sets this in the constructor:

```python
clf = TorchKMSVC(
    kernel="rbf",
    low_rank=True,
    num_landmarks=40,
    nys_k=20,
    nC=4,
    cv=5,
    device=device,
    probability=True,
    max_iter=40,
)

clf.fit(Xtr, ytr)
(clf.predict(Xte) == yte).mean()
```

For convenience, low-rank Nyström fitting can also be enabled at fit time:
`clf.fit(X, y, low_rank=True)`.

```python
clf = TorchKMSVC(kernel="rbf", Cs=Cs, cv=5, device=device, probability=True)
clf.fit(Xtr, ytr, low_rank=True, num_landmarks=40, nys_k=20)
```

### Kernel quantile regression

```python
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from torchkm.estimators import TorchKMKQR

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))
y = np.sin(X[:, 0]) + 0.2 * rng.normal(size=200)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

Cs = np.logspace(2, -2, 4)
device = "cuda" if torch.cuda.is_available() else "cpu"

qr = TorchKMKQR(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=3,
    tau=0.5,
    device=device,
    max_iter=40,
)
qr.fit(Xtr, ytr)
print("best C:", qr.best_C_)
print("predictions:", qr.predict(Xte[:3]))

qr_nys = TorchKMKQR(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=3,
    tau=0.5,
    low_rank=True,
    num_landmarks=40,
    nys_k=20,
    device=device,
    max_iter=40,
)
qr_nys.fit(Xtr, ytr)
print("Nyström predictions:", qr_nys.predict(Xte[:3]))
```

## What TorchKM provides

### sklearn-style estimators

- `TorchKMSVC` — kernel SVM classifier
- `TorchKMDWD` — kernel DWD classifier
- `TorchKMLogit` — kernel logistic regression
- `TorchKMKQR` — kernel quantile regression

`TorchKMKQR` provides kernel quantile regression. Use
`TorchKMKQR(low_rank=True)` for the Nyström approximation. There is no separate
`TorchKMNysKQR` class.

Common methods:

- `fit(X, y)`
- `decision_function(X)`
- `predict(X)`
- `predict_proba(X)` (when `probability=True`)
- `platt_plot(X, y)`

### Utilities

- `rbf_kernel`, `kernelMult`, `sigest`

## Device behavior

For portability, prefer:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

TorchKM is designed to run on GPU when CUDA is available and fall back safely to CPU otherwise.

## When to use TorchKM

TorchKM is a good fit when you want:

- nonlinear kernel classifiers without leaving the PyTorch ecosystem
- fast model selection across many regularization values
- exact LOOCV support for kernel SVM
- a wrapper API that feels familiar if you already use scikit-learn
- lower-level access to kernel matrices and solver internals

## Testing and coverage

```bash
python -m pytest -q
```

Run coverage locally with:

```bash
python -m pytest -q --cov=torchkm --cov-report=term-missing:skip-covered --cov-report=xml --cov-report=html
```

Current coverage: 66.57% on commit `a1b0489`, measured with Python 3.11.14 on
macOS arm64.

The GitHub Actions test workflow runs the test suite and uploads `coverage.xml`
and `htmlcov/` as artifacts.

CUDA smoke tests are marked with `pytest.mark.cuda` and skip automatically when
CUDA is unavailable. On a CUDA machine, run:

```bash
python -m pytest -q -m cuda
```

## Contributing

Issues, bug reports, benchmarks, documentation improvements, and pull requests are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, pull request guidelines, and bug-report instructions.

Good first contributions include:

- additional kernels
- multiclass wrappers
- more benchmark scripts
- expanded examples and tutorials

## Citation

If you use TorchKM in academic work, please cite:

```bibtex
@article{zhang2026torchkm,
  title   = {TorchKM: GPU-Accelerated Kernel Machines with Fast Model Selection in PyTorch},
  author  = {Zhang, Yikai and Jia, Gaoxiang and Ding, Jie and Wang, Boxiang},
  journal = {Journal of Machine Learning Research (MLOSS track)},
  year    = {2026},
  note    = {Software paper submission}
}
```

## License

MIT License. See `LICENSE`.

## Contact

Yikai Zhang  
skyezhang1995@gmail.com
