# TorchKM: Fast Kernel Machines in PyTorch

![PyPI](https://img.shields.io/pypi/v/torchkm)
![Python](https://img.shields.io/pypi/pyversions/torchkm)
![License](https://img.shields.io/github/license/YikaiZhang95/torchkm)
![GitHub Release](https://img.shields.io/github/v/release/YikaiZhang95/torchkm)

**TorchKM** is a PyTorch-based library for **kernel machines** with a focus on fast **train + tune** workflows.

It currently provides:

- **Kernel classification:** kernel SVM, kernel DWD, and kernel logistic regression
- **Fast model selection:** pathwise solutions over a grid of regularization values (`λ`)
- **Exact LOOCV support for kernel SVM**
- **GPU acceleration** via PyTorch/CUDA, with safe CPU fallback
- A **scikit-learn–style API** for easy integration into existing Python workflows

## Why TorchKM?

Kernel methods are still a strong choice when you want nonlinear decision boundaries, convex training objectives, and competitive performance on tabular or moderate-scale datasets. In practice, the bottleneck is often not training one model — it is training **and tuning many models**.

TorchKM is built for that workflow.
## Installation

### Minimal install

```bash
pip install torchkm
```

### Install the scikit-learn wrapper

```bash
pip install "torchkm[sklearn]"
```

### Development install

```bash
git clone https://github.com/YikaiZhang95/torchkm.git
cd torchkm
pip install -e ".[dev,sklearn]"
pytest -q
```

## Quickstart

### sklearn-style wrapper (recommended)

```python
import numpy as np
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMSVC

# Toy nonlinear classification task
X, y = make_circles(n_samples=1200, factor=0.4, noise=0.08, random_state=0)
X = StandardScaler().fit_transform(X)
y = np.where(y == 0, -1, 1)   # TorchKM accepts {-1, +1} labels

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

nfolds = 5
Cs = np.logspace(3, -3, num=12)

clf = TorchKMSVC(
    kernel="rbf",
    nC=len(Cs),
    Cs=Cs,
    nfolds=nfolds,
    device='cuda',
    probability=True,
    max_iter=200,
)

clf.fit(Xtr, ytr)

print("best lambda:", clf.best_C_)
print("test accuracy:", (clf.predict(Xte) == yte).mean())
print("first 3 probabilities:\n", clf.predict_proba(Xte[:3]))
```

### Low-rank approximation

Use the `low_rank=True` when you want to handle a large data set.

```python
clf = TorchKMSVC(
    kernel="rbf",
    low_rank=True,
    num_landmarks=500,
    nys_k=250,
    nC=20,
    nfolds=5,
    device='cuda',
    probability=True,
    platt_device='cuda',
)

clf.fit(Xtr, ytr)
(clf.predict(Xte) == yte).mean()
```

## What TorchKM provides

### High-level wrappers

- `TorchKMSVC` — kernel SVM classifier
- `TorchKMDWD` — kernel DWD classifier
- `TorchKMLogit` — kernel logistic regression

Common methods:

- `fit(X, y)`
- `decision_function(X)`
- `predict(X)`
- `predict_proba(X)` (when `probability=True`)
- `platt_plot(X, y)`

### Low-level solvers

- `cvksvm` — kernel SVM with cross-validation over `λ`
- `cvkdwd` — kernel DWD with cross-validation over `λ`
- `cvklogit` — kernel logistic regression with cross-validation over `λ`

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

## Testing

```bash
pytest -q
```

## Contributing

Issues, bug reports, benchmarks, documentation improvements, and pull requests are welcome.

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
  author  = {Zhang, Yikai and Jia, Gaoxiang and Wang, Boxiang},
  journal = {Journal of Machine Learning Research (MLOSS track)},
  year    = {2026},
  note    = {Software paper submission}
}
```

## License

MIT License. See `LICENSE`.

## Contact

Yikai Zhang  
yikai-zhang@uiowa.edu


