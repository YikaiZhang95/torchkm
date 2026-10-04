# Kernel SVM

`TorchKMSVC` is the high-level scikit-learn-style interface for kernel support vector classification.

## Basic usage

```python
import numpy as np
import torch

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMSVC

X, y = make_circles(n_samples=120, factor=0.4, noise=0.08, random_state=0)
X = StandardScaler().fit_transform(X)
y = np.where(y == 0, -1, 1)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

Cs = np.logspace(2, -2, num=4)
device = "cuda" if torch.cuda.is_available() else "cpu"

clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=5,
    device=device,
    max_iter=40,
)

clf.fit(Xtr, ytr)
pred = clf.predict(Xte)

print("best C:", clf.best_C_)
print("test accuracy:", (pred == yte).mean())
```

## Supported kernels

The high-level estimator supports common kernels such as:

- `"rbf"`
- `"linear"`
- `"poly"`
- `"precomputed"`

For the RBF kernel, TorchKM can estimate a kernel scale automatically when `rbf_sigma=None`.

## Important parameters

| Parameter | Description |
|---|---|
| `kernel` | Kernel type |
| `Cs` | Candidate regularization values |
| `nC` | Number of candidate regularization values |
| `cv` | Number of cross-validation folds |
| `device` | Device used for computation |
| `probability` | Whether to fit probability calibration |
| `low_rank` | Large-\(n\) mode: exact RBF kernel SVM without storing the kernel matrix |
| `max_iter` | Maximum number of optimization iterations (per lambda when `low_rank=True`) |
| `tol` | Numerical tolerance |

## Labels and fitted attributes

The high-level estimator accepts any two distinct class labels. Internally it
maps labels to `{-1, +1}` for the solver and maps predictions back to the
original labels.

After fitting, useful attributes include:

- `best_C_`: selected regularization value;
- `best_ind_`: selected grid index;
- `cv_mis_`: cross-validation misclassification scores;
- `classes_`: original class labels;
- `alpha_` and `intercept_`: selected model coefficients.

## Probability estimates

To enable class probabilities, set `probability=True` before fitting:

```python
clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    cv=5,
    device=device,
    probability=True,
    max_iter=40,
)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)
```

## Large-n SVM: `low_rank=True`

When the \(n \times n\) kernel matrix does not fit in memory, set
`low_rank=True`. This is not an approximation of the model: it fits the exact
RBF kernel SVM with the truncated-spectrum solver
(`torchkm.experimental.SpectralSVMPath`, as `spectrum="truncated"`), but the
kernel matrix is never stored. Every product with it is recomputed from the
training rows (`torchkm.experimental.RBFKernelOperator`), so memory grows like
\(n\) times the columns of a block (`spectrum_block` \(\times\) (`cv` + 1))
instead of \(n^2\).

```python
clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    cv=10,
    device="cuda",
    dtype="float32",
    low_rank=True,
    spectrum_rank=400,
    spectrum_block=10,
    max_iter=40,
)
clf.fit(Xtr, ytr)
print(clf.converged_)
```

- It needs `kernel="rbf"` on raw features (no `"precomputed"` kernel).
- On CUDA with `dtype="float32"` the kernel products run fused in one GPU
  kernel.
- `spectrum_rank`, `gap_tol` and `spectrum_block` apply; `spectrum` is ignored.
- `max_iter` is the iteration budget of each lambda. Fits that reach it are
  kept and reported as not converged in `converged_`.
- Each product pays the kernel's arithmetic again, so when the kernel fits in
  memory the stored-kernel modes (the default, or `spectrum="truncated"`) are
  faster.

On the whole covtype.binary training set (464,809 rows), 50 lambdas with
10-fold cross-validation at RBF \(\gamma = 32\), the call above
(`dtype="float32"`, `max_iter=40`, `spectrum_block=10`, `spectrum_rank=400`)
took 62 minutes and 6.6 GB of GPU memory on an NVIDIA L40S, with test accuracy
0.9609. cuML's SVC on the same job took 243 minutes and 4.5 GB, with test
accuracy 0.9620.
