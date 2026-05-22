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
| `low_rank` | Whether to use the Nyström approximation |
| `num_landmarks` | Number of Nyström landmark points |
| `nys_k` | Rank used in the Nyström approximation |
| `max_iter` | Maximum number of optimization iterations |
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
    nC=len(Cs),
    cv=5,
    device=device,
    probability=True,
    max_iter=40,
)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)
```

## Low-rank SVM

For larger data sets, use the Nyström approximation:

```python
clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=5,
    device=device,
    low_rank=True,
    num_landmarks=40,
    nys_k=20,
    max_iter=40,
)
clf.fit(Xtr, ytr)
```

Constructor-based configuration is recommended, but short examples may also
enable the Nyström path at fit time:

```python
clf = TorchKMSVC(kernel="rbf", Cs=Cs, cv=5, device=device, probability=True)
clf.fit(Xtr, ytr, low_rank=True, num_landmarks=40, nys_k=20)
```

The high-level low-rank path currently supports raw-feature RBF workflows. It
does not support `kernel="precomputed"`.
