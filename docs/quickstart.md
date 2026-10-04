# Quickstart

This example fits a kernel SVM with integrated model selection using the scikit-learn-style `TorchKMSVC` estimator.

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

# The high-level wrapper accepts any two labels. Using {-1, +1} keeps this
# example close to the low-level solver convention.
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

print("device:", device)
print("best C:", clf.best_C_)
print("test accuracy:", (clf.predict(Xte) == yte).mean())
print("first three probabilities:")
print(clf.predict_proba(Xte[:3]))
```

## What happens internally?

Instead of calling a separate `fit` for each regularization value and each cross-validation split, TorchKM solves the model-selection problem inside the estimator. Users pass a grid through `Cs`, and TorchKM selects the regularization parameter using cross-validation. The selected value is stored as `best_C_`.

## Using CPU

For a CPU-only run, set:

```python
device = "cpu"
```

## Enabling probability estimates

Set `probability=True` before fitting:

```python
clf = TorchKMSVC(kernel="rbf", Cs=Cs, nC=len(Cs), cv=5, probability=True)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)
```

## Data sets too large for exact mode

When the \(n \times n\) kernel matrix does not fit in memory, `TorchKMSVC`
has a large-\(n\) mode, `low_rank=True`. It fits the exact RBF kernel SVM (it
is not an approximation) but never stores the kernel matrix: every product with
it is recomputed from the training rows.

```python
clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    cv=5,
    device=device,
    dtype="float32",
    low_rank=True,
    max_iter=40,
)
clf.fit(Xtr, ytr)
```

Here `max_iter` is the iteration budget of each regularization value. See
[Kernel SVM](user_guide/svm.md) for details. `TorchKMDWD`, `TorchKMLogit` and
`TorchKMKQR` run in exact mode only.
