# Kernel Quantile Regression

TorchKM includes kernel quantile regression for continuous targets through
`TorchKMKQR`.

## Basic usage

```python
import numpy as np
import torch

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMKQR

X, y = make_regression(
    n_samples=100,
    n_features=5,
    noise=0.5,
    random_state=0,
)
X = StandardScaler().fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

Cs = np.logspace(2, -2, num=3)
device = "cuda" if torch.cuda.is_available() else "cpu"

reg = TorchKMKQR(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=5,
    tau=0.5,
    device=device,
    random_state=0,
    max_iter=80,
)

reg.fit(Xtr, ytr)
pred = reg.predict(Xte)

print("best C:", reg.best_C_)
print("first predictions:", pred[:3])
```

## Important parameters

| Parameter | Description |
|---|---|
| `kernel` | Kernel type for `TorchKMKQR`: `"rbf"`, `"linear"`, `"poly"`, or `"precomputed"` |
| `tau` | Quantile level in `(0, 1)` |
| `Cs` | Candidate regularization values under the `C` convention |
| `nC` | Number of candidate values when `Cs` is not supplied |
| `cv` | Number of cross-validation folds |
| `device` | `"cpu"`, `"cuda"`, or `None` for automatic selection |
| `max_iter` | Maximum solver iterations |

## Fitted attributes

After fitting, useful attributes include:

- `best_C_`: selected regularization value;
- `best_ind_`: selected grid index;
- `cv_loss_`: cross-validation check-loss values;
- `alpha_` and `intercept_`: selected model coefficients;
- `foldid_`: fold assignment used during fitting.

## Low-level solver

Advanced users can access the lower-level solver directly:

```python
from torchkm.cvkqr import cvkqr
```

`cvkqr` handles full-kernel quantile regression, and `cvknysqr` provides the
Nyström solver. See the [low-level solver API](../api/solvers.md) for exact
signatures.
