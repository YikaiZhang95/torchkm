# Distance-Weighted Discrimination

Distance-weighted discrimination, or DWD, is a margin-based classification method that can be useful in settings where support vector machines are affected by data piling or where a different margin geometry is desired.

TorchKM provides a high-level wrapper:

```python
from torchkm.estimators import TorchKMDWD
```

## Basic usage

```python
import numpy as np
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMDWD

X, y = make_classification(
    n_samples=120,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=0,
)
X = StandardScaler().fit_transform(X)
y = np.where(y == 0, -1, 1)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

Cs = np.logspace(2, -2, num=4)
device = "cuda" if torch.cuda.is_available() else "cpu"

clf = TorchKMDWD(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=5,
    device=device,
    max_iter=40,
)

clf.fit(Xtr, ytr)

print("best C:", clf.best_C_)
print("test accuracy:", (clf.predict(Xte) == yte).mean())
```

## Notes

The DWD estimator follows the same general interface as `TorchKMSVC`:

- instantiate the estimator;
- pass a grid of regularization values through `Cs`;
- call `fit`;
- call `predict`;
- inspect the selected regularization value through `best_C_`.

Like the other high-level binary classifiers, `TorchKMDWD` accepts any two
distinct labels, stores the original labels in `classes_`, and returns
predictions in the original label space. Cross-validation scores are stored in
`cv_mis_`.

Set `probability=True` to fit Platt calibration and enable `predict_proba`.
Set `low_rank=True` with an RBF kernel to use the Nyström backend.

## When to use

Use `TorchKMDWD` when you want a kernel-based large-margin classifier but would like to compare against an alternative to the SVM hinge-loss formulation.
