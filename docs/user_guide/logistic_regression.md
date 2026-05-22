# Kernel Logistic Regression

`TorchKMLogit` provides a high-level wrapper for kernel logistic regression with integrated model selection.

## Basic usage

```python
import numpy as np
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMLogit

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

clf = TorchKMLogit(
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

## Relationship to SVM

Kernel logistic regression uses a smooth logistic loss rather than the SVM hinge loss. The user-facing workflow is similar to `TorchKMSVC`, but the optimization problem and loss function are different.

## Suggested use

Kernel logistic regression is useful when you want a smooth-loss kernel
classifier while retaining the same integrated model-selection workflow.
`predict_proba` is available only when the estimator is fitted with
`probability=True`, in which case TorchKM fits Platt calibration on the selected
out-of-fold scores.

Useful fitted attributes include `best_C_`, `best_ind_`, `cv_mis_`, `classes_`,
`alpha_`, and `intercept_`.
