# Example: Nyström Approximation

This example enables the Nyström approximation with `low_rank=True`. It is kept
small enough to run as a smoke-test example; benchmark-scale runs belong in a
separate benchmark environment.

```python
import numpy as np
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMSVC

X, y = make_classification(
    n_samples=200,
    n_features=30,
    n_informative=15,
    random_state=0,
)
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
    low_rank=True,
    num_landmarks=40,
    nys_k=20,
    max_iter=40,
)

clf.fit(Xtr, ytr)

print("device:", device)
print("best C:", clf.best_C_)
print("test accuracy:", (clf.predict(Xte) == yte).mean())
```

Use this mode when the full kernel matrix is too large or too slow for exact computation.
