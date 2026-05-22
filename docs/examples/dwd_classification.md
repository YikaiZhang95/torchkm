# Example: DWD Classification

This example uses `TorchKMDWD` for binary classification.

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

clf = TorchKMDWD(kernel="rbf", Cs=Cs, nC=len(Cs), cv=5, device=device, max_iter=40)
clf.fit(Xtr, ytr)

print("device:", device)
print("best C:", clf.best_C_)
print("test accuracy:", (clf.predict(Xte) == yte).mean())
```
