# Example: SVM on make_circles

This example fits a nonlinear kernel SVM on a toy data set.

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

print("device:", device)
print("best C:", clf.best_C_)
print("test accuracy:", (clf.predict(Xte) == yte).mean())
```

Run the script:

```bash
python examples/svm_make_circles.py
```
