# Example: Probability Calibration

This example fits an SVM with Platt scaling and computes calibrated class probabilities.

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
    probability=True,
    max_iter=40,
)

clf.fit(Xtr, ytr)

print("first five probabilities:")
print(clf.predict_proba(Xte[:5]))

ax, stats = clf.platt_plot(Xte, yte)
ax.figure.savefig("calibration.png", dpi=150, bbox_inches="tight")

import matplotlib.pyplot as plt
plt.close(ax.figure)

print("ECE:", stats["ece"])
print("Brier score:", stats["brier"])
```
