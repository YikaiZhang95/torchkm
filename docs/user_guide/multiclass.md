# Multiclass Classification

TorchKM's classifiers are binary. For more than two classes, wrap an estimator
in scikit-learn's one-vs-rest meta-estimator: every TorchKM classifier follows
the scikit-learn interface (`get_params`, `set_params`, `fit`, `predict`,
`decision_function`), so `OneVsRestClassifier` trains one tuned binary model
per class and picks the class with the largest decision score.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMSVC

X, y = load_digits(return_X_y=True)          # 10 classes
X = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

clf = OneVsRestClassifier(
    TorchKMSVC(kernel="rbf", nC=10, cv=5, device="cuda", random_state=0)
)
clf.fit(Xtr, ytr)
print("test accuracy:", (clf.predict(Xte) == yte).mean())
```

Each binary problem runs the full train-and-tune pipeline, so the cost is the
number of classes times one binary fit; the fits are independent and
`OneVsRestClassifier(n_jobs=...)` can run them in parallel on a multi-GPU
machine by setting `device` per copy. Class probabilities need
`probability=True` on the inner estimator, after which `predict_proba` returns
per-class Platt probabilities normalised to sum to one.

One-vs-one (`OneVsOneClassifier`) works the same way and trains
`K (K - 1) / 2` smaller problems, which suits many classes with few samples
each.
