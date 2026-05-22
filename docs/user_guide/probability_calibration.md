# Probability Calibration

Raw decision scores are not calibrated probabilities. TorchKM supports
probability calibration for the high-level binary classifiers through Platt
scaling when `probability=True`.

## Basic usage

```python
clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=5,
    device=device,
    probability=True,
)

clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)
```

## Reliability plot

TorchKM also provides a reliability-curve plotting method:

```python
ax, stats = clf.platt_plot(Xte, yte)
ax.figure.savefig("calibration.png", dpi=150, bbox_inches="tight")

import matplotlib.pyplot as plt
plt.close(ax.figure)

print("ECE:", stats["ece"])
print("Brier score:", stats["brier"])
```

## When to use calibrated probabilities

Use probability calibration when you care about probability estimates rather than only class labels. Examples include:

- risk scoring;
- threshold selection;
- ranking examples by predicted risk;
- comparing predicted probabilities with observed frequencies.

## Notes

- `predict_proba` is available only when the model is fitted with `probability=True`.
- Probability calibration adds extra computation.
- Calibration quality should be evaluated on held-out data whenever possible.
- `platt_plot` returns a Matplotlib axis and a statistics dictionary. It does
  not call `plt.show()`, so scripts should save or close the figure explicitly.
