# Model Selection

Model selection is central to TorchKM. In a standard workflow, a kernel machine is trained repeatedly over a grid of tuning parameters and cross-validation folds. This can be expensive because each fit may require solving a large kernel system.

TorchKM changes this workflow by integrating training and tuning into the solver.

## Standard workflow

In a standard scikit-learn workflow, users might combine an estimator with `GridSearchCV`:

```python
# Conceptual standard workflow
# for C in grid:
#     for fold in folds:
#         fit a separate model
```

This is easy to use, but it can require many repeated kernel solves.

## TorchKM workflow

In TorchKM, users pass a sequence of candidate regularization values to the estimator:

```python
Cs = np.logspace(2, -2, num=4)

clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=5,
    device=device,
)
clf.fit(Xtr, ytr)
```

After fitting, the selected value is available as:

```python
clf.best_C_
```

The cross-validation scores are available as:

```python
clf.cv_mis_
```

## Parameters

| Parameter | Meaning |
|---|---|
| `Cs` | Candidate regularization values under the scikit-learn/LIBSVM convention |
| `nC` | Number of candidate regularization values |
| `cv` | Number of cross-validation folds |
| `foldid` | Optional user-specified fold assignments |
| `random_state` | Random seed for deterministic fold construction |
| `device` | `"cpu"`, `"cuda"`, or `None` for automatic selection |

## Notes

- Larger `nC` gives a finer regularization grid but increases computation.
- Larger `cv` can give a more stable estimate of predictive performance but also increases work.
- The selected parameter depends on the fold assignment and the candidate grid.
- For small examples and tests, use a short grid and a small number of folds.
