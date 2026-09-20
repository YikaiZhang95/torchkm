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

## What "exact" means, and the `is_exact` flag

Two different things are exact in TorchKM, and the `is_exact` argument
controls neither of them:

- **Exact cross-validation.** Every fold's solution is obtained from the same
  kernel matrix and the same eigendecomposition through a modified response
  vector (Wang and Zou, 2022), not by refitting on the reduced data. The fold
  solutions are the ones a refit would give, up to solver tolerance. This holds
  for every setting of `is_exact`.
- **Exact SVM solutions.** The hinge loss is replaced by a sequence of smoothed
  losses whose minimisers converge to the SVM solution (finite smoothing);
  the solver stops once the KKT conditions of the original problem hold to
  `tol`. Again independent of `is_exact`.

`is_exact=1` adds a final projection step in `cvksvm` and `cvkqr` that lands
the solution on the elbow set exactly, and switches the cross-validation loop
from the batched per-lambda implementation to a per-fold loop. It is slower
and rarely changes predictions; the default `is_exact=0` is what the paper's
benchmarks and the benchmark scripts use.

## Notes

- Larger `nC` gives a finer regularization grid but increases computation.
- Larger `cv` can give a more stable estimate of predictive performance but also increases work.
- The selected parameter depends on the fold assignment and the candidate grid.
- For small examples and tests, use a short grid and a small number of folds.
