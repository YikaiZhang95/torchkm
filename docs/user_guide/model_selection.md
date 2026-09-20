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

### The stopping rule and `KKTeps`

After each smoothing stage the solver checks the KKT conditions of the
original hinge-loss problem and stops when
`sum(KKT**2) / max(lambda, 1)**2 < KKTeps`. Each entry of the KKT residual
has natural scale `1/n`, so the squared norm shrinks like `1/n` as the sample
grows and the default `KKTeps=1e-3` becomes easy to satisfy: at `n` in the
thousands and weak regularization (small `lambda`, large `C`) the solver can
stop after a few passes with an objective noticeably above the optimum.
`benchmarks/bench_solver_quality.py` measures this against libsvm's solution
at a fixed `lambda`. Pass a tighter tolerance when the exact optimum matters:

```python
clf = TorchKMSVC(kernel="rbf", Cs=Cs, cv=5, device=device, KKTeps=1e-6)
```

The cost is a few more solver passes per regularization value, each an
`O(n^2)` matrix-vector product; the eigendecomposition is not repeated.

The inner step tolerance `tol` (the solver's `eps`) matters as much at weak
regularization: the smoothed problems are ill-conditioned when `lambda` is
small (large `C`), the proximal-gradient steps become short before the
optimum is reached, and the step criterion stops the loop. On a 3,000-sample
problem `tol=1e-8` with `KKTeps=1e-6` recovered the libsvm optimum to better
than 1% down to `lambda = 1e-4` at about three times the solver passes; at
`lambda = 1e-5` no setting tried got closer than 6%. Test accuracy of the
cross-validated model was the same under every setting. If the solution
itself matters (objective values, dual coefficients) rather than the
predictions, use `tol=1e-8, KKTeps=1e-6` and keep the grid's weak end at or
above `lambda = 1e-4`, that is `C_max` of about `1 / (2 n 1e-4)`.

`kkt_scaled=True` switches to a scale-aware rule that compares
`n * sum(KKT**2)` with `KKTeps`, so a given tolerance means the same relative
accuracy at every `n`: with the default `KKTeps=1e-3` each residual entry is
within about 3% of its natural unit `1/n`. It is available on the SVM and
quantile-regression solvers (`cvksvm`, `cvkqr`, `cvknyqr`) and the
corresponding estimators; `benchmarks/bench_solver_quality.py --kkt-scaled`
measures it against the absolute rule.

## Notes

- Larger `nC` gives a finer regularization grid but increases computation.
- Larger `cv` can give a more stable estimate of predictive performance but also increases work.
- The selected parameter depends on the fold assignment and the candidate grid.
- For small examples and tests, use a short grid and a small number of folds.
