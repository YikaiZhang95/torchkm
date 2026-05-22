# FAQ

## Does TorchKM require a GPU?

No. The high-level estimators choose CUDA when it is available and otherwise use
CPU. You can also pass `device="cpu"` explicitly for small examples, tests, and
debugging.

## What labels should I use for classification?

The low-level classification solvers use labels in `{-1, +1}`. The high-level
estimators accept any two distinct labels, map them internally, and return
predictions in the original label space.

## How is the tuning parameter selected?

Pass candidate values with `Cs` or let TorchKM create a log-spaced grid from
`C_max`, `C_min`, and `nC`. The estimator uses cross-validation with `cv` folds
and stores the selected value as `best_C_`.

## When should I use `low_rank=True`?

Use the Nyström approximation when the full kernel matrix is too large or too
slow for your workflow. The current high-level low-rank classifier path is for
raw-feature RBF-kernel workflows.

## Which estimators should I start with?

Start with the scikit-learn-style estimators in `torchkm.estimators`:

```python
from torchkm.estimators import (
    TorchKMSVC,
    TorchKMDWD,
    TorchKMLogit,
    TorchKMKQR,
)
```

Use `TorchKMSVC`, `TorchKMDWD`, and `TorchKMLogit` for binary classification,
and `TorchKMKQR` for kernel quantile regression.

## Will benchmark times match the paper exactly?

No. Wall-clock times depend on hardware, PyTorch and CUDA versions, system load,
and benchmark setup. Benchmark documentation should be read as a protocol, not a
promise of identical timings.
