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

`low_rank=True` is `TorchKMSVC`'s large-\(n\) mode, for problems whose kernel
matrix does not fit in memory. It fits the exact RBF kernel SVM (it is not an
approximation) but never stores the kernel matrix, recomputing every product
with it from the training rows. It needs `kernel="rbf"` on raw features. When
the kernel fits in memory, the stored-kernel modes are faster; try
`spectrum="truncated"` first if exact mode is just past its memory limit. See
[Kernel SVM](user_guide/svm.md) and the
[operating envelope](user_guide/operating_envelope.md). `TorchKMDWD`,
`TorchKMLogit` and `TorchKMKQR` run in exact mode only.

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
