# Estimators API

This page documents the high-level scikit-learn-style estimators in TorchKM.

The classification estimators provide a familiar interface:

- `fit(X, y, *, low_rank=None, num_landmarks=None, nys_k=None)`
- `predict(X)`
- `decision_function(X)`
- `predict_proba(X)` when `probability=True`
- `platt_plot(X, y)` when probability calibration is enabled

`TorchKMKQR` provides `fit(X, y, *, low_rank=None, num_landmarks=None, nys_k=None)`
and `predict(X)` for continuous targets.
Low-rank options are normally configured in the estimator constructor, but
`fit` also accepts keyword-only convenience arguments `low_rank`,
`num_landmarks`, and `nys_k`.

The classification wrappers accept NumPy arrays and torch tensors, map arbitrary
binary labels to the low-level `{-1, +1}` convention internally, choose
`best_C_` through cross-validation, and return predictions in the original label
space.

## TorchKMSVC

::: torchkm.estimators.TorchKMSVC

## TorchKMDWD

::: torchkm.estimators.TorchKMDWD

## TorchKMLogit

::: torchkm.estimators.TorchKMLogit

## TorchKMKQR

::: torchkm.estimators.TorchKMKQR
