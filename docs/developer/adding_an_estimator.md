# Adding an Estimator

A new high-level estimator should follow the scikit-learn-style pattern used by the existing TorchKM wrappers.

## Required methods

A new estimator should usually implement:

- `__init__`
- `fit`
- `predict` or another task-specific prediction method
- `decision_function`, when appropriate
- `predict_proba`, when probability estimates are available

## Expected behavior

A high-level estimator should:

- accept NumPy arrays and torch tensors when practical;
- validate input shapes;
- support CPU execution;
- support GPU execution when CUDA is available;
- expose selected tuning parameters after fitting;
- store fitted public attributes with trailing underscores, such as `best_C_`;
- preserve a stable public API;
- use clear error messages.

## Attributes

Attributes created during fitting should end with an underscore, following scikit-learn convention. Examples include:

- `classes_`
- `best_C_`
- `cv_mis_`
- `alpha_`
- `intercept_`
- `n_features_in_`

## Required tests

Add tests for:

- construction;
- fitting on a small synthetic data set;
- prediction shape;
- input-shape validation;
- label-format validation;
- CPU behavior;
- GPU behavior, skipped when CUDA is unavailable;
- reproducibility with fixed random seeds;
- probability prediction, if available;
- low-rank behavior, if supported.

## Required documentation

A new estimator should include:

- a class docstring;
- an API reference entry;
- a user guide page if it introduces a new method;
- at least one example script;
- a short description in the README or documentation index.

## Example docstring pattern

Use a NumPy-style docstring with sections for parameters, attributes, notes, and examples.

## Current estimator layer

The estimator module includes `TorchKMSVC`, `TorchKMDWD`, `TorchKMLogit`, and
`TorchKMKQR`. The binary classifiers share a base implementation that handles
kernel construction, label mapping, cross-validation selection, probability
calibration, and optional Nyström backends. The quantile-regression estimator
follows the same fit/predict and model-selection conventions for continuous
targets.

Do not add documentation for a new high-level estimator until the class exists
and the import path is stable.
