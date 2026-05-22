# Adding a Solver

Low-level solvers implement the numerical routines used by TorchKM estimators.

## What to document

A new solver should document:

- the mathematical objective;
- expected input shapes;
- label or response conventions;
- regularization-parameter format;
- cross-validation fold format;
- device behavior;
- convergence criterion;
- numerical tolerance;
- return values;
- relationship to any high-level estimator.

## Required tests

Add tests for:

- a small synthetic example;
- correct output shape;
- CPU execution;
- GPU execution, skipped when CUDA is unavailable;
- deterministic behavior with fixed random seeds;
- convergence on a simple problem;
- agreement with a simple baseline when appropriate.

## Performance-related changes

If a solver change is intended to improve runtime or memory use, include a small
benchmark or timing note and report:

- CPU model;
- GPU model, if used;
- RAM and GPU memory;
- operating system;
- Python version;
- PyTorch version;
- CUDA version;
- data set size;
- regularization grid;
- number of folds;
- whether preprocessing is included.

Do not put benchmark-scale workloads in the normal test suite or in examples
that users are expected to run casually.

## Documentation

A new solver should have:

- a source-level docstring;
- an API reference entry;
- a developer note explaining how it is called;
- a user-facing example if it is exposed directly.

Existing lower-level solver docstrings should be preserved and exposed through
the API reference with mkdocstrings.
