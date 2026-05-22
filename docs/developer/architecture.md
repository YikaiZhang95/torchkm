# Architecture

TorchKM has two main layers:

1. high-level scikit-learn-style estimators;
2. low-level numerical solvers.

The high-level estimators are the recommended entry point for most users. The
classification estimators provide familiar methods such as `fit`, `predict`,
`decision_function`, and, when `probability=True`, `predict_proba`. `TorchKMKQR`
provides `fit` and `predict` for quantile regression.

The low-level solvers implement the computational routines used by the estimators. They are useful for advanced users and for developers who want to add new kernel-machine methods.

## Main components

| Component | Purpose |
|---|---|
| `torchkm.estimators` | High-level estimator classes: `TorchKMSVC`, `TorchKMDWD`, `TorchKMLogit`, and `TorchKMKQR` |
| `torchkm.cvksvm` | Low-level kernel SVM solver |
| `torchkm.cvkdwd` | Low-level kernel DWD solver |
| `torchkm.cvklogit` | Low-level kernel logistic regression solver |
| `torchkm.cvkqr` | Low-level kernel quantile regression solver |
| `torchkm.cvknyssvm` | Nyström SVM solver |
| `torchkm.cvknysdwd` | Nyström DWD solver |
| `torchkm.cvknyslogit` | Nyström logistic regression solver |
| `torchkm.cvknysqr` | Nyström quantile regression solver |
| `torchkm.kernels` | Basic kernel functions |
| `torchkm.functions` | Kernel and numerical utility functions |
| `torchkm.platt` | Probability calibration utilities |
| `tests/` | Unit and integration tests |
| `examples/` | User-facing examples |
| `benchmarks/` | Benchmark protocol notes and, when added, benchmark scripts |

## Design principles

TorchKM is designed around a few principles:

- keep the user-facing API close to scikit-learn;
- preserve CPU fallback when CUDA is not available;
- use PyTorch tensors internally for CPU/GPU computation;
- avoid unnecessary CPU-GPU transfers;
- make model selection part of the estimator workflow;
- keep examples small and reproducible;
- document public functions and classes clearly.

## Adding new code

New code should be added with tests, documentation, and examples when it affects
the public API. Benchmark-scale work should stay out of normal tests and docs
builds.
