# Contributing to TorchKM

Thanks for your interest in TorchKM. Contributions are welcome, whether you are reporting a bug, improving documentation, adding tests, extending a solver, or sharing a benchmark result from your own hardware.

TorchKM is a PyTorch library for GPU-accelerated kernel machines with unified training and tuning. What we care about most is accuracy and practical performance. TorchKM supports both CPU execution and CUDA acceleration.

## Getting Started

Fork the repository on GitHub, then clone your fork and install in editable mode:

```bash
git clone https://github.com/<your-username>/torchkm.git
cd torchkm
python -m pip install -e ".[dev,examples,viz]"
```

The standard install includes scikit-learn because the high-level estimator API
is a core part of TorchKM. The development extras add testing, formatting,
packaging, example, and visualization dependencies.

Run the test suite:

```bash
pytest -q
```

Please target pull requests against the `main` branch.

## Reporting Bugs

Please include:

- the TorchKM version or commit hash
- your Python, PyTorch, operating system, and CUDA versions
- whether the issue happens on CPU, CUDA, or both
- a small reproducible example, preferably with synthetic data
- the full traceback or error message
- what you expected to happen and what actually happened

For numerical or performance issues, include the shape of the data, the kernel, the estimator or solver used, and any important parameters such as `nC`, `cv`, `low_rank`, `num_landmarks`, `nys_k`, `tol`, and `max_iter`.

## Tests

Tests live in `tests/` and are run with `pytest`. Please add or update tests when you change behavior.

Useful test targets include:

- solver output shapes and selected regularization values
- prediction behavior for the estimator wrappers (`TorchKMSVC`, `TorchKMDWD`, `TorchKMLogit`, `TorchKMKQR`)
- low-level quantile regression behavior through `cvkqr`
- CPU fallback behavior
- low-rank and exact solver paths
- edge cases for labels, folds, kernels, and input types

Keep tests reasonably small so they can run on CPU in a normal development environment. Use deterministic random seeds in examples and tests. CUDA-specific tests are welcome, but they should skip cleanly when CUDA is unavailable.

## Documentation

Documentation improvements are very helpful. Good contributions include:

- clearer README examples
- short examples for estimator parameters
- notes about device behavior or CPU fallback
- explanations of solver options
- benchmark instructions and result interpretation

When adding examples, prefer compact snippets that users can copy and run. Specify whether CUDA, a large dataset, or extra dependencies is required.

## Benchmarks

Benchmark guidance lives in `docs/developer/benchmarking.md`. Benchmark-related
pull requests should report:

- exact command used
- CPU model
- GPU model, if used
- operating system
- Python version
- PyTorch version
- CUDA version, if used
- TorchKM version or commit hash
- data set
- sample size and feature dimension
- number of cross-validation folds
- regularization grid
- whether timing includes preprocessing, training, and model selection

Do not compare a cold first run with a warmed run. For GPU timing, run at least one warmup iteration. When timing CUDA work, call `torch.cuda.synchronize()` before starting and after ending the timed region. Prefer repeated runs and report median/IQR or mean/standard deviation. Benchmark scripts should optionally write JSON or CSV output with both results and environment metadata.

## Adding Kernels

Kernel changes may touch both user-facing APIs and low-level computation paths. Before opening a pull request, check whether the new kernel should be available through:

- low-level helper functions in `torchkm/functions.py` or `torchkm/kernels.py`
- estimator options in `torchkm/estimators.py`
- tests for exact and low-rank paths, if applicable
- README examples or API notes

Please include tests that cover basic correctness and parameter handling. If the kernel has important numerical constraints, document them in the code or README.

## Extending Estimators and Solvers

Estimator and solver changes should preserve the existing public behavior unless the pull request clearly explains a breaking change.

- keep the scikit-learn-style API consistent across `TorchKMSVC`, `TorchKMDWD`, `TorchKMLogit`, and `TorchKMKQR`
- validate inputs with clear error messages
- keep CPU behavior working when CUDA is not available
- add tests for `fit`, `predict`, and `decision_function` when relevant
- add `predict_proba` tests when probability calibration is affected
- test both small and moderately sized synthetic examples
- check exact and low-rank paths when the change affects both
- be careful with device transfers inside iterative loops
- avoid unnecessary CPU/GPU synchronization in performance-sensitive code
- document any new tolerance, stopping, or regularization behavior

If a change improves speed, include a benchmark or a small timing comparison. If it changes numerical results, explain why the new behavior is expected.

## Pull Request Checklist

Before opening a pull request, please check:

- `pytest -q` passes locally, or any skipped checks are explained
- new behavior has tests
- public API changes are documented
- examples still run
- benchmark changes include commands and environment details
- the pull request description explains the motivation and tradeoffs

A good pull request description does not need to be long. A clear summary, a short test note, and any remaining caveats are enough.

## Code Style

Use clear names and keep changes focused. Avoid unrelated formatting-only changes, especially in performance-sensitive solver code.

Before opening a pull request, run:

```bash
python -m black --check .
python -m flake8 .
```

You may run this locally to format code:

```bash
python -m black .
```

Pull requests should avoid unrelated large formatting changes.
