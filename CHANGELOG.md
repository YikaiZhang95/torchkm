# Changelog

All notable changes to TorchKM are documented in this file.

## [Unreleased]

### Added
- `torchkm.memory`: `exact_mode_memory_estimate`, `max_exact_n`, and the
  out-of-memory message the estimators raise in exact mode (predicted
  requirement, device total, largest feasible `n`, and the `low_rank=True`
  alternative). Both helpers are exported from `torchkm`.
- Fitted attribute `peak_gpu_memory_bytes_` on every estimator: the PyTorch
  allocator peak over the whole `fit` (`None` on CPU).
- `random_state` now also seeds Nyström landmark sampling in `TorchKMSVC`,
  `TorchKMDWD` and `TorchKMLogit`; the low-level `cvknyssvm`, `cvknysdwd` and
  `cvknyslogit` take `random_state` and `sigma`. `sigest` takes an optional
  `generator`.
- `rbf_sigma` is honoured on the Nyström path of the binary classifiers.
- `KKTeps` (and `delta_len` for the SVM) are exposed on `TorchKMSVC`,
  `TorchKMDWD` and `TorchKMLogit`. The solvers' KKT stopping rule compares an
  absolute squared residual norm, whose natural scale shrinks like `1/n`, so
  the default `KKTeps=1e-3` can stop the SVM solver a few passes in at
  `n` in the thousands and weak regularization (objective up to 33% above the
  libsvm optimum at n=3000, lambda=1e-3 in `bench_solver_quality.py`);
  `KKTeps=1e-6` closes the gap at the same run time. Documented on the model
  selection page; the default is unchanged pending a scale-aware rule.
- Benchmark suite for the JMLR revision under `benchmarks/`: shared protocol
  helpers (peak memory from the PyTorch allocator and NVML, AUC and balanced
  accuracy, JSON results with environment snapshots), `bench_memory_envelope.py`,
  `bench_gpu_libraries.py` (scikit-learn, ThunderSVM, cuML, Falkon, linear
  baselines), `bench_covtype_rank.py`, `bench_kqr.py` and `bench_dwd.py` with
  R baselines, `bench_solver_quality.py`, and `make_tables.py`.
- User guide page on the exact-mode operating envelope; `is_exact` documented.

### Changed
- Exact-mode solvers no longer materialise the `n x n` matrix
  `diag(1/eigenvalues) U^T`; the projection applies it on the fly. Peak
  memory drops by `8 n^2` bytes.
- The training kernel is built on the target device instead of on the host,
  and `rbf_kernel` / `kernelMult` exponentiate in place.
- The Nyström backends no longer call `torch.manual_seed(0)` inside `fit`, so
  repeated fits can vary the landmarks and the global RNG is left untouched.
- `table2_simulation.py` also reports test accuracy, AUC and peak memory and
  writes JSON.

## [4.3.2] - 2026-08-02

### Added
- `cvksvm`, `cvkdwd`, and `cvklogit` now track a per-lambda `converged`
  boolean and emit an aggregated `ConvergenceWarning`
  (`sklearn.exceptions.ConvergenceWarning` when scikit-learn is
  installed, re-exported as `torchkm.ConvergenceWarning`) when the
  solver hits the `maxit` iteration cap without satisfying the
  convergence/KKT tolerance — including a separate warning when the
  cross-validation path hits its `nlam * maxit` cap. Previously
  `TorchKMSVC`/`TorchKMDWD`/`TorchKMLogit.fit()` completed silently on
  non-converged solutions.
- `TorchKMSVC`, `TorchKMDWD`, and `TorchKMLogit` expose the per-lambda
  convergence status as the fitted attribute `converged_`.

## [4.3.1] - 2026-06-08

### Added
- `torchkm.__version__`, resolved from the installed distribution metadata
  via `importlib.metadata` (no second hard-coded version). The CUDA
  snapshot in `scripts/run_cuda_tests.sh` and the self-hosted workflow now
  record `torchkm.__version__` and `torchkm.__file__`, so a run's
  `pip freeze` and checked-out build can always be reconciled.

## [4.3.0] - 2026-06-03

### Added
- `docs/developer/cuda_testing.md` describes how to validate the CUDA
  code paths on a GPU workstation and how to commit the log bundle
  under `benchmarks/cuda-runs/`.
- `.github/workflows/cuda-tests.yml` runs the same suite on a
  self-hosted `[self-hosted, linux, cuda]` runner via
  `workflow_dispatch` or a weekly cron, uploading a five-file artefact
  bundle (coverage XML, JUnit XML, full pytest log, `pytest -m cuda
  -v` log, runner snapshot).
- Behaviour-focused test files for solver validation, edge cases,
  estimator internals, and the Platt-plot contract.

### Changed
- Reorganised the previous `test_coverage_extras.py` into four
  behaviour-named files (`test_solver_validation_extras.py`,
  `test_solver_edge_cases.py`, `test_estimator_internals.py`,
  `test_platt_plot.py`); every test now reads as a behavioural
  assertion rather than a coverage-driven probe.

## [4.2.3] - 2026-05-21

### Added
- Test workflow and documentation workflow on GitHub Actions
  (`tests.yml`, `docs.yml`).
- Coverage gate (`--cov-fail-under=90`) on the Python 3.11 CI leg.

### Changed
- `nfolds` keyword renamed to `cv` across the scikit-learn-style
  estimators for compatibility with the standard sklearn convention.

## [4.2.2] - 2026-05-21

### Changed
- Internal cleanup; no public API changes.

## [4.2.1] - 2026-05-21

### Added
- `requires-python = ">=3.10"` and pinned minimum versions for
  `numpy`, `torch`, and `scikit-learn` in `setup.cfg`.
- Visualization extra (`torchkm[viz]`) for `matplotlib`-based
  calibration plotting.

## [4.1.0] - 2026-04-06

### Added
- Paper-driven performance optimizations on the GPU training paths
  (PR #1 by @jiagaoxiang).
- Benchmark accuracy checks documented.

### Changed
- README and quick-start guide updated to emphasize GPU acceleration
  and the integrated train+tune workflow.

## [4.0.x] - 2026-02-07

### Added
- Initial public release of TorchKM with:
  - Kernel classifiers: `cvksvm`, `cvkdwd`, `cvklogit`.
  - Kernel regressor: `cvkqr` (kernel quantile regression).
  - Nyström variants of each backend: `cvknyssvm`, `cvknysdwd`,
    `cvknyslogit`, `cvknyqr`.
  - scikit-learn-style estimators: `TorchKMSVC`, `TorchKMDWD`,
    `TorchKMLogit`, `TorchKMKQR`.
  - Platt-scaling calibration via `PlattScalerTorch`.
  - CPU + CUDA device selection with automatic fallback.

[4.3.1]: https://github.com/YikaiZhang95/torchkm/compare/v4.3.0...v4.3.1
[4.3.0]: https://github.com/YikaiZhang95/torchkm/compare/v4.2.3...v4.3.0
[4.2.3]: https://github.com/YikaiZhang95/torchkm/compare/v4.2.2...v4.2.3
[4.2.2]: https://github.com/YikaiZhang95/torchkm/compare/v4.2.1...v4.2.2
[4.2.1]: https://github.com/YikaiZhang95/torchkm/compare/v4.1.0...v4.2.1
[4.1.0]: https://github.com/YikaiZhang95/torchkm/compare/v4.0.0...v4.1.0
[4.0.x]: https://github.com/YikaiZhang95/torchkm/releases/tag/v4.0.0
