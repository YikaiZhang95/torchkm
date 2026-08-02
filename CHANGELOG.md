# Changelog

All notable changes to TorchKM are documented in this file.

## [Unreleased]

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
