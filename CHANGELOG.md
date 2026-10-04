# Changelog

All notable changes to TorchKM are documented in this file.

## [Unreleased]

### Removed
- The Nyström approximation: the solvers `cvknyssvm`, `cvknyslogit`,
  `cvknysdwd` and `cvknyqr`; the `low_rank`, `num_landmarks` and `nys_k`
  parameters of `TorchKMDWD`, `TorchKMLogit` and `TorchKMKQR` (exact mode
  only now); `num_landmarks` and `nys_k` of `TorchKMSVC`; the `low_rank`,
  `num_landmarks` and `nys_k` keywords of `fit` (now `fit(X, y)`); the
  Nyström benchmarks (`table4_nystrom.py`, `q3_covtype.py`,
  `bench_covtype_rank.py`), the example and the Nyström pages of the docs.

### Changed
- `TorchKMSVC(low_rank=True)` is the matrix-free large-n SVM, no longer a
  Nyström approximation: the exact RBF kernel model, fitted by the
  truncated-spectrum solver (as `spectrum="truncated"`), with the kernel never
  stored. Every product is recomputed from the training rows
  (`RBFKernelOperator`), fused on CUDA with `dtype="float32"`, so memory grows
  like n times the columns of a block instead of n^2; predictions use fused
  cross products. `spectrum_rank`, `gap_tol` and `spectrum_block` apply,
  `max_iter` is the iteration budget of each lambda, and `kernel="rbf"` is
  required. With the same bandwidth it gives the stored-kernel truncated fit
  (bitwise in float64 on the CPU). The whole covtype.binary (464,809 rows,
  50 lambdas x 10 folds, gamma 32, `dtype="float32"`, `max_iter=40`): 62.3
  min, 6.6 GB, test accuracy 0.9609 (cuML's SVC: 243.0 min, 4.5 GB, 0.9620);
  on a 60,000-row subsample the estimator reproduces the direct
  `SpectralSVMPath` call exactly.
- `TorchKMSVC` reports `duality_gaps_` and `fold_duality_gaps_` for
  `spectrum="truncated"` and `low_rank=True`.
- The truncated-spectrum solver is part of the package API:
  `torchkm.cvksvm.SpectralSVMPath` (with `hinge_duality_gap`) and
  `torchkm.functions.RBFKernelOperator`, copies of the experimental code. The
  estimators use these and no longer import `torchkm.experimental`, which
  keeps its own copy for exploring changes.

### Fixed
- `sigest` failed in `torch.quantile` when every sampled pair of rows was the
  same row (likely when `frac * n` is a handful of rows); it then uses every
  pair of distinct rows, and raises a `ValueError` if all rows are equal.
- Stopping of the exact kernel quantile regression solver (`cvkqr`,
  `TorchKMKQR`, `is_exact=0`). Each bandwidth's solve was accepted by a KKT
  test of the unsmoothed check loss that gave every row the subgradient of
  its residual's sign; the rows the fit interpolates keep a nonzero residual
  there at the optimum, so the test was never met. Every lambda and fold ran
  all `delta_len` bandwidths and printed "Exceeded maximum delta iterations"
  (once or twice per lambda in the Q2 run). The folds also stopped on a
  hard-coded step tolerance (1e-5) instead of `eps`. Now each lambda and fold
  stops once its certified relative duality gap is at most `gap_tol`
  (default 1e-3; the dual of the check loss, from the iterate's smoothed
  derivative, `lam * a` and its residual signs), the folds use `eps`, and one
  `ConvergenceWarning` names the fits that end above `gap_tol`; `gaps`,
  `fold_gaps`, `converged` and `fold_converged` record them. `max_tighten`
  (default 0) re-solves with a 100 times smaller `eps` to reach `gap_tol`.
  On cpusmall (n = 1,000, tau 0.1, 5 lambdas) the fits, objectives and time
  (19.6 s) are those of the previous version; the previous fits were within
  0.1% of the optimum except at lambda = 1e-7 (0.7%). With `max_tighten=6`
  every fit certifies, at about 20 times the time and with the same
  held-out quantiles to three decimals. The coverage gap in Q2 (0.156 for
  tau = 0.1 on cpusmall) is the estimator's, not the solver's: test coverage
  moves from 0.09 to 0.46 along the lambda grid identically for loose and
  certified fits.
- Intercept of the exact kernel DWD (`cvkdwd`, `TorchKMDWD`), logistic
  (`cvklogit`, `TorchKMLogit`) and squared-hinge (`cvksqsvm`) solvers. Each
  refines the intercept with a golden-section search over its objective
  helper, and the helper did not compute the solver's loss: DWD and logistic
  used `1 - y f` as the margin, and the squared hinge charged the points beyond
  the margin instead of those inside it. The coefficients were optimal but the
  intercept was not; on a 200-point RBF problem the true objective was 0.02% to
  0.5% above the optimum, all of it recovered by refitting the intercept. Fits
  now reach the optimum found by an independent L-BFGS solve to seven digits.
  The helpers of the hinge SVM, KQR, Huber and all Nystrom solvers were
  correct.
- Cross-validation of the exact kernel quantile regression solver (`cvkqr`,
  `TorchKMKQR`, and the Nystrom `cvknyqr`, which runs on it). The fold fits
  started at smoothing bandwidth 1 while reusing the step factors the path had
  built for bandwidth 0.125, so their steps were up to eight times too long:
  the fits diverged to NaN at the first lambda and spent the whole
  `nlam * maxit` iteration budget there, leaving at most one pass per fold for
  every later lambda. `TorchKMKQR` skipped the NaN, and in our checks it then
  chose the second-largest lambda. Separately, the fold fits' intercept search
  (and, with `is_exact=1`, every loss term) counted the held-out rows as
  responses of 0; zeroing `y` removes a row from a margin loss but not from the
  check loss.
  The fold fits now follow the path's bandwidth schedule and leave the held-out
  rows out. On test problems the cross-validation loss is within a few percent
  of exact fold solutions near its minimum and selects the same lambda
  (`tests/test_kqr_cv.py`); a 3-fold, 5-lambda fit on 400 points with
  `max_iter=100000` takes 7 s instead of 103 s.

### Added
- `RBFKernelOperator(fused=True)` (CUDA, float32): each product K B runs in a
  fused kernel, 128 columns at a time, and never writes a block of K to
  memory. With a_i = [4 sigma x_i, -2 sigma |x_i|^2, 1] and
  b_j = [x_j, 1, -2 sigma |x_j|^2], K B = exp(A B') B is attention without the
  softmax normalization, which PyTorch's memory-efficient attention computes
  in float32 together with each row's log-sum-exp. On whole covtype.binary a
  product takes about 2 s instead of 13 s (relative error 4e-5 against
  float64, 2e-5 for the blocks). `RBFKernelOperator.cross(Xq, B)` gives
  K(Xq, X) B for predictions, fused or in blocks.
- `benchmarks/covtype_full_cuml.py`, `covtype_full_trunc.py` and
  `covtype_full_cv.py`: cuML's SVC and the truncated-spectrum TorchKM on the
  whole covtype.binary training set, single fits and 10-fold CV over 50
  lambdas.
- Exact mode's size limit on the GPU. The exact solvers eigendecompose the
  kernel with cuSOLVER on CUDA, and cuSOLVER refuses n above 32,768 whatever
  the memory: in float32, 32,768 is accepted and 32,769 refused; float64
  accepts 32,768 and refuses 33,000 (PyTorch 2.6, CUDA 12.4, measured on an
  L40S). `torchkm.memory.EXACT_MODE_MAX_N_CUDA` records the limit. Above it
  the exact solvers raise a `torch.linalg.LinAlgError` that names the limit and
  the alternatives, instead of cuSOLVER's workspace-query error.
  `benchmarks/probe_eigh_size.py` finds the limit of another build in seconds.
- Experimental: `torchkm.experimental.SpectralSVMPath`, an exact-kernel SVM
  path with K-fold CV that stops every lambda and every fold at a certified
  duality gap (`hinge_duality_gap`), and that can replace the full
  eigendecomposition by a truncated spectrum. See
  `EIGENDECOMPOSITION_OPTIONS.md`.
  - The truncated spectrum keeps the top-r Ritz pairs from subspace iteration
    and a Lanczos bound on the rest. The eigendecomposition's O(n^3) work and
    its workspace are gone, and each iteration reads the kernel once instead
    of reading the kernel and the eigenvectors three times in all.
  - Steps are checked and fall back to a scalar majorizer when needed.
  - Iterations use FISTA momentum with restart.
  - `benchmarks/matched_accuracy.py` compares the two spectra at equal
    certified gaps on the GPU, on Table 2's simulation or a Q1 data set.
  - `TorchKMSVC(spectrum="truncated", spectrum_rank=400, gap_tol=1e-3)` runs
    it in place of `cvksvm`: the exact kernel, a peak of about 1.2 n x n
    matrices instead of 5, and no eigensolver size limit.
    `benchmarks/q1_full_kernel.py` reports it as the `torchkm_trunc` row.
  - `torchkm.experimental.RBFKernelOperator` is a matrix-free RBF kernel:
    `K @ B` recomputes K in blocks of rows and K is never stored, so with the
    truncated spectrum memory grows with n times the rank instead of n^2.
    `matched_accuracy.py --solvers matrix_free` measures it.
  - Not part of the stable API.
- `dtype` on the exact SVM solver (`cvksvm(dtype=torch.float32)`,
  `TorchKMSVC(dtype="float32")`): the kernel, its eigendecomposition and the
  solution path in single precision, half the memory of every `n x n` matrix.
  Three changes make the solver work in float32. (1) The intercept's step
  factors `n - sum(vvec)` and `rds - vvec . gamvec` subtracted numbers of size
  `n` that agree to within `4 n delta lambda`; they are now computed in an
  equivalent form without the difference. (2) The eigenvalues are clamped at
  zero and raised by twice the eigendecomposition's rounding error
  (`|U E U^T - K|`, estimated by power iteration: about 1e-13 `|K|` in float64,
  1e-6 `|K|` in float32). Without it the float32 curvature bound fell up to 9%
  below `K` at small `4 n delta lambda` and the accelerated steps oscillated
  until the iteration budget ran out. (3) In float32 a smoothing round also
  ends when its largest step has not reached a new low for 100 iterations: at
  weak regularization the steps stop shrinking at a level set by the rounding
  of `K alpha`, above `eps`. In float64 (1) and (2) change results only at
  rounding level and (3) is off. On simulated problems with 2,000 to 5,000
  rows (p from 10 to 1,000, and one-hot data with duplicate rows) and lambda
  down to 2e-5, float32 selected the same lambda as float64, test accuracy
  agreed within 0.005 at every lambda, and the objectives agreed to a median
  1e-5. At the smallest lambdas of a near-separable problem they differed by up
  to 5%, as much as float64 itself moves between `KKTeps=1e-6` and `1e-9`.
  `is_exact=1` needs float64; the other solvers and estimators stay in
  float64.
- Fitted attributes `fit_timing_` and `n_passes_` on the binary classifiers:
  seconds per phase of the last fit (kernel build, and for the exact SVM
  solver the eigendecomposition, its error check, the lambda path and the
  cross-validation fits; CUDA synchronised at each boundary) and the solver
  iterations of the path and of the cross-validation fits. `fit_profile_`
  has the same per lambda (path and fold-fit seconds, path iterations) and
  the iterations of every fold at every lambda. `cvksvm` records them in
  `timing`, `lambda_timing` and `fold_passes`.
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
- `kkt_scaled=True` on the SVM and quantile-regression solvers and estimators:
  a scale-aware KKT stopping rule (`n * sum(KKT**2) < KKTeps`).
- User guide pages on the exact-mode operating envelope and on multiclass
  classification through scikit-learn's one-vs-rest / one-vs-one wrappers
  (with tests); `is_exact`, `KKTeps` and `tol` documented on the model
  selection page.
- `benchmarks/run_campaign.sh` (the full revision campaign with its exact
  settings) and `benchmarks/make_figures.py` (the paper figures from the
  archived JSON results).

### Changed
- `spectrum="truncated"` is 2 to 18 times faster than cuML's SVC on the Q1
  tuning job (50 lambdas x 10 folds, L40S, float32); before, it ranged from
  7 times slower (w7a) to 2.8 times faster. Two changes to `SpectralSVMPath`:
  - Wide scheduling (`block`; the estimators' `spectrum_block`, default 10):
    the whole-data fit and the 10 folds of `block` consecutive lambdas are one
    set of columns, so each product with K serves up to 110 fits instead of 1
    (path) or 10 (folds). A product with K costs about the same for 1 column
    as for 110, because reading K dominates. Whole-data columns start from the
    previous block's last whole-data fit, each fold from the same fold's last
    fit (the `WideScheduler` of the large-n design).
  - Certification starts at a coarser smoothing level (`bias`, default 4,
    was 0.5 hard-coded). The old rule waited until the worst-case smoothing
    bias was half the gap tolerance, which at small lambda meant delta = 8^-4
    and a shift c = 4 n delta lambda about 10^4 times below the truncated
    tail, so the steps were tiny. The duality gap is exact, so starting
    earlier only helps; delta still shrinks when the gap stalls.
  Every lambda and fold still stops at a certified relative gap of 1e-3; at
  cuML's selected lambda the objective is within 0.06% of cuML's (lower on
  w7a and MNIST 3v8). w7a: 234 s to 18 s (cuML 34 s); a9a: 62 s to 8.4 s
  (cuML 45 s); n = 20,000 simulations: 12 s to 2.5 s (cuML 29 to 59 s). The
  wider blocks hold about n x 110 more float32 state, so peak memory grew,
  e.g. a9a 5.9 to 7.2 GB (measured before the memory changes below).
- `SpectralSVMPath` needs far less GPU memory, with the same arithmetic.
  Whole covtype.binary (464,809 rows, gamma 32, 50 lambdas x 10 folds, block
  10, rank 400, matrix-free fused kernel): 32.9 GB to 6.6 GB, 62.3 min both
  times, the same selected lambda, certified lambdas (28/50) and test
  accuracy (0.9609); cuML's SVC on the same job: 243.0 min, 4.5 GB, 0.9620.
  - The certificate scores its candidate dual points one at a time instead
    of stacking them (n x 5m float64), and keeps labels, masks, alpha and
    K alpha in their own dtypes.
  - The steps store the coupling vectors s, K s, v once per distinct lambda
    (block of them, not block x (folds + 1)), use views while every column is
    live, select with `torch.where` instead of blending, and free
    temporaries early. The float64 objective sums and the spectrum's residual
    norm run in row blocks.
  - Row tiling (`tile`, on by default when n x block x (folds + 1) exceeds
    2^24): the label matrix is never stored (`_Labels` builds any rows from y
    and the fold ids), and each step and certificate sweeps row tiles of
    about 2^21 elements, rebuilding the elementwise quantities, so only
    alpha, K alpha, their previous values and the buffers of each product
    with K are n x m. One step and one certificate match the untiled ones to
    rounding (tests); whole fits match in objective, gaps, decisions and CV
    error. Problems below the threshold (all of Q1) take the untiled path.
- `fit_cap` bounds all iterations of a lambda (or a wide block), loose
  smoothing rounds included; before, it was checked only after certifying
  rounds, so a fit that never reached them could run `round_cap` iterations
  per smoothing level. A fit that reaches the cap is certified once more and
  reported as not converged.
- `max_exact_n` stops at `EXACT_MODE_MAX_N_CUDA` by default; `size_limit=None`
  counts memory only.
- Exact mode factorizes the kernel in place. `torch.linalg.eigh` worked on a
  copy of the kernel; the estimators now let it overwrite the kernel's own
  storage with the eigenvectors and rebuild the kernel afterwards (one kernel
  evaluation), which takes one n x n matrix off the peak at the same speed.
  Fits are bitwise identical. The six exact solvers take `rebuild_kmat` for
  this; a precomputed kernel is the caller's array and is still factorized as
  a copy.
- `EXACT_MODE_COPIES` is 5, from the GPU, instead of the CPU sweep's 4. On an
  L40S cuSOLVER's eigendecomposition takes a 4.01-copy workspace, so a fit
  peaks at 5.01 copies of the n x n matrix with the kernel factorized in place
  (6.01, measured, with the copy). The 'Operating envelope' table is
  recomputed: a 48 GB card now stops at 32,768, the size limit, in float64
  and float32 alike.
- Fewer waits for the GPU. The solvers' intercept search (Brent's method, one
  copy per solver) kept its state in device tensors, so each of its steps
  launched several tiny kernels and made the CPU wait for the GPU several
  times; the nine copies are now one function, `functions.brent_minimize`,
  whose bookkeeping is in Python floats and which reads the objective once
  per evaluation. The solvers count their iterations on the host instead of
  summing a device counter on every iteration (`npass` and `cvnpass` are
  still int32 tensors after `fit`), and the unused row sums of the kernel
  (`Ksum`) are gone. On a 3,000-row SVM fit (50 lambdas, 10 folds) the
  device-to-host reads dropped from 159,850 to 17,811 with the same
  iterations. Results of the float64 SVM and of kernel quantile regression
  are unchanged; the DWD, logistic, squared-hinge, Huber and Nystrom solvers
  searched partly in float32 before (their constants were float32 tensors)
  and now search in float64, which moves their solutions within the solver
  tolerance (objectives within 1e-5 in our checks).
- Cross-validation of the exact SVM solver (`cvksvm`) finishes each smoothing
  round for all unfinished folds at once instead of fold by fold: one product
  with the kernel for all folds instead of one per fold, the intercept
  searches run in step with one objective evaluation per step for all folds
  (`functions.brent_minimize_batch`; each search takes the steps it would take
  alone), and one KKT test for all folds. On a 3,000-row fit (50 lambdas, 10
  folds, float32, CPU) the cross-validation took 1.76 s instead of 2.65 s,
  with 500 fewer full reads of the kernel and 4,089 device-to-host reads
  instead of 17,811; the iterations and the selected lambda are unchanged.
  The fold predictions can move where a fold's intercept objective is flat
  (the hinge loss often is, over an interval) and rounding picks a different,
  equally good point of it.
- Predictions in exact mode build the test kernel on the fit's device in
  blocks of test rows instead of on the CPU, and copy only the scores back.
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
