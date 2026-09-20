# Benchmarking

Benchmarks matter for TorchKM because the package is designed around the full
training-and-tuning pipeline. They are expensive, so they stay out of the test
suite and the documentation build and live under `benchmarks/`. This page is
for contributors adding a library, a dataset or a script; the user-facing
protocol and commands are on the
[reproduction page](../examples/reproduce_paper_benchmarks.md).

## Layout

```text
benchmarks/
  _common.py             protocol helpers: grids, folds, timing, peak memory, metrics, JSON, data
  _libraries.py          one runner per library (TorchKM, scikit-learn, ThunderSVM, cuML, Falkon, linear)
  bench_*.py             the experiments; every script has --smoke and --out
  table2_simulation.py   the submitted paper's Tables 2-4 with their original protocol
  table3_benchmarks.py
  table4_nystrom.py
  make_tables.py         Markdown / LaTeX tables from JSON results and R CSV rows
  r/                     R baselines (fastkqr, kernlab, kerndwd) on exported splits
  environment/           pinned environments for the comparison libraries
  results/               archived paper-scale runs (JSON, CSV, README per run)
  cuda-runs/             CUDA validation bundles of the test suite
```

## Principles

- **Same kernel, same grid, same folds for every library.** `_common.make_folds`
  produces the stratified fold ids; TorchKM takes them through `foldid`, the
  other libraries through `_common.cv_sweep`. Bandwidth conversions are in
  `_common.gamma_from_sigest` and `_common.falkon_sigma_from_sigest`.
- **End-to-end time.** The timed region starts before kernel or feature
  construction and ends after the final refit. Warm CUDA up first
  (`_common.warmup`) and synchronise around the region (`_common.timed`).
- **Two memory numbers.** `_common.PeakMemory` records the PyTorch allocator
  peak and an NVML sample of the process. Report the NVML figure when comparing
  libraries that do not allocate through PyTorch; report the allocator figure
  for TorchKM, where it is also available as `peak_gpu_memory_bytes_`.
- **Metrics that survive imbalance.** `_common.classification_metrics` returns
  accuracy, balanced accuracy and AUC from decision scores; report all three.
- **Every record is self-describing.** `_common.ResultWriter` writes an
  environment snapshot (GPU, driver, CUDA, PyTorch, library versions, TorchKM
  version and commit) plus one record per dataset x library x repeat, and
  rewrites the file after every record.
- **No silent failures.** A library that cannot be imported is reported and
  skipped; a fit that fails is a record with `status: failed` and the error;
  a sweep that hits `--time-cap` is `status: capped` with the number of grid
  values completed.

## Adding a library

Add a runner to `_libraries.py` with the shared signature
`run_x(data, sig, Cs, foldid, args, dev, seed, **options)` and register its
import in `EXTERNAL_IMPORTS`. Use `_sweep_record` when the library has no
integrated model selection: it runs the fold loop, the refit and the scoring
and returns a record in the common shape. Add the key to the `--libraries`
choices in `bench_gpu_libraries.py`.

## Adding a dataset

Add an entry to `_common.DATASETS` (LIBSVM file names, optional class pair for
multiclass files, optional stratified subsample sizes, group) and, if it
belongs to a suite, to `_common.SUITES`. The loader handles `.bz2` and `.xz`
files and single-file datasets (80/20 stratified split).

## R baselines

`bench_kqr.py --export-splits` and `bench_dwd.py --export-splits` write the
exact training and test splits with fold ids as CSV. The R scripts under
`benchmarks/r/` read them, so both sides tune on identical folds, and write
CSV rows that `make_tables.py --r-csv` merges with the JSON records. Record
`sessionInfo()` alongside the CSV when archiving.

## Reporting

Archive a paper-scale run under `benchmarks/results/<UTC-timestamp>/` with
the JSON files, the R CSV rows, and a `README` naming the commit and the
commands. Generate tables with `make_tables.py`; the paper and the
documentation quote those tables rather than hand-copied numbers.
