# TorchKM benchmarks

Scripts that reproduce every table and figure in the paper and its revision,
plus the protocol they share. Runs at paper scale take hours on a GPU; every
script also has `--smoke`, which finishes in seconds on a CPU and checks that
the pipeline runs end to end.

The user-facing protocol, commands and data sources are documented on the
[reproduction page](../docs/examples/reproduce_paper_benchmarks.md); the
contributor notes (adding a library or a dataset, the JSON schema) are in
[`docs/developer/benchmarking.md`](../docs/developer/benchmarking.md).

## Scripts

| Script | Produces |
| --- | --- |
| `bench_memory_envelope.py` | time and peak memory versus n in exact mode until the first OOM, Nyström beyond it, and the empirical memory constant |
| `bench_gpu_libraries.py` | TorchKM vs scikit-learn, ThunderSVM, cuML, Falkon and linear baselines on the scaling, exact, imbalanced and scale suites |
| `bench_covtype_rank.py` | accuracy versus landmarks x rank on covtype (optionally Falkon at the same centres) |
| `bench_kqr.py` + `r/bench_kqr.R` | kernel quantile regression vs fastkqr, kernlab::kqr and linear QR |
| `bench_dwd.py` + `r/bench_dwd.R` | kernel DWD vs kerndwd, SVM as reference |
| `bench_solver_quality.py` | SVM objective at fixed lambda, solver by solver, same kernel |
| `table2_simulation.py`, `table3_benchmarks.py`, `table4_nystrom.py` | the submitted paper's Tables 2-4 with their original protocol |
| `make_tables.py` | Markdown / LaTeX tables from JSON results and R CSV rows |

Shared code: `_common.py` (grids, folds, timing, `PeakMemory`, metrics,
`ResultWriter`, dataset registry) and `_libraries.py` (one runner per
library, lazy imports).

## Quick check

```bash
for s in bench_memory_envelope bench_gpu_libraries bench_covtype_rank bench_kqr bench_dwd bench_solver_quality; do
    python benchmarks/$s.py --smoke
done
```

## Q1 in one script: every method on the full kernel

`q1_full_kernel.py` answers the first reviewer question on its own: TorchKM
(`is_exact=0`), cuML `SVC`, Falkon with M = n, kernel ridge regression on a
KeOps `LazyTensor`, and EigenPro 2, all without any Nyström approximation, on
the same 5 folds, the same 50-value grid, in float64, three seeds per
dataset. It reports test accuracy, wall-clock time of the whole tuning run
and peak GPU memory, as JSON plus a Markdown table written next to it.

```bash
pip install pykeops
pip install --no-build-isolation git+https://github.com/EigenPro/EigenPro-pytorch.git
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
pip install falkon        # wheels: https://falkonml.github.io/falkon/install.html
python benchmarks/q1_full_kernel.py --data-dir ~/libsvm_data --out revision_results/q1.json
```

A method whose import fails is reported as unavailable and the rest still
run; re-running with the same `--out` resumes; `--smoke` checks the script on
a CPU in about a minute. The docstring at the top of the script states the
protocol and what each method solves.

## Paper-scale runs

See the reproduction page for the full command list. In short: LIBSVM files in
`--data-dir`, `--device cuda`, `--repeats 10` for the SVM suites and `5` for
KQR/DWD, `--time-cap` for libraries without integrated tuning, `--out` to a
file under `benchmarks/results/`, and `--thundersvm-path` when ThunderSVM is
built from source. Libraries that are not importable are reported once and
skipped.

## What every record contains

Each JSON file holds an environment snapshot (GPU, driver, CUDA, PyTorch,
library versions, TorchKM version and commit) and one record per dataset x
library x repeat with: dataset, n_train, n_test, p, class prior, library,
mode (exact / nystrom / linear), parameters (landmarks, rank, centres),
status (ok / capped / failed / exceeds_envelope), end-to-end time, peak memory
(PyTorch allocator and NVML process peak), the selected C, and the test
metrics (accuracy, balanced accuracy, AUC; pinball loss and coverage for KQR).

## Timing caveats

Wall-clock times vary across hardware, software versions, CUDA settings and
system load. Do not compare a cold first run with a warmed run: the scripts
run a warmup fit and synchronise CUDA around every timed region. Report the
number of repeats and the standard errors, which the scripts compute.

## Legacy protocol (Tables 2-4 of the submitted paper)

`table2_simulation.py`, `table3_benchmarks.py` and `table4_nystrom.py` follow
the source notebooks per method: 10-fold cross-validation over a 50-point
lambda grid transferred to LIBSVM `C` via `C = 1/(2*n*lambda)`; grids per
table (Table 2 `lambda in [1e-3, 1e3]`, Table 3 `[1e-5, 1e-1]`, Table 4
`[1e-3, 1e3]` except a9a `[1e-7, 1e-1]`); Table 3 times the solver only; Table
4's covtype cell uses rank `k=30`. Keep them for reproducing the submitted
numbers; use the `bench_*.py` scripts for the revision.

```bash
python benchmarks/table2_simulation.py --repeats 50 --device cuda --thundersvm-path /path/to/thundersvm/python
python benchmarks/table3_benchmarks.py --data-dir DATA_DIR --repeats 10 --device cuda --thundersvm-path /path/to/thundersvm/python
python benchmarks/table4_nystrom.py --data-dir DATA_DIR --datasets a9a w8a ijcnn1 --repeats 10 --device cuda
```
