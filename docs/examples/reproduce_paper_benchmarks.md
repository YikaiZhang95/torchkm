# Reproducing the Benchmarks

The scripts under `benchmarks/` reproduce every table and figure in the paper
and its revision. Runs at paper scale take hours on a GPU and should be
started deliberately on a machine reserved for the purpose; every script also
has a `--smoke` mode that finishes in seconds on a CPU and checks that the
pipeline runs end to end.

Timing results vary with hardware, CUDA and PyTorch versions, data-loading
behaviour and system load. The scripts reproduce the protocol; they do not
promise identical wall-clock times on another machine.

## Hardware and software used in the paper

- GPU: NVIDIA L40S with 48 GB memory
- CPU: AMD EPYC 9334, 32 cores
- System RAM: 768 GB
- Operating system: Ubuntu 22.04
- CUDA: 12.1
- Python: 3.11
- PyTorch: 2.4.1
- scikit-learn: 1.1.3
- NumPy: 1.25.2
- SciPy: 1.9.3
- ThunderSVM: 0.3.4

The revision adds cuML (RAPIDS) and Falkon on the same GPU, and the R packages
`fastkqr`, `kernlab` and `kerndwd` on the same CPU. The pinned environment is
in `benchmarks/environment/`, and every result file records the versions it
was produced with.

## Protocol

One protocol applies to every library and every script:

- **Kernel.** One RBF bandwidth per repeat from `torchkm.sigest` on the
  training features, shared by every library. TorchKM's kernel is
  \(\exp(-2\sigma d^2)\); libsvm-style baselines receive `gamma = 2 sigma`
  and Falkon receives the equivalent Gaussian width.
- **Grid.** 50 log-uniform values of \(C \in [10^{-3}, 10^3]\), converted to
  \(\lambda = 1/(2nC)\) for TorchKM and Falkon.
- **Model selection.** 10-fold cross-validation with identical stratified folds
  for every library. TorchKM tunes inside the solver; the others loop over the
  grid and the folds.
- **Timing.** End to end: kernel or feature construction, the whole
  cross-validation sweep, and the final refit. CUDA is warmed up and
  synchronised around the timed region.
- **Memory.** Peak device memory from the PyTorch allocator (also exposed on
  every fitted estimator as `peak_gpu_memory_bytes_`) and from NVML sampling
  of the process, which is the figure comparable across libraries that do not
  allocate through PyTorch.
- **Metrics.** Test accuracy, balanced accuracy and AUC for classification;
  pinball loss and empirical coverage for quantile regression. Means and
  standard errors over repeats.

## Scripts

| Script | What it produces |
| --- | --- |
| `bench_memory_envelope.py` | Time and peak memory versus \(n\) in exact mode until the first out-of-memory error, then the Nyström path beyond it; the empirical memory constant. The scaling figure. |
| `bench_gpu_libraries.py` | TorchKM against scikit-learn, ThunderSVM, cuML, Falkon and tuned linear baselines on the Adult scaling study, the exact-range problems (ijcnn1, MNIST pairs, covtype subsample, w7a), the imbalanced sets and the large sets. |
| `bench_covtype_rank.py` | Accuracy versus landmarks and rank on covtype: the accounting for the Nyström configuration. |
| `bench_kqr.py` and `r/bench_kqr.R` | Kernel quantile regression at three quantile levels against `fastkqr`, `kernlab::kqr` and a linear quantile regression. |
| `bench_dwd.py` and `r/bench_dwd.R` | Kernel DWD against `kerndwd`, with the SVM as the in-package reference. |
| `bench_solver_quality.py` | The SVM objective at fixed \(\lambda\), solver by solver, on the same kernel: the solver-quality check. |
| `table2_simulation.py`, `table3_benchmarks.py`, `table4_nystrom.py` | The submitted paper's Tables 2 to 4 with their original protocol (Table 2 now also reports accuracy, AUC and peak memory). |
| `make_tables.py` | Markdown or LaTeX tables from the JSON results and the R CSV rows. |

## Running

```bash
# data: LIBSVM files (train and .t test files, .bz2/.xz accepted) in one directory
DATA=~/libsvm

python benchmarks/bench_memory_envelope.py --device cuda --out benchmarks/results/envelope.json

python benchmarks/bench_gpu_libraries.py --data-dir $DATA --suite scaling \
    --libraries torchkm torchkm_nystrom sklearn_svc thundersvm cuml_svc linear \
    --repeats 10 --device cuda --time-cap 14400 --out benchmarks/results/scaling.json
python benchmarks/bench_gpu_libraries.py --data-dir $DATA --suite exact \
    --libraries torchkm sklearn_svc thundersvm cuml_svc linear \
    --repeats 10 --device cuda --time-cap 14400 --out benchmarks/results/exact.json
python benchmarks/bench_gpu_libraries.py --data-dir $DATA --suite imbalanced \
    --libraries torchkm_nystrom falkon linear --falkon-centers 2000 10000 \
    --repeats 10 --device cuda --out benchmarks/results/imbalanced.json
python benchmarks/bench_gpu_libraries.py --data-dir $DATA --suite scale \
    --libraries torchkm_nystrom falkon linear --falkon-centers 2000 10000 20000 \
    --repeats 10 --device cuda --time-cap 14400 --out benchmarks/results/scale.json

python benchmarks/bench_covtype_rank.py --data-dir $DATA --device cuda --with-falkon \
    --repeats 3 --out benchmarks/results/covtype_rank.json

python benchmarks/bench_kqr.py --data-dir $DATA --datasets synthetic cadata abalone cpusmall \
    --repeats 5 --device cuda --export-splits benchmarks/results/kqr_splits \
    --out benchmarks/results/kqr.json
Rscript benchmarks/r/bench_kqr.R benchmarks/results/kqr_splits benchmarks/results/kqr_r.csv

python benchmarks/bench_dwd.py --data-dir $DATA --datasets gisette ijcnn1_30k mnist_3v8 \
    --repeats 5 --device cuda --export-splits benchmarks/results/dwd_splits \
    --out benchmarks/results/dwd.json
Rscript benchmarks/r/bench_dwd.R benchmarks/results/dwd_splits benchmarks/results/dwd_r.csv

python benchmarks/bench_solver_quality.py --device cuda --repeats 5 \
    --out benchmarks/results/solver_quality.json

python benchmarks/make_tables.py benchmarks/results/exact.json
python benchmarks/make_tables.py benchmarks/results/kqr.json --r-csv benchmarks/results/kqr_r.csv --latex
```

Add `--smoke` to any Python script for the CPU check. Pass
`--thundersvm-path /path/to/thundersvm/python` when ThunderSVM is built from
source. Libraries that are not importable are reported once and skipped, so a
CPU-only machine can still run every TorchKM, scikit-learn and linear row.

## Data

Tables use the standard LIBSVM releases from
<https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/>: `a1a` to `a9a`
(Adult), `w7a`, `w8a`, `ijcnn1`, `covtype.libsvm.binary.scale`, `mnist.scale`
and `mnist8m.scale` (multiclass files; pairs of digits are selected at load
time), `gisette_scale`, and the regression sets `cadata`, `abalone`,
`cpusmall`, `space_ga` and `YearPredictionMSD`. Compressed files can stay
compressed. The dataset registry in `benchmarks/_common.py` records the
class prior and any subsampling for every named problem.

## Results

Archive each paper-scale run under `benchmarks/results/<UTC-timestamp>/`
with the JSON files, the R CSV rows and a short `README` naming the commit,
following the convention of `benchmarks/cuda-runs/`. Regenerate the tables
from the archive with `make_tables.py`; do not hand-edit numbers into the
paper or the documentation.
