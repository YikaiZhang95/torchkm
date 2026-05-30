# TorchKM Benchmark Protocol

This directory is for benchmark protocols, scripts, and result files. Benchmark
runs are intentionally separate from the normal test suite and documentation
build because they may take a long time and can depend heavily on hardware.

## What To Report

When adding or reporting a benchmark, include:

- exact command used;
- CPU model;
- GPU model, if used;
- RAM and GPU memory, if relevant;
- operating system;
- Python version;
- PyTorch version;
- CUDA version, if used;
- TorchKM version or commit hash;
- data set;
- sample size and feature dimension;
- number of cross-validation folds;
- regularization grid;
- whether timing includes preprocessing, training, and model selection.

## Timing Caveats

Wall-clock times vary across hardware, software versions, CUDA settings, and
system load. Do not expect identical times on different machines.

Do not compare a cold first run with a warmed run. For GPU benchmarks, CUDA
initialization and first-use overhead can make the first run misleading. Run at
least one warmup iteration before timing. When timing CUDA work, call
`torch.cuda.synchronize()` before starting and after ending the timed region.

Report the number of repeats. Prefer median and IQR, or mean and standard
deviation. Avoid reporting a single timing as a strong benchmark claim.

Benchmark scripts should optionally write JSON or CSV output. Output should
include both results and environment metadata. Do not commit large generated
benchmark outputs.

## Development Guidance

Start with a smoke-sized benchmark while developing a script. Run paper-scale
benchmarks only when the command, output files, and expected runtime are clearly
documented.

## Reproducing Paper Tables 2-4

Scripts for the paper's three benchmark tables live next to this README. They
reproduce the source notebooks' protocol per method. ThunderSVM is imported on
demand (build it, then `cd thundersvm/python/` or pass `--thundersvm-path`) and
skipped with guidance if unavailable.

Each method is tuned by 10-fold cross-validation over a 50-point regularization
grid, with end-to-end wall-clock timing and a CUDA warmup. The grid is in the
lambda parameterization, transferred to LIBSVM `C` via `C = 1/(2*n*lambda)`.
The exact grid is per-table (and per-dataset in Table 4) to match the source
notebooks: Table 2 uses `lambda in [1e-3, 1e3]`, Table 3 uses
`lambda in [1e-5, 1e-1]`, Table 4 uses `lambda in [1e-3, 1e3]` for w8a/ijcnn1/
covtype/mnist8m and `lambda in [1e-7, 1e-1]` for a9a. `--repeats` defaults to a
small smoke value; pass the paper count (50 for Table 2, 10 for Tables 3-4).
Use `--device cuda` on a GPU; timings differ on other hardware.

| Script | Table | Data | Reports |
| --- | --- | --- | --- |
| `table2_simulation.py` | 2 | synthetic (`torchkm.data_gen`) | objective + time: scikit-learn, ThunderSVM, TorchKM |
| `table3_benchmarks.py` | 3 | a7a, a8a, w7a | accuracy + time: TorchKM (exact RBF SVM via `cvksvm`), ThunderSVM |
| `table4_nystrom.py` | 4 | a9a, w8a, ijcnn1, covtype, mnist8m (4-vs-6) | accuracy + time: TorchKM Nystrom (`cvknyssvm`), scikit-learn Nystrom (manual transform + `LinearSVC`) |

```bash
# Table 2 (no external data needed)
python benchmarks/table2_simulation.py --repeats 50 --device cuda \
    --thundersvm-path /path/to/thundersvm/python

# Table 3 (LIBSVM files in DATA_DIR, train + ".t" test file each)
python benchmarks/table3_benchmarks.py --data-dir DATA_DIR --repeats 10 --device cuda \
    --thundersvm-path /path/to/thundersvm/python

# Table 4 (Nystrom; covtype/mnist8m are single files and get a split)
python benchmarks/table4_nystrom.py --data-dir DATA_DIR --datasets a9a w8a ijcnn1 \
    --repeats 10 --device cuda
```

**Data.** Tables 3-4 read standard LIBSVM files from `--data-dir`. Download from
`https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/`. Table 4 opens
`.bz2`/`.xz` files transparently, so `ijcnn1.bz2`, `covtype.libsvm.binary.scale.bz2`,
and `mnist8m.scale.xz` can sit in `--data-dir` as-downloaded with no manual
decompression. (`bunzip2 -k` / `unxz -k` is still faster if you'll rerun many
times.) The mnist8m 4-vs-6 subset is extracted from `mnist8m.scale` at load time.

**Runtime scale.** Table 2 at `n=20000, p=1000` builds a 20k x 20k kernel and
needs a GPU; scikit-learn does not finish within hours at that size
(`--skip-sklearn` to drop it). Table 4 on `covtype` (~580k) and `mnist8m`
(~1.27M after filtering 4-vs-6) is heavy and is where the Nystrom approximation
matters most; the a9a column also runs with `maxit=10_000_000` (the notebook
value). These scripts print results to stdout and do not write output files.

**Convenience.** `benchmarks/run_table3.sh [DATA_DIR] [THUNDERSVM_PYTHON_PATH]`
downloads the Table 3 LIBSVM files and runs the benchmark in one shot.
