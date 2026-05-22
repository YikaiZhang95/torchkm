# TorchKM Benchmark Protocol

This directory is for benchmark protocols, scripts, and result files. Benchmark
runs are intentionally separate from the normal test suite and documentation
build because they may take a long time and can depend heavily on hardware.

## What To Report

When adding or reporting a benchmark, include:

- data set name, source, and preprocessing steps;
- number of samples and features;
- train/test split and random seed;
- estimator or low-level solver used;
- kernel and kernel parameters;
- `C` grid or solver regularization grid;
- number of cross-validation folds;
- exact command used;
- CPU model and core count;
- GPU model and GPU memory, if CUDA is used;
- system RAM;
- operating system;
- Python, PyTorch, CUDA, NumPy, and scikit-learn versions;
- whether data loading and preprocessing are included in the timing;
- number of repetitions, warmup policy, and summary statistic.

## Timing Caveats

Wall-clock times vary across hardware, software versions, CUDA settings, and
system load. Do not expect identical times on different machines.

For GPU runs, use warmup runs before timed runs. First-use CUDA initialization
and kernel compilation can make a cold run unrepresentative.

## Development Guidance

Start with a smoke-sized benchmark while developing a script. Run paper-scale
benchmarks only when the command, output files, and expected runtime are clearly
documented.
