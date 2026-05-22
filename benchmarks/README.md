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
