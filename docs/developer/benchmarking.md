# Benchmarking

Benchmarks are important for TorchKM because the package is designed around the
full training-and-tuning pipeline. Benchmark runs can be expensive, so they
should be separate from documentation builds, unit tests, and quick examples.

## Benchmark principles

A benchmark should report enough information for another user to understand the result:

- data set;
- number of samples;
- number of features;
- train/test split;
- regularization grid;
- number of folds;
- CPU model;
- GPU model;
- RAM and GPU memory;
- operating system;
- Python version;
- PyTorch version;
- CUDA version;
- whether preprocessing is included;
- number of repetitions;
- mean runtime and uncertainty.

## Repository structure

```text
benchmarks/
  README.md
  # optional benchmark scripts and saved outputs
```

## Timing

Use wall-clock time for end-to-end training and model selection when comparing
with the paper. State clearly whether data loading and preprocessing are
included. Do not promise identical times across machines.

For GPU benchmarks, use warmup runs before timed runs. First-use CUDA kernel
initialization can make cold timings misleading.

## Accuracy

For classification benchmarks, report test accuracy or another clearly defined metric. If cross-validation is used for tuning, evaluate final performance on held-out test data when available.

## Reproducibility

Use fixed random seeds where possible. For GPU benchmarks, exact timing may still
vary because of hardware and system-level differences. Always report the
hardware and software fields listed above with benchmark results.
