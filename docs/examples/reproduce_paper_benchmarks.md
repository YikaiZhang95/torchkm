# Reproducing Paper Benchmarks

This page describes the benchmark protocol used for paper-style comparisons.
Benchmark-scale runs can take a long time and should be run deliberately on a
machine reserved for that purpose.

Timing results can vary substantially across hardware, CUDA versions, PyTorch builds, data-loading behavior, and system load. The goal of these scripts is to reproduce the experimental protocol, not to guarantee identical wall-clock times on every machine.

## Hardware and software used in the paper

The paper reports experiments on:

- GPU: NVIDIA L40s with 48 GB memory
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

## Cross-validation and tuning grid

The paper uses a grid of 50 candidate regularization values. The values are log-spaced under the scikit-learn/LIBSVM \(C\)-parameterization, with \(C \in [10^{-3}, 10^3]\).

For the paper benchmarks, model selection is performed through cross-validation, and reported times are end-to-end wall-clock times for the training-and-tuning pipeline.

## Running benchmarks

This checkout does not require documentation builds or examples to run benchmark
scripts. If benchmark scripts are added under `benchmarks/`, document the exact
command, expected runtime scale, data source, and output files before asking
users to run them.

Start with a smoke-sized run when developing a benchmark harness, then run the
paper-sized protocol only on suitable hardware.

## Reporting results

When reporting new benchmark results, include:

- data set name;
- number of samples and features;
- CPU model;
- GPU model;
- RAM and GPU memory;
- operating system;
- Python version;
- PyTorch version;
- CUDA version;
- regularization grid;
- number of folds;
- whether preprocessing time is included;
- mean runtime and number of repetitions.
