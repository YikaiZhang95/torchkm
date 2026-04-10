# TorchKM Benchmark And Optimization Summary

## Overview

This directory contains the benchmarking harness, warmed benchmark artifacts, and a summary of the performance work done for TorchKM on GPU-enabled systems.

The main goals were:

- Measure end-to-end `fit()` performance for the public wrapper APIs in `torchkm/estimators.py`
- Identify the dominant setup and solver bottlenecks for exact and Nyström backends
- Apply a small number of low-risk optimizations
- Re-run the same benchmark matrix and keep only changes that improved performance without breaking output parity

## Benchmark Environment

- Container image: GPU-enabled benchmark container
- GPU: benchmark accelerator
- PyTorch: GPU-enabled development build
- Device used for benchmarks: `cuda`
- Warmup policy: one untimed warmup run per case to avoid first-use GPU compilation noise

The main benchmark driver is:

- `benchmarks/benchmark_torchkm.py`

The durable benchmark result files are:

- `benchmarks/results/baseline_warm_smoke_medium.json`
- `benchmarks/results/baseline_warm_stress_svc.json`
- `benchmarks/results/final_warm_smoke_medium.json`
- `benchmarks/results/final_warm_stress_svc.json`
- `benchmarks/results/paper_final_warm_smoke_medium.json`
- `benchmarks/results/paper_final_warm_stress_svc.json`

## What Was Changed

### 1. Benchmark Harness

Added a reusable wrapper-level benchmark harness in `benchmarks/benchmark_torchkm.py` that:

- exercises `TorchKMSVC`, `TorchKMDWD`, and `TorchKMLogit`
- covers both exact and `low_rank=True` paths
- optionally profiles low-level backend phases such as:
  - kernel build
  - `torch.linalg.eigh`
  - `torch.linalg.svd`
  - `rbf_kernel`
  - `kernelMult`
  - `golden_section_search`
- writes machine-readable JSON so results are easy to compare

### 2. Solver Hot-Loop Cleanup

Updated:

- `torchkm/cvksvm.py`
- `torchkm/cvkdwd.py`
- `torchkm/cvklogit.py`
- `torchkm/cvknyssvm.py`
- `torchkm/cvknysdwd.py`
- `torchkm/cvknyslogit.py`

Key changes:

- reduced `.item()` usage in hot iterative loops by keeping step-state tensors on device
- reused step buffers instead of allocating fresh tensors every iteration
- removed unnecessary CPU round-trips in the DWD and logit objective paths

### 3. Shared Kernel And Nyström Setup Optimization

Updated:

- `torchkm/functions.py`
- `torchkm/cvknyssvm.py`
- `torchkm/cvknysdwd.py`
- `torchkm/cvknyslogit.py`

Key changes:

- optimized `rbf_kernel()` and `kernelMult()` to use a more efficient distance assembly pattern
- clamped pairwise distances before exponentiation for numerical stability
- fixed `sigest()` for newer PyTorch quantile dtype behavior
- moved Nyström preprocessing to the device path instead of forcing CPU preprocessing
- kept learned Nyström state on device for faster `transform()` and prediction

## Why These Changes Speed Things Up

### Exact Backends

The exact solvers still spend meaningful time in eigendecomposition and Python-driven control flow, so the gains are moderate rather than dramatic. The main wins came from:

- fewer host/device synchronizations in the inner loops
- less Python scalar extraction from device tensors
- fewer per-iteration temporary allocations

This is why exact-mode gains are real but modest.

### Nyström Backends

The biggest wins came from eliminating unnecessary CPU preprocessing in the Nyström path.

Before the optimization, the low-rank backends spent a large fraction of total time in:

- CPU-side landmark kernel construction
- CPU-side `kernelMult`
- CPU-side SVD and feature-map assembly

After the optimization, those steps moved onto the device path and dropped sharply.

For example, in the `medium` `svc low_rank` case, backend phase time changed from:

- `linalg_svd`: `0.6859s -> 0.0172s`
- `kernelMult`: `0.1585s -> 0.0003s`
- `rbf_kernel`: `0.0212s -> 0.0001s`
- total backend fit: `0.9749s -> 0.0852s`

That phase reduction is the main reason low-rank speedups are much larger than exact-mode speedups.

### Why Warmup Matters

Initial GPU runs showed misleading one-time latency from first-use kernel compilation. The final comparison therefore uses one untimed warmup per case before taking measurements. The warmed results are the numbers that should be used for fair comparison.

## Final Benchmark Results

The table below compares the warmed untouched baseline against the warmed optimized implementation.

| Size | Backend | Mode | `is_exact` | Baseline Fit (s) | Final Fit (s) | Speedup | Accuracy Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| smoke | svc | exact | 0 | 0.142 | 0.104 | 1.37x | +0.0000 |
| smoke | svc | exact | 1 | 0.144 | 0.105 | 1.38x | +0.0000 |
| smoke | svc | low_rank | 0 | 0.464 | 0.052 | 8.87x | +0.0104 |
| smoke | dwd | exact | 0 | 0.221 | 0.174 | 1.27x | +0.0000 |
| smoke | dwd | low_rank | 0 | 0.558 | 0.117 | 4.75x | +0.0000 |
| smoke | logit | exact | 0 | 0.233 | 0.156 | 1.49x | +0.0000 |
| smoke | logit | low_rank | 0 | 0.554 | 0.132 | 4.19x | +0.0000 |
| medium | svc | exact | 0 | 0.201 | 0.155 | 1.30x | +0.0000 |
| medium | svc | exact | 1 | 0.186 | 0.148 | 1.25x | +0.0000 |
| medium | svc | low_rank | 0 | 0.968 | 0.085 | 11.45x | +0.0000 |
| medium | dwd | exact | 0 | 0.437 | 0.374 | 1.17x | +0.0000 |
| medium | dwd | low_rank | 0 | 1.226 | 0.299 | 4.09x | +0.0050 |
| medium | logit | exact | 0 | 0.417 | 0.380 | 1.10x | +0.0000 |
| medium | logit | low_rank | 0 | 1.104 | 0.232 | 4.76x | -0.0050 |
| stress | svc | exact | 0 | 0.221 | 0.180 | 1.23x | +0.0000 |
| stress | svc | exact | 1 | 0.223 | 0.185 | 1.21x | +0.0000 |
| stress | svc | low_rank | 0 | 2.205 | 0.131 | 16.79x | -0.0036 |

## Interpretation

- Exact paths improved by about `1.10x` to `1.49x`
- Low-rank paths improved by about `4.09x` to `16.79x`
- The largest gains came from device-side Nyström setup and faster shared kernel helpers
- Accuracy stayed effectively stable across the matrix, with only very small low-rank deltas

Representative wins:

- `medium` `TorchKMSVC(low_rank=True)`: `0.968s -> 0.085s` (`11.45x`)
- `medium` `TorchKMDWD(low_rank=True)`: `1.226s -> 0.299s` (`4.09x`)
- `medium` `TorchKMLogit(low_rank=True)`: `1.104s -> 0.232s` (`4.76x`)
- `stress` `TorchKMSVC(low_rank=True)`: `2.205s -> 0.131s` (`16.79x`)

## Paper-Aligned Follow-Up

After reading the TorchKM paper, a second optimization round focused on the two ideas emphasized there:

- exact cross-validation reuse via modified response vectors
- spectral reuse that keeps the heavy decomposition outside the tuning and CV loops

The follow-up changes were:

- replaced generic Nyström `svd(W)` with symmetric `eigh(W)` in:
  - `torchkm/cvknyssvm.py`
  - `torchkm/cvknysdwd.py`
  - `torchkm/cvknyslogit.py`
- batched the fold updates for:
  - exact `cvksvm` when `is_exact=0`
  - exact `cvkdwd`
  - exact `cvklogit`
  - low-rank `cvknysdwd`
  - low-rank `cvknyslogit`
- removed the explicit CPU staging from the main SVM intercept/objective path

The new benchmark artifacts for this pass are:

- `benchmarks/results/paper_final_warm_smoke_medium.json`
- `benchmarks/results/paper_final_warm_stress_svc.json`

### What Paid Off

Compared with the first optimized baseline in `final_warm_*.json`, the paper-aligned changes improved:

| Size | Backend | Mode | `is_exact` | Previous Fit (s) | Paper-Aligned Fit (s) | Extra Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| smoke | svc | low_rank | 0 | 0.052 | 0.042 | 1.24x |
| smoke | dwd | exact | 0 | 0.174 | 0.153 | 1.14x |
| smoke | dwd | low_rank | 0 | 0.117 | 0.101 | 1.16x |
| smoke | logit | exact | 0 | 0.156 | 0.145 | 1.08x |
| smoke | logit | low_rank | 0 | 0.132 | 0.082 | 1.60x |
| medium | svc | exact | 0 | 0.155 | 0.138 | 1.12x |
| medium | svc | low_rank | 0 | 0.085 | 0.069 | 1.23x |
| medium | dwd | exact | 0 | 0.374 | 0.284 | 1.32x |
| medium | dwd | low_rank | 0 | 0.299 | 0.157 | 1.90x |
| medium | logit | exact | 0 | 0.380 | 0.231 | 1.65x |
| medium | logit | low_rank | 0 | 0.232 | 0.155 | 1.49x |
| stress | svc | exact | 0 | 0.180 | 0.175 | 1.02x |
| stress | svc | low_rank | 0 | 0.131 | 0.097 | 1.36x |

### What Did Not Move Much

- `svc exact` with `is_exact=1` showed little improvement and slight small-case regression.
- This is expected because that path still relies on the more complex exact projection refinement, which remains much more serial and less amenable to simple fold batching.

### Why The Paper-Aligned Changes Helped

- For the Nyström backends, `W` is symmetric PSD, so `eigh` is a better structural match than generic SVD.
- For the DWD and logit CV loops, batching folds converts repeated fold-by-fold matrix-vector work into larger matrix-matrix operations, which better matches the paper’s GPU-oriented design.
- The phase data shows the expected effect:
  - low-rank DWD/logit remove the `linalg_svd` phase entirely and cut `backend_fit_total` substantially
  - exact DWD/logit reduce `iterative_solve_estimate` sharply after fold batching

Representative medium-case phase changes versus the first optimized baseline:

- `medium dwd exact`: `backend_fit_total 0.3166s -> 0.1963s`
- `medium logit exact`: `backend_fit_total 0.3243s -> 0.1716s`
- `medium dwd low_rank`: `backend_fit_total 0.2988s -> 0.1567s`
- `medium logit low_rank`: `backend_fit_total 0.2328s -> 0.1548s`

Overall, the paper-aligned follow-up especially helped the non-SVM backends and the low-rank pipelines, while exact SVM `is_exact=1` remains the least improved path.

## Validation

Targeted validation after the optimization changes:

```bash
pytest tests/test_estimators.py tests/test_cvksvm.py tests/test_cvkdwd.py tests/test_cvklogit.py -q
```

Result:

- `9 passed`

## Not Done

This work intentionally did not prioritize:

- `torch.compile` integration
- vendor-specific graph capture APIs
- major algorithmic rewrites of the exact SVM intercept search

Those may still be worth exploring later, but the changes committed here were chosen because they were lower-risk and produced clear measured wins, especially for the Nyström backends.
