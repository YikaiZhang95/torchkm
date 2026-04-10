# TorchKM Performance Findings

## Scope

This note summarizes only the performance methods that produced measurable speedups in TorchKM on GPU-enabled systems. It focuses on changes that improved end-to-end training and tuning time for the exact and Nyström backends.

## Benchmark Setup

- Container: GPU-enabled benchmark container
- GPU: benchmark accelerator
- PyTorch: GPU-enabled development build
- Device: `cuda`
- Benchmark driver: `benchmarks/benchmark_torchkm.py`
- Methodology: one untimed warmup run per case to remove first-use GPU compilation noise

Reference artifacts:

- `benchmarks/results/baseline_warm_smoke_medium.json`
- `benchmarks/results/baseline_warm_stress_svc.json`
- `benchmarks/results/paper_final_warm_smoke_medium.json`
- `benchmarks/results/paper_final_warm_stress_svc.json`

## Performance Methods That Worked

### 1. Hot-loop cleanup in the solvers

Updated files:

- `torchkm/cvksvm.py`
- `torchkm/cvkdwd.py`
- `torchkm/cvklogit.py`
- `torchkm/cvknyssvm.py`
- `torchkm/cvknysdwd.py`
- `torchkm/cvknyslogit.py`

Successful changes:

- kept step-state tensors on device instead of repeatedly extracting Python scalars with `.item()`
- reused temporary step buffers instead of allocating new tensors every iteration
- removed unnecessary CPU staging from hot objective/intercept paths where it was safe

Why it helped:

- fewer host/device synchronizations
- less per-iteration allocation overhead
- better GPU utilization inside the iterative solver loops

### 2. Device-side Nyström preprocessing

Updated files:

- `torchkm/cvknyssvm.py`
- `torchkm/cvknysdwd.py`
- `torchkm/cvknyslogit.py`
- `torchkm/functions.py`

Successful changes:

- moved landmark kernel construction and feature-map setup onto the device path
- kept learned Nyström state on device for faster transform and prediction
- optimized shared kernel helpers in `functions.py`

Why it helped:

- removed the largest CPU-side bottleneck in the low-rank pipeline
- turned setup into GPU-friendly dense linear algebra
- reduced setup overhead before the actual solver iterations even begin

### 3. Use symmetric eigendecomposition for Nyström landmark kernels

Updated files:

- `torchkm/cvknyssvm.py`
- `torchkm/cvknysdwd.py`
- `torchkm/cvknyslogit.py`

Successful change:

- replaced generic `svd(W)` with symmetric `eigh(W)` for the Nyström landmark kernel `W`

Why it helped:

- the landmark kernel is symmetric positive semidefinite
- `eigh` matches that structure directly and avoids paying for a more general decomposition
- this reduced spectral setup cost in the low-rank path

### 4. Batch fold updates for exact and low-rank CV

Updated files:

- `torchkm/cvksvm.py` for `is_exact=0`
- `torchkm/cvkdwd.py`
- `torchkm/cvklogit.py`
- `torchkm/cvknysdwd.py`
- `torchkm/cvknyslogit.py`

Successful change:

- replaced repeated fold-by-fold CV updates with fold-batched tensor operations

Why it helped:

- converts repeated per-fold matrix-vector work into larger matrix-matrix operations
- better matches the paper’s “exact cross-validation reuse” idea
- gives the GPU more work per kernel launch and reduces Python-loop overhead

## What Boosted Performance The Most

### Largest end-to-end wins

Compared with the warmed untouched baseline:

| Size | Backend | Mode | Baseline Fit (s) | Final Fit (s) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| smoke | svc | low_rank | 0.464 | 0.042 | 11.00x |
| medium | svc | low_rank | 0.968 | 0.069 | 14.05x |
| medium | dwd | low_rank | 1.226 | 0.157 | 7.79x |
| medium | logit | low_rank | 1.104 | 0.155 | 7.11x |
| stress | svc | low_rank | 2.205 | 0.097 | 22.81x |
| medium | svc | exact | 0.201 | 0.138 | 1.46x |
| medium | dwd | exact | 0.437 | 0.284 | 1.54x |
| medium | logit | exact | 0.417 | 0.231 | 1.81x |

## Phase-Level Findings

### Nyström path

The low-rank speedups came primarily from shrinking setup time before the solver loop.

Representative example: `medium svc low_rank`

- backend fit total: `0.9749s -> 0.0680s`
- `linalg_svd` disappeared from the critical path
- `kernelMult` and `rbf_kernel` dropped to near-negligible time

Representative example: `medium dwd low_rank`

- backend fit total: `1.2354s -> 0.1567s`

Representative example: `medium logit low_rank`

- backend fit total: `1.1033s -> 0.1548s`

### Exact path

The exact path still has unavoidable dense-kernel and eigendecomposition cost, so gains are smaller but still meaningful.

Representative example: `medium dwd exact`

- backend fit total: `0.3437s -> 0.1963s`

Representative example: `medium logit exact`

- backend fit total: `0.3200s -> 0.1716s`

Representative example: `medium svc exact`

- fit time improved from `0.201s -> 0.138s`

## Why These Methods Worked

The methods that boosted performance all shared the same pattern:

- push more work into GPU-friendly dense linear algebra
- reuse expensive spectral work across folds and tuning values
- remove repeated setup and Python overhead from the hot path

This lines up with the TorchKM paper’s main performance direction:

- exact cross-validation reuse
- spectral reuse across the regularization path
- GPU-oriented algorithm design instead of only GPU offloading

## Practical Takeaways

- The biggest payoff comes from improving the low-rank/Nyström path, especially setup and fold reuse.
- Batched CV is a strong improvement for DWD and logit.
- Exact backends can still improve, but the remaining wins are smaller because dense kernel build and eigendecomposition remain fundamental costs.

## Validation

Targeted validation used during this performance work:

```bash
pytest tests/test_estimators.py tests/test_cvksvm.py tests/test_cvkdwd.py tests/test_cvklogit.py -q
```

Observed result on the benchmark environment:

- `9 passed`
