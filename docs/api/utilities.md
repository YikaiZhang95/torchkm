# Kernels and Utilities API

This page documents public kernel and utility functions.

## Kernel functions

::: torchkm.kernels.linear_kernel

::: torchkm.kernels.polynomial_kernel

::: torchkm.kernels.rbf_kernel

## Utility functions

::: torchkm.functions.sigest

::: torchkm.functions.rbf_kernel

::: torchkm.functions.kernelMult

## Memory envelope and the eigendecomposition

::: torchkm.linalg.kernel_eigh

::: torchkm.memory.exact_mode_memory_estimate

::: torchkm.memory.max_exact_n

## Probability calibration

::: torchkm.platt.PlattScalerTorch

## Notes

The utility API is lower-level than the estimator API. Most users should begin with the high-level estimators and use these functions only when they need custom kernels, kernel matrices, or direct solver access.
