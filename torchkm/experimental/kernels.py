# SPDX-License-Identifier: MIT
"""A kernel matrix that is never stored (experimental).

:class:`RBFKernelOperator` stands in for the n x n RBF kernel matrix wherever
only products ``K @ B`` are needed, as in
:class:`torchkm.experimental.SpectralSVMPath` with the truncated spectrum. Each
product recomputes K in blocks of rows, uses each block once and frees it.
Memory is then O(n p + block + n k) for k right-hand columns instead of n^2,
and each product costs the kernel's arithmetic again (about 2 n^2 p flops for
the distances) instead of one read of a stored K.
"""

from __future__ import annotations

import torch


class RBFKernelOperator:
    """K = exp(-2 sigma |x_i - x_j|^2) over the rows of ``X``, as an operator.

    ``K @ B`` equals ``torchkm.functions.rbf_kernel(X, sigma) @ B`` (the same
    formula, block by block, so the entries agree to rounding). ``B`` may be a
    vector or a matrix in ``X``'s dtype and on its device.

    Parameters
    ----------
    X : tensor (n, p)
        Training rows, on the device and in the dtype to compute in.
    sigma : float
        Bandwidth, as in ``rbf_kernel``.
    block_bytes : int, default 2**30
        Size of one block of kernel rows; the rows per block follow from it.
    """

    def __init__(self, X: torch.Tensor, sigma: float, block_bytes: int = 2**30):
        n = X.shape[0]
        self.X, self.sigma = X, float(sigma)
        self.shape = (n, n)
        self.dtype, self.device = X.dtype, X.device
        self.x_norm = (X * X).sum(dim=1)
        self.block_rows = max(1, min(n, int(block_bytes) // (n * X.element_size())))

    def __matmul__(self, B: torch.Tensor) -> torch.Tensor:
        vector = B.dim() == 1
        B2 = B.unsqueeze(1) if vector else B
        n, rows = self.shape[0], self.block_rows
        out = torch.empty(n, B2.shape[1], dtype=B2.dtype, device=B2.device)
        for i in range(0, n, rows):
            # rbf_kernel's arithmetic on rows i .. i + rows
            D = self.x_norm[i : i + rows, None] + self.x_norm[None, :]
            D.addmm_(self.X[i : i + rows], self.X.T, beta=1.0, alpha=-2.0)
            D.clamp_min_(0.0)
            out[i : i + rows] = D.mul_(-2.0 * self.sigma).exp_() @ B2
        return out.squeeze(1) if vector else out
