# SPDX-License-Identifier: MIT
"""A kernel matrix that is never stored (experimental).

:class:`RBFKernelOperator` stands in for the n x n RBF kernel matrix wherever
only products ``K @ B`` are needed, as in
:class:`torchkm.experimental.SpectralSVMPath` with the truncated spectrum. Each
product recomputes K in blocks of rows, uses each block once and frees it.
Memory is then O(n p + block + n k) for k right-hand columns instead of n^2,
and each product costs the kernel's arithmetic again (about 2 n^2 p flops for
the distances) instead of one read of a stored K.

With ``fused=True`` (CUDA, float32) a product runs in one fused kernel per 128
columns and never writes a block of K to memory. With
a_i = [4 sigma x_i, -2 sigma |x_i|^2, 1] and b_j = [x_j, 1, -2 sigma |x_j|^2],
a_i'b_j = -2 sigma |x_i - x_j|^2, so K B = exp(A B') V: attention without the
softmax normalization. PyTorch's memory-efficient attention computes it in
float32 and returns the log-sum-exp of each row, which undoes the
normalization. Building the blocks of K instead writes and reads each entry
about ten times, which dominates the block path's time.
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
    fused : bool, default False
        Products in one fused kernel per ``FUSED_MAX_COLUMNS`` columns (CUDA,
        float32 only) instead of blocks of K. Entries agree
        with the block path to float32 rounding (relative error about 4e-5
        against float64, 2e-5 for the blocks), not bitwise.
    """

    FUSED_MAX_COLUMNS = 128

    def __init__(
        self,
        X: torch.Tensor,
        sigma: float,
        block_bytes: int = 2**30,
        fused: bool = False,
    ):
        n = X.shape[0]
        self.X, self.sigma = X, float(sigma)
        self.shape = (n, n)
        self.dtype, self.device = X.dtype, X.device
        self.x_norm = (X * X).sum(dim=1)
        self.block_rows = max(1, min(n, int(block_bytes) // (n * X.element_size())))
        self.fused = bool(fused)
        if self.fused:
            if X.device.type != "cuda" or X.dtype != torch.float32:
                raise ValueError("fused=True needs float32 rows on a CUDA device")
            s, nx = self.sigma, self.x_norm[:, None]
            one = torch.ones_like(nx)
            self._q = _pad8(torch.cat([4.0 * s * X, -2.0 * s * nx, one], dim=1))
            self._k = _pad8(torch.cat([X, one, -2.0 * s * nx], dim=1))

    def _fused(self, B: torch.Tensor, q: torch.Tensor = None) -> torch.Tensor:
        q = self._q if q is None else q
        n, w = q.shape[0], B.shape[1]
        out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
            q[None, None],
            self._k[None, None],
            _pad8(B)[None, None],
            None,
            True,  # return the log-sum-exp of each row
            0.0,
            False,
            scale=1.0,
        )[:2]
        return out[0, 0, :, :w] * lse[0, 0, :n, None].exp()

    def cross(self, Xq: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """K(Xq, X) @ B for other rows ``Xq`` (prediction), in blocks of query
        rows, or fused (no block of the kernel is formed) with ``fused``."""
        vector = B.dim() == 1
        B2 = B.unsqueeze(1) if vector else B
        s = self.sigma
        if self.fused:
            nq = (Xq * Xq).sum(dim=1, keepdim=True)
            q = _pad8(torch.cat([4.0 * s * Xq, -2.0 * s * nq, torch.ones_like(nq)], 1))
            w = self.FUSED_MAX_COLUMNS
            out = torch.cat(
                [self._fused(B2[:, j : j + w], q) for j in range(0, B2.shape[1], w)],
                dim=1,
            )
            return out.squeeze(1) if vector else out
        rows = self.block_rows
        out = torch.empty(Xq.shape[0], B2.shape[1], dtype=B2.dtype, device=B2.device)
        for i in range(0, Xq.shape[0], rows):
            Xb = Xq[i : i + rows]
            D = (Xb * Xb).sum(dim=1)[:, None] + self.x_norm[None, :]
            D.addmm_(Xb, self.X.T, beta=1.0, alpha=-2.0)
            D.clamp_min_(0.0)
            out[i : i + rows] = D.mul_(-2.0 * s).exp_() @ B2
        return out.squeeze(1) if vector else out

    def __matmul__(self, B: torch.Tensor) -> torch.Tensor:
        vector = B.dim() == 1
        B2 = B.unsqueeze(1) if vector else B
        if self.fused:
            w = self.FUSED_MAX_COLUMNS
            out = torch.cat(
                [self._fused(B2[:, j : j + w]) for j in range(0, B2.shape[1], w)], dim=1
            )
            return out.squeeze(1) if vector else out
        n, rows = self.shape[0], self.block_rows
        out = torch.empty(n, B2.shape[1], dtype=B2.dtype, device=B2.device)
        for i in range(0, n, rows):
            # rbf_kernel's arithmetic on rows i .. i + rows
            D = self.x_norm[i : i + rows, None] + self.x_norm[None, :]
            D.addmm_(self.X[i : i + rows], self.X.T, beta=1.0, alpha=-2.0)
            D.clamp_min_(0.0)
            out[i : i + rows] = D.mul_(-2.0 * self.sigma).exp_() @ B2
        return out.squeeze(1) if vector else out


def _pad8(M: torch.Tensor) -> torch.Tensor:
    """Columns zero-padded to a multiple of 8 (the fused kernel's alignment)."""
    return torch.nn.functional.pad(M, (0, (-M.shape[1]) % 8)).contiguous()
