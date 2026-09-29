# SPDX-License-Identifier: MIT
"""Memory-envelope helpers for TorchKM's exact (full-kernel) mode.

Exact mode eigendecomposes the full ``n x n`` kernel matrix, so peak device
memory grows with ``n**2`` and is the binding constraint on the largest problem
a given GPU can handle. These helpers make that envelope explicit: they predict
the peak for a given ``n``, invert the prediction to give the largest ``n`` a
memory budget supports, and format the message raised when a fit runs out of
memory.

Memory is not the only bound. On CUDA the eigendecomposition runs in
cuSOLVER, which refuses ``n`` above :data:`EXACT_MODE_MAX_N_CUDA` whatever the
memory; :func:`kernel_eigh` turns that refusal into a clear error.

The prediction is ``n_copies * n**2 * itemsize`` plus the ``n x nlam``
solution path. ``n_copies`` counts the ``n x n`` float64 matrices resident at
the peak: the kernel matrix, the eigenvector matrix, and the eigensolver
workspace. It is a calibrated constant; ``benchmarks/bench_memory_envelope.py``
reports the empirical value for a PyTorch/CUDA build so it can be checked
against :data:`EXACT_MODE_COPIES`.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

#: Number of ``n x n`` matrices' worth of memory that exact mode holds at its
#: peak: the kernel matrix, eigh's copy of it (which becomes the eigenvectors)
#: and cuSOLVER ``syevd``'s workspace of 4.01 more. Measured 6.01 on an L40S
#: (PyTorch 2.6, CUDA 12.4), in float32 at n = 20,000 and in float64 at
#: n = 16,100. A CPU sweep (LAPACK ``syevd``) measured 4.2, so on the CPU the
#: estimate is conservative. ``benchmarks/bench_memory_envelope.py`` measures
#: a build.
EXACT_MODE_COPIES: float = 6.0

#: Largest ``n`` whose eigendecomposition cuSOLVER accepts, whatever the device
#: memory. Measured with PyTorch 2.6 and CUDA 12.4 on an L40S
#: (``benchmarks/probe_eigh_size.py``): in float32 n = 32,768 (2**15) is
#: accepted and 32,769 refused; float64 accepts 32,768 and refuses 33,000.
#: cuSOLVER refuses in its workspace-size query, before allocating anything.
EXACT_MODE_MAX_N_CUDA: int = 32_768

DeviceLike = Union[str, torch.device, None]


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def exact_mode_memory_estimate(
    n_samples: int,
    *,
    dtype: torch.dtype = torch.float64,
    n_copies: float = EXACT_MODE_COPIES,
    nlam: int = 50,
) -> int:
    """Predicted peak device memory, in bytes, for an exact-mode fit.

    Parameters
    ----------
    n_samples : int
        Number of training samples ``n``.
    dtype : torch.dtype, default=torch.float64
        Element type of the kernel matrix. ``TorchKMSVC(dtype="float32")``
        runs the exact SVM solver in float32; the other solvers run in float64.
    n_copies : float, default=EXACT_MODE_COPIES
        Number of ``n x n`` matrices resident at the peak.
    nlam : int, default=50
        Length of the regularization path (the ``n x nlam`` coefficient and
        out-of-fold prediction matrices are included in the estimate).

    Returns
    -------
    int
        Estimated peak in bytes.
    """
    n = int(n_samples)
    if n < 0:
        raise ValueError("n_samples must be non-negative.")
    size = _itemsize(dtype)
    square = float(n_copies) * n * n * size
    path = 2.0 * n * int(nlam) * size
    return int(math.ceil(square + path))


def max_exact_n(
    memory_bytes: int,
    *,
    dtype: torch.dtype = torch.float64,
    n_copies: float = EXACT_MODE_COPIES,
    usable_fraction: float = 0.9,
    size_limit: Optional[int] = EXACT_MODE_MAX_N_CUDA,
) -> int:
    """Largest ``n`` whose exact-mode fit fits in ``memory_bytes``.

    ``usable_fraction`` leaves headroom for the CUDA context, fragmentation,
    and the solution path; the default keeps 10% back. The answer is at most
    ``size_limit``, the largest ``n`` cuSOLVER's eigendecomposition accepts
    (:data:`EXACT_MODE_MAX_N_CUDA`); ``size_limit=None`` counts memory only.
    """
    if memory_bytes <= 0:
        return 0
    size = _itemsize(dtype)
    budget = float(memory_bytes) * float(usable_fraction)
    n = int(math.floor(math.sqrt(budget / (float(n_copies) * size))))
    return n if size_limit is None else min(n, int(size_limit))


def device_total_memory(device: DeviceLike) -> Optional[int]:
    """Total memory of a CUDA device in bytes, or ``None`` for CPU / no CUDA."""
    if device is None:
        return None
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(index).total_memory)


def format_bytes(n_bytes: float) -> str:
    """Human-readable size: ``12.3 GB`` or ``512 MB``."""
    gb = float(n_bytes) / 1e9
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{float(n_bytes) / 1e6:.0f} MB"


def exact_mode_oom_message(
    n_samples: int, device: DeviceLike, dtype: torch.dtype = torch.float64
) -> str:
    """Message for a CUDA out-of-memory error raised by an exact-mode fit."""
    n = int(n_samples)
    size = _itemsize(dtype)
    est = exact_mode_memory_estimate(n, dtype=dtype)
    total = device_total_memory(device)
    msg = (
        f"TorchKM exact mode ran out of GPU memory at n_samples={n:,}. "
        f"Exact mode eigendecomposes the full n x n kernel matrix and needs "
        f"about {format_bytes(est)} for n={n:,} "
        f"({EXACT_MODE_COPIES:g} x {size} x n^2 bytes)"
    )
    if total is not None:
        msg += (
            f"; the device reports {format_bytes(total)} in total, which supports "
            f"exact mode up to roughly n={max_exact_n(total, dtype=dtype):,}"
        )
    msg += (
        ". Use low_rank=True (Nyström approximation, memory grows with "
        "n x num_landmarks instead of n^2), reduce the training size, or fit on "
        "a device with more memory. See the 'Operating envelope' page of the "
        "user guide."
    )
    return msg


def exact_mode_size_message(n_samples: int) -> str:
    """Message for cuSOLVER refusing an exact-mode eigendecomposition."""
    return (
        f"TorchKM exact mode cannot eigendecompose the kernel matrix at "
        f"n_samples={int(n_samples):,}: cuSOLVER's eigendecomposition refuses "
        f"n above {EXACT_MODE_MAX_N_CUDA:,} (measured with PyTorch 2.6 and CUDA "
        f"12.4), whatever the GPU's memory. Use low_rank=True (Nyström "
        f"approximation) or reduce the training size. See the 'Operating "
        f"envelope' page of the user guide."
    )


def kernel_eigh(K: torch.Tensor):
    """``torch.linalg.eigh(K)`` for the exact solvers.

    When cuSOLVER refuses the size (``n`` above :data:`EXACT_MODE_MAX_N_CUDA`),
    the error names the limit and the alternatives instead of cuSOLVER's
    workspace query.
    """
    try:
        return torch.linalg.eigh(K)
    except torch.linalg.LinAlgError as err:
        n = K.shape[-1]
        if n > EXACT_MODE_MAX_N_CUDA and "cusolver" in str(err).lower():
            raise torch.linalg.LinAlgError(exact_mode_size_message(n)) from err
        raise
