# SPDX-License-Identifier: MIT
"""Memory-envelope helpers for TorchKM's exact (full-kernel) mode.

Exact mode eigendecomposes the full ``n x n`` kernel matrix, so peak device
memory grows with ``n**2`` and is the binding constraint on the largest problem
a given GPU can handle. These helpers make that envelope explicit: they predict
the peak for a given ``n``, invert the prediction to give the largest ``n`` a
memory budget supports, and format the message raised when a fit runs out of
memory.

The prediction is ``n_copies * n**2 * itemsize`` plus the ``n x nlam``
solution path. ``n_copies`` counts the ``n x n`` matrices resident at the
peak, which is set by the one eigendecomposition of the kernel and so by where
it runs (``eigh_backend``, see :mod:`torchkm.linalg`). The measured values are
in :data:`EXACT_MODE_COPIES_BY_BACKEND`; ``benchmarks/bench_memory_envelope.py``
re-measures them on other hardware.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import torch

#: Peak memory of an exact-mode fit in units of one ``n x n`` matrix, by where
#: the kernel eigendecomposition runs. Peaks of PyTorch's allocator, measured
#: for float64 kernels of 16,100 to 22,696 rows on an NVIDIA L40S (PyTorch 2.6,
#: CUDA 12.4) and, for ``"lapack"``, in a CPU sweep (PyTorch 2.14):
#:
#: * ``"cusolver"`` 6.1: the kernel, cuSOLVER's working copy and its
#:   workspace. The driver (NVML) reports 6.4 to 6.8 for the whole process,
#:   which adds the CUDA context and the allocator's cached blocks.
#: * ``"magma"`` 2.0: the kernel and the eigenvectors. MAGMA allocates part of
#:   its device workspace itself, outside PyTorch, so the driver's figure is
#:   higher; it is recorded by ``benchmarks/q1_full_kernel.py``.
#: * ``"cpu"`` 2.0: the kernel and the eigenvectors on the device; the
#:   factorisation runs in host memory.
#: * ``"lapack"`` 4.2: a fit that runs on the CPU, in host memory.
EXACT_MODE_COPIES_BY_BACKEND: Dict[str, float] = {
    "cusolver": 6.1,
    "magma": 2.0,
    "cpu": 2.0,
    "lapack": 4.2,
}

#: Peak of a default GPU fit (cuSOLVER), the figure the envelope is quoted in.
EXACT_MODE_COPIES: float = EXACT_MODE_COPIES_BY_BACKEND["cusolver"]

#: Peak of the low-memory eigendecomposition that ``eigh_backend="auto"``
#: falls back to; it bounds the largest ``n`` exact mode can reach at all.
LOW_MEMORY_COPIES: float = EXACT_MODE_COPIES_BY_BACKEND["cpu"]

DeviceLike = Union[str, torch.device, None]


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _copies(n_copies: Optional[float], backend: str) -> float:
    """Explicit ``n_copies``, else the measured value for ``backend``.

    ``"auto"`` counts as ``"cusolver"``, the backend it uses whenever that
    fits; pass ``"cpu"`` for the largest problem its fallback reaches.
    """
    if n_copies is not None:
        return float(n_copies)
    name = "cusolver" if str(backend).lower() == "auto" else str(backend).lower()
    if name not in EXACT_MODE_COPIES_BY_BACKEND:
        raise ValueError(
            f"backend must be one of auto, {', '.join(EXACT_MODE_COPIES_BY_BACKEND)}; "
            f"got {backend!r}."
        )
    return EXACT_MODE_COPIES_BY_BACKEND[name]


def exact_mode_memory_estimate(
    n_samples: int,
    *,
    dtype: torch.dtype = torch.float64,
    n_copies: Optional[float] = None,
    nlam: int = 50,
    backend: str = "cusolver",
) -> int:
    """Predicted peak device memory, in bytes, for an exact-mode fit.

    Parameters
    ----------
    n_samples : int
        Number of training samples ``n``.
    dtype : torch.dtype, default=torch.float64
        Element type of the kernel matrix. The solvers currently run in
        float64.
    n_copies : float, optional
        Number of ``n x n`` matrices resident at the peak; overrides
        ``backend``.
    nlam : int, default=50
        Length of the regularization path (the ``n x nlam`` coefficient and
        out-of-fold prediction matrices are included in the estimate).
    backend : {"cusolver", "magma", "cpu", "lapack", "auto"}, default="cusolver"
        Where the eigendecomposition runs; selects the measured copy count
        from :data:`EXACT_MODE_COPIES_BY_BACKEND`.

    Returns
    -------
    int
        Estimated peak in bytes.
    """
    n = int(n_samples)
    if n < 0:
        raise ValueError("n_samples must be non-negative.")
    size = _itemsize(dtype)
    square = _copies(n_copies, backend) * n * n * size
    path = 2.0 * n * int(nlam) * size
    return int(math.ceil(square + path))


def max_exact_n(
    memory_bytes: int,
    *,
    dtype: torch.dtype = torch.float64,
    n_copies: Optional[float] = None,
    usable_fraction: float = 0.9,
    backend: str = "cusolver",
) -> int:
    """Largest ``n`` whose exact-mode fit fits in ``memory_bytes``.

    ``usable_fraction`` leaves headroom for the CUDA context, fragmentation,
    and the solution path; the default keeps 10% back. ``backend`` (or an
    explicit ``n_copies``) selects the peak model, as in
    :func:`exact_mode_memory_estimate`.
    """
    if memory_bytes <= 0:
        return 0
    size = _itemsize(dtype)
    budget = float(memory_bytes) * float(usable_fraction)
    return int(math.floor(math.sqrt(budget / (_copies(n_copies, backend) * size))))


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


def exact_mode_oom_message(n_samples: int, device: DeviceLike) -> str:
    """Message for a CUDA out-of-memory error raised by an exact-mode fit."""
    n = int(n_samples)
    fast = exact_mode_memory_estimate(n, backend="cusolver")
    low = exact_mode_memory_estimate(n, backend="cpu")
    total = device_total_memory(device)
    msg = (
        f"TorchKM exact mode ran out of GPU memory at n_samples={n:,}. "
        f"Exact mode eigendecomposes the full n x n kernel matrix: with cuSOLVER "
        f"it needs about {format_bytes(fast)} for n={n:,} "
        f"({EXACT_MODE_COPIES:g} x 8 x n^2 bytes), with the low-memory "
        f"eigendecomposition (eigh_backend='magma' or 'cpu', which the default "
        f"eigh_backend='auto' falls back to) about {format_bytes(low)} "
        f"({LOW_MEMORY_COPIES:g} x 8 x n^2 bytes)"
    )
    if total is not None:
        msg += (
            f"; the device reports {format_bytes(total)} in total, which supports "
            f"exact mode up to roughly n={max_exact_n(total):,} with cuSOLVER and "
            f"n={max_exact_n(total, backend='cpu'):,} with the low-memory "
            f"eigendecomposition"
        )
    msg += (
        ". Use low_rank=True (Nyström approximation, memory grows with "
        "n x num_landmarks instead of n^2), reduce the training size, or fit on "
        "a device with more memory. See the 'Operating envelope' page of the "
        "user guide."
    )
    return msg
