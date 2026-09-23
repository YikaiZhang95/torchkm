# SPDX-License-Identifier: MIT
"""Eigendecomposition of the kernel matrix, the one O(n^3) step of exact mode.

Exact mode factorises the ``n x n`` kernel once, ``K = U diag(e) U^T``, and
reuses ``U`` and ``e`` for the whole regularization path and the exact
cross-validation formula. Where that factorisation runs decides the peak
device memory of the whole fit. The eigenpairs are the same on every backend
up to rounding error, so the choice changes time and memory, never the
solution.

Backends for a kernel on a CUDA device (``eigh_backend`` on the estimators):

``"cusolver"``
    PyTorch's default GPU eigensolver. Fastest. Its peak is about six
    ``n x n`` matrices: the kernel, cuSOLVER's working copy and its
    workspace.
``"magma"``
    MAGMA's hybrid solver, which keeps its large workspace in host memory:
    PyTorch's allocator sees two ``n x n`` matrices (the kernel and the
    eigenvectors). MAGMA also allocates part of its device workspace itself,
    outside PyTorch, so the process peak reported by the driver is somewhat
    higher. When PyTorch was built without MAGMA this runs ``"cpu"``.
``"cpu"``
    LAPACK in host memory; the eigenvectors are then copied to the device.
    The device holds exactly two ``n x n`` matrices; the speed is the host
    CPU's.
``"auto"`` (default)
    ``"cusolver"``, repeated with the low-memory backends (``"magma"``, then
    ``"cpu"``) when the device runs out of memory. A fit that fits keeps the
    fast path; a fit that does not is no longer an out-of-memory error.

Measured on an NVIDIA L40S (PyTorch 2.6, CUDA 12.4) for a 16,100-row float64
kernel: cuSOLVER 21.5 s at 6.0 matrices of allocator peak, MAGMA 85.4 s at
2.0, host LAPACK 253.6 s at 2.0, largest eigenvalue difference 7e-15. On a CPU
tensor every backend is the same LAPACK call.

``torch.linalg.eigh`` factorises a working copy of its input. When the caller
can rebuild the kernel (``rebuild``), :func:`kernel_eigh` instead lets the
eigensolver overwrite the kernel's own storage with the eigenvectors, which
removes that copy from the peak at no cost in speed or accuracy: the same
routine runs on the same values. The caller then rebuilds the kernel, a
single matrix product, for the rest of the fit.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

EIGH_BACKENDS: Tuple[str, ...] = ("auto", "cusolver", "magma", "cpu")

EighResult = Tuple[torch.Tensor, torch.Tensor]


def check_eigh_backend(backend: str) -> str:
    """Return ``backend`` normalised to lower case, or raise ``ValueError``."""
    name = str(backend).lower()
    if name not in EIGH_BACKENDS:
        raise ValueError(
            f"eigh_backend must be one of {', '.join(EIGH_BACKENDS)}; got {backend!r}."
        )
    return name


def has_magma() -> bool:
    """Whether this PyTorch build can run the MAGMA eigensolver."""
    return bool(getattr(torch.cuda, "has_magma", False))


def low_memory_backend() -> str:
    """``"magma"`` when available in this PyTorch build, otherwise ``"cpu"``."""
    return "magma" if has_magma() else "cpu"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _eigh_with_library(K: torch.Tensor, library: str) -> EighResult:
    """``torch.linalg.eigh`` on a CUDA tensor with the given GPU library.

    PyTorch selects the GPU linear-algebra library through a process-wide
    preference; it is set for this call only and restored afterwards.
    """
    previous = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library(library)
    try:
        return torch.linalg.eigh(K)
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)


def _eigh_into(K: torch.Tensor) -> EighResult:
    """``torch.linalg.eigh`` writing the eigenvectors into ``K``'s own storage.

    ``K`` is symmetric, so its transpose ``K.mT`` is the same matrix in the
    column-major layout the eigensolvers work in; passed as both input and
    eigenvector output it is factorised in place, without the working copy.
    The eigenvectors come back as that column-major view of ``K``'s storage,
    the layout ``torch.linalg.eigh`` returns them in.
    """
    V = K.mT
    w = torch.empty(K.shape[-1], dtype=K.dtype, device=K.device)
    torch.linalg.eigh(V, out=(w, V))
    return w, V


def _eigh_with_library_into(K: torch.Tensor, library: str) -> EighResult:
    """:func:`_eigh_into` on a CUDA tensor with the given GPU library."""
    previous = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library(library)
    try:
        return _eigh_into(K)
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)


def _eigh_on_host(K: torch.Tensor, overwrite: bool = False) -> EighResult:
    """Factorise a device tensor with host LAPACK; return results on its device.

    With ``overwrite`` the eigenvectors are copied into ``K``'s storage, in the
    same column-major layout, instead of into a new device tensor.
    """
    K_host = K.detach().cpu()
    w, U = torch.linalg.eigh(K_host)
    del K_host
    if not overwrite:
        return w.to(K.device), U.to(K.device)
    V = K.mT
    V.copy_(U)
    return w.to(K.device), V


def _eigh_once(K: torch.Tensor, backend: str, overwrite: bool = False) -> EighResult:
    """One factorisation of a CUDA tensor with an explicit backend, no retry."""
    if backend in ("cusolver", "magma"):
        if overwrite:
            return _eigh_with_library_into(K, backend)
        return _eigh_with_library(K, backend)
    if backend == "cpu":
        return _eigh_on_host(K, overwrite)
    raise ValueError(f"no single backend called {backend!r}")


def _attempt_order(backend: str) -> List[str]:
    """Backends to try, in order, for a kernel on a CUDA device."""
    if backend == "auto":
        return ["cusolver", "magma", "cpu"] if has_magma() else ["cusolver", "cpu"]
    if backend == "magma" and not has_magma():
        return ["cpu"]
    return [backend]


def _release_cached_memory() -> None:
    """Hand the allocator's cached blocks back to the driver before a retry.

    MAGMA allocates part of its workspace with its own ``cudaMalloc`` calls,
    which cannot use memory PyTorch's caching allocator is holding on to.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_attempts(
    K: torch.Tensor,
    order: List[str],
    rebuild: Optional[Callable[[], torch.Tensor]] = None,
) -> Tuple[EighResult, str, List[str]]:
    """Try each backend in turn; return the result, its backend and the failures.

    A backend is abandoned when the device runs out of memory, and MAGMA also
    when it reports any other error (a failed device allocation reaches Python
    as a plain ``RuntimeError`` carrying MAGMA's error code). Anything else,
    and any failure of the last backend, propagates. With ``rebuild`` each
    attempt factorises ``K`` in place, and ``K`` is restored from ``rebuild``
    before the next attempt, so a failed attempt can never leave a partly
    overwritten kernel behind.
    """
    overwrite = rebuild is not None
    failed: List[str] = []
    for i, name in enumerate(order):
        last = i == len(order) - 1
        try:
            return _eigh_once(K, name, overwrite), name, failed
        except torch.cuda.OutOfMemoryError:
            if last:
                raise
        except RuntimeError:
            if last or name != "magma":
                raise
        failed.append(name)
        _release_cached_memory()
        if overwrite:
            K.copy_(rebuild())
    raise RuntimeError("no eigendecomposition backend to try")  # pragma: no cover


def kernel_eigh(
    K: torch.Tensor,
    backend: str = "auto",
    *,
    return_info: bool = False,
    rebuild: Optional[Callable[[], torch.Tensor]] = None,
) -> Union[EighResult, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    """Eigendecomposition ``K = U diag(w) U^T`` of a symmetric kernel matrix.

    Parameters
    ----------
    K : torch.Tensor
        Symmetric ``n x n`` matrix; results are returned on its device.
    backend : {"auto", "cusolver", "magma", "cpu"}, default="auto"
        Where the factorisation runs when ``K`` is on a CUDA device; see the
        module docstring. Ignored for a CPU tensor.
    rebuild : callable, optional
        Returns the kernel matrix again, with the same values. When given,
        ``K`` must be symmetric and its storage is overwritten with the
        eigenvectors (the returned ``U`` is a view of it), which saves the
        working copy ``torch.linalg.eigh`` would make; the caller rebuilds
        ``K`` afterwards if it still needs it.
    return_info : bool, default=False
        Also return a dict with the backend that produced the result
        (``"used"``), the backends that failed before it, normally by
        running out of device memory (``"failed"``), and the wall-clock
        seconds of the factorisation including any failed attempt
        (``"seconds"``).

    Returns
    -------
    (w, U) or (w, U, info)
        Eigenvalues in ascending order and the matching eigenvectors, as
        ``torch.linalg.eigh`` returns them.
    """
    requested = check_eigh_backend(backend)
    device = K.device
    failed: List[str] = []
    _synchronize(device)
    t0 = time.perf_counter()
    if device.type != "cuda":
        w, U = _eigh_into(K) if rebuild is not None else torch.linalg.eigh(K)
        used = "lapack"
    else:
        (w, U), used, failed = _run_attempts(K, _attempt_order(requested), rebuild)
    _synchronize(device)
    seconds = time.perf_counter() - t0
    if not return_info:
        return w, U
    info = {
        "requested": requested,
        "used": used,
        "failed": failed,
        "seconds": seconds,
        "in_place": rebuild is not None,
    }
    return w, U, info
