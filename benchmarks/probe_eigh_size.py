#!/usr/bin/env python
"""Which sizes cuSOLVER's eigh accepts, whatever the memory.

At n = 60,000 (float32, L40S) torch.linalg.eigh stops in cuSOLVER's
workspace-size query (cusolverDnXsyevd_bufferSize: INVALID_VALUE), before any
allocation. This finds where that starts.

For each n the process may hold 2.5 n x n matrices. The input and eigh's copy
of it fit, the workspace (about 4 more) does not. So a size cuSOLVER accepts
stops at the workspace allocation with "out of memory", and a refused size
stops in the query. Nothing is factorized: each size takes about a second.

Usage:
  python benchmarks/probe_eigh_size.py
  python benchmarks/probe_eigh_size.py --sizes 46000 46500 --dtype float64
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16000, 17000, 20000, 23000, 23500, 32500, 33000, 46000, 46500, 60000],
    )
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    total = torch.cuda.get_device_properties(0).total_memory
    print(
        f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"{torch.cuda.get_device_name()}, {args.dtype}",
        flush=True,
    )
    for n in args.sizes:
        unit = torch.finfo(dtype).bits // 8 * n * n
        if 2.05 * unit > total:
            print(f"n = {n}: not tested, two n x n matrices do not fit", flush=True)
            continue
        torch.cuda.set_per_process_memory_fraction(min(1.0, 2.5 * unit / total))
        K = torch.eye(n, dtype=dtype, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.linalg.eigh(K)
            result = "factorized"
        except torch.cuda.OutOfMemoryError:
            # past the query only if eigh's copy of K was allocated
            copied = torch.cuda.max_memory_allocated() > 1.5 * unit
            result = (
                "accepted (stopped at the workspace allocation)"
                if copied
                else "not tested (out of memory before the query)"
            )
        except torch.linalg.LinAlgError as err:
            result = "refused: " + str(err).split(",")[0]
        print(f"n = {n}: {result}", flush=True)
        del K
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
