# Operating Envelope

TorchKM's exact mode eigendecomposes the full \(n \times n\) kernel matrix
once and reuses it across every fold and every regularization value. That is
where the speed comes from, and it is also the binding constraint: peak device
memory grows with \(n^2\), so a given GPU supports exact mode only up to some
\(n\). This page states that envelope, shows how to read it off a fitted
model, and says what to do beyond it.

## Memory model

The peak of an exact-mode fit is set by its one eigendecomposition of the
kernel. At that moment the device holds the kernel, the eigenvectors being
computed, and whatever working copy and workspace the eigensolver needs. The
rest of the fit, the whole regularization path and the exact cross-validation,
adds only \(n \times L\) matrices for \(L\) candidate values. The prediction is

\[
\text{peak} \approx c \cdot 8 n^2 + 16\, n L \ \text{bytes},
\]

with \(c\) the number of resident \(n \times n\) float64 copies, which depends
on where the eigendecomposition runs (`eigh_backend`):

| `eigh_backend` | \(c\) (measured) | Factorisation time, 16,100 rows on an L40S |
|---|---|---|
| `"cusolver"` | 6.1 | 21.5 s |
| `"magma"` | 2.0 (PyTorch's allocator; MAGMA adds some device workspace of its own) | 85.4 s |
| `"cpu"` | 2.0 (the factorisation itself runs in host memory) | 253.6 s |

The figures are PyTorch allocator peaks measured on an NVIDIA L40S with
PyTorch 2.6 and CUDA 12.4, for kernels of 16,100 to 22,696 rows; the driver
(NVML) reports about half a copy more for cuSOLVER, which includes the CUDA
context and cached blocks. The three backends return the same eigenpairs to
rounding error (largest eigenvalue difference 7e-15 in that test), so the
choice changes time and memory, never the fitted model.

The default, `eigh_backend="auto"`, runs cuSOLVER and, when the device runs
out of memory, repeats the factorisation with MAGMA and then on the host. A
problem that fits keeps the fast path; a problem that does not fit it is no
longer an out-of-memory error, at several times the factorisation time.
`eigh_backend_` and `eigh_seconds_` on the fitted estimator say which backend
ran and how long it took.

```python
clf = TorchKMSVC(kernel="rbf", cv=10, device="cuda", eigh_backend="magma").fit(X, y)
print(clf.eigh_backend_, clf.eigh_seconds_)
```

The helpers are public:

```python
from torchkm import exact_mode_memory_estimate, max_exact_n

exact_mode_memory_estimate(30_000)                 # cuSOLVER peak in bytes
max_exact_n(48e9)                                  # largest n with cuSOLVER
max_exact_n(48e9, backend="cpu")                   # largest n with the fallback
```

Predicted ceilings, from `max_exact_n` with 10% headroom for the CUDA context:

| Device memory | Largest \(n\), cuSOLVER | Largest \(n\), low-memory eigendecomposition |
|---|---|---|
| 16 GB | ≈ 17,200 | ≈ 30,000 |
| 24 GB | ≈ 21,000 | ≈ 36,700 |
| 48 GB | ≈ 29,800 | ≈ 52,000 |
| 80 GB | ≈ 38,400 | ≈ 67,100 |

`benchmarks/bench_memory_envelope.py` (with `--eigh-backend`) fits exact mode
at increasing \(n\) until the first out-of-memory error and reports the
measured constant for the hardware in use; its numbers supersede these. A fit
that runs entirely on the CPU (LAPACK) peaks at about 4.2 copies of host
memory.

Every exact-mode estimator (`TorchKMSVC`, `TorchKMDWD`, `TorchKMLogit`,
`TorchKMKQR`) shares the same decomposition, so the envelope is the same for
all of them.

## Reading the envelope off a fitted model

After `fit` on a CUDA device, `peak_gpu_memory_bytes_` holds the peak memory
the PyTorch allocator recorded during the whole call: kernel construction, the
solver, and Platt calibration. It is `None` after a CPU fit.

```python
clf = TorchKMSVC(kernel="rbf", cv=5, device="cuda").fit(X, y)
print(clf.peak_gpu_memory_bytes_ / 1e9, "GB")
```

When an exact-mode fit does run out of memory even with the low-memory
eigendecomposition, the `torch.cuda.OutOfMemoryError` TorchKM raises names the
training size, the requirement with each eigendecomposition, the device's
total memory, the largest \(n\) it supports with each, and the alternatives
below.

## Beyond the envelope: the Nyström path

`low_rank=True` replaces the \(n \times n\) kernel with a rank-\(k\) feature
map built from \(m\) landmark rows (`num_landmarks`, `nys_k`). Memory then grows
with \(n \times m\) for the landmark kernel block and \(n \times k\) for the
features, so problems in the hundreds of thousands to millions of rows fit on
one card. The single \(m \times m\) decomposition happens once, outside the
fold and path loops, exactly as in exact mode.

The trade is approximation quality: accuracy rises with \(m\) and \(k\), and a
low rank on a problem with a slowly decaying kernel spectrum leaves accuracy
on the table. `benchmarks/bench_covtype_rank.py` measures that curve. Start
with the defaults, then raise `nys_k` before `num_landmarks` if held-out
accuracy is short of the exact-mode result on a subsample.

## Time

Exact mode pays one \(O(n^3)\) eigendecomposition and then \(O(n^2)\) per
solver iteration for every fold and regularization value. The eigendecomposition
runs in float64, so GPUs with a low float64 rate spend proportionally longer in
it: the L40S computes float64 at 1/64 of its float32 rate, and there the
cuSOLVER factorisation of a 16,100-row kernel took 21.5 s, against 62 s for a
complete tuned fit (10-fold CV, 50 values) of a7a at the same size.
Data-centre cards with full float64 rates (A100, H100) shorten that part. `eigh_seconds_` reports the split for any fit.
