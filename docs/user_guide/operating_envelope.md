# Operating Envelope

TorchKM's exact mode eigendecomposes the full \(n \times n\) kernel matrix
once and reuses it across every fold and every regularization value. That is
where the speed comes from, and it is also the binding constraint: peak device
memory grows with \(n^2\), so a given GPU supports exact mode only up to some
\(n\), and on any GPU the eigendecomposition itself stops at
\(n = 32{,}768\). This page states that envelope, shows how to read it off a
fitted model, and says what to do beyond it.

## Memory model

At its peak an exact-mode fit holds a small number of \(n \times n\) float64
matrices on the device: the kernel matrix, the eigenvector matrix, and the
eigensolver's workspace. The regularization path adds two \(n \times L\)
matrices (coefficients and out-of-fold predictions for \(L\) candidate
values). The prediction TorchKM uses is

\[
\text{peak} \approx c \cdot 8 n^2 + 16\, n L \ \text{bytes},
\]

with \(c\) the number of resident \(n \times n\) copies,
`torchkm.memory.EXACT_MODE_COPIES`. `TorchKMSVC(dtype="float32")` stores
every \(n \times n\) matrix in 4 bytes per entry instead of 8, which halves
the peak and raises the largest feasible \(n\) by about \(\sqrt{2}\), up to
the size limit below; pass
`dtype=torch.float32` to `exact_mode_memory_estimate` and `max_exact_n` for
its envelope. The other exact solvers run in float64. The constant is calibrated, not derived:
`benchmarks/bench_memory_envelope.py` fits exact mode at increasing \(n\)
until the first out-of-memory error and reports the empirical \(c\) for the
PyTorch and CUDA build in use. Run it once on your hardware; its measured
ceiling supersedes the predictions below.

The helpers are public:

```python
from torchkm import exact_mode_memory_estimate, max_exact_n

exact_mode_memory_estimate(30_000)          # predicted peak in bytes
max_exact_n(48e9)                           # largest n for a 48 GB card
```

Predicted ceilings for common cards, from `max_exact_n` with the default
constant, 10% headroom for the CUDA context and the eigensolver's size limit
(below). Where the size limit binds, the value memory alone would allow
(`size_limit=None`) is in parentheses:

| Device memory | Largest \(n\), float64 | Largest \(n\), float32 (`TorchKMSVC`) |
|---|---|---|
| 8 GB | ≈ 12,200 | ≈ 17,300 |
| 16 GB | ≈ 17,300 | ≈ 24,500 |
| 24 GB | ≈ 21,200 | ≈ 30,000 |
| 48 GB | ≈ 30,000 | 32,768 (≈ 42,400) |
| 80 GB | 32,768 (≈ 38,700) | 32,768 (≈ 54,800) |

The default constant of 6 is the GPU measurement. On an L40S (PyTorch 2.6,
CUDA 12.4) a fit peaks at 6.01 copies, in float32 and float64 alike: the
kernel matrix, cuSOLVER's copy of it, which becomes the eigenvectors, and a
workspace of 4.01 more. A CPU sweep of the same code path (LAPACK eigensolver)
measured 4.2 copies, so on the CPU the prediction is conservative.

Every exact-mode estimator (`TorchKMSVC`, `TorchKMDWD`, `TorchKMLogit`,
`TorchKMKQR`) shares the same decomposition, so the envelope is the same for
all of them.

## Size limit of the GPU eigensolver

Memory is not the only limit. On CUDA, exact mode eigendecomposes the kernel
matrix with cuSOLVER (through `torch.linalg.eigh`), and cuSOLVER refuses large
sizes outright. With PyTorch 2.6 and CUDA 12.4 on an NVIDIA L40S, it accepts
\(n = 32{,}768\) and refuses \(n = 32{,}769\) in float32; float64 accepts
32,768 and refuses 33,000. The refusal comes from cuSOLVER's workspace-size
query, before anything is allocated, so it applies on every card.

Exact mode on a GPU therefore stops at \(n = 32{,}768\)
(`torchkm.memory.EXACT_MODE_MAX_N_CUDA`), however much memory the card has.
It binds wherever memory would allow more: from 48 GB up in float32 and at
80 GB in float64 (the values in parentheses above). `max_exact_n` caps its
answer at this size; pass `size_limit=None` to count memory alone. Above it, the exact solvers raise a
`torch.linalg.LinAlgError` that names the limit and the alternatives, instead
of cuSOLVER's own message.

Other PyTorch and CUDA builds may differ. `benchmarks/probe_eigh_size.py`
checks a build in seconds, without factorizing anything.

## Reading the envelope off a fitted model

After `fit` on a CUDA device, `peak_gpu_memory_bytes_` holds the peak memory
the PyTorch allocator recorded during the whole call: kernel construction, the
solver, and Platt calibration. It is `None` after a CPU fit.

```python
clf = TorchKMSVC(kernel="rbf", cv=5, device="cuda").fit(X, y)
print(clf.peak_gpu_memory_bytes_ / 1e9, "GB")
```

When an exact-mode fit does run out of memory, the `torch.cuda.OutOfMemoryError`
TorchKM raises names the training size, the predicted requirement, the device's
total memory, the largest \(n\) it supports, and the alternatives below.

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
runs in float64 by default, so GPUs with a low float64 rate (consumer cards and
the L40S) spend proportionally longer in it than data-centre cards; the
envelope script reports the split. `TorchKMSVC(dtype="float32")` runs it at the
single-precision rate.
