# Nyström Approximation

Kernel methods often require an \(n \times n\) kernel matrix. For large data sets, storing and manipulating the full matrix can become the main computational bottleneck.

TorchKM supports a Nyström approximation for larger problems. This provides a lower-rank representation of the kernel matrix using a subset of landmark points.

## Basic usage

```python
clf = TorchKMSVC(
    kernel="rbf",
    Cs=Cs,
    nC=len(Cs),
    cv=5,
    device=device,
    low_rank=True,
    num_landmarks=40,
    nys_k=20,
    max_iter=40,
    probability=True,
)
clf.fit(Xtr, ytr)
```

## Important parameters

| Parameter | Meaning |
|---|---|
| `low_rank` | Enables the Nyström approximation |
| `num_landmarks` | Number of landmark points used to build the approximation |
| `nys_k` | Rank used in the low-rank representation |
| `device` | CPU or GPU device |
| `kernel` | The high-level low-rank path currently supports RBF-kernel workflows |

## When to use

Use `low_rank=True` when:

- the full kernel matrix is too large for memory;
- training with the exact kernel is too slow;
- an approximate solution is acceptable;
- the data set is large enough that full-kernel methods become impractical.

## Practical advice

Start with a modest number of landmarks, then increase `num_landmarks` and `nys_k` if accuracy is not sufficient. Larger values may improve approximation quality but increase memory use and runtime.

The high-level low-rank classifier path requires raw feature input and does not
support `kernel="precomputed"`.
