# Device and Precision

TorchKM uses PyTorch internally, so computation can run on either CPU or GPU.

## Choosing a device

A portable way to select the device is:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
```

Then pass it to the estimator:

```python
clf = TorchKMSVC(kernel="rbf", Cs=Cs, nC=len(Cs), cv=5, device=device)
```

## CPU behavior

CPU execution is useful for:

- small examples;
- unit tests;
- debugging;
- systems without CUDA.

## GPU behavior

GPU execution is useful for:

- larger kernel matrices;
- repeated matrix-vector operations;
- pathwise model selection;
- Nyström approximation workflows.

## Practical tips

- Avoid unnecessary CPU/GPU transfers inside custom code.
- Keep input arrays in a consistent dtype.
- Use smaller examples for tests.
- Use fixed random seeds for reproducibility.
- Report CUDA and PyTorch versions in performance benchmarks.

## Troubleshooting

If CUDA is unavailable, TorchKM should still be able to run with `device="cpu"`. If you expected CUDA to be available, check:

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```
