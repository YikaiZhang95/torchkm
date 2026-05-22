# Installation

TorchKM can be installed from PyPI or from source.

## Basic installation

```bash
pip install torchkm
```

The default installation includes the high-level scikit-learn-style estimator
API used throughout the examples.

## Development installation

For development, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/YikaiZhang95/torchkm.git
cd torchkm
pip install -e ".[dev,examples,viz]"
```

Then run the test suite:

```bash
pytest -q
```

## GPU support

TorchKM uses PyTorch for tensor computation. To run on a GPU, install a PyTorch build that is compatible with your CUDA version. A typical device-selection pattern is:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
```

Then pass the device to the estimator:

```python
clf = TorchKMSVC(kernel="rbf", Cs=Cs, nC=len(Cs), cv=5, device=device)
```

## CPU fallback

TorchKM is designed to run on CPU when CUDA is not available. This is useful for testing, examples, and smaller data sets. For large kernel matrices, GPU execution is usually preferred when available.

## Verifying the installation

```python
import torch
from torchkm.estimators import TorchKMSVC

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("TorchKMSVC:", TorchKMSVC)
```
