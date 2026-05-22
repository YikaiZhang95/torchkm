# Testing

Run the test suite from the repository root:

```bash
pytest -q
```

For a quick environment check, collect tests first:

```bash
pytest --collect-only -q
```

## Coverage

To run tests with coverage:

```bash
pytest --cov=torchkm --cov-report=term-missing --cov-report=xml
```

## Test categories

Useful test categories include:

- estimator interface tests;
- low-level solver tests;
- CPU fallback tests;
- GPU tests when CUDA is available;
- input validation tests;
- label convention tests;
- probability calibration tests;
- Nyström approximation tests;
- reproducibility tests with fixed random seeds.

## CUDA tests

Tests that require CUDA should be skipped when CUDA is not available. For example:

```python
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
```

## Small tests are better

Keep unit tests small and fast. Large benchmark tests should live in
`benchmarks/`, not in the normal test suite.

## Documentation build

The documentation build should be lightweight and should not run examples or
benchmarks:

```bash
mkdocs build --strict
```
