# Testing

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The default CI path runs CPU-safe tests and skips CUDA-only tests:

```bash
python -m pytest -q -m "not cuda"
```

For a quick environment check, collect tests first:

```bash
python -m pytest --collect-only -q
```

## Coverage

To run tests with coverage:

```bash
python -m pytest -q --cov=torchkm --cov-report=term-missing:skip-covered --cov-report=xml --cov-report=html
```

Current coverage: 66.57% on commit `a1b0489`, measured with Python 3.11.14 on
macOS arm64.

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

CUDA smoke tests are marked with `pytest.mark.cuda` and skip automatically when
CUDA is unavailable. GitHub-hosted runners usually may not have CUDA GPUs, so
run the CUDA smoke tests on a CUDA machine with:

```bash
python -m pytest -q -m cuda
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
