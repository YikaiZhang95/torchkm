# Benchmark environment

The comparison libraries need CUDA builds that pip cannot resolve on its own.
This directory pins an environment that has run the full suite; every result
file also records the versions it was produced with (`environment` block).

```bash
conda env create -f benchmarks/environment/environment.yml
conda activate torchkm-bench
pip install -e ".[dev,examples,viz]"
Rscript benchmarks/environment/r-packages.R      # fastkqr, kernlab, kerndwd
```

Notes per library:

- **cuML** comes from the RAPIDS channel; match the `cuda-version` pin to the
  driver on the machine (`nvidia-smi` shows the supported CUDA version).
- **Falkon** is installed from PyPI and compiles its CUDA extensions against
  the installed PyTorch; install PyTorch first and keep the CUDA versions
  aligned.
- **ThunderSVM 0.3.4** is a source build (`cmake` with CUDA); pass the path of
  its `python/` directory to the scripts with `--thundersvm-path`.
- **R packages**: `fastkqr` and `kerndwd` are on CRAN; `kernlab` provides
  `kqr` and `sigest`.

Record `pip freeze`, `conda list` and `sessionInfo()` next to archived
results.
