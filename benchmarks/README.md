# TorchKM Benchmark Protocol

This directory is for benchmark protocols, scripts, and result files. Benchmark
runs are intentionally separate from the normal test suite and documentation
build because they may take a long time and can depend heavily on hardware.

## What To Report

When adding or reporting a benchmark, include:

- exact command used;
- CPU model;
- GPU model, if used;
- RAM and GPU memory, if relevant;
- operating system;
- Python version;
- PyTorch version;
- CUDA version, if used;
- TorchKM version or commit hash;
- data set;
- sample size and feature dimension;
- number of cross-validation folds;
- regularization grid;
- whether timing includes preprocessing, training, and model selection.

## Timing Caveats

Wall-clock times vary across hardware, software versions, CUDA settings, and
system load. Do not expect identical times on different machines.

Do not compare a cold first run with a warmed run. For GPU benchmarks, CUDA
initialization and first-use overhead can make the first run misleading. Run at
least one warmup iteration before timing. When timing CUDA work, call
`torch.cuda.synchronize()` before starting and after ending the timed region.

Report the number of repeats. Prefer median and IQR, or mean and standard
deviation. Avoid reporting a single timing as a strong benchmark claim.

Benchmark scripts should optionally write JSON or CSV output. Output should
include both results and environment metadata. Do not commit large generated
benchmark outputs.

## Development Guidance

Start with a smoke-sized benchmark while developing a script. Run paper-scale
benchmarks only when the command, output files, and expected runtime are clearly
documented.

## Reproducing Paper Tables 2-4

Minimal, lean scripts for the paper's three benchmark tables live next to this
README. They are TorchKM-focused: the ThunderSVM and 1D-CNN columns from the
paper are omitted (they need a from-source CUDA build / a separate training
loop), and the directly pip-installable scikit-learn baselines are kept.

All three follow the paper protocol (Appendix B.1): a 50-point grid of `C`
values log-uniform over `[1e-3, 1e3]`, 10-fold cross-validation, and end-to-end
wall-clock timing with a CUDA warmup. `--repeats` defaults to a small smoke
value; pass the paper count (50 for Table 2, 10 for Tables 3-4) for full runs.
Use `--device cuda` on a GPU; timings on other hardware will differ.

| Script | Table | Data | Reports |
| --- | --- | --- | --- |
| `table2_simulation.py` | 2 | synthetic (`torchkm.data_gen`) | objective value + time, TorchKM vs scikit-learn SVC |
| `table3_benchmarks.py` | 3 | a7a, a8a, w7a | accuracy + time, TorchKM (exact RBF SVM) |
| `table4_nystrom.py` | 4 | a9a, w8a, ijcnn1, covtype, MNIST8m (4-vs-6) | accuracy + time, TorchKM Nystrom vs scikit-learn Nystrom |

```bash
# Table 2 (no external data needed)
python benchmarks/table2_simulation.py --repeats 50 --device cuda

# Table 3 (LIBSVM files in DATA_DIR, train + ".t" test file each)
python benchmarks/table3_benchmarks.py --data-dir DATA_DIR --repeats 10 --device cuda

# Table 4 (Nystrom; covtype/mnist8m are single files and get a split)
python benchmarks/table4_nystrom.py --data-dir DATA_DIR --datasets a9a w8a ijcnn1 \
    --repeats 10 --device cuda
```

**Data.** Tables 3-4 read standard LIBSVM files from `--data-dir`; download from
the LIBSVM datasets page and decompress any `.bz2`/`.xz` first:
`https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/`. The MNIST8m 4-vs-6
subset is extracted from `mnist8m.scale` at load time.

**Runtime scale.** Table 2 at `n=20000, p=1000` builds a 20k x 20k kernel and
needs a GPU; scikit-learn does not finish within hours at that size (`--skip-sklearn`
to drop it). Table 4 on `covtype` (581k) and `mnist8m` (1.27M) is heavy and is
where the Nystrom approximation matters most. These scripts print results to
stdout and do not write output files.
