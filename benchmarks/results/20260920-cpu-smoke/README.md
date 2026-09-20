# 2026-09-20 CPU runs (no GPU available)

Produced on a 4-vCPU Intel Xeon @ 2.10 GHz cloud container with 15 GB RAM,
PyTorch 2.14 (CPU), scikit-learn 1.9.1, TorchKM at the commit these files
were added in. Not the paper's hardware: the accuracy and objective numbers
are hardware-independent, the times are not, and there is no GPU memory
figure. They are here because two of the results do not depend on the card.

| File | Command | What it shows |
| --- | --- | --- |
| `solver_quality_default_kkteps.json` | `bench_solver_quality.py --device cpu --sizes 3000,10 3000,100 --lambdas 1e-1 1e-2 1e-3 --repeats 2` | With the default `KKTeps=1e-3`, TorchKM's SVM objective at fixed lambda is above libsvm's optimum by up to 33% (n=3000, p=100, lambda=1e-3) and 2.5% at lambda=1e-2; equal to 1e-5 at lambda=1e-1. The solver had declared convergence after 2 to 5 passes: the squared KKT residual scales like 1/n, so the absolute threshold is loose at this n. |
| `solver_quality_kkteps_1e-6.json` | same with `--kkt-eps 1e-6` | Largest remaining relative gap 4e-3; run time unchanged (about 2 s per fit on this CPU). |
| `envelope_cpu.json` | `bench_memory_envelope.py --device cpu --sizes 1000 2000 4000 6000 8000 --nystrom-sizes 8000 16000 --p 100 --folds 3 --grid-size 5 --max-iter 2000 --landmarks 500 --rank 100` | Host RSS grows by 4.2 x 8 n^2 bytes per exact-mode fit (n = 2,000 to 8,000; LAPACK eigensolver). Nyström at n=16,000 stays under 0.9 GB. |

Generate tables with `python benchmarks/make_tables.py benchmarks/results/20260920-cpu-smoke/<file>`.
