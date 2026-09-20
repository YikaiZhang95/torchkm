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

## Solver tolerance probe (CPU, n = 1,000 and 3,000, p = 100, synthetic mixture)

`cvksvm` at a single fixed lambda against libsvm (`SVC`, `tol=1e-6`) on the
same kernel; gap = (TorchKM objective − libsvm objective) / libsvm objective;
"conv" is the solver's own convergence flag; test accuracy of both solutions
on the held-out split. `max_iter=100000`, `is_exact=0` unless stated.

| n | lambda | setting | gap | passes | conv | test acc (TorchKM / libsvm) |
|---|---|---|---|---|---|---|
| 1000 | 1e-3 | tol 1e-5, KKTeps 1e-6 | +5.9e-3 | 52 | no | 0.980 / 0.980 |
| 1000 | 1e-3 | tol 1e-8, KKTeps 1e-6 | +2.4e-4 | 254 | no | 0.980 / 0.980 |
| 1000 | 1e-4 | tol 1e-5, KKTeps 1e-6 | +2.7e-1 | 161 | yes | 0.980 / 0.980 |
| 1000 | 1e-4 | tol 1e-8, KKTeps 1e-6 | +9.4e-3 | 1072 | no | 0.980 / 0.980 |
| 1000 | 1e-5 | tol 1e-5, KKTeps 1e-6 | +1.4e+0 | 158 | yes | 0.980 / 0.980 |
| 1000 | 1e-5 | tol 1e-8, KKTeps 1e-6 | +7.2e-2 | 3512 | yes | 0.980 / 0.980 |
| 3000 | 1e-3 | tol 1e-5, KKTeps 1e-6 | +3.5e-3 | 39 | no | 0.992 / 0.992 |
| 3000 | 1e-3 | tol 1e-8, KKTeps 1e-6 | +8.4e-5 | 186 | no | 0.992 / 0.992 |
| 3000 | 1e-4 | tol 1e-5, KKTeps 1e-6 | +3.6e-1 | 94 | yes | 0.995 / 0.993 |
| 3000 | 1e-4 | tol 1e-8, KKTeps 1e-6 | +6.4e-3 | 904 | no | 0.993 / 0.993 |
| 3000 | 1e-5 | tol 1e-5, KKTeps 1e-6 | +1.2e+0 | 159 | yes | 0.993 / 0.993 |
| 3000 | 1e-5 | tol 1e-8, KKTeps 1e-6 | +6.1e-2 | 3288 | yes | 0.993 / 0.993 |

`KKTeps=1e-8`, `delta_len=12` and `is_exact=1` gave the same gaps as the
`tol=1e-8` rows. Times at n = 3,000: 2.6 s (tol 1e-5) to 8 s (tol 1e-8) at
lambda = 1e-4, 23 s at lambda = 1e-5.
