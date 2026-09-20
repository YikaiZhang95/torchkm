# Experiment design for the JMLR MLOSS revision

One experiment per question in the decision letter, with the exact settings,
the command that produces it, where the result goes in the paper, and what
outcome would and would not support the claim. Every command is in
`benchmarks/run_campaign.sh`, which runs the whole campaign in this order and
archives the results; the experiment ids (E1 to E10) are the ids in that
script.

## 0. Protocol shared by every experiment

| Element | Setting | Why |
|---|---|---|
| Hardware | NVIDIA L40S 48 GB, AMD EPYC 9334, 768 GB RAM; the paper's machine | Numbers comparable with the submitted tables |
| Software | `benchmarks/environment/environment.yml`; versions recorded in every JSON | Reproducibility |
| Kernel | RBF, one bandwidth per repeat from `sigest` on the training features; TorchKM uses exp(−2σd²), the libsvm-style baselines `gamma = 2σ`, Falkon the equivalent Gaussian width | Every library sees the same kernel |
| Regularization grid | 50 log-uniform C in [1e−3, 1e3]; λ = 1/(2nC) for TorchKM and Falkon | The paper's grid |
| Model selection | 10-fold stratified CV, folds seeded per repeat and identical across libraries (`_common.make_folds`) | Paired comparisons |
| Repeats | 10 for TorchKM and the linear baselines; 3 for libraries whose grid sweep costs hours; 5 for KQR and DWD; 3 for sweeps. Seeds 52, 53, … | Standard errors without weeks of GPU time |
| Timing | End to end: kernel/feature construction + full CV sweep + final refit; CUDA warmed up and synchronised | The user's cost of a tuned model |
| Memory | NVML process peak (comparable across libraries) and the PyTorch allocator peak (`peak_gpu_memory_bytes_`) | Reviewer item 6 |
| Metrics | Accuracy, balanced accuracy, AUC (classification); pinball loss and coverage (quantile regression); mean ± SE | Reviewer item 4 |
| Time cap | 2 h per library × dataset × repeat for libraries without integrated tuning (4 h on the million-row sets); a capped run is reported with the fraction of the grid completed | Bounded campaign |
| TorchKM solver | Main tables: `tol=1e-5`, `max_iter=100000`, `is_exact=0`, `KKTeps=1e-6` (`$KKT`). Converged-solution setting for E1e and E10d: `tol=1e-8`, `KKTeps=1e-6` (`$TIGHT`). Both are reported; see E10 | Accuracy is the same under both (measured on CPU); the tight setting costs about 3× the solver passes |
| Nyström defaults | 2,000 landmarks, rank 300, seeded per repeat | The paper's setting, now with varying landmarks across repeats |
| Statistics | Paired differences on the shared folds; two methods are "equal" when the paired difference is within 2 SE; every table shows SE | Reviewer item 5, language |

Estimated GPU time for the whole campaign: 3 to 5 days, dominated by the
cuML/ThunderSVM/Falkon grid sweeps. Steps are independent JSON files, so the
campaign can be split across sessions and re-run per step.

---

## Q1. Compare against the GPU kernel libraries (cuML, Falkon; KeOps, EigenPro)

**Claim to support.** On problems that fit exact mode, TorchKM returns a tuned
kernel SVM with the same test accuracy as cuML/ThunderSVM/libsvm at a small
fraction of the tuning time; on the Nyström path it is competitive with Falkon
at equal centres, and Falkon at 5 to 10× the centres shows what a larger
budget buys.

**Experiments.**

| Id | Suite | Libraries | Repeats | Cap | Command |
|---|---|---|---|---|---|
| E1a | exact (ijcnn1-30k, MNIST 3v8, MNIST 4v9, covtype-30k, w7a) | TorchKM exact, logistic regression, LinearSVC | 10 | — | `bench_gpu_libraries.py --suite exact --libraries torchkm linear --kkt-eps 1e-6 --repeats 10` |
| E1b | exact | cuML SVC, ThunderSVM (float32 inputs, their native precision) | 3 | 2 h | `--libraries cuml_svc thundersvm --repeats 3 --time-cap 7200 --float32-baselines` |
| E1c | exact | cuML SVC, budget-matched 10-value grid | 3 | — | `--libraries cuml_svc --grid-size 10` |
| E1d | exact | scikit-learn SVC (CPU reference) | 1 | 2 h | `--libraries sklearn_svc --repeats 1 --time-cap 7200` |
| E1e | exact | TorchKM at the converged-solution setting (`--kkt-eps 1e-6 --tol 1e-8`), same folds and seeds as E1a | 10 | — | `--libraries torchkm --kkt-eps 1e-6 --tol 1e-8 --repeats 10` |
| E3b | imbalanced (w8a, ijcnn1 full) | Falkon at M = 2,000 and 10,000 | 3 | 2 h | `--suite imbalanced --libraries falkon --falkon-centers 2000 10000` |
| E4b | scale (covtype, MNIST8m 4v6) | Falkon at M = 2,000, 10,000, 20,000 | 3 | 4 h | `--suite scale --libraries falkon --falkon-centers 2000 10000 20000 --time-cap 14400` |

**Design notes.**
- cuML runs the identical fold × grid loop (500 fits per repeat). E1c repeats
  it on a 10-value grid so the table shows both the like-for-like number and
  what a practitioner would actually pay.
- Falkon is a squared-loss (KRR) classifier; the table says so, and its λ grid
  is the same λ = 1/(2nC) because its objective has the same
  (loss + λ·penalty) form.
- KeOps: related-work paragraph, no run. It is a kernel-operation engine, not
  a model-selection library, and TorchKM materialises K by design (it
  eigendecomposes it). Future work: a KeOps/Falkon-style backend for building
  the Nyström features.
- EigenPro: cited. A runner (`_libraries.py`, same signature as `run_falkon`)
  is a small addition if the reviewers ask for a third GPU KRR solver.

**Paper.** Table 2 (exact suite): rows = datasets with n, p, class prior;
columns = linear baseline, scikit-learn, ThunderSVM, cuML, TorchKM; cells =
accuracy or AUC ± SE, time ± SE, peak memory. Table 3 (Nyström suite): Falkon
at each M, TorchKM at 2,000 and 5,000 landmarks. Table 1 gains cuML, Falkon
and KeOps columns and rows for exact CV and for KQR/DWD.

**Expected outcome.** Accuracy equal within 2 SE across the SMO solvers (same
kernel, same grid); TorchKM time 5 to 50× lower than cuML on the full grid.
If cuML on the full grid is within 2× of TorchKM on some dataset, that is
reported as such; the integrated-CV advantage is what is being measured.

---

## Q2. The benchmark suite

### Q2a. Adult and Web as scaling studies, not eight benchmarks

| Id | Suite | Libraries | Repeats | Command |
|---|---|---|---|---|
| E2a | scaling (a1a 1,605; a3a 3,185; a5a 6,414; a7a 16,100; a8a 22,696; a9a 32,561) | TorchKM exact (skipped automatically above the envelope), TorchKM Nyström, linear | 10 | `--suite scaling --libraries torchkm torchkm_nystrom linear --kkt-eps 1e-6 --repeats 10` |
| E2b | scaling | scikit-learn, cuML, ThunderSVM | 3 | `--suite scaling --libraries sklearn_svc cuml_svc thundersvm --repeats 3 --time-cap 7200 --float32-baselines` |

**Paper.** Figure 2: end-to-end time versus n, one line per library
(`make_figures.py`, `fig2_scaling`). Adult appears nowhere else; w7a and w8a
are in the imbalanced set only.

### Q2b. Adult accuracy is close to logistic regression

Every row of every table carries the tuned linear baselines (E1a, E2a, E3a,
E4c). The text states the kernel-versus-linear gap per dataset; on Adult it
is expected to be small, on ijcnn1 and the MNIST pairs several points, on
covtype large. No claim rests on Adult accuracy.

### Q2c. AUC and balanced accuracy where accuracy is uninformative

| Id | Suite | Libraries | Repeats | Command |
|---|---|---|---|---|
| E3a | imbalanced (w8a 3% positive, ijcnn1 10% positive) | TorchKM Nyström (2,000 and 5,000 landmarks), linear | 10 | `--suite imbalanced --libraries torchkm_nystrom linear --landmarks 2000 5000 --kkt-eps 1e-6 --repeats 10` |

The harness computes accuracy, balanced accuracy and AUC for every record; the
dataset table lists the class prior; AUC is the headline for w7a, w8a and
ijcnn1, accuracy for the balanced sets.

### Q2d. MNIST8m 4-vs-6 is a scale test

| Id | Suite | Libraries | Repeats | Command |
|---|---|---|---|---|
| E4a | scale (covtype 581k, MNIST8m 4v6 1.27M) | TorchKM Nyström (2,000 and 5,000 landmarks) | 5 | `--suite scale --libraries torchkm_nystrom --landmarks 2000 5000 --repeats 5` |
| E4c | scale | linear | 3 | `--suite scale --libraries linear --repeats 3 --time-cap 7200` |

Reported as time and memory versus n with accuracy/AUC as a sanity column,
labelled a scale test.

### Q2e. A problem where the kernel is the natural choice and n fits exact mode

Covered by E1a: ijcnn1 (30k stratified subsample of the 49,990 training rows,
tested on the full 91,701 test rows), MNIST 3-vs-8 and 4-vs-9 (≈12k each,
tested on the MNIST test digits), covtype (30k stratified subsample, 20k
test subsample), each next to its linear baseline. The expected gap
(kernel − linear) is the concrete example the introduction needs.

### Q2f. The covtype result

**Answer already known.** `table4_nystrom.py` used rank 30 on 2,000 landmarks
for covtype while every other dataset used rank 300; 0.807 is that
configuration.

| Id | Setting | Repeats | Command |
|---|---|---|---|
| E5 | covtype full (465k train / 116k test), landmarks {2,000; 5,000; 10,000} × rank {30; 300; 1,000; = landmarks}; Falkon at the same centres | 3 | `bench_covtype_rank.py --landmarks 2000 5000 10000 --ranks 30 300 1000 full --with-falkon --repeats 3` |

20,000 landmarks is excluded on purpose: the n × m float32 landmark kernel
block would be 37 GB at n = 465k, above what the card leaves free; that
limit is itself part of the envelope statement. The exact-mode reference is
the covtype-30k row of E1a.

**Paper.** Figure 3: accuracy versus landmarks, one line per rank, Falkon
points, the exact-mode reference as a horizontal line, with time and memory
per point in the appendix table. Text: covtype's kernel spectrum decays
slowly; accuracy tracks the budget; the submitted number was a low-budget
point.

### Q2g. The Gaussian-mixture simulation

| Id | Setting | Repeats | Command |
|---|---|---|---|
| E9 | Table 2 cells (n ∈ {10k, 20k}, p ∈ {10, 100, 1000}), matched kernel for the baselines, test accuracy + AUC + peak memory added | 20 | `table2_simulation.py --repeats 20 --matched-kernel --max-iter 100000` |

Moves to the appendix as a controlled scaling experiment with the sentence
that the mixture's fast spectral decay favours spectral and Nyström methods.

---

## Q3. Kernel quantile regression and DWD

**Claim to support.** GPU-accelerated KQR and DWD with exact cross-validation
along the path do not exist elsewhere; TorchKM matches the reference CPU
implementations' predictive quality and tunes in a small fraction of their
time.

### KQR (E6, E6b)

| Element | Setting |
|---|---|
| Datasets | synthetic heteroscedastic (n = 10,000, p = 5, known conditional quantiles); cadata (20,640 × 8); abalone (4,177 × 8); cpusmall (8,192 × 12); space_ga (3,107 × 6); YearPredictionMSD (463,715 × 90, Nyström path only) |
| Splits | 80/20 per repeat; features standardised on the training split; targets centred and scaled by the training SD for the solvers, scored in original units |
| Quantiles | τ ∈ {0.1, 0.5, 0.9} |
| Methods | TorchKM KQR exact; TorchKM KQR Nyström (2,000 landmarks, rank 300); linear `QuantileRegressor` (highs); R: `fastkqr` (cv.kqr on the exported folds), `kernlab::kqr` (same fold loop) |
| Metrics | pinball loss on the test split; empirical coverage P(y ≤ q̂) against τ; RMSE to the true quantile on the synthetic set; time; memory |
| Repeats | 5 (3 on YearPredictionMSD) |
| Commands | `bench_kqr.py --datasets synthetic cadata abalone cpusmall space_ga --taus 0.1 0.5 0.9 --repeats 5 --synthetic-n 10000 --export-splits …/kqr_splits`; `Rscript benchmarks/r/bench_kqr.R …/kqr_splits …/kqr_r.csv`; `bench_kqr.py --datasets YearPredictionMSD --methods torchkm_kqr_nystrom linear_qr --repeats 3` |

**Paper.** Table 4: rows = dataset × τ; columns = linear QR, kernlab, fastkqr,
TorchKM exact, TorchKM Nyström; cells = pinball ± SE, coverage, time.

**Expected outcome.** Pinball loss equal to fastkqr within 2 SE (same
algorithm family, same kernel); coverage within ±0.02 of τ for all kernel
methods; linear QR visibly worse on the synthetic and cadata rows; TorchKM
time one to two orders of magnitude below the R packages on n ≥ 8,000.

### DWD (E7)

| Element | Setting |
|---|---|
| Datasets | gisette (6,000 × 5,000 train, 1,000 test; HDLSS); ijcnn1-30k; MNIST 3-vs-8 |
| Methods | TorchKM DWD exact; TorchKM DWD Nyström (2,000 / 300); TorchKM SVM (reference); R: `kerndwd` (cv.kerndwd on the exported folds, qval = 1) |
| Metrics | accuracy, balanced accuracy, AUC, time, memory |
| Repeats | 5 |
| Commands | `bench_dwd.py --datasets gisette ijcnn1_30k mnist_3v8 --repeats 5 --export-splits …/dwd_splits`; `Rscript benchmarks/r/bench_dwd.R …/dwd_splits …/dwd_r.csv` |

**Paper.** Table 5: rows = datasets; columns = kerndwd, TorchKM DWD, TorchKM
SVM; cells = accuracy/AUC ± SE, time.

**Before the first R run.** Neither R script has executed yet (no R on the
machine used for this branch). Check `?fastkqr::cv.kqr`, `?fastkqr::kqr`,
`?kerndwd::cv.kerndwd` for the argument names; the scripts are written to the
documented signatures and print the exact call on failure.

---

## Q4. The operating envelope

**Claim to support.** Peak exact-mode memory is c · 8n² + 16nL bytes with
c measured, so a reader can compute the largest n a card supports; the L40S
ceiling is stated and matches where the tables switch to Nyström; the
Nyström path continues to 10⁶ rows with memory growing as n × landmarks.

| Id | Setting | Repeats | Command |
|---|---|---|---|
| E8 | synthetic (p = 100), exact mode at n ∈ {5k, 10k, 15k, 20k, 25k, 30k, 35k, 40k, 45k, 50k} until the first OOM, paper protocol (50 C values, 10 folds); Nyström at n ∈ {10k, 50k, 100k, 250k, 500k, 1M} with 2,000 landmarks / rank 300 | 3 | `bench_memory_envelope.py --sizes … --nystrom-sizes … --p 100 --repeats 3 --kkt-eps 1e-6` |
| E8b | the same code path for SVM, DWD, logistic and KQR at n ∈ {5k, 10k, 20k} (10 C values, 5 folds) | 1 | `--estimators svm dwd logit kqr --sizes 5000 10000 20000 --skip-nystrom --grid-size 10 --folds 5` |
| E8c (optional) | E8 on a 16 GB consumer card | 1 | same command on that machine |

The script records, per fit, wall time, the allocator peak, the NVML peak,
the predicted peak, and the empirical c = (peak − 16nL)/(8n²); it prints the
median c and the first OOM size. Update `torchkm.memory.EXACT_MODE_COPIES`
to the measured c (the CPU sweep gave 4.2; the constant is 4 until then).

**Paper.** Figure 1: time and peak memory versus n, exact and Nyström, first
OOM marked, 16/24/48/80 GB ceilings drawn (`fig1_envelope`). One sentence in
the text: the formula, c, the L40S ceiling, and that Table 3's 24,692 and
Table 4's 32,561 are on either side of it. The user-guide page carries the
per-card table.

**Expected outcome.** First OOM between n = 35k and 45k on the L40S with
c ≈ 3.5 to 4.5; the E8b curves coincide.

---

## Q5. Table 2's headline quantity and the language

### Table 2 reports accuracy (E9, above)

Test accuracy and AUC replace objective values in the main-text table; time
and peak memory stay. The objective comparison moves to the documentation.

### Solver quality at fixed λ (E10)

| Id | Setting | Repeats | Command |
|---|---|---|---|
| E10a | (n, p) ∈ {(10k, 10), (10k, 100), (10k, 1000), (20k, 100)}, λ ∈ {1e−1, 1e−2, 1e−3, 1e−4, 1e−5}, default `KKTeps=1e-3`, `tol=1e-5`; scikit-learn, ThunderSVM, cuML at C = 1/(2nλ), gamma = 2σ | 3 | `bench_solver_quality.py --sizes … --lambdas … --repeats 3` |
| E10b | TorchKM with `KKTeps=1e-6` | 3 | `--kkt-eps 1e-6 --solvers torchkm` |
| E10c | TorchKM with the scale-aware rule (`kkt_scaled=True`, `KKTeps=1e-3`) | 3 | `--kkt-scaled --solvers torchkm` |
| E10d | TorchKM with `KKTeps=1e-6` and the tight inner step tolerance `tol=1e-8` | 3 | `--kkt-eps 1e-6 --tol 1e-8 --solvers torchkm` |

Every solution is scored with the same objective on the same kernel; the
record holds the objective, the gap to the best solver, solver passes and
time. E1e runs the exact suite at the E10d setting so the main table can
show the accuracy (expected identical) and the time cost of converged
solutions.

**Already measured on CPU** (archived under
`benchmarks/results/20260920-cpu-smoke/`, n = 1,000 and 3,000, p = 100):

| λ | default (`tol=1e-5`, `KKTeps=1e-3`) | `KKTeps=1e-6` | `KKTeps=1e-6`, `tol=1e-8` | test accuracy |
|---|---|---|---|---|
| 1e−2 | gap 2.5%, 2 passes | 6e−4, 11 passes | — | identical to libsvm |
| 1e−3 | 33%, 5 passes | 3.5e−3, 39 passes | 8e−5, 186 passes | identical |
| 1e−4 | 36%, 94 passes | 36% (rule reports convergence) | 6e−3, 904 passes | identical |
| 1e−5 | 123%, 159 passes | 123% | 6%, 3,288 passes | identical |

Two conclusions. First, the tuned model's test accuracy is the same as
libsvm's in every cell, so the predictive claims do not depend on the
tolerance; the gap concerns the phrase "exact solution", not the results.
Second, the finite-smoothing solver loses accuracy as λ shrinks (the smoothed
problems' conditioning scales like 1/λ), and the inner step tolerance `tol`
matters more than the KKT rule there: `tol=1e-8` recovers the optimum to
better than 1% down to λ = 1e−4 at about 3 to 10× the passes, and λ = 1e−5
is not solved to better than 6% by any setting tried. The paper's grid
(λ ∈ [1e−3, 1e3]) stays inside the accurate range; the estimators' default
grid (`C_max=1e3`, so λ down to 5e−8 at n = 10k) does not.

**Paper.** Figure 4: relative gap versus λ for the four settings; a sentence
that all main-table TorchKM runs are reported at both the default and the
converged setting with identical accuracy, and that solutions are exact to
the stated tolerance for λ ≥ 1e−4; the "lowest objective" claim withdrawn.
The docs page "solver quality" carries the full table.

**Decisions the experiment informs.** (1) Whether to change the defaults:
`tol=1e-8` with `KKTeps=1e-6` (or the scale-aware rule) in the release that
accompanies the resubmission, with the time cost from E1e stated. (2) Whether
to narrow the estimators' default `C` grid so its weak-regularization end
stays where the solver is accurate (λ ≥ 1e−4 means `C_max ≈ 1/(2n·1e−4)`),
or to document the range instead.

### Language

Delete "consistently superior accuracy", "attains the best accuracy",
"lowest objective values". Report SE on every mean; describe differences
within 2 SE as equal; the claim is equal accuracy at a fraction of the run
time, with covtype as the case where accuracy depends on the budget.

---

## The appendix

No experiment. A.1 to A.4 become one paragraph citing Wang and Zou (2022)
and Tang et al. (2026). A.5 → `docs/user_guide/nystrom.md`, A.6 →
`docs/user_guide/probability_calibration.md`, B.1 →
`docs/examples/reproduce_paper_benchmarks.md` (all rewritten on this branch).
The multiclass section describes behaviour the package does not have; the
branch adds `docs/user_guide/multiclass.md` showing
`OneVsRestClassifier(TorchKMSVC(...))` with tests, so the section can be
replaced by one sentence and a pointer.

## Further points

- **State-of-the-art claim.** Replace with "remain competitive on
  small-to-mid-size tabular data and the method of choice for specific
  tasks", citing the kernel-versus-linear gaps of E1a as the concrete example.
- **Agentic passage.** Delete.
- **Stale reproduction page.** Rewritten; regenerated from the results
  archive by `make_tables.py`.

## Running the campaign

```bash
DATA=~/libsvm THUNDERSVM=~/thundersvm/python bash benchmarks/run_campaign.sh
# or a subset:
ONLY="E8 E1a E5" DATA=~/libsvm bash benchmarks/run_campaign.sh
```

Outputs land in `benchmarks/results/<UTC timestamp>/` with one JSON and one
log per step, the R CSV rows, Markdown tables from `make_tables.py`, and the
four figures from `make_figures.py`. Commit that directory; the paper quotes
it.
