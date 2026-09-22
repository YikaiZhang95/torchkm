# Response to the JMLR MLOSS decision letter (draft)

Point-by-point, in the letter's order. Each item is marked **Answered**
(the answer is known now, from the code or from runs already done),
**Ready to run** (the experiment is implemented on this branch and needs the
GPU workstation), or **Decision** (the authors must choose). Commands for
every run are on `docs/examples/reproduce_paper_benchmarks.md`.

---

## 1. The comparison omits the GPU kernel libraries a reader would weigh TorchKM against

**Ready to run.** `benchmarks/q1_full_kernel.py` runs one protocol for
TorchKM, cuML, Falkon, KeOps and EigenPro with every method on the full
kernel (no Nyström centres anywhere): the same RBF bandwidth, the same
50-value λ grid from 1e-2 down to 2e-5, identical stratified 5-fold splits, float64 throughout,
end-to-end timing of the cross-validation sweep plus the final fit and the
test predictions, peak GPU memory from NVML, and test accuracy. Three seeds
per dataset give standard errors. Datasets: the paper's a7a, a8a, a9a, w7a,
MNIST 3v8 and 4v9, ijcnn1 and covtype (30k subsamples), the sizes at which
every full-kernel method fits one 48 GB GPU.

- **cuML `SVC`** is the direct competitor: the same hinge objective, solved
  by SMO, tuned by the same fold loop (251 fits per repeat).
- **Falkon** with M = n centres is exact kernel ridge regression
  (preconditioned conjugate gradient); its loss is squared, which the table
  states.
- **KeOps** is a kernel-operation engine rather than a model-selection
  library; the kernel method run with it is kernel ridge regression solved
  matrix-free by conjugate gradient on a `LazyTensor` kernel, with O(n)
  memory. The text says what KeOps is and what it is not.
- **EigenPro 2** (all training rows as centres, preconditioned SGD) has no
  regularization parameter; its 50-value grid is the number of epochs,
  chosen on the same folds.
- **ThunderSVM** and **scikit-learn** stay as SMO references in the
  exact-suite runs of `bench_gpu_libraries.py`.
- Table 1 gains columns for cuML, Falkon, KeOps and EigenPro and rows for
  exact CV and for the losses beyond hinge (KQR, DWD, logistic).

## 2. The benchmark suite is thinner than it appears

**2a. Adult at three sizes and Web at two are five problems, not eight.**
Answered: agreed. Adult (a1a to a9a) becomes one scaling study, a single
figure of time and memory versus n for every library, with exact mode
running while it fits the envelope. w7a and w8a are kept as the imbalanced
set, not as separate benchmarks.

**2b. Adult accuracy is close to logistic regression.**
Answered: agreed, and the revision makes it visible. Every row of every
table gets tuned linear baselines (logistic regression and LinearSVC on the
same folds and grid), so the reader sees what the kernel buys. Adult is
scaling data only.

**2c. w7a / w8a are 97% negative; report AUC or balanced accuracy.**
Answered: implemented. All tables report accuracy, balanced accuracy and
AUC from the decision scores, and the dataset table lists each class prior.
For w7a, w8a and ijcnn1 the headline metric is AUC.

**2d. MNIST8m 4-vs-6 is a scale test.**
Answered: it is labelled as one. It stays in the "scale" suite for the
Nyström path, where time and memory are the point, and its accuracy column
is not used to support any claim.

**2e. ijcnn1 is the row that exercises the nonlinearity.**
Answered: it becomes central. The "exact" suite is built from problems where
the kernel matters and n fits exact mode: ijcnn1 on a 30k stratified
subsample, MNIST 3-vs-8 and 4-vs-9, covtype on a 30k subsample, and w7a for
imbalance; ijcnn1 at full size runs on the Nyström path. Each row carries the
linear baselines.

**2f. The covtype result needs explanation.**
Answered, with the curve ready to run. The Table 4 covtype cell used Nyström
rank `k = 30` on 2,000 landmarks (`benchmarks/table4_nystrom.py`), while every
other dataset used `k = 300` and the scikit-learn baseline used 300
everywhere. The 0.807 is a property of that configuration, not of the
kernel method: covtype's kernel spectrum decays slowly, so a low-rank
approximation on 2,000 landmarks is intrinsically limited.
`benchmarks/bench_covtype_rank.py` sweeps landmarks in {2k, 5k, 10k, 20k} and
rank in {30, 300, 1000, full}, with Falkon at the same centre counts, and
the exact-mode 30k subsample gives the reference. The paper will show the
accuracy-versus-budget curve with time and memory, and drop the single point.

**2g. The simulation is an isotropic Gaussian mixture, a favourable regime.**
Answered: acknowledged in the text. The simulation moves to the appendix as
a controlled scaling experiment in n and p, with the sentence that its fast
spectral decay favours spectral and Nyström methods, and gains a test
accuracy column. No claim rests on it; the real-data tables carry the case.

## 3. KQR and DWD are not benchmarked

**Ready to run.** These are the capabilities no GPU library offers, and the
revision benchmarks both against the CPU packages their users run today.

- **Kernel quantile regression**: `benchmarks/bench_kqr.py` on cadata
  (California housing, 20,640), abalone, cpusmall and space_ga (exact
  mode), YearPredictionMSD (Nyström), and a synthetic heteroscedastic model
  with known conditional quantiles; τ ∈ {0.1, 0.5, 0.9}; pinball loss,
  empirical coverage, RMSE to the true quantile on the synthetic set, time and
  memory. Baselines: `fastkqr` and `kernlab::kqr` through
  `benchmarks/r/bench_kqr.R` on identical exported splits and folds, and
  scikit-learn's linear `QuantileRegressor` so the kernel's contribution is
  visible.
- **Kernel DWD**: `benchmarks/bench_dwd.py` on gisette (6,000 × 5,000, the
  HDLSS regime DWD was designed for), ijcnn1-30k and MNIST 3-vs-8, with
  `kerndwd` through `benchmarks/r/bench_dwd.R` and TorchKMSVC as the
  in-package reference.
- The R scripts have not run yet (no R on the machine used for this branch);
  the `fastkqr` and `kerndwd` call signatures need checking against the
  installed versions before the first run.

## 4. The operating envelope is never characterised

**Answered in form, GPU numbers pending.** Peak memory in exact mode is
`c · 8 n² + 16 n L` bytes: c resident n×n float64 matrices (kernel,
eigenvectors, eigensolver workspace) plus the n×L path. A CPU sweep of the
same code measured c = 4.2 for n = 2,000 to 8,000; the CUDA eigensolver's
workspace differs, so `benchmarks/bench_memory_envelope.py` on the L40S is
the number to quote. With c = 4 the predicted ceiling on a 48 GB card is
about n = 37,000, which is why Table 3 stopped at 24,692 and Table 4 began at
32,561 on the Nyström path.

Package changes on this branch make the envelope explicit: one of the n×n
copies is no longer materialised (it was only used by the projection step,
and never read at all in two solvers), the kernel is built on the device
instead of the host, every fitted estimator reports `peak_gpu_memory_bytes_`,
`torchkm.exact_mode_memory_estimate` and `torchkm.max_exact_n` predict the
requirement and the largest feasible n, and an exact-mode out-of-memory error
now states the requirement, the device total, the feasible n and the
`low_rank=True` alternative. A user-guide page ("Operating envelope") states
the model and a table per card size.

The new Figure 1 will be time and peak memory versus n, exact mode up to the
first out-of-memory error and the Nyström path beyond it, with the ceilings
for 16, 24, 48 and 80 GB cards marked. A 16 GB consumer card gives a second
measured point.

## 5. Table 2 reports the wrong quantity; temper the language

**Answered.** Table 2 will report test accuracy (with AUC), run time and peak
memory; the script already does. The objective comparison leaves the paper
and becomes a solver-quality check in the documentation, done properly: every
solver minimises the same objective at the same fixed λ on the same kernel
(`benchmarks/bench_solver_quality.py`), instead of at whatever λ each library
selected.

Doing that check exposed a solver issue, which the revision reports rather
than hides. At n = 3,000 with the default tolerances, TorchKM's objective sat
above libsvm's optimum by 33% at λ = 1e-3 and 36% at λ = 1e-4 after a handful
of solver passes, while the stopping rules reported convergence: the KKT
threshold is absolute (its natural scale shrinks like 1/n) and the inner
step tolerance stops the smoothed proximal-gradient iterations early when
the problem is ill-conditioned, which it is at weak regularization. Two
facts bound the consequence. The tuned model's test accuracy was identical
to libsvm's in every cell, so no predictive result depends on this. And
tighter tolerances recover the optimum: `KKTeps=1e-6` alone brings λ ≥ 1e-3
to within 4e-3, and `tol=1e-8` with it brings λ ≥ 1e-4 to within 1% at about
three times the solver passes; λ = 1e-5 is not solved to better than 6%, so
the paper states the range over which solutions are exact to tolerance. The
tolerances are constructor parameters, a scale-aware KKT rule is available,
every benchmark takes them as flags, and the main tables report TorchKM at
both the default and the converged setting with the time cost. The claim
"TorchKM attains the lowest objective values" is withdrawn.

Language: "consistently superior accuracy", "attains the best accuracy" and
"lowest objective values" go. With standard errors on every mean (Tables 3
and 4 had none), differences within two standard errors are described as
equal. The supported claim is equal accuracy at a fraction of the run time,
with the covtype curve as the one case where accuracy genuinely depends on
the budget.

## On the appendix

**Answered; edits pending.** A.1 to A.4 shrink to one paragraph stating the
two ideas and citing Wang and Zou (2022) for the exact-CV lemma and finite
smoothing and Tang et al. (2026) for KQR. A.5 (Nyström design) moves to the
Nyström user-guide page, A.6 (Platt scaling and Figure 2) to the calibration
page, and B.1 to the reproduction page, which has been rewritten around the
scripts that exist and is generated from the archived results. The appendix
keeps only the protocol summary and any table that does not fit.

**Decision.** The submitted manuscript's appendix on multiclass handling
describes behaviour the package does not have: the estimators accept two
classes and the README lists a one-vs-rest wrapper as a first contribution.
Either drop the section, or add a short user-guide page showing
`OneVsRestClassifier(TorchKMSVC(...))`, which works because the estimators
are scikit-learn compatible, with a test. The section cannot stay as is.

## Further points

- **"Continue to define the state-of-the-art in many fields."** Softened to
  "remain competitive on small-to-mid-size tabular and structured data, and
  the method of choice for specific tasks", with the concrete example drawn
  from the revision's own tables: the kernel-versus-linear gap on ijcnn1 and
  the MNIST pairs, and DWD in the HDLSS regime.
- **The agentic / automated-ML passage.** Removed from the paper.
- **The stale reproduction page.** Rewritten; the reference to a script that
  did not exist is gone; the page now lists every script and command and is
  regenerated from the results archive.

## What the resubmission will contain, against the letter's list

| # | Requested | Where it comes from | Status |
|---|---|---|---|
| 1 | cuML and Falkon with accuracy, time, peak memory | `bench_gpu_libraries.py`, exact and Nyström suites | ready to run |
| 2 | One KQR and one DWD benchmark | `bench_kqr.py` + `r/bench_kqr.R`; `bench_dwd.py` + `r/bench_dwd.R` | ready to run; R untested |
| 3 | covtype accounted for | rank-30 configuration (answered); `bench_covtype_rank.py` curve | answered; curve ready to run |
| 4 | AUC / balanced accuracy under imbalance | harness metrics on every table | done |
| 5 | Kernel-natural problem in the exact range | ijcnn1-30k, MNIST pairs, covtype-30k, each with linear baselines | ready to run |
| 6 | Memory envelope | `bench_memory_envelope.py`, `peak_gpu_memory_bytes_`, envelope page | model and helpers done; GPU sweep pending |
| 7 | Reduced appendix, derivations cited, behaviour documented online | docs pages done; manuscript edits pending | pending |

## Decisions for the authors

1. **KKT stopping rule.** Keep the default and report runs at `KKTeps=1e-6`,
   or make the rule scale-aware (compare `n · Σ KKT²`), which changes every
   solver's stopping behaviour and the paper's timings. Recommendation:
   scale-aware rule in the release that accompanies the resubmission, with
   the trade-off measured by the solver-quality script.
2. **Multiclass appendix.** Drop, or document the one-vs-rest wrapper (see
   above).
3. **Neural-network column.** The 1D CNN on tabular features in Tables 3 and
   4 is a weak baseline that invites the next round of criticism.
   Recommendation: drop it in favour of the linear baselines; if a neural
   baseline is wanted, a tuned MLP in the documentation.
