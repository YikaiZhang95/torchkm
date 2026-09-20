# JMLR MLOSS revision plan for TorchKM

## Status

Implemented on this branch (CPU-verified; GPU numbers still to be produced
on the L40S):

- Phase 0 harness: `benchmarks/_common.py` (peak memory from the PyTorch
  allocator and NVML, accuracy / balanced accuracy / AUC, JSON results with
  environment snapshots, dataset registry with subsampling, shared folds,
  time-capped CV sweeps), `benchmarks/_libraries.py` (runners for TorchKM,
  scikit-learn, ThunderSVM, cuML, Falkon, linear baselines),
  `benchmarks/make_tables.py`, `benchmarks/environment/`,
  `benchmarks/results/README.md`; stale docs rewritten.
- Phase 1 package changes: `torchkm/memory.py` (estimate, `max_exact_n`, OOM
  message), `peak_gpu_memory_bytes_`, the `eU` copy removed from all six exact
  solvers, kernel built on device with in-place exponentiation, Nyström
  seeding from `random_state` (and an explicit `sigma`), `is_exact` and the
  operating envelope documented; 16 new tests.
- Scripts for Phases 1 to 3: `bench_memory_envelope.py`,
  `bench_gpu_libraries.py`, `bench_covtype_rank.py`, `bench_solver_quality.py`,
  `bench_kqr.py` + `r/bench_kqr.R`, `bench_dwd.py` + `r/bench_dwd.R`;
  `table2_simulation.py` reports accuracy, AUC, memory and JSON. Every Python
  script has a `--smoke` mode that passed on CPU.

First results (CPU only, archived under `benchmarks/results/20260920-cpu-smoke/`):

- **Solver quality.** At fixed lambda on the shared kernel, TorchKM's SVM
  objective with the default `KKTeps=1e-3` sits above libsvm's optimum by
  33% (n=3000, p=100, lambda=1e-3), 2.5% (lambda=1e-2) and 1e-5 (lambda=1e-1).
  The solver had declared convergence after 2 to 5 passes: the KKT rule
  compares an absolute squared norm whose scale shrinks like 1/n. With
  `KKTeps=1e-6` the largest gap is 4e-3 at unchanged run time. `KKTeps` is now
  exposed on the classifiers and `--kkt-eps` on every script; the GPU runs
  should use it and report the trade-off. Decide before resubmission whether
  to make the rule scale-aware (e.g. compare `n * sum(KKT**2)`), which changes
  every solver's stopping behaviour and the paper's timings.
- **Envelope (CPU).** Host memory grows by 4.2 x 8 n^2 bytes per exact-mode
  fit for n = 2,000 to 8,000; `EXACT_MODE_COPIES` is set to 4 until the CUDA
  sweep calibrates it.

Still to do: run the scripts on the GPU workstation (commands on the
reproduction page), run the R scripts (untested here: no R in this
environment; check the `fastkqr` / `kerndwd` argument names against the
installed versions), calibrate `EXACT_MODE_COPIES` from the CUDA envelope
sweep, decide on the KKT rule, the float32 stretch goal, and Phase 4
(manuscript and response letter).

Working plan for answering the JMLR MLOSS decision letter (`rev.txt`). The
letter accepts the engineering and returns the manuscript on the empirical
case. This document maps every point in the letter to concrete work in the
package, the benchmark suite, the documentation, and the paper, in the order
the work should be done.

The plan is written against `paper.pdf` in this repository (13 pages: 4 main,
9 appendix). The LaTeX source is not in the repository. Two things the letter
mentions, the "agentic systems" passage in Section 1 and an appendix section on
multiclass handling (the letter's A.5 to A.7), are not in `paper.pdf`, so the
submitted manuscript is a later revision. First step of the paper work is to
check the LaTeX source into `paper/` so the revision is tracked with the code.

---

## 0. What the repository already tells us

These are findings from reading the code and scripts. Several of them are the
direct answer to a reviewer point and should be stated plainly in the response.

| Finding | Where | Bears on |
| --- | --- | --- |
| The covtype Nyström run uses **rank `k=30`** with 2000 landmarks, versus `k=300` for every other dataset (the scikit-learn baseline also used 300). | `benchmarks/table4_nystrom.py`, `DATASETS["covtype"]` | Point 2, resubmission item 3. The 0.807 is a rank-30 approximation, not a kernel-method result. |
| Exact mode keeps three n×n float64 matrices on the GPU (`Kmat`, `Umat` from `eigh`, and `eU = (einv * Umat).T`) plus the cuSOLVER `eigh` workspace, which is itself O(n²). Peak is therefore about 4 to 5 × 8n² bytes: ≈24 GB at n=24,692 and ≈42 GB at n=32,561. | `torchkm/cvksvm.py` `fit()`; `eU` is built in all six exact solvers (`cvksvm`, `cvkdwd`, `cvklogit`, `cvkqr`, `cvkhuber`, `cvksqsvm`), used only by the `is_exact=1` projection in `cvksvm` and `cvkqr`, and never read in `cvkdwd` and `cvklogit` | Point 4, item 6. On the 48 GB L40S the exact-mode ceiling is n≈33k, which is exactly why Table 3 stops at 24,692 and Table 4 starts at 32,561. Computing the projection on the fly instead of storing `eU` would move the ceiling to n≈40k+. |
| The kernel matrix is built on the **CPU** in float64 and then moved to the device, so host RAM also needs 8n² bytes and the build is not on the GPU. | `torchkm/estimators.py` `_compute_K_train` | Point 4, and timing fairness for the new GPU comparisons. |
| Table 3 timings exclude the kernel build for TorchKM ("kernel build is outside the timed region, as in the notebook") but include everything for ThunderSVM. | `benchmarks/table3_benchmarks.py` docstring | Must be end-to-end for every library in the revision. |
| Table 2's baseline objective is evaluated at `ulam[-1]` (λ = 1e-3, the notebook's leftover loop variable), not at the baseline's selected λ or at a common λ, while TorchKM's objective uses its own λ*. The paper says all values are "evaluated at the same tuning parameter". | `benchmarks/table2_simulation.py`, `--baseline-lambda` | Point 5. The objective comparison needs a clean protocol before it can be kept even in the docs. |
| Tables 3 and 4 report no standard errors although they are averages of 10 runs. | `paper.pdf` Tables 3, 4 | Point 5 (language). Needed to justify "within noise". |
| The Nyström classifier backends call `torch.manual_seed(0)` inside `fit()`, so landmark sampling is identical across the "10 independent runs" and the global RNG is reset as a side effect. `cvknyqr` already does this correctly with a `random_state` generator. | `torchkm/cvknyssvm.py:125`, `cvknysdwd.py:125`, `cvknyslogit.py:125` | Point 2 (variance is understated) and a package fix. |
| `is_exact` defaults to 0 in the estimators and in all benchmark scripts. The flag controls a final projection onto the elbow set and which CV loop runs; both settings use the exact-CV formula. The name invites the misreading that the benchmarks ran an approximate mode. | `torchkm/cvksvm.py`, `estimators.py` | Documentation: define the term "exact" once in the user guide. |
| No multiclass support exists: the estimators raise on more than two classes and the README lists a one-vs-rest wrapper as a "good first contribution". | `torchkm/estimators.py` `_check_binary_y` | If the submitted manuscript has a multiclass appendix, it describes behaviour the package does not have. |
| The benchmark scripts print to stdout only; the JSON/CSV output with environment metadata that `benchmarks/README.md` and `CONTRIBUTING.md` ask for does not exist. No peak-memory measurement anywhere. | `benchmarks/*.py` | Items 1, 6. |
| Stale documentation: `docs/examples/reproduce_paper_benchmarks.md` ("If benchmark scripts are added…"), `docs/developer/benchmarking.md` and `docs/developer/architecture.md` ("when added, benchmark scripts"), and `benchmarks/README.md` references a `run_table3.sh` that does not exist. | docs | Further point 3. |
| The neural-network column in Tables 3 and 4 is a "lightweight 1D CNN" on tabular features. The letter does not mention it, but it is a weak baseline that invites the next reviewer's criticism, and it occupies space better spent on a linear baseline that shows what the kernel buys. | `paper.pdf` B.2 | Point 2 (Adult accuracies near logistic regression). |

---

## 1. Phase 0: benchmark infrastructure (do first, everything else depends on it)

Goal: every benchmark script reports the same metrics in the same machine-
readable form, so the new tables can be regenerated from archived JSON.

1. **Shared harness** in `benchmarks/_common.py`:
   - `--out results/<name>.json` on every script; each record carries the
     environment snapshot (GPU, driver, CUDA, torch, torchkm version and
     commit, cuML/Falkon/ThunderSVM versions, CPU, RAM), dataset name, n, p,
     class prior, protocol (folds, grid), repeats, and per-run metrics.
   - **Peak GPU memory** two ways: `torch.cuda.reset_peak_memory_stats()` +
     `max_memory_allocated()` / `max_memory_reserved()` for TorchKM, and an
     NVML sampler thread (`pynvml`, ~50 ms) for process-level peak so that
     cuML, Falkon and ThunderSVM are measured on the same basis. Report the
     NVML number for all libraries and the torch number as a cross-check.
   - **Metrics**: test accuracy, **balanced accuracy**, **AUC** (from
     `decision_function`), wall-clock time (mean and SE over repeats),
     peak memory. Every table in the paper gets accuracy plus AUC or balanced
     accuracy; the script computes all three every time.
   - Timing is **end to end for every library**: kernel or feature
     construction, CV over the full grid, and the final refit are all inside
     the timed region. Keep the CUDA warmup and `synchronize()`.
   - Standard errors on every reported mean; repeats default to the paper
     count when `--paper` is passed.
2. **Environment specification** under `benchmarks/environment/`: a conda
   `environment.yml` (RAPIDS `cuml-cu12`, `falkon`, `thundersvm` build recipe,
   `pynvml`), an R package list for `fastkqr`, `kernlab`, `kerndwd` with
   versions, and a short README. Record the resolved versions in every JSON.
3. **Results archive** `benchmarks/results/<UTC-timestamp>/` with the JSON
   files, a `runner-snapshot.txt` like `benchmarks/cuda-runs/`, and an
   `INDEX.md`. The paper tables are generated from this directory by a small
   `benchmarks/make_tables.py` (LaTeX and Markdown output) so the manuscript,
   the docs page and the archive cannot drift.
4. **Fix the documentation debt now** (further point 3): rewrite
   `docs/examples/reproduce_paper_benchmarks.md` around the scripts that
   exist, update `docs/developer/benchmarking.md` and
   `docs/developer/architecture.md`, and either add `benchmarks/run_table3.sh`
   or delete the reference in `benchmarks/README.md`.

Effort: 2 to 3 days.

---

## 2. Phase 1: memory envelope and the package changes it motivates (item 6)

### 2.1 Measure

`benchmarks/bench_memory_envelope.py`:

- Synthetic data (`data_gen`, p=100) at n ∈ {5k, 10k, 15k, 20k, 25k, 30k,
  35k, 40k, 50k} in exact mode until the first CUDA OOM, then Nyström at the
  same n and beyond (100k, 250k, 500k, 1M). Record time, `max_memory_allocated`,
  `max_memory_reserved`, NVML peak, and host RSS.
- Repeat the exact-mode sweep for `TorchKMDWD`, `TorchKMLogit`, `TorchKMKQR`
  (same eigendecomposition, so the curve should be the same; confirm).
- Output feeds **a new Figure 1 in the paper**: time and peak memory versus n,
  exact and Nyström on the same axes, with the OOM point marked and horizontal
  lines for 16, 24, 48 and 80 GB cards.
- A one-line fitted model `peak ≈ a·n² + b` and a table "GPU memory → largest
  n for exact mode" go into the docs and the paper.

### 2.2 Package changes (bounded, each with a test)

1. **Drop the `eU` n×n copy** in `cvksvm` (and the same pattern in `cvkdwd`,
   `cvklogit`, `cvkqr` if present). It is only used in the `is_exact=1`
   projection as `Umat @ (eU @ theta)`; compute `Umat @ (einv * (Umat.T @
   theta))` instead. Saves 8n² bytes and lifts the ceiling by roughly 15%.
   Verify bitwise-close results on the CPU test suite and re-run the CUDA
   protocol.
2. **Build the kernel on the target device** in `_compute_K_train` (currently
   CPU float64 then `.to(dev)`); removes an 8n² host allocation and a
   host-to-device copy from the timed pipeline.
3. **Expose peak memory** as a fitted attribute `peak_gpu_memory_bytes_`
   (reset peak stats before `backend.fit()`, read after; `None` on CPU) and a
   helper `torchkm.utils.exact_mode_memory_estimate(n, dtype)` returning the
   predicted peak; both documented on the new envelope page.
4. **Informative OOM**: catch `torch.cuda.OutOfMemoryError` around
   `backend.fit()` in exact mode and re-raise with the estimate and the
   suggestion `low_rank=True` or a smaller `n`.
5. **Nyström seeding**: replace `torch.manual_seed(0)` in `cvknyssvm`,
   `cvknysdwd`, `cvknyslogit` with a `torch.Generator` seeded from
   `random_state`, the way `cvknyqr` already does, so repeats vary the
   landmarks and the global RNG is untouched. Test that two fits with the same
   `random_state` agree and different seeds differ.
6. **Stretch, only if it validates**: `dtype="float32"` for the kernel and
   the eigendecomposition. Would double the envelope (≈12n² bytes). Report
   objective and accuracy differences versus float64 on the envelope sweep;
   ship it only if the loss is negligible, otherwise document why not.
7. **Document `is_exact`** in `docs/user_guide/model_selection.md`: "exact"
   in the paper means the exact-CV formula and the exact SVM solution from
   finite smoothing; the flag adds a final projection step. State which
   setting the benchmarks use.

### 2.3 Docs

New page `docs/user_guide/operating_envelope.md`: the formula, the table per
GPU size, how to read `peak_gpu_memory_bytes_`, when to switch to Nyström, and
what Nyström costs (O(n·m + n·k) instead of O(n²)).

Effort: 4 to 6 days plus a few GPU hours.

---

## 3. Phase 2: the SVM comparison the reviewer asked for (items 1, 3, 4, 5)

### 3.1 Restructure the dataset suite

Replace the eight-name list with a suite that separates scaling from
difficulty and shows the class prior for every set.

| Role | Datasets | Mode | Notes |
| --- | --- | --- | --- |
| **Scaling study** (replaces a7a/a8a/a9a and w7a/w8a as separate rows) | Adult at a1a (1,605), a3a (3,185), a5a (6,414), a7a (16,100), a8a (22,696), a9a (32,561) | exact up to the envelope, Nyström above | One figure or one compact table: time and memory versus n for every library. Adult stays as the scaling data, not as six benchmarks. |
| **Nonlinearity in the exact range** (item 5) | ijcnn1 stratified subsample n=30,000; MNIST 3-vs-8 and 4-vs-9 from `mnist.scale` (≈12k each); covtype stratified subsample n=30,000 | exact | Each row also gets a tuned linear baseline (`LogisticRegression` and `LinearSVC`) so the gap between linear and kernel is visible. On ijcnn1 and the MNIST pairs the kernel gap is several points; on covtype it is large. |
| **Imbalanced** | w8a (≈3% positive), ijcnn1 full (≈10% positive) | Nyström | Report AUC and balanced accuracy as the headline, accuracy secondary. |
| **Scale** | covtype full (581k), MNIST8m 4-vs-6 (1.27M) | Nyström | Keep, labelled as scale tests. covtype gets the rank sweep below. |

Drop the 1D-CNN column. If a neural baseline is wanted, use a tuned MLP with
early stopping and put it in the docs, not the paper.

### 3.2 Libraries

`benchmarks/bench_gpu_libraries.py`, one protocol for every library: RBF
kernel with the same bandwidth, 50-value grid, 10-fold CV, end-to-end timing,
NVML peak memory, 10 repeats.

| Library | Role | How it is tuned | Caveats to state |
| --- | --- | --- | --- |
| **cuML `SVC`** (RAPIDS) | Direct competitor on the exact path | `cross_val_score` over the same grid (cuML estimators are scikit-learn compatible) | 500 SMO fits per dataset; put a per-cell time cap (e.g. 4 h) and report "> cap". Also report a budget-matched row (cuML on a 10-value grid) so the reader sees both the like-for-like and the practical number. |
| **Falkon** | Direct competitor on the Nyström path | `Falkon`/`LogisticFalkon` with M = 2000 centres (matched to `num_landmarks`) and with M = 10,000 and 20,000 to show accuracy at a larger budget; λ chosen by the same fold loop | Squared-loss classifier, not hinge; say so. Falkon is where covtype will look best, and that is the honest picture. |
| **ThunderSVM 0.3.4** | Keep | as today | Latest release; note the zero-support-vector failure at the strong-regularisation end that the script already handles. |
| **scikit-learn `SVC`** | CPU reference | as today | Only on the small end of the scaling study. |
| **EigenPro 3** | Optional third GPU KRR solver on the Nyström table | its own preconditioned SGD, λ by fold loop | Include if it installs cleanly; otherwise related work. |
| **KeOps** | Related work only | | It is a kernel-operation engine, not a model-selection library. TorchKM materialises K by design because it eigendecomposes it; KeOps avoids materialisation. Mention that a KeOps or Falkon-style backend for building Nyström features is future work. Cite the JMLR MLOSS paper. |

### 3.3 The covtype accounting (item 3)

`benchmarks/bench_covtype_rank.py`: TorchKM Nyström on covtype full at
landmarks m ∈ {2000, 5000, 10000, 20000} × rank k ∈ {30, 300, 1000, m}, plus
the exact-mode covtype-30k row from 3.1, plus Falkon at the same m. Report
accuracy, AUC, time, memory. Expected outcome and the paragraph the paper
needs: covtype has a slowly decaying kernel spectrum, so a rank-30 (or even
rank-300) approximation on 2000 landmarks is intrinsically limited; TorchKM's
accuracy rises with rank at a stated cost in time and memory, and the exact
30k subsample shows what the kernel itself achieves. The Table 4 number is
explained by configuration, and the revision reports the curve instead of one
point.

### 3.4 What happens to the old Table 2 and Figure 1 (item 5, point 2)

- The simulation gains a **test-accuracy column** (the script already builds a
  test split) plus time and memory, and moves to the appendix as a controlled
  scaling experiment in n and p, with a sentence noting that the Gaussian
  mixture has fast eigenvalue decay and is the favourable regime.
- The **objective comparison** becomes a solver-quality check in the docs
  (`docs/developer/solver_quality.md`): fix three λ values, fit every solver at
  that λ with no CV, and compare objectives. This removes the different-λ
  problem and the `ulam[-1]` quirk. Report it once; TorchKM should match the
  exact optimum to solver tolerance, and any gap is a real finding.
- The CPU-versus-GPU proximal-gradient figure (old Figure 1) moves to the same
  docs page as the mechanism illustration.

### 3.5 Language

Search-and-replace the claims: "consistently superior accuracy", "attains the
best accuracy", "lowest objective values" go. The supported claim is **equal
accuracy at a fraction of the run time, with standard errors shown**, plus the
covtype curve. Every accuracy difference within two SEs is described as equal.

Effort: 1.5 to 2 weeks; GPU time dominated by cuML and Falkon CV grids on
covtype and MNIST8m (budget the time caps accordingly).

---

## 4. Phase 3: kernel quantile regression and DWD benchmarks (item 2, point 3)

These are the capabilities no GPU library offers, so they are the strongest
part of the case and currently absent.

### 4.1 KQR (`benchmarks/bench_kqr.py` + `benchmarks/r/bench_kqr.R`)

- **Datasets** (mix of exact and Nyström range, all standard and public):
  California housing (20,640; exact), bike sharing hourly (17,379; exact),
  CASP protein (45,730; Nyström, and a 30k exact subsample), plus one synthetic
  heteroscedastic set where the true conditional quantiles are known.
- **Quantiles** τ ∈ {0.1, 0.5, 0.9}.
- **Baselines**: `fastkqr` (R, CPU; the reference implementation the letter
  names), `kernlab::kqr` (R, QP solver; slow, capped), and scikit-learn's
  linear `QuantileRegressor` so the value of the kernel is visible.
- **Metrics**: test check (pinball) loss, empirical coverage of the fitted
  quantile, time (end to end including the λ path and CV), peak memory. The
  synthetic set additionally reports error against the true quantile.
- R baselines run from a script that records `sessionInfo()`; timing is done
  inside R, and the JSON merges both sides.

### 4.2 DWD (`benchmarks/bench_dwd.py` + `benchmarks/r/bench_dwd.R`)

- **Datasets**: one high-dimension low-sample-size case where DWD is the
  method of choice (gisette, 6,000 × 5,000; or a gene-expression set), and
  two mid-n sets from the SVM suite (ijcnn1-30k, MNIST 3-vs-8) to show the
  speed of exact CV on the DWD path.
- **Baselines**: `kerndwd` (R, CPU, with its own `cv.kerndwd`), the Python
  `dwd` package if it installs cleanly, and `TorchKMSVC` as the in-package
  reference.
- **Metrics**: accuracy, AUC, time, memory.

### 4.3 Paper

One table for KQR and one for DWD, both in the main text if space allows
(they are the differentiating result), each with a two-sentence statement of
the constituency served: GPU-accelerated KQR and DWD with exact CV do not
otherwise exist.

Effort: 1 week, including the R environment.

---

## 5. Phase 4: the manuscript (item 7, further points, and the response)

### 5.1 New shape of the paper (4 pages main, appendix cut to the tables)

1. **Introduction**: soften "continue to define the state-of-the-art in many
   fields" to "remain competitive on small-to-mid-size tabular and structured
   data and remain the method of choice for specific tasks", and point to the
   ijcnn1/MNIST linear-versus-kernel gap in the new tables as the concrete
   example. Delete the agentic/automated-ML passage (the README may keep a
   toned-down version; the paper should not carry an undeveloped claim).
2. **Related work** (new short paragraph): ThunderSVM, cuML, Falkon, KeOps,
   EigenPro, with one sentence each on what they do and how TorchKM differs
   (integrated exact CV along the path; losses beyond hinge). Table 1 gains
   columns for cuML, Falkon and KeOps and rows for exact CV and for KQR/DWD.
3. **Package overview**: unchanged in substance; add `low_rank`, memory
   attribute and the envelope statement in one sentence with a pointer to the
   docs page.
4. **Algorithms**: one paragraph. State the two ideas, cite Wang and Zou (2022)
   for the exact-CV lemma and finite smoothing, Tang et al. (2026) for KQR,
   and give the single formula for the eigendecomposition reuse. Everything
   else in A.1 to A.4 goes.
5. **Benchmarks** (the bulk of the space): scaling figure with the memory
   envelope; exact-range table (linear, sklearn, ThunderSVM, cuML, TorchKM);
   Nyström table (sklearn Nyström, Falkon, TorchKM, with the covtype rank
   curve); KQR table; DWD table. Each table carries accuracy plus AUC or
   balanced accuracy where imbalance warrants it, time with SE, and peak
   memory.
6. **Appendix**: only the protocol summary (hardware, grid, folds, data
   sources with class priors) and any table that does not fit. Simulation
   study with accuracy goes here.

### 5.2 Moves from appendix to documentation

| Appendix section | Destination |
| --- | --- |
| A.1 to A.4 (SVM formulation, exact CV, spectral algorithm, Algorithm 1) | Cite. Optionally a short `docs/developer/algorithms.md` that restates the update and links the papers. |
| A.5 Nyström design | `docs/user_guide/nystrom.md` (expand: landmark sampling, rank, the single decomposition outside the fold/λ loops, seeding). |
| Multiclass handling (submitted version) | Either drop the section, or add a short `docs/user_guide/multiclass.md` showing `OneVsRestClassifier(TorchKMSVC(...))` with a test, since the estimators are scikit-learn compatible. Do not describe behaviour the package lacks. |
| A.6 Platt scaling and Figure 2 | `docs/user_guide/probability_calibration.md` (already has the reliability plot API; add the figure and the ECE/Brier numbers). |
| B.1 experimental setup | `docs/examples/reproduce_paper_benchmarks.md`, rewritten in Phase 0 and regenerated from the results archive. |

### 5.3 Response letter

Point-by-point, in the letter's order, each answer stating what changed and
where (table, figure, docs URL, script path). The covtype answer cites the
rank-30 configuration and the new curve. The memory answer cites the formula,
the figure and the fitted attribute. The language answer lists the removed
phrases.

Effort: 1 week after the numbers are in.

---

## 6. Order of work and dependencies

```
Phase 0 (harness, memory, AUC, JSON, docs debt)        2–3 days
   └─ Phase 1 (envelope sweep, eU removal, seeding,     4–6 days
   │           memory attribute, OOM message)
   │     └─ Phase 2 (cuML/Falkon/ThunderSVM suite,      1.5–2 weeks
   │     │           covtype rank curve, Table 2 → acc)
   │     └─ Phase 3 (KQR vs fastkqr, DWD vs kerndwd)     1 week   (parallel to 2)
   └─ Phase 4 (paper restructure, docs moves, response)  1 week
```

Roughly five to six weeks of focused work; the long pole is GPU time for
the cuML and Falkon cross-validation grids on the large sets.

Package release: cut `v4.4.0` once Phases 1 and 2 land (memory attribute,
OOM message, Nyström seeding, kernel-on-device, `eU` removal, envelope docs,
benchmark harness), with a fresh `benchmarks/cuda-runs/` bundle on the release
commit, and cite that version in the resubmission.

---

## 7. Checklist against the letter

| Letter item | Deliverable | Section above |
| --- | --- | --- |
| 1. cuML and Falkon, with accuracy, time, peak memory | `bench_gpu_libraries.py`, exact and Nyström tables | 3.2 |
| 2. One KQR and one DWD benchmark | `bench_kqr.py`, `bench_dwd.py`, R scripts, two tables | 4 |
| 3. covtype accounted for | rank-30 finding, `bench_covtype_rank.py`, curve | 0, 3.3 |
| 4. AUC / balanced accuracy under imbalance | harness metrics, w8a and ijcnn1 rows | 1, 3.1 |
| 5. Kernel-natural problem in the exact range | ijcnn1-30k, MNIST pairs, covtype-30k with linear baseline | 3.1 |
| 6. Memory envelope | `bench_memory_envelope.py`, new Figure 1, envelope docs page, `peak_gpu_memory_bytes_` | 2 |
| 7. Reduced appendix, derivations cited, behaviour documented online | 5.1, 5.2 | 5 |
| Table 2 headline → accuracy | simulation gains accuracy; objective check to docs with fixed-λ protocol | 3.4 |
| Temper language | 3.5 | 3.5 |
| Adult/Web as scaling not benchmarks | scaling study | 3.1 |
| Simulation not neutral | stated, moved to appendix | 3.4 |
| State-of-the-art claim | softened, concrete example from new tables | 5.1 |
| Agentic passage | removed | 5.1 |
| Stale reproduce page | rewritten, generated from results | 1 |
