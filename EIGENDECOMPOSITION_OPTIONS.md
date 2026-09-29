# Large kernel eigendecompositions in TorchKM: less memory, same speed

**Question.** Can the eigendecomposition of the n x n kernel use less GPU
memory without getting slower?

**How to read this.** This is a proposal for review. Sections 1 and 7 are
measured on the L40S (48 GB). Section 4.6 comes from a CPU emulation at
n = 3,000. Anything marked *projected* or *untested* has not been run. A "unit" means one n x n
matrix: 4n² bytes in float32, 8n² in float64.

## Summary

- **The eigendecomposition sets the memory ceiling.** A fit peaks at 6 units:
  - the kernel K (1);
  - the eigenvectors U (1);
  - one 4.01-unit cuSOLVER workspace (measured, section 7).

  Once the factorization is done, only K and U stay resident (2 units).
- **It also sets a size ceiling, whatever the memory.** cuSOLVER's eigh
  (PyTorch 2.6, CUDA 12.4) accepts n = 32,768 (2¹⁵) and refuses 32,769 in
  float32; float64 accepts 32,768 and refuses 33,000 (measured, section 7.3).
  - So the float32 exact solver stops at n = 32,768 on the L40S, not at
    the 43,000 its memory would allow, and at the same n on a larger card.
  - The truncated spectrum calls no dense eigensolver on the n × n matrix,
    so only memory limits it.
- **Keeping a full eigendecomposition, the cheapest cut is 6 → 5 units.**
  Factorize in place and rebuild K afterwards. This was measured at the same
  speed and has since been reverted. MAGMA or host LAPACK get the peak down to
  2 units, but run 4× and 12× slower (measured). We know of no full dense
  eigensolver that gives both lower memory and cuSOLVER's speed.
  - Under the size ceiling, the cut no longer raises the largest n in
    float32, and in float64 only from 30,600 to about 32,800.
- **Proposal: don't compute the full eigendecomposition.**
  - The solver only needs a curvature matrix K̃ ⪰ K for which (K̃ + cI)⁻¹ is
    cheap for every c.
  - Keep the top r eigenpairs exactly and replace the rest of the spectrum by
    one constant τ ≥ e_{r+1}.
  - The kernel is not approximated. The iteration keeps the same fixed
    point, and every λ is accepted by the same test on the true K.
    - That test is the shipped solver's heuristic, not a certificate. 4.11
      proposes a duality-gap certificate.
  - Descent is guaranteed only with exact eigenvectors. With Ritz vectors a
    step can go uphill (a reviewer's 2×2 example). 4.7 adds a safeguard that
    costs no extra product with K.
  - The O(n³) eigendecomposition becomes O(n² r) matrix products. At
    n = 20,000 on the L40S, the top 400 eigenpairs take 0.09 s against 10.2 s
    for the full eigh (measured).
  - The eigensolver workspace disappears. The peak is K plus 0.16 units
    (measured), against 6.01.
  - Each iteration reads one n × n matrix (K) instead of three (K once, U
    twice).
- **Evidence** (CPU emulation, n = 3,000, 50 λ × 10 folds, p = 10 and
  100):
  - **Q1 λ grid, r = 100–800 (3–27% of n).**
    - Same selected λ and test accuracy.
    - 0–1.3% more iterations.
    - Objectives change by at most 2e-5 from the full solver with exact
      eigenvectors and r = 400; 9e-5 with Ritz vectors; 1.5e-4 with r = 100.
    - Ritz vectors from 2 cheap subspace passes do nearly as well as exact
      eigenvectors.
  - **Harder grid (λ down to 2e-5, KKTeps 1e-6).**
    - At the same eps, the truncated solver stops earlier and less
      accurately, because its steps are smaller.
    - With eps ten times smaller in the rounds where c < τ, it is more
      accurate than the full solver at the default eps.
    - Per unit of accuracy it takes about the same number of iterations
      (p = 10) or about 30% more (p = 100), and each iteration reads one
      n × n matrix instead of three.
    - On the Q1 grid that rule changes nothing.
    - Relative suboptimality below ~1e-4 needs a larger r.
- **Prototype built:** `torchkm.experimental.SpectralSVMPath` (section 7.2).
  Every λ and every fold stops at the same certified duality gap.
  - CPU results, n = 3,000: on the Q1 grid and the p = 10 hard grid the
    truncated spectrum is 1.6–1.8× faster than the full one. It reaches the
    same accuracy against libsvm with about 3× fewer n × n reads.
  - Its limit: separable data at tiny λ, where it does not certify in
    reasonable time.
  - **Measured on the L40S** (n = 20,000, float32, 50 λ × 10 folds; section
    7.3): at the same certified gap (1e-3 or 1e-4), truncated is 2.8× faster
    than full and peaks at 1.25 units of memory against 6.11.
  - It selects the same λ and reaches the same accuracy.
  - At gap 1e-4 it takes the same time as the shipped, uncertified solver
    (16.7 against 16.8 s).
  - At n = 60,000 cuSOLVER's eigh refused the problem size outright
    (measured), and would need about 82 GiB anyway. There, truncated
    certified every λ and fold at 1e-3 in 82 s, peaking at 1.15 units
    (16.6 GiB), with about as many n × n reads as at n = 20,000.
- **Review.** Round 1, with the corrections it led to, is in
  `EIGENDECOMPOSITION_REVIEW_REPLY.md`.

## 1. Where the memory and time go (measured, L40S 48 GB)

| run | n | dtype | eigh time | peak during eigh (units) | resident after |
|---|---:|---|---:|---:|---:|
| whole fit, cuSOLVER (torch.linalg.eigh) | 20,000 | float32 | 21.4 s of 66.0 s (32%) | 6.01 (8.98 GiB) | 2.01 (K, U) |
| eigh alone, cuSOLVER (probe, section 7) | 20,000 | float32 | 10.2 s | 6.01: K 1 + working copy 1.00 + workspace 4.01 | 2 |
| eigh alone, cuSOLVER | 16,100 | float64 | 21.5 s | 6.01 | 2 |
| eigh alone, MAGMA (one-stage) | 16,100 | float64 | 85.4 s | 2.00 | 2 |
| eigh alone, host LAPACK (IDAS CPU) | 16,100 | float64 | 253.6 s | 2.00 on the GPU | 2 |

- The table uses the PyTorch allocator peak. NVML reads 0.4–0.7 units more
  (float64 Q1 runs on a7a, a8a and w7a); that difference is allocator slack.
- The same float32 eigh at n = 20,000 took 10.2 s in the probe and 21.4 s
  inside the profiled fit. The matrix and data were the same, and the
  difference is not explained; other work on the GPU during the profile is
  one possibility.
- The rest of the float32 fit at n = 20,000:
  - path: 11.7 s over 410 iterations;
  - CV: 32.5 s over 2,525 fold-iterations;
  - one matrix-vector product with a unit: 5 ms (about 320 GB/s).

Largest n on a 48 GB card (about 45 GB usable) for a given peak:

| peak (units) | float32 | float64 |
|---:|---:|---:|
| 6 (now) | 43,000 | 30,600 |
| 5 (in place) | 47,400 | 33,500 |
| 2 | 75,000 | 53,000 |
| 1 | 106,000 | 75,000 |

This counts memory only. cuSOLVER's eigh also has a size limit: in float32
it accepts n = 32,768 and refuses 32,769; float64 accepts 32,768 and refuses
33,000 (measured, section 7.3).
- In the rows that call it (6 and 5 units), the float32 limit is therefore
  32,768, and the float64 limit at 5 units is between 32,768 and 33,000.
- MAGMA and host LAPACK (2 units) usually take the workspace size as a 32-bit
  integer too, so they may hit the same limit (*untested*).
- The truncated spectrum (about 1 unit) calls no dense eigensolver on the
  n × n matrix.

## 2. What the solver needs from the eigendecomposition

For each λ and smoothing level δ, `cvksvm` minimises

  F(b, α) = Σᵢ φ_δ(yᵢ(b + Kᵢα)) + nλ αᵀKα + nε b²

where φ_δ is the smoothed hinge. Its curvature is at most 1/(2δ).

- **The curvature bound.** Write A = [1, K]. The bound is
  M = AᵀA/(2δ) + diag(2nε, 2nλK). Its α-block is K(K + cI)/(2δ), with
  c = 4nδλ.
- **The step.** Solving MΔ = −∇F cancels one factor of K, so the step needs
  only
  - Δα = −2δ·mul·(K + cI)⁻¹γ − Δb·s,
  - with γ = z + 2nλα (z = y∘φ′_δ) and s = (K + cI)⁻¹1,
  - plus a scalar intercept step.
  - mul ∈ [1, 2) is the over-relaxation factor (`torchkm/cvksvm.py:415-426`).
- **What the eigendecomposition buys.** K = U diag(e) Uᵀ gives
  (K + cI)⁻¹ = U diag(1/(e + c)) Uᵀ for every c at once. A fit uses 50 λ
  values, up to 8 δ rounds and 10 folds, so one O(n³) factorization serves
  hundreds of values of c.
- **Cost per iteration.** Three unit reads: Kα, Uᵀγ, and U(·).

The float32 work already showed two things. First, the step only needs its
curvature matrix to dominate K; float32 adds 2‖UEUᵀ − K‖ to the eigenvalues
for exactly this reason. Second, the fixed point does not depend on that
matrix. The proposal in section 4 builds on both.

## 3. Options that keep the full eigendecomposition (same iterates)

| option | peak | resident | speed | status | effort |
|---|---:|---:|---|---|---|
| a. In place: hand eigh K's own storage, rebuild K afterwards (one GEMM, 0.04 s at n = 20k) | 5 | 2 | same | measured, fits bitwise identical (commit c4c0cac; reverted in b9ca1a3) | small: restore |
| b. Spectral coordinates: drop K after eigh and keep β = Uᵀα. One pass U[β, e∘β] gives α and Kα; a second gives Uᵀγ | 5–6 | 1 | 2 unit reads per iteration instead of 3, about 1.5× faster path and CV (*projected*) | *untested* | moderate |
| c. Call cuSOLVER directly: in place with an explicit workspace, or syevdx / syevj | 5 with syevd | 1–2 | same (syevj slower) | syevd's workspace is one 4.01-unit allocation (probe A), so an in-place syevd call equals (a); syevdx / syevj *untested* | moderate |
| d. MAGMA one-stage or host LAPACK | 2 | 2 | 4× / 12× slower | measured | in git history |
| e. Two-stage tridiagonalization (MAGMA dsyevdx_2stage, ELPA) | 2–3 | 2 | unknown; two-stage beats one-stage at large n | *untested*; not exposed by PyTorch | high |
| f. Several GPUs (cusolverMg, cuSOLVERMp) | 6/G per GPU | | similar | *untested*; not exposed by PyTorch | high |

- **cuSOLVER's size limit caps options a–c at n = 32,768** (section 7.3),
  whatever their peak. Option d may share it (*untested*).
- **Caveat for option b.** It replaces K by UEUᵀ: a 1e-13 relative change in
  float64 and about 1e-6 in float32. In exchange, U(e + γ)Uᵀ dominates that
  kernel exactly, so float32 would no longer need the error shift.
- **Conclusion.** With a full eigendecomposition, the peak cannot drop below
  the eigensolver's output plus its workspace. With cuSOLVER syevd that is
  5 units: the 1-unit output and 4.01 units of workspace, both measured.
  Going lower (MAGMA, host) costs speed.

## 4. Proposal: truncated-spectrum majorizer (no full eigendecomposition)

### 4.1 Construction

Let V (n × r) hold the top-r eigenvectors of K and Θ = VᵀKV = diag(θ₁…θ_r).
Let τ be at least the largest eigenvalue of K on the orthogonal complement of
V; for exact eigenvectors, τ = e_{r+1}. Define

  K̃ = VΘVᵀ + τ(I − VVᵀ).

Then, for every c > 0,

  (K̃ + cI)⁻¹ = (τ + c)⁻¹ I + V [(Θ + cI)⁻¹ − (τ + c)⁻¹ I] Vᵀ,

so each application costs two n × r products. With exact eigenvectors,
K̃ ⪰ K and K̃ commutes with K.

### 4.2 The step

Replace (K + cI)⁻¹ by P̃ = (K̃ + cI)⁻¹:

- Δα = −2δ·mul·P̃γ − Δb·s̃, with s̃ = P̃1.
- Δb = −2δ·mul·(1ᵀz + 2nεb − ṽᵀγ) / (n + 4nδε − 1ᵀṽ), with ṽ = P̃(K1).
- K1 is computed once, at setup.

The intercept coupling uses the true K (ṽ = P̃K1, not K̃P̃1). This makes
the step the exact solution of M̃Δ = −∇F, where

  M̃ = M + diag(0, K(K̃ − K)/(2δ)).

### 4.3 What stays exact, and what is proven

1. **Majorization (exact eigenvectors).** K̃ commutes with K and K̃ ⪰ K, so
   K(K̃ − K) ⪰ 0 and M̃ ⪰ M ⪰ ∇²F. Each step minimises a quadratic upper
   bound of F. With over-relaxation,
   F(θ − mul·M̃⁻¹g) ≤ F(θ) − mul(1 − mul/2)·gᵀM̃⁻¹g, so F still decreases
   monotonically for mul < 2. For singular K the argument holds on range(K).
   **This needs K̃ to commute with K.** With Ritz vectors K(K̃ − K) is not
   symmetric, and K̃ ⪰ K alone does not give descent (4.7).
2. **Fixed point.** The update is zero exactly when ∇F = 0, for any invertible
   M̃. So r, τ and the accuracy of V change the route to the solution, not the
   solution.
3. **Same acceptance test.** A λ, or a fold, is accepted only by the existing
   test on the unsmoothed hinge, KKT = z/n + 2λα. That test uses residuals
   y(b + Kα) computed with the true K and never touches the eigendecomposition.
   It is a finite-tolerance heuristic, not a certificate of optimality: on the
   Q1 grid it stops 1.9–29% above the optimum at the smallest λ (4.6). 4.11
   proposes a certificate.

### 4.4 What truncation costs in speed

Let D mark the margin rows (φ″ = 1/(2δ) there). One step multiplies the error
by I − mul·P̃(DK + cI). Its eigenvalues μ solve

  (KDK + cK)x = μ(KK̃ + cK)x,

which is a symmetric pencil when K̃ commutes with K.

- **Worst case is unchanged** [proven, for commuting K̃ ⪰ K and any
  0 ⪯ D ⪯ I, intercept left out].
  - Full spectrum: μ ∈ [c/(e₁ + c), 1].
  - Truncated: μ ∈ [c/(e₁ + c), 1] too, since K̃'s largest eigenvalue is
    still e₁.
  - So the worst-case contraction per iteration is the same.
- **More directions are slow** [heuristic: this treats D as diagonal in K's
  eigenbasis, which it is not].
  - A direction with eigenvalue eᵢ < τ moves from μ = (dᵢeᵢ + c)/(eᵢ + c) to
    (dᵢeᵢ + c)/(τ + c), where dᵢ ∈ [0, 1] is the direction's weight on
    margin rows.
  - When c = 4nδλ ≪ τ, many directions are slow instead of a few.
  - r changes speed, and where a finite tolerance or iteration cap stops. It
    does not change the limit point.
- **Choosing r.** The relevant ratio is c/τ = 4nδλ/e_{r+1}. Kernel eigenvalues
  grow like n·μₖ, where μₖ are the eigenvalues of the kernel's integral
  operator.
  - For a fixed λ grid (as in the Q1 simulation), c/τ ≈ 4δλ/μ_{r+1} does not
    depend on n, so r need not grow with n.
  - For a fixed C grid (λ = 1/(2nC)), c = 2δ/C is fixed while τ grows like n,
    so r must grow with n. That growth is slow when μₖ decays fast, as it does
    for an RBF kernel on low-dimensional data. (*projected*)

### 4.5 The stopping rule

- **Why it stops early.** The inner loop stops when max Δ² < eps·mul². A step
  is 2δ·mul times the preconditioned gradient, and truncation shrinks its
  tail components by (eᵢ + c)/(τ + c) ≥ c/(τ + c). So the same eps stops
  earlier, further from the optimum. The hard grid in 4.6 shows this.
- **What works: eps ten times smaller in the rounds where c < τ.**
  - On the Q1 grid this changes nothing: those rounds start close to
    converged.
  - On the hard grid the truncated solver ends up more accurate than the full
    solver at its default eps. Per unit of accuracy it takes about the same
    number of passes (p = 10) or about 30% more (p = 100); see 4.6.
- **What does not work: scaling eps by the curvature ratio.**
  - The idea was eps′ = eps·(c/(τ + c))². In a decoupled model that keeps the
    stopping error no larger than the full method's in every eigendirection.
  - Unfloored, it is unusable. In the last smoothing rounds c = 4nδλ falls to
    about 1e-7, so eps′ drops below float64 resolution and the loop runs to
    its iteration cap.
  - Floored at 0.01, it costs more passes than eps/10 for similar accuracy.
- **Open.** A test that does not depend on the preconditioner may be better;
  see question 4.

### 4.6 Evidence: CPU emulation

- **Setup.**
  - Data: Table 2's Gaussian mixture, n = 3,000 (600 test rows), seed 42,
    sigest bandwidth.
  - Protocol: 50 λ, 10 folds, float64, rank r out of n; eps 1e-5 unless a
    row says otherwise.
- **How the emulation works.** torch.linalg.eigh returns a full orthonormal
  basis U = [W, V] with eigenvalues [τ, …, τ, Θ]. Then U diag Uᵀ = K̃
  exactly, so the arithmetic is that of the rank-r method.
  - The intercept coupling uses ṽ = P̃K1 (4.2).
  - The solver code is otherwise the shipped one. Only the inner stopping
    tolerance varies by row, either for the whole fit or per smoothing round.
    The CV has its own fixed tolerance of 1e-5 and is scaled by the same
    factor.
- **Configurations.**
  - "exact r": the true top-r eigenpairs, with τ = e_{r+1}.
  - "Ritz q, r": randomized subspace iteration with q passes and block
    r + 20, then Rayleigh–Ritz. It then shifts Θ + ρ and uses
    τ = 1.05·(power-iteration estimate) + ρ, with ρ = ‖KV − VΘ‖. This makes
    K̃ ⪰ K; min eig(K̃ − K) > 0 was checked in every run.
- **Suboptimality.** (F − F*)/F*, where F* is the optimum from libsvm
  (sklearn SVC, precomputed kernel, tol 1e-7).

**Q1 grid** (λ 1e3 → 1e-3, KKTeps 1e-3, eps 1e-5), as in the Q1 simulation
runs. The |F − F_full| column is the largest change in objective against
the full solver, across the 50 λ.

| p | solver | τ | path passes | CV passes | max \|F − F_full\| / F_full | selected λ | test accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | full eigendecomposition | | 396 | 2,505 | | 0.001 | 0.830 |
| 10 | exact r = 800 | 0.34 | 396 | 2,505 | 5.5e-7 | 0.001 | 0.830 |
| 10 | exact r = 400 | 0.83 | 396 | 2,507 | 5.3e-7 | 0.001 | 0.830 |
| 10 | exact r = 200 | 2.12 | 396 | 2,515 | 3.1e-6 | 0.001 | 0.830 |
| 10 | exact r = 100 | 3.66 | 396 | 2,521 | 1.7e-6 | 0.001 | 0.830 |
| 10 | exact r = 50 | 11.2 | 394 | 2,571 | 3.6e-4 | 0.001 | 0.830 |
| 10 | Ritz q = 2, r = 400 | 1.33 | 396 | 2,510 | 5.4e-6 | 0.001 | 0.830 |
| 10 | Ritz q = 2, r = 100 | 5.19 | 395 | 2,533 | 2.1e-5 | 0.001 | 0.830 |
| 10 | Ritz q = 0, r = 400 | 3.72 | 395 | 2,524 | 3.4e-5 | 0.001 | 0.830 |
| 100 | full eigendecomposition | | 467 | 2,519 | | 0.0126 | 1.000 |
| 100 | exact r = 400 | 0.89 | 467 | 2,521 | 2.0e-5 | 0.0126 | 1.000 |
| 100 | exact r = 100 | 5.1 | 466 | 2,551 | 1.5e-4 | 0.0126 | 1.000 |
| 100 | Ritz q = 2, r = 400 | 1.37 | 467 | 2,530 | 9.0e-5 | 0.0126 | 1.000 |

- Spectra (n = 3,000):
  - p = 10: e₁ = 412, e₁₀₀ = 3.66, e₄₀₀ = 0.83.
  - p = 100: e₁ = 407, e₁₀₀ = 5.1, e₄₀₀ = 0.89.
- At the last λ, c = 12δ. So c ≥ τ in the first smoothing round for every r
  shown, and in the second round too for r ≥ 400.
- Noise floor: with the full spectrum, the helpers in the appendix differ from
  the shipped solver only in rounding. That gives changes up to 2.6e-6.
- **Context on this grid.** Both solvers stop at the same points. Those points
  sit above the libsvm optimum for λ ≤ 0.04: by up to 1.9e-2 at p = 10, and up
  to 2.9e-1 at p = 100 (at λ = 1e-3: 0.129 against 0.0997). KKTeps = 1e-3 is
  loose there. Truncation does not change that.

**Hard grid** (λ 1e-2 → 2e-5, KKTeps 1e-6): accuracy against work, where
accuracy is how far each solver ends above the libsvm optimum, across the
50 λ. At the last λ, c = 0.24δ, so c < τ from the first smoothing round.

| p | solver | eps | path passes | CV passes | above optimum, max / median | selected λ | test accuracy |
|---:|---|---|---:|---:|---:|---:|---:|
| 10 | full | 1e-5 | 7,586 | 4,666 | 1.1e-3 / 4.1e-4 | 3.7e-4 | 0.825 |
| 10 | full | 1e-6 | 14,247 | 6,372 | 1.3e-4 / 8.6e-5 | 4.2e-4 | 0.825 |
| 10 | full | 1e-7 | 27,035 | 8,255 | 5.1e-5 / 2.5e-5 | 4.8e-4 | 0.825 |
| 10 | exact r = 400 | 1e-5 | 4,961 | 4,729 | 1.7e-3 / 1.0e-3 | 3.7e-4 | 0.825 |
| 10 | exact r = 400 | 1e-6 | 9,831 | 6,444 | 3.0e-4 / 1.6e-4 | 3.7e-4 | 0.825 |
| 10 | exact r = 400 | 1e-7 | 20,199 | 8,183 | 2.0e-4 / 5.1e-5 | 4.2e-4 | 0.823 |
| 10 | exact r = 400, eps/10 where c < τ | 1e-5 | 9,839 | 5,610 | 4.7e-4 / 2.4e-4 | 4.8e-4 | 0.825 |
| 10 | exact r = 400, eps·max(0.01, (c/(τ+c))²) | 1e-5 | 19,873 | 5,871 | 3.6e-4 / 1.2e-4 | 4.2e-4 | 0.823 |
| 10 | exact r = 100 | 1e-6 | 10,348 | 7,266 | 1.6e-3 / 3.3e-4 | 4.2e-4 | 0.825 |
| 10 | exact r = 100 | 1e-7 | 20,275 | 9,106 | 3.3e-4 / 8.1e-5 | 4.2e-4 | 0.823 |
| 10 | Ritz q = 2, r = 400 | 1e-6 | 9,518 | 6,611 | 4.7e-4 / 2.0e-4 | 4.2e-4 | 0.825 |
| 10 | Ritz q = 2, r = 400, eps/10 where c < τ | 1e-5 | 9,549 | 5,980 | 4.8e-4 / 3.0e-4 | 4.8e-4 | 0.825 |
| 100 | full | 1e-5 | 3,189 | 3,739 | 1.1 / 1.1e-2 | 0.01 | 1.000 |
| 100 | full | 1e-6 | 8,915 | 8,193 | 5.3e-1 / 3.6e-3 | 0.01 | 1.000 |
| 100 | exact r = 400 | 1e-5 | 2,532 | 3,375 | 1.2 / 1.8e-2 | 0.01 | 1.000 |
| 100 | exact r = 400 | 1e-6 | 7,298 | 7,863 | 8.8e-1 / 5.6e-3 | 0.01 | 1.000 |
| 100 | exact r = 400, eps/10 where c < τ | 1e-5 | 7,169 | 7,020 | 8.8e-1 / 7.0e-3 | 0.01 | 1.000 |
| 100 | Ritz q = 2, r = 400 | 1e-6 | 7,170 | 7,835 | 8.9e-1 / 6.4e-3 | 0.01 | 1.000 |

Reading the hard grid:

- **Same eps: the truncated solver stops earlier and less accurately.** At
  p = 10 it takes 9.7k passes instead of 12.3k, and ends 1.7e-3 above the
  optimum instead of 1.1e-3. That is the stopping rule of 4.5.
- **eps/10 in the smoothing rounds where c < τ (the proposed rule).** The
  truncated solver is then more accurate than the full solver at the default
  eps. Per unit of accuracy it costs about as many passes (p = 10) or about
  30% more (p = 100).
  - p = 10: 4.7e-4 / 2.4e-4 (max / median) in 15.4k passes. The full solver
    needs about 15.6k / 15.1k passes for the same max / median
    (log-interpolated between its 1e-5 and 1e-6 runs).
  - p = 100: median 7.0e-3 in 14.2k passes, against about 11k for the full
    solver.
  - Each truncated pass reads one n × n matrix instead of three.
  - eps/10 in every round gives similar results (p = 10: 3.0e-4 in 16.3k
    passes).
  - On the Q1 grid the rule changes nothing: pass counts are identical for
    exact r = 400 and r = 100, and for Ritz q = 2, r = 400.
- **Very tight accuracy needs a larger r.** Below about 1e-4, r = 400
  plateaus at 2e-4 on its worst λ, where the full solver reaches 5e-5.
  r = 100 is too small for this grid.
- **The p = 100 problem is separable.** The optimum at the smallest λ is
  0.0025, so the worst relative gaps are large for every solver. The median
  is the informative number there.
- **Model selection is unaffected.** The selected λ and test accuracy vary
  with eps over the same range for both solvers.
- **Ritz vectors behave like exact eigenvectors.** With q = 2 subspace passes
  they are close in passes and accuracy, even though ‖KK̃ − K̃K‖ ≈ 0.26
  (K̃ ⪰ K holds by the ρ shift). With q = 0 on the Q1 grid, ‖KK̃ − K̃K‖ ≈ 390
  and the solver still converged, with 0.8% more CV passes.

### 4.7 Computing V and τ at scale

- **Top-r eigenpairs.** Use randomized subspace iteration or block Krylov
  (Halko–Martinsson–Tropp; Musco–Musco), then Rayleigh–Ritz.
  - Cost: (q + 2) products of K with an n × (r + 20) block, plus thin QRs,
    i.e. O(n²r) flops, all GEMM.
  - Memory: a few n × (r + 20) blocks.
  - **Measured** at n = 20,000, float32, on the L40S (probe B, section 7),
    against 10.2 s for the full eigh:
    - r = 400: 0.09 s with q = 2 (ρ/e₁ = 2.7e-4) and 0.13 s with q = 4
      (ρ/e₁ = 1.6e-4);
    - r = 1000: 0.24–0.33 s;
    - extra memory: 0.16 units for r = 400, 0.37 for r = 1000.
  - **That spectrum** (p = 100): e₁ = 2693, e₁₀₁ ≈ 38, e₄₀₁ ≈ 2.6,
    e₁₀₀₁ ≈ 1.8.
    - The tail is flat beyond about 400, so a larger r lowers τ very little.
    - At the last Q1 λ, c = 80δ. So c ≥ τ in the first two smoothing rounds,
      as in the n = 3,000 emulation.
- **Notation.** τ₀ ≥ λ_max(WᵀKW) is the unshifted bound on the complement.
  K̃ uses Θ + ρ on V and τ₀ + ρ on the complement, so ρ is added once to
  each block. In the formulas of 4.1 and 4.2, τ then stands for τ₀ + ρ, and
  Θ for Θ + ρ.
- **τ₀.** Run k Lanczos steps on (I − VVᵀ)K(I − VVᵀ) from a random start and
  take τ₀ = θ/(1 − ε).
  - Kuczyński–Woźniakowski: θ ≥ (1 − ε)λ_max with probability at least
    1 − 1.648 √n e^{−√ε (2k − 1)}. With ε = 0.05, k = 30 and n = 10⁵, that is
    about 1 − 1e-3; k = 60 gives about 1 − 1e-9.
  - Cost: k K-products.
  - A deterministic bound: λ_max(WᵀKW) ≤ ‖QKQ‖_F, with
    ‖QKQ‖_F² = ‖K‖_F² − ‖Θ‖_F² − 2‖R‖_F², computed in one pass over K.
    - It equals the root of the sum of squared tail eigenvalues, so on a flat
      tail it is many times e_{r+1}.
    - It is useful only as a cap.
  - The emulation's 1.05 × (50 power iterations) has no such certificate. The
    emulation checked K̃ ⪰ K directly instead (4.6).
- **Ritz vectors do not commute with K.**
  - The shift keeps K̃ ⪰ K exactly: in the [V, W] basis,
    K̃ − K = [[ρI, −B], [−Bᵀ, (τ₀ + ρ)I − WᵀKW]] with ‖B‖ = ‖KV − VΘ‖ = ρ.
  - The shift can be split: any a·b ≥ ρ² works, with +a on V and +b on the
    tail. A small b keeps the tail close to τ₀.
  - But K(K̃ − K) is no longer symmetric, so argument 1 of 4.3 does not apply.
  - K̃ ⪰ K alone does not give descent. A reviewer's example:
    - K = [[99, √98], [√98, 2]], V = e₁, ρ = √98, τ₀ = 2, c = 0.1;
    - γ = [1, −2]ᵀ gives γᵀK(K̃ + cI)⁻¹γ = −0.257 < 0;
    - with the intercept, the full 4.2 step raises F from 674.02 to 976.63.
  - The mechanism: in the [V, W] basis, K = diag(Θ, C) + [[0, B], [Bᵀ, 0]].
    The coupling B can outweigh the positive terms in complement directions
    of small curvature.
  - On realistic kernels it did not happen [observation, n = 3,000]:
    - sym(K(K̃ + cI)⁻¹) was positive definite for every tested c, at p = 10
      and 100, and for Ritz vectors as crude as q = 0, r = 100
      (ρ ≈ 0.9 τ₀);
    - none of 23,659 emulated path steps was an ascent step.
  - Arguments 2 and 3 (fixed point, same acceptance test) still hold.
- **Alternative with K̃ ⪰ K by construction:** a randomized Nyström factor
  (*untested* here).
  - Â = (KΩ)(ΩᵀKΩ)⁺(KΩ)ᵀ = VΛ̂Vᵀ satisfies Â ⪯ K.
  - With τ₀ ≥ λ_max(K − Â) (Lanczos on the PSD matrix K − Â), the matrix
    K̃ = Â + τ₀I = V(Λ̂ + τ₀I)Vᵀ + τ₀(I − VVᵀ) dominates K without any
    residual shift.
  - It has the same shape and cost as above.
  - It does not commute with K either, so it also needs the safeguard.
  - "Nyström" here is only the curvature bound. The problem stays the exact
    one.
- **Safeguard, with exact gradients at no extra product.**
  - Replace the per-iteration product Kα by Kz, and keep Kα up to date: with
    KV stored from the Rayleigh–Ritz setup,
    KP̃γ = (τ + c)⁻¹Kγ + (KV)D_c(Vᵀγ) costs O(nr).
  - One product per iteration then gives:
    - the exact gradient g = (g_b, Kγ), with Kγ = Kz + 2nλKα;
    - each candidate step's directional derivative;
    - F at any step length, in O(n).
  - The fallback step uses K̃ = σI with σ = max(θ₁, τ₀) + ρ ≥ λ_max(K). It
    commutes with K, is a valid majorizer, and its K-image is O(n).
  - **Rule.** Take the fast step (backtracking t ∈ {1, ½, ¼}) if both hold:
    - gᵀΔ_f ≤ η·gᵀΔ_s;
    - Armijo holds on the step actually taken.

    Otherwise take the fallback and reset mul.
  - **Guarantee** [proof sketch]. Every step then lowers F by at least
    κ·gᵀM_s⁻¹g, so ‖g‖ → 0 within a round.
  - With exact eigenvectors M̃ ⪯ M_s, so the fast step always passes. The
    safeguard costs nothing there.
  - Refresh Kα with a real product at the end of each round, against
    rounding drift.
  - Details and the proof sketch: `EIGENDECOMPOSITION_REVIEW_REPLY.md`,
    item 2.
- **Matrix-free K** (optional, for n beyond one stored unit): form products Kx
  in tiles on the fly, KeOps-style.
  - Cost per product: 2n²p flops plus n² exps, compute-bound, instead of
    reading 4n² bytes.
  - *Projected* at n = 100,000 and p = 100: roughly 0.05–0.1 s per product,
    versus 0.125 s to read a stored float32 K at 320 GB/s.
  - Memory: O(n(p + r)).

### 4.8 Cost at n = 20,000, float32 (factorization measured; iterations *projected*)

| | now (measured) | truncated, r = 400 |
|---|---|---|
| factorization | 10.2 s (probe) to 21.4 s (profiled fit) eigh, + 0.3 s error check | 0.09 s for V (q = 2, measured), + ~0.15 s for τ (30 K-products, *projected*) |
| peak memory | 6.01 units (8.98 GiB) | 1.16 units during the factorization (measured): K plus 0.16 units of n × (r + 20) blocks |
| unit reads per iteration | 3 | 1, plus 2 products of n·r |
| whole fit | 66 s (profiled) | roughly 20–30 s if iteration counts hold (*projected*) |

The path's non-matvec overhead (Brent search, KKT test, elementwise work) is
about 13 ms per iteration and does not change. That is why the range is
20–30 s and not 66/3.

### 4.9 Scope

- **Other solvers.** The same structure appears in `cvkdwd`, `cvklogit`,
  `cvkqr`, `cvksqsvm` and `cvkhuber`: their MM steps also apply (K + cI)⁻¹.
  So the change carries over.
- **`is_exact=1`.** It applies K⁻¹ through the eigendecomposition
  (`einv`, `cvksvm.py:568`, 898). It would keep the full eigendecomposition or
  use a CG solve.
- **CV.** The batched CV reuses the whole-data V and τ, just as it now reuses
  U.

### 4.10 Related work

- EigenPro (Ma & Belkin) preconditions kernel gradient descent by flattening
  the top eigendirections, using a subsample's eigendecomposition.
- Falkon preconditions with a Nyström factor.
- Randomized Nyström preconditioning (Frangella, Tropp & Udell, SIAM J.
  Matrix Anal. Appl. 2023) builds the same shape for CG on (A + μI)x = b:
  rank-r Nyström plus a flat tail, valid for any μ.
- Deflation preconditioners do similar things for CG.

**What is different here.**
- The preconditioner is a majorizer: steps are monotone and need no step size.
  This holds for exact eigenvectors (4.3); with Ritz vectors it needs the
  safeguard of 4.7.
- One factorization serves every c = 4nδλ, i.e. all λ, all δ and all folds,
  as the full eigendecomposition does now.

### 4.11 A certificate: the duality gap

The shipped acceptance test (4.3, point 3) does not bound suboptimality. The
unsmoothed problem has a dual that does, and it works for any positive
semidefinite K, singular included.

- **Primal:** P(α, b) = mean(max(0, 1 − yᵢ(Kᵢα + b))) + λαᵀKα.
- **Dual:** D(β) = 1ᵀβ − (β∘y)ᵀK(β∘y)/(4λ), with 0 ≤ β ≤ 1/n and βᵀy = 0.
- **Weak duality:** D(β) ≤ P* ≤ P(α, b). It needs no subgradient choice at the
  elbows, and βᵀy = 0 covers the intercept.
- **Dual point from the iterate.**
  - Margin rows (|rᵢ − 1| ≤ t) get βᵢ = 2λyᵢαᵢ, violators 1/n, the rest 0.
  - Clip to [0, 1/n], and restore βᵀy = 0 with a monotone shift found by
    bisection.
  - Keep the best over t ∈ {1e-2, 1e-3, 1e-4}. Each costs one product with K.
- **Tested against libsvm optima** [observation, n = 3,000]: the Q1 grid at
  p = 10 and 100, and the hard grid at p = 10.
  - D stayed at or below the optimum, and the gap at or above the true excess,
    at every λ.
  - Where the excess is above 1e-5, the gap is 1–4× the excess on the Q1
    grid, and 3–30× on the hard grid.
  - The gap is loose at the two or three largest λ (up to 2.9e-3 against a
    tiny excess). A few projected-gradient steps on D should tighten it
    (*untested*).
- **Rule.** Accept a λ, and each fold, when (P − D)/P ≤ tol_gap.
- **Inner loop.** It can stop on the smoothed stationarity, (g_b, Kγ), which
  the safeguard of 4.7 computes anyway. That test does not depend on the
  preconditioner.

## 5. Approximations that change the problem

These are not recommended for the exact-mode claim:
- Nyström (`low_rank=True`, already in the package);
- random features;
- hierarchical-matrix compression of K.

All three solve a different problem, so the results are no longer exact.

## 6. Recommendation

1. **Restore the in-place factorization (3a).** It takes one unit off the
   peak at no cost in speed and has already been tested. Under cuSOLVER's
   size limit it no longer raises the largest float32 n on a 48 GB card.
2. **The truncated majorizer, prototyped** as
   `torchkm.experimental.SpectralSVMPath` (section 7.2).
   - On the L40S (section 7.3) it matches the full spectrum at the same
     certified gap, 2.8× faster and with a fifth of the memory. It is also
     the only route past cuSOLVER's size limit.
   - Next: the same comparison on real data (the Q1 data sets).
   - If it holds up there, give `cvksvm` an opt-in, e.g. `spectrum_rank=r`,
     with today's full eigendecomposition as the default below the size
     limit.
3. **If the full eigendecomposition stays the default, add spectral
   coordinates (3b)** for faster iterations.

## 7. The GPU probe, the prototype, and the GPU run

### 7.1 The probe (measured, L40S)

`benchmarks/probe_eigh.py`, run at n = 20,000, p = 100, float32. It uses the
same data as `profile_gpu.py`.

**A. Allocations inside `torch.linalg.eigh`** (10.2 s): 1.00 unit, which is
the working copy that becomes U, and 4.01 units of cuSOLVER workspace. So the
6.01-unit peak is K + 1.00 + 4.01. An in-place call can remove only the copy,
which gives 5 units; option 3c gains nothing over 3a.

**B. The top-r eigenpairs by subspace iteration** (block r + 20, then
Rayleigh–Ritz). Errors are relative to e₁ = 2693.

| r | q | time (s) | share of eigh | extra memory (units) | ρ/e₁ | eigenvalue error/e₁ | e_{r+1}/e₁ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0 | 0.03 | 0.3% | 0.050 | 1.1e-2 | 1.0e-2 | 1.4e-2 |
| 100 | 2 | 0.02 | 0.2% | 0.050 | 1.5e-3 | 3.3e-5 | 1.4e-2 |
| 100 | 4 | 0.04 | 0.3% | 0.050 | 9.1e-5 | 9.2e-7 | 1.4e-2 |
| 400 | 0 | 0.06 | 0.6% | 0.157 | 3.8e-3 | 1.4e-3 | 9.5e-4 |
| 400 | 2 | 0.09 | 0.9% | 0.157 | 2.7e-4 | 2.0e-4 | 9.5e-4 |
| 400 | 4 | 0.13 | 1.3% | 0.157 | 1.6e-4 | 1.1e-4 | 9.5e-4 |
| 1000 | 0 | 0.14 | 1.4% | 0.374 | 1.7e-3 | 4.4e-4 | 6.5e-4 |
| 1000 | 2 | 0.24 | 2.3% | 0.374 | 2.1e-4 | 1.3e-4 | 6.5e-4 |
| 1000 | 4 | 0.33 | 3.2% | 0.374 | 1.4e-4 | 8.2e-5 | 6.5e-4 |

- The factorization the proposal needs costs about 1% of the eigh's time and
  a sixth of one unit of memory.
- Its residuals are smaller, relative to e₁, than those of the Ritz vectors
  that worked in the emulation (q = 2, r = 400: 7.0e-4 at n = 3,000).

To rerun the probe:

```bash
pkill -f probe_eigh.py
cd ~/torchkm-revision/repo && git pull --ff-only
conda activate q1
export RESULTS=/home/yzhang705/torchkm-revision/repo/revision_results
nohup python benchmarks/probe_eigh.py --n 20000 --p 100 --dtype float32 > $RESULTS/probe_eigh.log 2>&1 &
head -3 $RESULTS/probe_eigh.log
```

### 7.2 The prototype: `torchkm.experimental.SpectralSVMPath`

It solves the same problem as `cvksvm`: the smoothed hinge, smoothing
rounds, and folds that zero the held-out labels and start from the
whole-data fit at the same λ. `spectrum="full"` uses eigh as `cvksvm` does;
`spectrum="truncated"` uses sections 4.1–4.7. What differs from `cvksvm`:

- **Every λ and every fold stops at a certified duality gap (4.11)**, not at
  the KKT threshold.
- **FISTA momentum with a function-value restart** replaces the relaxation
  factor mul. On one hard λ it needed 4.5–11× fewer iterations at small δ.
- **Smoothing.** δ stops shrinking once the smoothing bias is below half the
  tolerance. The bias bound is δ/4 times the share of rows in the smoothing
  band. From then on, iterations run in chunks, each followed by the
  certificate.
- **Inner stops are scale-free**: a step's predicted decrease against the
  objective. An absolute step-size test calls every step converged when α is
  about 1e-7, as it is at large λ.
- **Warm-started fits** (the next λ, the folds) start one smoothing level
  above the level where their starting point was certified.
- **Bounded work.** A λ, or a batch of folds, stops certifying after
  `fit_cap` iterations and is reported as not converged; a non-finite
  iterate raises an error.
- **Truncated spectrum:**
  - r = 400 Ritz pairs from 4 subspace passes;
  - τ₀ from 40 Lanczos steps with factor 1/0.95;
  - the ρ shift;
  - exact gradients from K z and K V;
  - the safeguard with the scalar fallback.
- Not part of the stable API. Tests are in `tests/test_experimental_spectral.py`.

**CPU validation** (n = 3,000, float64, 50 λ, 10 folds; gap target 1e-3
unless marked). "Excess" is how far the path's objective ends above libsvm's
optimum, as max / median over the 50 λ.

| setting | solver | time (s) | excess, max / median | all certified | n × n reads in iterations | selected λ | CV error | test acc |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Q1 grid, p = 10 | shipped `cvksvm` | 5.2 | 1.9e-2 / 5.6e-7 | (no certificate) | | 0.001 | 0.1683 | 0.830 |
| | full | 26.5 | 1.3e-4 / 4.9e-8 | yes | 10,815 | 0.001 | 0.1643 | 0.828 |
| | truncated | 15.9 | 1.3e-4 / 4.9e-8 | yes | 3,655 | 0.001 | 0.1643 | 0.828 |
| Q1 grid, p = 10, gap 1e-4 | full | 41.9 | 2.9e-5 / 4.9e-8 | yes | 15,257 | 0.001 | 0.1647 | 0.828 |
| | truncated | 23.5 | 3.4e-5 / 4.9e-8 | yes | 4,717 | 0.001 | 0.1643 | 0.828 |
| hard grid, p = 10 | shipped `cvksvm` | 26.7 | 1.1e-3 / 4.1e-4 | (no certificate) | | 3.7e-4 | 0.1640 | 0.825 |
| | full | 64.8 | 9.3e-4 / 2.3e-4 | yes | 23,786 | 7.9e-4 | 0.1640 | 0.827 |
| | truncated | 38.6 | 9.2e-4 / 2.1e-4 | yes | 7,958 | 4.8e-4 | 0.1637 | 0.825 |
| Q1 grid, p = 100 | shipped `cvksvm` | 5.4 | 2.3e-1 / 8.2e-7 | (no certificate) | | 0.0126 | 0.000 | 1.000 |
| | full | 24.9 | 2.0e-4 / 1.9e-7 | yes | 9,251 | 0.0168 | 0.000 | 1.000 |
| | truncated | 15.1 | 1.5e-4 / 1.9e-7 | yes | 3,226 | 0.0168 | 0.000 | 1.000 |
| hard grid, p = 100 (separable) | shipped `cvksvm` | 14.3 | 5.3e-1 / 1.1e-2 | (no certificate) | | 0.01 | 0.000 | 1.000 |
| | full | 384.4 | 9.2e-4 / 3.6e-4 | yes | 116,302 | 0.01 | 0.000 | 1.000 |
| | truncated | 289.6 | 9.4e-4 / 2.7e-4 | path yes; some folds end at 5.5e-3 | 43,537 | 0.01 | 0.000 | 1.000 |

Reading the table:

- **At matched certified accuracy, truncated is 1.3–1.8× faster than full on
  this CPU**, and reads the n × n matrices about 3× less often.
  - It also keeps the same selected λ, CV error and test accuracy, except at
    p = 10 on the hard grid, where the two pick different λ at CV errors
    0.1640 and 0.1637. Iteration counts are within a few percent.
  - No step fell back in any run.
  - On the GPU the reads dominate and the eigendecomposition disappears, so
    the gap there should be larger (*projected*; 7.3 measures it).
- **Certification costs time.** On the first three settings the certified
  solvers take 1.4–5× as long as the shipped solver at its default
  tolerances. On the separable hard grid they take 20–27× as long. The
  shipped solver, for its part, stops 1.9% (p = 10) and 23% (p = 100) above
  the optimum at the smallest λ of the Q1 grid.
- **The separable hard grid is the expensive case, and truncation's limit.**
  - The objective goes down to 0.0025, so a relative gap of 1e-3 needs
    δ ≈ 4e-6. Then c = 4nδλ ≈ 1e-6, far below τ ≈ 1.2, and the truncated
    spectrum's tail directions move at a rate of about c/τ per step.
  - The run in the table ended its folds early. With the final exit rule
    (tighten, then cap each fit at `fit_cap` = 20,000 iterations), a rerun of
    the last 10 λ went past 30 minutes without certifying every fold, and was
    stopped.
  - The full spectrum certifies this grid, in 384 s.
  - So in this regime the truncated spectrum is not the method to use; the
    Q1 grid is where it pays off.

### 7.3 The GPU run

The matched-accuracy experiment: n = 20,000, p = 100, float32, the Q1 grid
(50 λ, 10 folds), gap 1e-3 and then 1e-4. It compares the shipped solver,
full and truncated.

**Measured on the L40S** (`benchmarks/matched_accuracy.py`):

| solver | gap target | time (s) | peak (n × n units) | peak NVML (GiB) | n × n reads | path gap max | fold gap max | fallbacks | selected λ | CV error | test acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped `cvksvm` | its tolerances | 16.8 | 6.12 | 10.3 | | 1.8e-1 | | | 0.00954 | 0.0003 | 1.0000 |
| full | 1e-3 | 37.5 | 6.11 | 10.3 | 9,880 | 7.2e-4 | 8.6e-4 | 0 | 0.0126 | 0.0003 | 1.0000 |
| full | 1e-4 | 47.1 | 6.11 | 10.3 | 13,188 | 8.9e-5 | 9.6e-5 | 0 | 0.0126 | 0.0003 | 1.0000 |
| truncated | 1e-3 | 13.5 | 1.25 | 3.1 | 3,603 | 7.7e-4 | 9.2e-4 | 279 | 0.0126 | 0.0003 | 1.0000 |
| truncated | 1e-4 | 16.7 | 1.25 | 3.1 | 4,481 | 9.3e-5 | 9.9e-5 | 83 | 0.0126 | 0.0003 | 1.0000 |

Reading it:

- **At the same certified gap, truncated is 2.8× faster than full** (13.5
  against 37.5 s at 1e-3; 16.7 against 47.1 s at 1e-4).
  - It needs 2.7–2.9× fewer n × n reads.
  - It selects the same λ, with the same CV error and test accuracy.
- **Peak memory falls from 6.11 to 1.25 units** (10.3 to 3.1 GiB in NVML).
  At 1.25 units, a 48 GB card holds about n = 95,000 in float32 (about
  100,000 at the 1.15 units measured at n = 60,000, below). With eigh the
  limit is 32,768, cuSOLVER's size limit (below); memory alone would
  allow 43,000.
- **Against the shipped solver:** truncated at gap 1e-4 takes the same time,
  16.7 against 16.8 s. The shipped solver ends up to 18% above the optimum
  on its path (certified afterwards), and uses 5× the memory.
- **Fallbacks.** In float32 the safeguard took the fallback step in 279 and
  83 column steps. It never did in the float64 CPU runs.
  - The rate needs `column_iterations` from the run's JSON. The reads column
    is not the denominator: one n × n product serves up to 10 fold columns,
    and reads also count the spectrum, the certificate and the refreshes.
  - Float32 Ritz vectors commute with K only to rounding, and the step checks
    see float32 noise.
  - Every λ and fold still certified.
  - A noise floor for the check in float32 might remove most of them
    (*untested*).
- **Limits of this evidence.**
  - This is one simulated data set, nearly separable at the selected λ (CV
    error 0.0003).
  - Real data (the Q1 data sets) is the next run.

**n = 60,000** (measured, L40S, gap 1e-3):

| solver | gap target | time (s) | peak (n × n units) | peak NVML (GiB) | n × n reads | path gap max | fold gap max | fallbacks | selected λ | CV error | test acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped `cvksvm` | its tolerances | refused by cuSOLVER | | | | | | | | | |
| full | 1e-3 | refused by cuSOLVER | | | | | | | | | |
| truncated | 1e-3 | 81.7 | 1.15 | 16.6 | 3,399 | 8.7e-4 | 8.1e-4 | 83 | 0.001 | 0.0004 | 0.9995 |

- **Only the truncated spectrum runs.** cuSOLVER's eigh refused n = 60,000
  (measured).
  - The shipped solver and the full spectrum both stopped in cuSOLVER's
    workspace-size query (`cusolverDnXsyevd_bufferSize`:
    CUSOLVER_STATUS_INVALID_VALUE), before anything was allocated.
  - So the refusal is about the problem size, not the memory. Even with the
    size accepted, the eigh-based solvers would peak at 6.1 units (one unit is
    13.4 GiB), about 82 GiB: more than the card's 48 GB (*projected*).
  - cuSOLVER refuses every n above 32,768 in float32 (below).
- **Every λ and fold certified:** path gaps up to 8.7e-4, fold gaps up to
  8.1e-4.
- **The number of n × n reads did not grow with n:** 3,399, against 3,603 at
  n = 20,000.
  - The time grew 6.1× (13.5 → 81.7 s) for 9× the entries.
  - Streaming K once takes at least 16.7 ms at the card's 864 GB/s, so the
    reads take at least 57 of the 82 s (arithmetic, not measured). At
    n = 20,000 the same bound is 6.7 of 13.5 s.
- **Memory: 1.15 units** (15.4 GiB allocated).
  - K and the benchmark's test kernel take 1.1 units. The solver's own share
    fell from 0.15 units at n = 20,000 to 0.05 here. Both are about seven
    n × 420 blocks, so the share is O(nr), not O(n²).
  - On a 48 GB card that puts the float32 limit near n = 100,000
    (*projected*).
- **The selected λ is the smallest on the grid (1e-3).** The CV optimum may
  lie below the grid. This does not affect the time and memory comparison.
- **Fallbacks:** 83, against 279 at n = 20,000 and the same gap.

**cuSOLVER's size limit** (measured with `benchmarks/probe_eigh_size.py`;
PyTorch 2.6.0+cu124, CUDA 12.4, L40S):

| n | float32 | float64 |
|---|---|---|
| 16,000; 17,000; 20,000; 23,000; 23,500; 32,500 | accepted | accepted |
| 32,766 | accepted | not tested |
| 32,767; 32,768 | accepted | accepted |
| 32,769; 32,800; 32,900 | refused | not tested |
| 33,000; 46,000; 46,500 | refused | refused |
| 60,000 | refused | not tested (two n × n matrices do not fit) |

- "Accepted" means the workspace-size query passed. The probe caps memory so
  that nothing is factorized.
- The limit is the same in both precisions, so it counts elements, not
  bytes.
- In float32 the limit is exactly n = 32,768 = 2¹⁵. A power of two points to
  a fixed bound inside cuSOLVER, not to memory. Neither 32-bit count guessed
  earlier matches it: LAPACK's syevd workspace, 1 + 6n + 2n², overflows from
  n = 32,767 and 2n² from 32,768, yet both sizes are accepted.
- TorchKM now records the limit (`torchkm.memory.EXACT_MODE_MAX_N_CUDA`).
  Above it the exact solvers raise an error that names it, instead of
  cuSOLVER's, and `max_exact_n` stops at it.
- Newer CUDA releases may differ (*untested*).

```bash
pkill -f matched_accuracy.py
cd ~/torchkm-revision/repo && git pull --ff-only
conda activate q1
export RESULTS=/home/yzhang705/torchkm-revision/repo/revision_results
nohup python benchmarks/matched_accuracy.py --n 20000 --gaps 1e-3 1e-4 --out $RESULTS/matched_20k.json > $RESULTS/matched_20k.log 2>&1 &
head -3 $RESULTS/matched_20k.log
```

Then n = 60,000, which only the truncated spectrum fits (about 1 unit of
14.4 GB, against 6 units, 86 GB, for eigh):

```bash
nohup python benchmarks/matched_accuracy.py --n 60000 --solvers truncated --gaps 1e-3 --out $RESULTS/matched_60k.json > $RESULTS/matched_60k.log 2>&1 &
```

At n = 60,000 the eigh-based solvers stop with cuSOLVER's error, which the
script records as a row:

```bash
nohup python benchmarks/matched_accuracy.py --n 60000 --solvers shipped full --gaps 1e-3 --out $RESULTS/matched_60k_eigh.json > $RESULTS/matched_60k_eigh.log 2>&1 &
```

Where cuSOLVER starts refusing: memory is capped so that nothing is
factorized, so each size takes about a second.

```bash
python benchmarks/probe_eigh_size.py > $RESULTS/probe_eigh_size.log 2>&1
python benchmarks/probe_eigh_size.py --dtype float64 >> $RESULTS/probe_eigh_size.log 2>&1
cat $RESULTS/probe_eigh_size.log
```

Each log ends with one table:
- time;
- peak memory in n × n units and in NVML GiB;
- n × n reads;
- certified gaps;
- fallbacks;
- selected λ, CV error, test accuracy.

The hard grid is optional (`--grid hard`). For p = 100 it is the separable
stress case above.

## 8. Questions for the reviewer

Round 1 answers, with the corrections they led to, are in
`EIGENDECOMPOSITION_REVIEW_REPLY.md`.

1. Is the majorization argument in 4.3 correct, including over-relaxation
   with mul ∈ [1, 2)? The claim: M̃ = M + diag(0, K(K̃ − K)/(2δ)) ⪰ M when K̃
   commutes with K and K̃ ⪰ K.
2. With Ritz vectors, K̃ does not commute with K. Which guarantee survives? Is
   the ρ-shift plus a monotonicity restart enough? Or is there a construction
   that commutes exactly and is still cheap for every c, for example K̃ = f(K)
   via a polynomial filter?
3. What is the best practical upper bound on λ_max((I − VVᵀ)K(I − VVᵀ))?
   Randomized Lanczos gives a probabilistic one (4.7). Is there a cheap
   deterministic one?
4. What stopping test should the inner loop use? The step-size test fires
   earlier under a looser majorizer. eps/10 in the rounds where c < τ works on
   these grids. The scaling eps·(c/(τ + c))² fails when c → 0 (4.5). Should
   the test use a preconditioner-free quantity instead, such as the smoothed
   KKT residual ‖γ‖/n?
5. Is anything known about the iteration count of MM with a truncated spectral
   majorizer as c → 0 (small λ, late smoothing rounds)?
6. cuSOLVER syevd's workspace is one 4.01-unit allocation (float32,
   n = 20,000). Would syevdx or a two-stage solver need less without losing
   speed?
7. cuSOLVER's syevd refuses n > 32,768 (CUDA 12.4, section 7.3). Is there a
   dense GPU eigensolver without this limit and at similar speed (syevj,
   syevdx, an ILP64 MAGMA, a newer cuSOLVER)? Or is the truncated spectrum
   the practical route past it?

## Appendix: the emulation's core (CPU, float64)

The solver code is unchanged. `torch.linalg.eigh` and the two intercept
helpers are replaced as follows; `c = 4 n delta lambda`.

```python
# exact r: keep the top r eigenpairs, flatten the rest to tau = e_{r+1}
e, U = torch.linalg.eigh(K)              # ascending
e[: n - r] = e[n - r - 1]                # U diag(e) U' = Kt exactly
# Ritz q: V from q passes of subspace iteration (block r + 20), T = V'KV,
# rho = ||K V - V T||, tau = 1.05 * lambda_max estimate on the complement + rho,
# U = [W, V] with W an orthonormal basis of the complement,
# e = [tau, ..., tau, T + rho]

w = U.T @ (K @ ones)                     # so that v = (Kt + cI)^-1 K 1 = U (w / (e + c))

def gval(Usum, lpUsum_d, delta, lam, n, eps):     # 1 / (n + 4 n delta eps - 1'v)
    return 1 / (n + 4 * n * delta * eps - lpUsum_d @ w)

def hval(z, alp, s_d, v_d, delta, lam, n, eps):   # 1'z + 2 n eps b - v'(z + 2 n lam alpha)
    v = U @ (w / (e + 4 * n * delta * lam))
    return z.sum(0) - v @ z - 2 * n * lam * (v @ alp[1:]) + 2 * n * eps * alp[0]
```

With the full spectrum (r = n), these helpers reproduce the shipped solver:
identical iteration counts, and objectives equal to within 3e-6.

The Ritz rows used τ = 1.05 × (50 power iterations) + ρ. That carries no
probabilistic certificate. Instead, the emulation computed λ_max(WᵀKW)
exactly and checked min eig(K̃ − K) > 0 in every run. A method should use
Lanczos with 1/(1 − ε) (4.7).
