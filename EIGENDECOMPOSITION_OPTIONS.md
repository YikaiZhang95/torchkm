# Large kernel eigendecompositions in TorchKM: less memory, same speed

**Question.** Can the eigendecomposition of the n x n kernel use less GPU
memory without getting slower?

**How to read this.** This is a proposal for review. Section 1 is measured on
the L40S (48 GB). Section 4.6 comes from a CPU emulation at n = 3,000. Anything
marked *projected* or *untested* has not been run. A "unit" means one n x n
matrix: 4n² bytes in float32, 8n² in float64.

## Summary

- **The eigendecomposition sets the memory ceiling.** A fit peaks at 6 units:
  the kernel K (1), the eigenvectors U (1), and about 4 units of cuSOLVER
  workspace. Once the factorization is done, only K and U stay resident
  (2 units).
- **Keeping a full eigendecomposition, the cheapest cut is 6 → 5 units.**
  Factorize in place and rebuild K afterwards. This was measured at the same
  speed and has since been reverted. MAGMA or host LAPACK get the peak down to
  2 units, but run 4× and 12× slower (measured). We know of no full dense
  eigensolver that gives both lower memory and cuSOLVER's speed.
- **Proposal: don't compute the full eigendecomposition.**
  - The solver only needs a curvature matrix K̃ ⪰ K for which (K̃ + cI)⁻¹ is
    cheap for every c.
  - Keep the top r eigenpairs exactly and replace the rest of the spectrum by
    one constant τ ≥ e_{r+1}.
  - The iteration keeps the same fixed point. Every λ is still accepted by the
    same KKT test on the true K, so the answer is exact, not approximated.
  - The O(n³) eigendecomposition becomes O(n² r) matrix products.
  - The eigensolver workspace disappears: the peak is about 1 unit plus n·r.
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
- **Open points for review** (section 8):
  - the majorization argument;
  - Ritz vectors, which commute with K only approximately;
  - how to get a rigorous τ;
  - the stopping rule.

## 1. Where the memory and time go (measured, L40S 48 GB)

| run | n | dtype | eigh time | peak during eigh (units) | resident after |
|---|---:|---|---:|---:|---:|
| whole fit, cuSOLVER (torch.linalg.eigh) | 20,000 | float32 | 21.4 s of 66.0 s (32%) | 6.01 (8.98 GiB) | 2.01 (K, U) |
| eigh alone, cuSOLVER | 16,100 | float64 | 21.5 s | 6.01 | 2 |
| eigh alone, MAGMA (one-stage) | 16,100 | float64 | 85.4 s | 2.00 | 2 |
| eigh alone, host LAPACK (IDAS CPU) | 16,100 | float64 | 253.6 s | 2.00 on the GPU | 2 |

- The table uses the PyTorch allocator peak. NVML reads 0.4–0.7 units more
  (float64 Q1 runs on a7a, a8a and w7a); that difference is allocator slack.
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
| c. Call cuSOLVER directly: in place with an explicit workspace, or syevdx / syevj | 1 + workspace | 1–2 | same (syevj slower) | *untested*; probe A in section 7 measures the workspace | moderate |
| d. MAGMA one-stage or host LAPACK | 2 | 2 | 4× / 12× slower | measured | in git history |
| e. Two-stage tridiagonalization (MAGMA dsyevdx_2stage, ELPA) | 2–3 | 2 | unknown; two-stage beats one-stage at large n | *untested*; not exposed by PyTorch | high |
| f. Several GPUs (cusolverMg, cuSOLVERMp) | 6/G per GPU | | similar | *untested*; not exposed by PyTorch | high |

- **Caveat for option b.** It replaces K by UEUᵀ: a 1e-13 relative change in
  float64 and about 1e-6 in float32. In exchange, U(e + γ)Uᵀ dominates that
  kernel exactly, so float32 would no longer need the error shift.
- **Conclusion.** With a full eigendecomposition, the peak cannot drop below
  the eigensolver's output plus its workspace, which is about 5 units with
  cuSOLVER. Going lower (MAGMA, host) costs speed.

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

### 4.3 Why the answer stays exact

1. **Majorization (exact eigenvectors).** K̃ commutes with K and K̃ ⪰ K, so
   K(K̃ − K) ⪰ 0 and M̃ ⪰ M ⪰ ∇²F. Each step minimises a quadratic upper
   bound of F. With over-relaxation,
   F(θ − mul·M̃⁻¹g) ≤ F(θ) − mul(1 − mul/2)·gᵀM̃⁻¹g, so F still decreases
   monotonically for mul < 2.
2. **Fixed point.** The update is zero exactly when ∇F = 0, for any invertible
   M̃. So r, τ and the accuracy of V change the route to the solution, not the
   solution.
3. **Certificate.** A λ, or a fold, is accepted only by the existing test on
   the unsmoothed hinge, KKT = z/n + 2λα. That test uses residuals
   y(b + Kα) computed with the true K and never touches the eigendecomposition.

### 4.4 What truncation costs in speed

Let D mark the margin rows (φ″ = 1/(2δ) there). One step multiplies the error
by I − mul·P̃(DK + cI). Its eigenvalues μ solve

  (KDK + cK)x = μ(KK̃ + cK)x,

which is a symmetric pencil when K̃ commutes with K.

- **Worst case is unchanged.** Full spectrum: μ ∈ [c/(e₁ + c), 1].
  Truncated: μ ∈ [c/(e₁ + c), 1] too, since K̃'s largest eigenvalue is still
  e₁. So the worst-case contraction per iteration is the same.
- **More directions are slow.** A direction with eigenvalue eᵢ < τ moves from
  μ = (dᵢeᵢ + c)/(eᵢ + c) to (dᵢeᵢ + c)/(τ + c), where dᵢ ∈ [0, 1] is the
  direction's weight on margin rows. When c = 4nδλ ≪ τ, many directions are
  slow instead of a few.
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
  - *Projected* at n = 20,000, r = 400, q = 2: about 1.3e12 flops, well under
    a second on the L40S, versus 21.4 s for eigh. Probe B in section 7
    measures it.
- **τ.** Run 30 Lanczos steps on (I − VVᵀ)K(I − VVᵀ) from a random start and
  take τ = θ/0.95 + ρ.
  - Kuczyński–Woźniakowski: θ ≥ 0.95 λ_max with probability at least
    1 − 1.648 √n e^{−√0.05 (2k − 1)}. For k = 30 and n = 10⁵ that is about
    1 − 1e-3.
  - Cost: 30 K-products.
- **Ritz vectors do not commute with K.**
  - The shift keeps K̃ ⪰ K exactly: in the [V, W] basis,
    K̃ − K = [[ρI, −B], [−Bᵀ, (τ + ρ)I − WᵀKW]] with ‖B‖ = ‖KV − VΘ‖ = ρ.
  - But K(K̃ − K) is no longer symmetric, so argument 1 of 4.3 does not apply.
  - Arguments 2 and 3 (fixed point, KKT certificate) still hold.
- **Alternative with K̃ ⪰ K by construction:** a randomized Nyström factor
  (*untested* here).
  - Â = (KΩ)(ΩᵀKΩ)⁺(KΩ)ᵀ = VΛ̂Vᵀ satisfies Â ⪯ K.
  - With τ₀ ≥ λ_max(K − Â) (Lanczos on the PSD matrix K − Â), the matrix
    K̃ = Â + τ₀I = V(Λ̂ + τ₀I)Vᵀ + τ₀(I − VVᵀ) dominates K without any
    residual shift.
  - It has the same shape and cost as above.
  - "Nyström" here is only the curvature bound. The problem stays the exact
    one.
- **Safeguard.** F is available every iteration for O(n) extra work: the loss
  from the residuals, and αᵀKα from the Kα already computed. If a step raises
  F, reject it, reset mul (told = 1) and inflate τ. Accepted steps then never
  raise F, whatever V and τ are.
- **Matrix-free K** (optional, for n beyond one stored unit): form products Kx
  in tiles on the fly, KeOps-style.
  - Cost per product: 2n²p flops plus n² exps, compute-bound, instead of
    reading 4n² bytes.
  - *Projected* at n = 100,000 and p = 100: roughly 0.05–0.1 s per product,
    versus 0.125 s to read a stored float32 K at 320 GB/s.
  - Memory: O(n(p + r)).

### 4.8 Cost at n = 20,000, float32 (*projected* from the measured unit costs)

| | now (measured) | truncated, r = 400 |
|---|---|---|
| factorization | 21.4 s eigh + 0.3 s error check | ~0.5 s (subspace iteration and τ) |
| peak memory | 6.01 units (8.98 GiB) | ~1.1 units: K (1.5 GiB) plus n × (r + 20) blocks (~0.1 GiB) |
| unit reads per iteration | 3 | 1, plus 2 products of n·r |
| whole fit | 66 s | roughly 20–30 s if iteration counts hold |

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
- One factorization serves every c = 4nδλ, i.e. all λ, all δ and all folds,
  as the full eigendecomposition does now.

## 5. Approximations that change the problem

These are not recommended for the exact-mode claim:
- Nyström (`low_rank=True`, already in the package);
- random features;
- hierarchical-matrix compression of K.

All three solve a different problem, so the results are no longer exact.

## 6. Recommendation

1. **Restore the in-place factorization (3a).** It takes one unit off the
   peak at no cost in speed and has already been tested.
2. **Prototype the truncated majorizer (section 4) in `cvksvm`** behind an
   opt-in, e.g. `spectrum_rank=r` (default: today's full eigendecomposition).
   Use eps/10 in the smoothing rounds where c < τ, and add the objective
   safeguard (4.7).
   - Validate it on the L40S at n = 20,000 against the full method.
   - Then run n = 50,000–100,000, where the full method cannot run.
3. **If the full eigendecomposition stays the default, add spectral
   coordinates (3b)** for faster iterations.

## 7. Next measurements (GPU)

`benchmarks/probe_eigh.py`, added with this note, takes a few minutes on the
L40S. It measures two things:

- **A. Every device allocation inside `torch.linalg.eigh`.** This shows
  whether the ~4 extra units are one cuSOLVER workspace (then option 3c gains
  nothing over 3a) or something PyTorch adds.
- **B. The cost of the top-r eigenpairs by subspace iteration**, for
  r = 100, 400, 1000 and q = 0, 2, 4, next to the full eigh time: time, extra
  memory, residual ρ and eigenvalue error. These replace the projection in 4.8.

```bash
pkill -f probe_eigh.py
cd ~/torchkm-revision/repo && git pull --ff-only
conda activate q1
export RESULTS=/home/yzhang705/torchkm-revision/repo/revision_results
nohup python benchmarks/probe_eigh.py --n 20000 --p 100 --dtype float32 > $RESULTS/probe_eigh.log 2>&1 &
head -3 $RESULTS/probe_eigh.log
```

After that, build a prototype of section 4 in `cvksvm` and measure it against
the full method at n = 20,000: time, peak memory, passes, objective, selected
λ. Then run it at n = 50,000–100,000.

## 8. Questions for the reviewer

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
   these grids. The scaling eps·(c/(τ + c))² fails when c → 0 (4.5). Should the test use a
   preconditioner-free quantity instead, such as the smoothed KKT residual
   ‖γ‖/n?
5. Is anything known about the iteration count of MM with a truncated spectral
   majorizer as c → 0 (small λ, late smoothing rounds)?
6. Is cuSOLVER syevd's ~4-unit workspace inherent? Would syevdx or a
   two-stage solver reduce it without losing speed?

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
