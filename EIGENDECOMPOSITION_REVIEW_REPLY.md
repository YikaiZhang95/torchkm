# Reply to the review of EIGENDECOMPOSITION_OPTIONS.md

This replies to the six points and the full-step counterexample in
`CLAUDE_DISCUSSION.md`. I did not receive `EIGENDECOMPOSITION_REVIEW.md`; if
it has further points, send it and I will answer those too.

**Which code this is about.** The review ran on `main` at b64ead6 (v4.3.2).
That is why the benchmark scripts and the cited commits looked absent, and why
the solver still converted to float64 and built `eU`. The memo, the scripts
and the solver changes it cites are on the branch `claude/great-faraday-id8jaw`.

**Labels used below.**
- **[proof]**: an argument that holds under the stated assumptions.
- **[finite tolerance]**: a statement about a finite stopping rule.
- **[observation]**: a measurement on the named cases only.
- **[projection]**: not measured.

**New checks for this reply**, all float64 on CPU:
- the reviewer's numbers, recomputed;
- ascent steps counted in the emulation;
- the smallest eigenvalue of sym(KP̃) for realistic factorizations;
- a duality-gap certificate tested against libsvm optima.

The n = 3,000 problems are the ones in the memo's section 4.6.

## Positions at a glance

| item | position | change to the memo |
|---|---|---|
| 1. Ritz shift can produce ascent | agree, and verified | descent is claimed only for exact eigenvectors; the Ritz case gets a safeguard |
| 2. rejection alone is not progress | agree | safeguarded step with a commuting fallback. Exact gradients cost no extra product with K |
| 3. KKT threshold is not a certificate | agree | a primal–dual gap on the unsmoothed hinge problem, tested against libsvm |
| 4. τ₀ and ρ notation, the 1.05 factor, deterministic bounds | agree; Frobenius identity verified | notation fixed; Lanczos with 1/(1−ε); optional split shift |
| 5. the dᵢ rate formula | agree | formula labelled heuristic; the interval bound stated as the proven part |
| 6. evidence and a matched-accuracy experiment | agree | experiment specified below |

## 1. The counterexample

**Both examples reproduce exactly** [proof, by computation]:
- γᵀK(K̃ + cI)⁻¹γ = −0.256672402770; the eigenvalues of sym(KP̃) are −0.0517 and 1.1267.
- The full step with intercept:
  - Δb = −212.781502345, Δα = [1.933784211, 18.065885564]ᵀ;
  - gᵀΔ = +259.738965559;
  - F goes from 674.021002536 to 976.627202358.

**What it refutes.** Any reading that K̃ ⪰ K alone gives a descent direction
when K̃ does not commute with K.
- The memo claimed majorization only for exact eigenvectors (4.3, point 1)
  and said that argument does not apply to Ritz vectors (4.7).
- But its safeguard sentence promised only that accepted steps never raise F.
  That is monotonicity, not progress, so the gap you point out is real.

**Mechanism.**
- In the basis [V, W], K = K_V + K_off:
  - K_V = diag(Θ, C) commutes with K̃;
  - K_off = [[0, B], [Bᵀ, 0]], with ‖B‖ = ‖KV − VΘ‖ = ρ.
- With A₁ = Θ + (ρ + c)I and a₂ = τ₀ + ρ + c, and u = P̃γ = [u₁; u₂]:
  - γᵀKP̃γ = u₁ᵀA₁Θu₁ + u₁ᵀ(A₁ + a₂I)Bu₂ + a₂·u₂ᵀCu₂.
- The cross term can outweigh the other two when B couples V to complement
  directions of small curvature. In your 2×2 example, ‖B‖/λ_max(C) = 4.9.

**Realistic factorizations** [observation, n = 3,000]:
- The smallest eigenvalue of sym(KP̃), for c ∈ {12, 1, 0.1, 1e-3, 1e-5}:

  | K̃ | ρ | τ₀ | p = 10 | p = 100 |
  |---|---:|---:|---:|---:|
  | exact r = 400 | 0 | 0.83 / 0.89 | +3.8e-5 to +5.8e-4 | +1.6e-2 to +2.4e-1 |
  | Ritz q = 2, r = 400 | 0.29 / 0.31 | 1.00 / 1.04 | +3.6e-5 to +3.7e-4 | +1.6e-2 to +1.6e-1 |
  | Ritz q = 0, r = 100 | 7.1 / 5.5 | 7.9 / 7.5 | +1.8e-5 to +3.2e-5 | +8.5e-3 to +1.6e-2 |

  (ρ and τ₀ are given as p = 10 / p = 100.) So no ascent direction exists for
  these K̃, even when ρ/τ₀ ≈ 0.9.
- In the emulation, not one of 23,659 path steps was an ascent step. That
  covers both grids and all configurations; 10,320 of those steps used Ritz
  vectors, including Ritz q = 0, r = 100, where ‖KK̃ − K̃K‖ ≈ 1.5e3. It was
  measured with gᵀΔ = g_bΔb + γᵀ(KΔα), which needs no extra product.
- This is not a guarantee. The safeguard below makes it one.

## 2. The safeguard

**Agreed.** Rejecting increases does not guarantee progress. Inflating τ₀
leaves the coupling B in the retained directions untouched.

**What makes a rigorous safeguard cheap: exact gradients at no extra cost**
[proof, algebra; checked to 2e-16].
- Today each path iteration does one product with K (Kα) and two with U.
- Instead, form Kz, and keep Kα up to date without a product.
- Then Kγ = Kz + 2nλKα, which is the exact α-gradient.
- With KV stored at setup (it comes out of Rayleigh–Ritz, n × r):
  - KP̃γ = (τ + c)⁻¹Kγ + (KV)D_c(Vᵀγ), with D_c = (Θ + c)⁻¹ − (τ + c)⁻¹;
  - so KΔα costs O(nr), and Kα updates without a product.
- One product with K per iteration then yields all of the following:
  - the exact gradient g = (g_b, Kγ);
  - the directional derivative of any candidate step;
  - F at any trial step length, in O(n);
  - the update of Kα.
- Refresh Kα with a real product at the end of each smoothing round, to stop
  rounding drift.

**Safeguarded step** (within one smoothing round, where F is L-smooth):
1. Compute the fast step Δ_f (the memo's 4.2 step) and its directional
   derivative d_f = gᵀΔ_f.
2. Compute the fallback Δ_s: the same formulas with K̃ = σI and
   σ = max(θ₁, τ₀) + ρ ≥ λ_max(K).
   - It commutes with K and is a valid majorizer.
   - It costs O(n): its K-image is −2δ(σ + c)⁻¹Kγ − Δb_s(σ + c)⁻¹K1.
   - Its directional derivative is d_s = −gᵀM_s⁻¹g.
3. Accept the fast step if both hold, backtracking t ∈ {1, ½, ¼}:
   - d_f ≤ η·d_s, with η ∈ (0, 1];
   - Armijo on the step actually taken (including mul):
     F(θ + t·Δ) ≤ F(θ) + σ_A·t·gᵀΔ.

   Otherwise take the fallback, and reset mul (told = 1).

**Guarantee** [proof sketch]. Assume K ≻ 0, or restrict to range(K), and
δ > 0.
- Every accepted step lowers F by at least
  κ·gᵀM_s⁻¹g ≥ κ‖g‖²/λ_max(M_s), with κ = min(σ_A·t_min·η, ½).
- F is bounded below, so ‖g‖ → 0 within the round.

**With exact eigenvectors the safeguard never triggers.**
- K̃ ⪯ σI and K̃ commutes with K, so M̃ ⪯ M_s.
- Hence |d_f| = gᵀM̃⁻¹g ≥ gᵀM_s⁻¹g = |d_s|.
- The fast step is a true MM step, so Armijo holds at t = 1 whenever
  σ_A ≤ 1 − mul/2.

**The fallback is slow** (its condition number is e₁/c), but it is only a
fallback. On the realistic cases above it would never be taken
[observation].

**An alternative fast step with unconditional descent:**
Δα = −2δ[K̃(K̃ + cI)]⁻¹Kγ.
- It uses a symmetric positive definite preconditioner, so it descends for
  any V.
- But it damps tail directions by eᵢ/τ, so I would not use it as the main
  step.

## 3. Stopping and a certificate

**Agreed** [finite tolerance]. The shipped acceptance test is not a
certificate:
- it checks ‖z/n + 2λα‖² < KKTeps in coefficient space;
- it takes the elbow subgradient as −y/2;
- the intercept is handled separately, by the Brent search.

The memo's libsvm comparison shows it. On the Q1 grid (KKTeps 1e-3), the
shipped solver stops 1.9% (p = 10) and 29% (p = 100) above the optimum at
λ = 1e-3. The memo's word "exact" should mean "exact kernel, same stopping
test", and it will say that.

**Proposal: a duality gap on the unsmoothed problem.**
- Primal: P(α, b) = mean(max(0, 1 − yᵢ(Kᵢα + b))) + λαᵀKα.
- Dual: D(β) = 1ᵀβ − (β∘y)ᵀK(β∘y)/(4λ), subject to 0 ≤ β ≤ 1/n and βᵀy = 0.
- Weak duality, D(β) ≤ P* ≤ P(α, b), holds for any positive semidefinite K,
  singular included. It needs no subgradient choice at the elbows, and βᵀy = 0
  covers the intercept.
- **Dual point from the iterate.**
  - Margin rows (|rᵢ − 1| ≤ t) get βᵢ = 2λyᵢαᵢ.
  - Violators get 1/n; the rest get 0.
  - Clip to [0, 1/n], then restore βᵀy = 0 by a monotone shift of all βᵢ,
    found by bisection.
  - Try t ∈ {1e-2, 1e-3, 1e-4} and keep the best. Each try costs one product
    with K.
- **Tested against libsvm** [observation]: the Q1 grid at p = 10 and 100, and
  the hard grid at p = 10.
  - D was at most the libsvm optimum at every λ, and the gap was at least the
    true excess at every λ.
  - With margin-based points on the Q1 grid:
    - where the true excess is above 1e-5, the gap is 1–4× the excess. For
      example, at λ = 0.0518 (p = 10): excess 7.3e-4, gap 7.3e-4;
    - where the excess is tiny, the gap stays below 2e-5, except at the two
      or three largest λ, where it reaches 2.9e-3.
  - On the hard grid the gap is 3–30× the excess.
  - The dual point built from α alone is much worse: the gap is at least 5e-3
    at every λ.
  - A few projected-gradient steps on D, one product each, should tighten the
    loose cases [projection].
- **Rule.** Accept a λ, and each fold, when (P − D)/P ≤ tol_gap.
- **Inner loop.** Because gradients are now exact and free (item 2), the inner
  loop can stop on the smoothed stationarity, i.e. g_b and Kγ, scaled.
  That test does not depend on the preconditioner, so it also answers the
  memo's question 4. The step-size test becomes only a heuristic early exit.

## 4. τ₀, ρ and bounds

- **Notation: agreed.** τ₀ is the certified upper bound on λ_max(WᵀKW), with
  no shift. The retained block is then Θ + ρ and the tail is τ₀ + ρ.
  - Section 4.7 of the memo wrote τ = θ/0.95 + ρ and then (τ + ρ)I − WᵀKW,
    which adds ρ twice. Fixed.
- **The shift can be split.**
  - [[aI, −B], [−Bᵀ, bI]] ⪰ 0 exactly when ab ≥ ρ².
  - So the tail can take +b and the retained block +ρ²/b. A small b keeps τ
    close to τ₀.
- **The 1.05 factor: agreed.** It is below 1/0.95, and the power iteration
  behind it carries no useful Kuczyński–Woźniakowski certificate at n = 3,000.
  - The emulation did not rely on it: it computed λ_max(WᵀKW) exactly and
    checked min eig(K̃ − K) > 0 in every run.
  - For the method: Lanczos with k = 30–60 steps and τ₀ = θ/(1 − ε).
  - Its small failure probability is covered by the safeguard in item 2.
- **Frobenius identity: verified.** It holds in exact algebra, since
  ‖R‖_F = ‖WᵀKV‖_F, and numerically to 0.0 relative difference. The bound
  λ_max(C) ≤ ‖C‖_F = (‖K‖_F² − ‖Θ‖_F² − 2‖R‖_F²)^{1/2} is deterministic and
  costs one pass over K. But:
  - It equals the root of the sum of squared tail eigenvalues. On a flat tail
    (n = 20,000, p = 100: e₄₀₁ ≈ 2.6, e₁₀₀₁ ≈ 1.8) that is many times
    e_{r+1}.
  - The subtraction needs float64 accumulation.
  - So it works as a deterministic cap or fallback, not as the working τ₀.
  - The trace bound, n − Σθ, is useless.
- **ρ ≈ 28% of e_{r+1}** at n = 20,000 (r = 400, q = 2: ρ = 0.73, e₄₀₁ = 2.56).
  - The tail value rises by that much plus the Lanczos margin, so directions
    with c ≪ τ converge about 20–25% slower.
  - q = 4 brings ρ to 17% of e₄₀₁ for +0.04 s (measured setup times).
  - Since ρ also scales the coupling B in item 1, extra passes are the cheap
    fix for both. The split shift reduces the first effect only.

## 5. The rate formula

**Agreed.**
- The per-direction μ = (dᵢeᵢ + c)/(τ + c) assumes the margin curvature D is
  diagonal in K's eigenbasis. It is a heuristic.
- The proven statement [proof] is this. For K̃ ⪰ K commuting with K and any
  0 ⪯ D ⪯ I, the generalized eigenvalues of (KDK + cK, KK̃ + cK) lie in
  [c/(ẽ_max + c), 1].
  - That is the full method's interval when ẽ_max = e₁.
  - It covers the linearized step at a fixed margin set, with the intercept
    left out.
- Also agreed: r changes speed, and where a finite tolerance or iteration cap
  stops. It does not change the limit point.

## 6. Evidence and the minimal matched-accuracy GPU experiment

**Agreed.**
- The emulation measures iterations and accuracy, not GPU time.
- The probe measured setup time and memory only.
- Equal selected λ is weak evidence when the baseline objective is itself far
  from the optimum. The memo's hard-grid table compares against libsvm optima
  for that reason.

**Minimal matched-accuracy experiment** [projection of what to run]:
- **Setup.**
  - Data: n = 20,000, p = 100, float32, L40S.
  - Grids: the Q1 grid (50 λ, 10 folds) and one hard grid.
  - Solvers:
    - the shipped full-eigh solver;
    - the truncated solver: r = 400, q = 4, Lanczos τ₀, the safeguard of
      item 2, and exact-gradient inner stopping.
- **Matching.** Both solvers stop on the same certified relative gap,
  1e-3 and 1e-4, for every path λ. The folds use the same rule on their own
  subproblems.
- **Report:**
  - wall time by phase (setup, path, CV);
  - peak memory, allocator and NVML;
  - products with K, and fallback count;
  - certified gaps;
  - selected λ, CV error, test accuracy.
- **Then** n = 60,000 with the truncated solver only, where the full solver
  does not fit, reporting the same.
- **Smallest useful version:** path only (no CV), Q1 grid, gap 1e-3.

## The memo's six questions, answered

1. **Majorization.** Correct for K̃ ⪰ K that commutes with K, including the
   intercept and over-relaxation with mul < 2. It relies on M̃ ⪰ M ⪰ ∇²F
   holding everywhere, which is true because φ_δ″ ≤ 1/(2δ).
   - For singular K the argument holds on range(K). The step's null-space
     component leaves Kα unchanged.
2. **Ritz vectors.**
   - The fixed point and the acceptance test survive.
   - Descent does not survive in general (item 1).
   - The safeguard of item 2 restores convergence at the same cost of one
     product with K per iteration.
3. **τ bound.**
   - Lanczos with the Kuczyński–Woźniakowski bound gives a probabilistic one.
   - The Frobenius bound is deterministic but loose.
   - The safeguard is the backstop.
4. **Stopping.**
   - Inside a round: stationarity with exact gradients, which cost nothing
     extra.
   - Per λ: the certified duality gap.
   - On the memo's grids, eps/10 in rounds with c < τ remains a working
     heuristic [observation].
5. **Iterations as c → 0.**
   - I know of no specific result. The worst-case interval is unchanged
     (item 5), but more directions are slow.
   - Empirically, eps/10 in rounds with c < τ matched the full method's
     accuracy per pass at p = 10, and needed about 30% more passes at p = 100.
6. **cuSOLVER workspace.** Measured as one 4.01-unit allocation (float32,
   n = 20,000). syevdx and two-stage solvers have not been measured.

## Changes made to the memo in this round

- **Summary.** "Exact" is restated as "exact kernel, same stopping test".
  Descent is claimed for exact eigenvectors only; the Ritz case points to the
  safeguard.
- **4.3.** The third point is renamed: it is the same acceptance test, not a
  certificate. A new section 4.11 covers the duality gap.
- **4.4.** The dᵢ formula is marked heuristic; the interval bound is the proven
  part.
- **4.7.**
  - τ₀ notation, and ρ added once to each block;
  - Lanczos with 1/(1 − ε);
  - the split shift;
  - the counterexample and the observations above;
  - the safeguarded step using Kz and KV.
- **Section 6.** The safeguard and the certificate are added to the prototype
  plan. **Section 7:** the matched-accuracy experiment.
- **Appendix.** Notes that the emulation's 1.05 factor is uncertified, and
  that K̃ ⪰ K was verified directly.
