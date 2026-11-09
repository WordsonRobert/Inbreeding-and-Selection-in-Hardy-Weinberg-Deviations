# Lean 4 formalisation

Lean 4 + Mathlib proofs of the algebraic results of the model: Propositions 1–12 of the
manuscript. There are **81** main theorems. None uses `sorry` or `admit`, and none adds an
axiom: every theorem depends only on `propext`, `Classical.choice` and `Quot.sound`.
`BastionTheory/Checks.lean` prints these axiom dependencies.

The definitions follow the Python implementation in `../bastion/model.py`, and the
docstring of each Lean file names the Python functions it covers.

## Verify

You need the Lean toolchain manager [`elan`](https://github.com/leanprover/elan).

```bash
cd lean
lake exe cache get     # download Mathlib's prebuilt .olean files (do not build Mathlib from source)
lake build             # must finish with zero errors and zero warnings
./verify.sh            # build + axiom audit
```

The expected last line of `./verify.sh` is

```
OK: 81 theorems, standard axioms only
```

`verify.sh` runs `lake build` and then re-elaborates `BastionTheory/Checks.lean`. It fails
in any of these cases:

* a theorem depends on `sorryAx`;
* a theorem depends on an axiom other than the three standard ones;
* the number of axiom reports differs from the number of `#print axioms` lines.

**Versions:** `leanprover/lean4:v4.31.0` and Mathlib `v4.31.0`, manifest revision
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`. They are pinned in `lean-toolchain`,
`lakefile.toml` and `lake-manifest.json`. `.lake/` is not tracked.

## Numbering

The formalisation numbers its statements P1–P13. The manuscript merges them into
Propositions 1–12. `Statements.md` gives every statement in LaTeX with its Lean names.

| manuscript | formalisation | Lean file |
|---|---|---|
| Proposition 1 (coordinates and admissibility) | P1, P2, and `x2_eq` from P5(a) | `Basic.lean`, `Selection.lean` |
| Proposition 2 (closed form) | P3 | `Selection.lean` |
| Proposition 3 (κ decomposition) | P4 | `Selection.lean` |
| Proposition 4 (sign rules) | P5 | `Selection.lean` |
| Proposition 5 (mutation) | P6 | `Mutation.lean` |
| Proposition 6 (remaining operators) | P7 | `Operators.lean` |
| Proposition 7 (life cycle) | P8 | `LifeCycle.lean` |
| Proposition 8 (Shapley attribution) | P9 | `Shapley.lean`, `Attribution.lean` |
| Proposition 9 (level split) | P10 | `Shapley.lean` |
| Proposition 10 (rare allele) | P11 | `RareAllele.lean` |
| Proposition 11 (Fisher-information rank) | P12 | `Identifiability.lean` |
| Proposition 12 (drift and F_IS) | P13 | `Drift.lean` |

## Files

| file | contents |
|---|---|
| `BastionTheory/Basic.lean` | genotype vectors, `p`, `D`, the simplex, `vec`, `zyg` (P1, P2) |
| `BastionTheory/Selection.lean` | fitnesses, `sel`, `κ`, closed form and sign rules (P3, P4, P5) |
| `BastionTheory/Mutation.lean` | genotype-level mutation (P6) |
| `BastionTheory/Operators.lean` | `Y0`, `I`, `Dr`, null operators (P7) |
| `BastionTheory/LifeCycle.lean` | composition of the five operators, agreement with the Python at every step (P8) |
| `BastionTheory/Shapley.lean` | generic Shapley values (P9) and the two-player level split (P10) |
| `BastionTheory/Attribution.lean` | the BASTION attribution game; P9 applied |
| `BastionTheory/RareAllele.lean` | rare-allele dynamics, `s_eff`, first-order identification (P11) |
| `BastionTheory/Identifiability.lean` | Fisher-information rank bound (P12) |
| `BastionTheory/Drift.lean` | Wright's `F` vs `F_IS`, binomial moments (P13) |
| `BastionTheory/Checks.lean` | `#print axioms` for every main theorem, and only those |
| `verify.sh` | build + axiom audit in one command |

## Propositions

| # | statement | Lean name | certifies (`bastion/model.py`) | status |
|---|---|---|---|---|
| P1 | On a vector of unit total, `D = x1·x3 − x2²/4`. | `Geno.D_eq_of_total`, `Geno.D_eq` | `hw_dev` | proved |
| P2 | `p(vec(p,D)) = p`, `D(vec(p,D)) = D`, `vec(p(x),D(x)) = x`; `vec(p,D)` is a genotype vector iff `−min(p,q)² ≤ D ≤ pq`; `zyg(p,F)` is one iff `−min(p,q)/max(p,q) ≤ F ≤ 1`. | `p_vec`, `D_vec`, `vec_p_D_self`, `inSimplex_vec_iff`, `inSimplex_zyg_iff` | `_vec_with`, `zygotes` | proved (the first two are unconditional — stronger than asked) |
| P3 | `D(sel_w x) = (w1w3·x1x3 − w2²x2²/4)/w̄²`. | `D_sel_closed` | `adult_D_closed` | proved (needs only `w̄ ≠ 0`, not the simplex) |
| P4 | `D(sel_w x) = (w1w3·D(x) + κ·(x2/2)²)/w̄²`, and `κ = s(2h−1) − h²s²` for the BASTION fitnesses. | `D_sel_kappa`, `kappa_bastion` | `adult_D_closed`; `kap = w3 - w2*w2` in `e01_theory.identities` | proved |
| P5(a) | From HW zygotes the whole deviation is the distortion term, so its sign is the sign of `κ`; and `x2 = 2(pq − D)`, so `D < 0` is exactly heterozygote excess. | `D_sel_of_D_eq_zero`, `D_sel_pos_iff`, `D_sel_eq_zero_iff`, `D_sel_neg_iff`, `x2_eq` | `adult_D_closed`; figure `selection_induced_FIS.png` | proved (as three iffs, see Corrections 4) |
| P5(b) | Overdominance ⇒ `κ < 0` ⇒ heterozygote excess. | `kappa_neg_of_overdominance`, `D_sel_vec_zero_neg` | same | proved |
| P5(c) | Underdominance ⇒ `κ > 0` ⇒ homozygote excess. | `kappa_pos_of_underdominance`, `D_sel_vec_zero_pos` | same | proved |
| P5(d) | Multiplicative fitness (`w2² = w1w3`) ⇒ `κ = 0` ⇒ `D` stays 0. | `kappa_eq_zero_of_multiplicative`, `D_sel_vec_zero_eq_zero` | same | proved |
| P5(e) | `D ≥ 0` and `κ ≥ 0` ⇒ `D(sel_w x) ≥ 0`, and the mirrored version. | `D_sel_nonneg`, `D_sel_nonpos` | same | proved |
| P6 | Mutation preserves the simplex; `p(Mx) = (1−u)p + v(1−p)`; `D(Mx) = (1−u−v)²·D(x)`. | `inSimplex_mutate`, `p_mutate`, `D_mutate` | `_apply` (`op == 0`), `mutate_allele` | proved |
| P7 | `D(Y0 x) = 0`; `I_F` adds `F·p·q`; `Dr_δ` moves `p` by `δ` at fixed `F_g`; all four null operators are the identity. | `p_reset`, `D_reset`, `p_inbreed`, `D_inbreed`, `p_drift`, `D_drift`, `mutate_zero`, `inbreed_zero`, `sel_one`, `sel_bastion_zero`, `drift_zero` | `_apply` (all branches) | proved |
| P8 | `S ∘ Dr ∘ I ∘ Y0 ∘ M = sel_w ∘ zyg(p′, F)` with `p′ = m + δ`; the intermediate vectors are `vec(m,0)`, `zyg(m,F)`, `zyg(p′,F)`; and under admissibility of `F` at **both** `m` and `p′` **every** intermediate vector is in the simplex, i.e. the Python's `_vec_with` clip is inactive at every step. | `reset_mutate`, `inbreed_reset_mutate`, `drift_inbreed_reset_mutate`, `lifecycle`, `inSimplex_lifecycle`, `lifecycle_python_agreement` | `_apply` order vs `adults_at`/`gamete_mean`; `_vec_with` | proved (the equation needs fewer hypotheses than asked — see Corrections 5) |
| P9 | Shapley efficiency, dummy, symmetry; sequential attribution along any fixed order telescopes. | `ShapleyGame.sum_shapley`, `…shapley_dummy`, `…shapley_symm`, `…sum_seq_telescope` | `shapley_step` (`phi` and `seq`) | proved for the subset-weight definition; permutation-average equivalence **not formalised** (Corrections 6) |
| P9 (applied) | The five operators' Shapley shares sum to the observed change in `D`; each of the four null operators gets exactly 0. | `sum_shapley_game`, `sum_shapley_game_lifecycle`, `shapley_selection_eq_zero`, `shapley_mutation_eq_zero`, `shapley_inbreeding_eq_zero`, `shapley_drift_eq_zero` | `shapley_step`, `attribute_path` | proved (drift needs two interiority hypotheses — Corrections 10) |
| P10 | `Dz + DS = vIS`; `F = 0 ⇒ Dz = 0`; `s = 0 ⇒ DS = 0`. | `Dz_add_DS`, `Dz_eq_zero`, `DS_eq_zero`, `DS_eq_zero_bastion` | `attribute_path` (level split) | proved |
| P11(a) | `q′ − q = −s·q(1−q)·[F + (1−F)h + (1−F)(1−2h)q] / w̄`, exactly. | `qAdult_sub` | `zygotes` + `select` at `p = 1−q` | proved with an added hypothesis `w̄ ≠ 0` (Corrections 7) |
| P11(b) | `q ↦ q′` has derivative `1 − s_eff` at `q = 0`, `s_eff = s(F + (1−F)h)`; equal `s_eff` ⇒ equal linearisation; `s_eff = hs` under random mating. | `hasDerivAt_qAdult`, `sEff_determines_linearisation`, `sEff_zero` | E01d `rare_allele_seff` | proved |
| P11(c) | The first-order coefficients are `c2 = 2(1−F)(1−hs)` and `c3 = F(1−s)`.  Under `F = 0` they depend on `(s,h)` only through `hs` — with an explicit non-identifiable pair `(0.2, 0.5)` vs `(0.1, 1)` — but for `0 < F < 1` they determine `s` and `hs`, and `h` too when `s ≠ 0`. | `tendsto_rareAdult_x2_div`, `tendsto_rareAdult_x3_div`, `tendsto_c2`, `tendsto_c3`, `first_order_hs_only`, `first_order_not_identified`, `first_order_injective`, `first_order_injective_h`, `first_order_limits_injective` | identifiability claims of E02 | proved |
| P12 | If every column of `J` sums to 0 then `rank(JᵀWJ) ≤ m − 1`; `m = 3 ⇒ ≤ 2`, `m = 2 ⇒ ≤ 1`; and the Jacobian of any differentiable family of probability vectors has zero column sums. | `rank_le_of_col_sum_zero`, `rank_quadratic_le`, `rank_fisher_genotype_le_two`, `rank_fisher_phenotype_le_one`, `sum_deriv_eq_zero_of_probability_family` | `fim`, `identifiability` (`cross_sectional_rank_max`) | proved |
| P13(a) | `F_{t+1} = c + (1−c)F_t ⇒ 1 − F_t = (1−c)^t(1 − F_0)`. | `Drift.wright_closed_form` | legacy Layer 0 recursion (`legacy_v2.py`) | proved |
| P13(b) | `Σ_k C(n,k)p^k q^{n−k}(k/n)(1−k/n) = pq(1 − 1/n)`. | `Drift.binom_var_identity`, `Drift.binom_moment1`, `Drift.binom_moment2` | E01a `drift_vs_fis` | proved |
| P13(c) | `1 − E[H_obs]/E[2p̂q̂] = −1/(2N−1)`, both expectations written out as binomial sums, independent of `t`. | `Drift.fis_ratio_full`, `Drift.expected_het_fraction` | E01a `drift_vs_fis` | proved |


## Corrections

Deviations from the Python, and from the statements as originally posed.  Nothing was
silently weakened; the Lean statement is in every case at least as strong as the one asked
for, except where explicitly flagged.

1. **Clipping is replaced by an admissibility hypothesis, and that hypothesis is real.**
   `_vec_with` in `bastion/model.py` clips the HW deviation to `[-min(p,q)², p·q]`.  The Lean `vec`
   never clips, and by P2 "in the simplex" is exactly "the clip is inactive".
   `lifecycle_python_agreement` proves that *every* intermediate vector of the generation
   — `M a`, `Y0(M a) = vec(m,0)`, `I_F(…) = zyg(m,F)`, `Dr_δ(…) = zyg(p′,F)` and the final
   `S(…)` — is in the simplex, provided `F` is admissible at **both** `m` and `p′`.
   When `F` is inadmissible at `m` (possible for negative `F` near fixation) `bastion/model.py`
   clips at the `I` step, so its attribution can differ from the Lean statement.  This is
   the source of the small Shapley-efficiency residuals for negative `F` reported by the
   numerical checks.
   Two further numerical guards in the Python are **not** covered by this agreement
   theorem: the drift-branch guard `pq > 1e-15` (below which `_apply` uses `F_g = 0`
   instead of `D/(pq)`), and the clamp of the post-drift frequency to
   `[1e-12, 1 − 1e-12]`.  Both are inactive for interior `p`, which is what the Lean
   statements assume, but neither is formalised.
2. **The drift guard is replaced by `p ∈ (0,1)`.**  `drift_zero` and the drift operator
   are stated for an interior `p`, so the `pq > 1e-15` branch is never taken.
3. **P3 needs less than asked.**  `D_sel_closed` holds for any `x` with `w̄ ≠ 0` — the
   simplex and positive fitnesses are not needed.  The simplex version follows.
4. **P5(a) is stated as three `iff`s** (`0 < D(sel) ↔ 0 < κ`, `= 0 ↔ κ = 0`,
   `< 0 ↔ κ < 0`) rather than with a `sign` function — the same content without dragging
   in `SignType`.  The hypothesis is `w̄ > 0` rather than `w > 0`; on the simplex
   positive fitnesses imply positive mean fitness, so the two are interchangeable there.
5. **P8's equation needs fewer hypotheses than asked.**  The identity
   `S(Dr(I(Y0(M a)))) = sel_w(zyg(p′,F))` needs only `a` of unit total and `m ∈ (0,1)`.
   Admissibility of `F` is what makes the intermediate and final vectors genotype
   vectors, and that is `inSimplex_lifecycle` and `lifecycle_python_agreement`, which do
   take the admissibility bounds.
6. **P9: the Shapley value is defined by the subset-weight formula, not by averaging over
   permutations.**  `ShapleyGame.shapley` is
   `Σ_{S ∌ i} |S|!(n−|S|−1)!/n! · (v(S∪{i}) − v(S))`, i.e. literally what `shapley_step`
   computes, and efficiency, dummy and symmetry are proved for *that* definition.  The
   permutation-average characterisation
   `φ_i = (1/n!) Σ_σ [v(P_σ(i) ∪ {i}) − v(P_σ(i))]`, with `P_σ(i) = {j : σ(j) < σ(i)}`,
   is **stated here but not machine-checked**; proving it needs the fibre count
   `#{σ : P_σ(i) = S} = |S|!(n−|S|−1)!`, which was out of budget.  Nothing in the paper
   depends on it.  Separately, `sum_seq_telescope` proves the telescoping claim for an
   *arbitrary* chain `∅ = S_0 ⊆ … ⊆ S_n = univ`, which covers every fixed operator order,
   not just the life-cycle one.
7. **P11(a) needs `w̄ ≠ 0`.**  As posed the identity is false without it: Lean's `x/0 = 0`,
   so at `w̄ = 0` the left side is `−q` and the right side is `0`.  `w̄ ≠ 0` holds
   automatically whenever the fitnesses are positive and the zygotes are admissible.
8. **P12 uses truncated subtraction.**  `rank ≤ m − 1` is stated in `ℕ`; for `m = 0` it
   reads `rank ≤ 0`, which is still true, so no hypothesis `m ≥ 1` is needed.
9. **P13(c) is about the ratio of expectations, and says so.**  `fis_ratio_full` proves
   `1 − E[H_obs]/E[2p̂q̂] = −1/(2N−1)` with both expectations written out as explicit
   binomial sums.  It does **not** say `E[F_IS] = −1/(2N−1)`; the expectation of the ratio
   is not that number, and no such claim is made or proved here.  The simulated
   `F_IS_mean ≈ −0.0102` at `N = 50` in `theory_summary.json` sits near, but is not equal
   to, `−1/(2N−1) = −0.0101…` and `−1/(2N) = −0.01`.
10. **The drift null-operator share needs two interiority hypotheses.**
    `shapley_drift_eq_zero` assumes `0 < a.p < 1` *and* `0 < m < 1`.  These are minimal:
    `Dr_0` is the identity only at an interior allele frequency, and the frequency
    reaching the drift step is `a.p` when the mutation operator is absent from the
    coalition and `m` when it is present, so both must be interior.  The other three null operators (`S` at `s = 0`, `M` at `u = v = 0`, `I` at
    `F = 0`) need only `a` of unit total.


## Not formalised

Deliberately out of scope.  None of the following is machine-checked, and it is not
claimed as such:

* **The HMM likelihood** — `hmm_forward`, `build_transition`, `ffbs`,
  `predictive_residuals`: the grid construction, the mean-preserving hat-function
  projection of the Wright–Fisher kernel, the claimed variance match "within a few
  percent", and the discretisation error of "~0.01 log-units".
* **Grid discretisation** — `make_grid`, `make_levels`, `choose_K`, and the
  interpolation-floor bound `c²/6`.
* **MCMC** — everything in `bastion/mcmc.py`: adaptive Metropolis, the change-of-variables
  Jacobian, chain pruning, split-R̂, ESS, HDI.
* **All numerical results** — every number in `results/`, including the identity
  residuals in `theory_summary.json` (numerical evidence for P1–P8, now superseded by
  proof), the Fisher-information CRLBs and correlations, the operator-order ranges of E06,
  SBC coverage, the migration ROC, and the empirical fits.
* **The permutation-average characterisation of the Shapley value** (Corrections 6).
* **The Python's two remaining numerical guards** — the `pq > 1e-15` drift guard and the
  `[1e-12, 1 − 1e-12]` frequency clamp (Corrections 1).
* **Sampling theory beyond the two binomial moments** — P13 proves the first and second
  factorial moments of the binomial and the identities built on them, inside ordinary real
  algebra.  No measure-theoretic probability space, no Wright–Fisher process, and no
  statement about `F_IS` as a random variable is formalised.
* **Layer 1, Layer 4 and the legacy neural ODE** — nothing in `legacy/`.
