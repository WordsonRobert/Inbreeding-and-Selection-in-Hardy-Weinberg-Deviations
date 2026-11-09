/-
# Rare-allele selection (P11)

Ground truth: `model.py` (`zygotes`, `select`, `adults_at`) specialised to the BASTION
fitnesses, and `experiments/e01_theory.py : identities`, block `rare_allele_seff`, which
checks the `q → 0` approximation numerically.

Here `q` is the frequency of the *deleterious* allele `a` in the zygotes `zyg (1-q) F`,
and `qAdult` is its frequency after viability selection.
-/
import BastionTheory.Selection
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Topology.Algebra.Order.Field

namespace Bastion

open Geno Filter Topology

/-- Effective selection coefficient on a rare allele.  Python: `s * (F + (1 - F) * h)`
in `e01_theory.identities` and `g_se` in `identifiability`. -/
noncomputable def sEff (s h F : ℝ) : ℝ := s * (F + (1 - F) * h)

/-- Zygotes carrying the rare allele `a` at frequency `q`.  Python: `zygotes(1-q, F)`. -/
noncomputable def rareZyg (F q : ℝ) : Geno := zyg (1 - q) F

/-- Adults after viability selection with the BASTION fitnesses.
Python: `adults_at(1-q, F, s, h)`. -/
noncomputable def rareAdult (s h F q : ℝ) : Geno := sel (bastionFit s h) (rareZyg F q)

/-- Mean fitness at rare-allele frequency `q`.  Python: `wb` inside `select`. -/
noncomputable def wbarRare (s h F q : ℝ) : ℝ := (bastionFit s h).bar (rareZyg F q)

/-- Frequency of the rare allele among the adults.  Python: `qn = a[2] + a[1] / 2` in
`e01_theory.identities`. -/
noncomputable def qAdult (s h F q : ℝ) : ℝ :=
  (rareAdult s h F q).x3 + (rareAdult s h F q).x2 / 2

/-! ### Explicit polynomial forms -/

theorem wbarRare_poly (s h F q : ℝ) :
    wbarRare s h F q
      = 1 + (s * (2 * F * h - F - 2 * h)) * q + (-s * (F - 1) * (2 * h - 1)) * q ^ 2 := by
  simp only [wbarRare, Fit.bar, rareZyg, zyg, vec_x1, vec_x2, vec_x3, bastionFit]
  ring

@[simp] theorem wbarRare_zero (s h F : ℝ) : wbarRare s h F 0 = 1 := by
  rw [wbarRare_poly]; ring

theorem rareAdult_x2 (s h F q : ℝ) :
    (rareAdult s h F q).x2
      = (1 - h * s) * (2 * ((1 - q) * q - F * (1 - q) * q)) / wbarRare s h F q := by
  simp only [rareAdult, sel_x2, wbarRare, rareZyg, zyg, vec_x2, bastionFit]
  ring_nf

theorem rareAdult_x3 (s h F q : ℝ) :
    (rareAdult s h F q).x3 = (1 - s) * (q ^ 2 + F * (1 - q) * q) / wbarRare s h F q := by
  simp only [rareAdult, sel_x3, wbarRare, rareZyg, zyg, vec_x3, bastionFit]
  ring_nf

/-- Numerator of the adult rare-allele frequency, as a polynomial in `q`. -/
noncomputable def numRare (s h F q : ℝ) : ℝ :=
  (1 - sEff s h F) * q + (-s * (F - 1) * (h - 1)) * q ^ 2

theorem add_half_div (a b c : ℝ) : a / c + b / c / 2 = (a + b / 2) / c := by
  rcases eq_or_ne c 0 with rfl | hc
  · simp
  · field_simp

theorem qAdult_eq (s h F q : ℝ) :
    qAdult s h F q = numRare s h F q / wbarRare s h F q := by
  rw [qAdult, rareAdult_x2, rareAdult_x3, add_half_div]
  congr 1
  simp only [numRare, sEff]
  ring

/-! ### P11(a): the exact change in allele frequency -/

/-- **P11(a).**  The exact one-generation change of a rare-allele frequency under
viability selection. -/
theorem qAdult_sub (s h F q : ℝ) (hw : wbarRare s h F q ≠ 0) :
    qAdult s h F q - q
      = -s * q * (1 - q) * (F + (1 - F) * h + (1 - F) * (1 - 2 * h) * q)
          / wbarRare s h F q := by
  have h1 : qAdult s h F q - q
      = (numRare s h F q - q * wbarRare s h F q) / wbarRare s h F q := by
    rw [qAdult_eq, sub_div, mul_div_assoc, div_self hw, mul_one]
  rw [h1]
  congr 1
  simp only [numRare, sEff, wbarRare_poly]
  ring

/-! ### P11(b): the linearisation at `q = 0` -/

theorem hasDerivAt_quad (a b c x : ℝ) :
    HasDerivAt (fun t : ℝ => a + b * t + c * t ^ 2) (b + 2 * c * x) x := by
  have hsq : HasDerivAt (fun t : ℝ => t ^ 2) (2 * x) x := by
    simpa using hasDerivAt_pow 2 x
  have h1 : HasDerivAt (fun t : ℝ => b * t) (b * 1) x := (hasDerivAt_id x).const_mul b
  have h2 : HasDerivAt (fun t : ℝ => c * t ^ 2) (c * (2 * x)) x := hsq.const_mul c
  have h3 : HasDerivAt (fun t : ℝ => a + b * t + c * t ^ 2) (0 + b * 1 + c * (2 * x)) x :=
    HasDerivAt.add (HasDerivAt.add (hasDerivAt_const x a) h1) h2
  have hv : (0 : ℝ) + b * 1 + c * (2 * x) = b + 2 * c * x := by ring
  rw [hv] at h3
  exact h3

theorem hasDerivAt_numRare (s h F : ℝ) :
    HasDerivAt (fun q => numRare s h F q) (1 - sEff s h F) 0 := by
  have hfun : (fun q => numRare s h F q)
      = fun t : ℝ => 0 + (1 - sEff s h F) * t + (-s * (F - 1) * (h - 1)) * t ^ 2 := by
    funext t; simp only [numRare]; ring
  rw [hfun]
  simpa using hasDerivAt_quad 0 (1 - sEff s h F) (-s * (F - 1) * (h - 1)) 0

theorem hasDerivAt_wbarRare (s h F : ℝ) :
    HasDerivAt (fun q => wbarRare s h F q) (s * (2 * F * h - F - 2 * h)) 0 := by
  have hfun : (fun q => wbarRare s h F q)
      = fun t : ℝ => 1 + (s * (2 * F * h - F - 2 * h)) * t
          + (-s * (F - 1) * (2 * h - 1)) * t ^ 2 := by
    funext t; exact wbarRare_poly s h F t
  rw [hfun]
  simpa using hasDerivAt_quad 1 (s * (2 * F * h - F - 2 * h)) (-s * (F - 1) * (2 * h - 1)) 0

/-- **P11(b).**  The adult-frequency map has derivative `1 - s_eff` at `q = 0`. -/
theorem hasDerivAt_qAdult (s h F : ℝ) :
    HasDerivAt (fun q => qAdult s h F q) (1 - sEff s h F) 0 := by
  have hfun : (fun q => qAdult s h F q)
      = fun q => numRare s h F q / wbarRare s h F q := funext (qAdult_eq s h F)
  rw [hfun]
  have hw : wbarRare s h F 0 ≠ 0 := by rw [wbarRare_zero]; norm_num
  have hn0 : numRare s h F 0 = 0 := by simp only [numRare]; ring
  have key := (hasDerivAt_numRare s h F).div (hasDerivAt_wbarRare s h F) hw
  have hv : ((1 - sEff s h F) * wbarRare s h F 0
      - numRare s h F 0 * (s * (2 * F * h - F - 2 * h))) / wbarRare s h F 0 ^ 2
      = 1 - sEff s h F := by rw [wbarRare_zero, hn0]; ring
  rw [hv] at key
  exact key

/-- **P11(b)**, corollary: the linearised dynamics at `q = 0` depend on `(s, h, F)` only
through `s_eff`, so parameter triples with equal `s_eff` are indistinguishable there. -/
theorem sEff_determines_linearisation {s₁ h₁ F₁ s₂ h₂ F₂ : ℝ}
    (hEq : sEff s₁ h₁ F₁ = sEff s₂ h₂ F₂) :
    HasDerivAt (fun q => qAdult s₁ h₁ F₁ q) (1 - sEff s₂ h₂ F₂) 0 := by
  rw [← hEq]; exact hasDerivAt_qAdult s₁ h₁ F₁

/-- Under random mating `s_eff` is exactly `h s`. -/
theorem sEff_zero (s h : ℝ) : sEff s h 0 = s * h := by simp only [sEff]; ring

/-! ### P11(c): first-order genotype frequencies -/

theorem continuous_wbarRare (s h F : ℝ) : Continuous (fun q => wbarRare s h F q) := by
  simp only [wbarRare_poly]
  fun_prop

/-- **P11(c)**, heterozygotes: `a2 / q → 2(1-F)(1-hs)`. -/
theorem tendsto_rareAdult_x2_div (s h F : ℝ) :
    Tendsto (fun q => (rareAdult s h F q).x2 / q) (𝓝[≠] (0 : ℝ))
      (𝓝 (2 * (1 - F) * (1 - h * s))) := by
  have hw : wbarRare s h F 0 ≠ 0 := by rw [wbarRare_zero]; norm_num
  have hc : ContinuousAt
      (fun q : ℝ => 2 * (1 - h * s) * (1 - F) * (1 - q) / wbarRare s h F q) 0 :=
    ContinuousAt.div (by fun_prop) (continuous_wbarRare s h F).continuousAt hw
  have hval : 2 * (1 - h * s) * (1 - F) * (1 - (0 : ℝ)) / wbarRare s h F 0
      = 2 * (1 - F) * (1 - h * s) := by rw [wbarRare_zero]; ring
  have hlim : Tendsto (fun q : ℝ => 2 * (1 - h * s) * (1 - F) * (1 - q) / wbarRare s h F q)
      (𝓝[≠] (0 : ℝ)) (𝓝 (2 * (1 - F) * (1 - h * s))) := by
    rw [← hval]
    exact hc.tendsto.mono_left nhdsWithin_le_nhds
  refine hlim.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with q hq
  have hq' : q ≠ 0 := hq
  rw [rareAdult_x2]
  rcases eq_or_ne (wbarRare s h F q) 0 with h0 | h0
  · rw [h0]; simp
  · field_simp

/-- **P11(c)**, recessive homozygotes: `a3 / q → F(1-s)`. -/
theorem tendsto_rareAdult_x3_div (s h F : ℝ) :
    Tendsto (fun q => (rareAdult s h F q).x3 / q) (𝓝[≠] (0 : ℝ)) (𝓝 (F * (1 - s))) := by
  have hw : wbarRare s h F 0 ≠ 0 := by rw [wbarRare_zero]; norm_num
  have hc : ContinuousAt
      (fun q : ℝ => (1 - s) * (q + F * (1 - q)) / wbarRare s h F q) 0 :=
    ContinuousAt.div (by fun_prop) (continuous_wbarRare s h F).continuousAt hw
  have hval : (1 - s) * ((0 : ℝ) + F * (1 - 0)) / wbarRare s h F 0 = F * (1 - s) := by
    rw [wbarRare_zero]; ring
  have hlim : Tendsto (fun q : ℝ => (1 - s) * (q + F * (1 - q)) / wbarRare s h F q)
      (𝓝[≠] (0 : ℝ)) (𝓝 (F * (1 - s))) := by
    rw [← hval]
    exact hc.tendsto.mono_left nhdsWithin_le_nhds
  refine hlim.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with q hq
  have hq' : q ≠ 0 := hq
  rw [rareAdult_x3]
  rcases eq_or_ne (wbarRare s h F q) 0 with h0 | h0
  · rw [h0]; simp
  · field_simp

/-! ### P11(c): what the first-order coefficients identify -/

/-- First-order coefficient of the heterozygote class, `lim a2/q`.
This is the limit proved by `tendsto_rareAdult_x2_div`. -/
noncomputable def c2 (s h F : ℝ) : ℝ := 2 * (1 - F) * (1 - h * s)

/-- First-order coefficient of the recessive-homozygote class, `lim a3/q`.
This is the limit proved by `tendsto_rareAdult_x3_div`. -/
noncomputable def c3 (s F : ℝ) : ℝ := F * (1 - s)

theorem tendsto_c2 (s h F : ℝ) :
    Tendsto (fun q => (rareAdult s h F q).x2 / q) (𝓝[≠] (0 : ℝ)) (𝓝 (c2 s h F)) :=
  tendsto_rareAdult_x2_div s h F

theorem tendsto_c3 (s h F : ℝ) :
    Tendsto (fun q => (rareAdult s h F q).x3 / q) (𝓝[≠] (0 : ℝ)) (𝓝 (c3 s F)) :=
  tendsto_rareAdult_x3_div s h F

/-- **P11(c)**, the `F = 0` case: under random mating the first-order coefficients depend
on `(s, h)` only through the product `h s`, and the `aa` class is absent at first order. -/
theorem first_order_hs_only {s₁ h₁ s₂ h₂ : ℝ} (he : h₁ * s₁ = h₂ * s₂) :
    c2 s₁ h₁ 0 = c2 s₂ h₂ 0 ∧ c3 s₁ 0 = c3 s₂ 0 ∧ c3 s₁ 0 = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [c2]; rw [he]
  · simp only [c3]; ring
  · simp only [c3]; ring

/-- **P11(c)**, the `F = 0` case is genuinely non-identifiable: an explicit pair of
distinct `(s, h)` with the same first-order coefficients under random mating.  So
random-mating first-order data identify `h s` and nothing more. -/
theorem first_order_not_identified :
    ((0.2 : ℝ), (0.5 : ℝ)) ≠ ((0.1 : ℝ), (1 : ℝ))
      ∧ c2 0.2 0.5 0 = c2 0.1 1 0
      ∧ c3 0.2 0 = c3 0.1 0 := by
  refine ⟨?_, ?_, ?_⟩
  · norm_num [Prod.ext_iff]
  · simp only [c2]; norm_num
  · simp only [c3]; norm_num

/-- **P11(c)**, the `F > 0` case: with any strictly interior inbreeding the first-order
coefficients determine `s` and `h s`.  So the paper may **not** claim that only `h s` is
identified once `F > 0`. -/
theorem first_order_injective {s₁ h₁ s₂ h₂ F : ℝ} (hF0 : 0 < F) (hF1 : F < 1)
    (e2 : c2 s₁ h₁ F = c2 s₂ h₂ F) (e3 : c3 s₁ F = c3 s₂ F) :
    s₁ = s₂ ∧ h₁ * s₁ = h₂ * s₂ := by
  simp only [c3] at e3
  have hs : s₁ = s₂ := by
    have := mul_left_cancel₀ (ne_of_gt hF0) e3
    linarith
  refine ⟨hs, ?_⟩
  simp only [c2] at e2
  have hF : (1 : ℝ) - F ≠ 0 := by linarith
  have h2 : (2 : ℝ) * (1 - F) ≠ 0 := by
    simpa using mul_ne_zero (two_ne_zero) hF
  have := mul_left_cancel₀ h2 e2
  linarith

/-- **P11(c)**, the `F > 0` case: if in addition `s ≠ 0`, the dominance coefficient itself
is determined. -/
theorem first_order_injective_h {s₁ h₁ s₂ h₂ F : ℝ} (hF0 : 0 < F) (hF1 : F < 1)
    (hs : s₁ ≠ 0) (e2 : c2 s₁ h₁ F = c2 s₂ h₂ F) (e3 : c3 s₁ F = c3 s₂ F) :
    h₁ = h₂ := by
  obtain ⟨hseq, hhs⟩ := first_order_injective hF0 hF1 e2 e3
  rw [← hseq] at hhs
  exact mul_right_cancel₀ hs hhs

/-- **P11(c)**, stated directly in terms of the limits: two parameter triples whose
first-order genotype limits agree, at any strictly interior `F`, have the same `s` and the
same `h s`. -/
theorem first_order_limits_injective {s₁ h₁ s₂ h₂ F L2 L3 : ℝ} (hF0 : 0 < F) (hF1 : F < 1)
    (hA2 : Tendsto (fun q => (rareAdult s₁ h₁ F q).x2 / q) (𝓝[≠] (0 : ℝ)) (𝓝 L2))
    (hB2 : Tendsto (fun q => (rareAdult s₂ h₂ F q).x2 / q) (𝓝[≠] (0 : ℝ)) (𝓝 L2))
    (hA3 : Tendsto (fun q => (rareAdult s₁ h₁ F q).x3 / q) (𝓝[≠] (0 : ℝ)) (𝓝 L3))
    (hB3 : Tendsto (fun q => (rareAdult s₂ h₂ F q).x3 / q) (𝓝[≠] (0 : ℝ)) (𝓝 L3)) :
    s₁ = s₂ ∧ h₁ * s₁ = h₂ * s₂ := by
  have e2 : c2 s₁ h₁ F = c2 s₂ h₂ F := by
    rw [tendsto_nhds_unique (tendsto_c2 s₁ h₁ F) hA2,
      tendsto_nhds_unique (tendsto_c2 s₂ h₂ F) hB2]
  have e3 : c3 s₁ F = c3 s₂ F := by
    rw [tendsto_nhds_unique (tendsto_c3 s₁ h₁ F) hA3,
      tendsto_nhds_unique (tendsto_c3 s₂ h₂ F) hB3]
  exact first_order_injective hF0 hF1 e2 e3

end Bastion
