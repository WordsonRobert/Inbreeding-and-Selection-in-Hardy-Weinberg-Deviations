/-
# Drift versus the within-population F_IS (P13)

Ground truth: `experiments/e01_theory.py : drift_vs_fis` (experiment E01a), and the legacy
Layer 0 recursion reproduced in `bastion/legacy_v2.py`.

The point of E01a is that the two coefficients the legacy pipeline identified are different
quantities: Wright's `F_t`, which grows towards 1 under drift, and the within-population
`F_IS`, which does not.  P13(c) proves the exact statement: the *ratio of expectations*
`1 - E[H_obs] / E[2 p̂ q̂]` equals `-1/(2N-1)`, independently of `t`.

Note on scope: `-1/(2N-1)` is the ratio of expectations, not the expectation of the ratio
`F_IS`; the latter is not this number and is not claimed here (see README, "Corrections").
-/
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

namespace Bastion
namespace Drift

open Finset

/-! ### P13(a): the Wright recursion -/

/-- **P13(a).**  The legacy Layer 0 recursion `F_{t+1} = c + (1-c) F_t` has the closed form
`1 - F_t = (1-c)^t (1 - F_0)`; with `c = 1/(2N)` this is Wright's inbreeding coefficient,
which converges to 1. -/
theorem wright_closed_form (c : ℝ) (F : ℕ → ℝ) (hrec : ∀ t, F (t + 1) = c + (1 - c) * F t)
    (t : ℕ) : 1 - F t = (1 - c) ^ t * (1 - F 0) := by
  induction t with
  | zero => simp
  | succ k ih =>
    have h : 1 - (c + (1 - c) * F k) = (1 - c) * (1 - F k) := by ring
    rw [hrec k, h, ih, pow_succ]
    ring

/-! ### Binomial moments -/

private theorem binom_sum_one (m : ℕ) (p : ℝ) :
    ∑ j ∈ range (m + 1), (m.choose j : ℝ) * (p ^ j * (1 - p) ^ (m - j)) = 1 := by
  have h1 : p + (1 - p) = 1 := by ring
  have h2 : ∑ j ∈ range (m + 1), (m.choose j : ℝ) * (p ^ j * (1 - p) ^ (m - j))
      = ∑ k ∈ range (m + 1), p ^ k * (1 - p) ^ (m - k) * (m.choose k : ℝ) :=
    Finset.sum_congr rfl fun j _ => by ring
  rw [h2, ← add_pow, h1, one_pow]

/-- The index shift behind every binomial moment: `k * C(m+1, k) = (m+1) * C(m, k-1)`. -/
private theorem choose_shift (m : ℕ) (g : ℕ → ℝ) :
    ∑ k ∈ range (m + 2), (k : ℝ) * ((m + 1).choose k : ℝ) * g k
      = (m + 1 : ℝ) * ∑ j ∈ range (m + 1), (m.choose j : ℝ) * g (j + 1) := by
  rw [Finset.sum_range_succ']
  simp only [Nat.cast_zero, zero_mul]
  rw [add_zero, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  have hnat := Nat.add_one_mul_choose_eq m j
  have hcast : (m + 1 : ℝ) * (m.choose j : ℝ)
      = ((m + 1).choose (j + 1) : ℝ) * ((j : ℝ) + 1) := by exact_mod_cast hnat
  push_cast
  rw [← mul_assoc, hcast]
  ring

theorem binom_moment1 (n : ℕ) (p : ℝ) :
    ∑ k ∈ range (n + 1), (k : ℝ) * (n.choose k : ℝ) * (p ^ k * (1 - p) ^ (n - k))
      = (n : ℝ) * p := by
  cases n with
  | zero => simp
  | succ m =>
    rw [choose_shift m (fun k => p ^ k * (1 - p) ^ (m + 1 - k))]
    have h : ∀ j ∈ range (m + 1),
        (m.choose j : ℝ) * (p ^ (j + 1) * (1 - p) ^ (m + 1 - (j + 1)))
          = p * ((m.choose j : ℝ) * (p ^ j * (1 - p) ^ (m - j))) := by
      intro j _
      rw [Nat.succ_sub_succ]
      ring
    rw [Finset.sum_congr rfl h, ← Finset.mul_sum, binom_sum_one]
    push_cast
    ring

theorem binom_moment2 (n : ℕ) (p : ℝ) :
    ∑ k ∈ range (n + 1), (k : ℝ) * ((k : ℝ) - 1) * (n.choose k : ℝ) * (p ^ k * (1 - p) ^ (n - k))
      = (n : ℝ) * ((n : ℝ) - 1) * p ^ 2 := by
  match n with
  | 0 => simp
  | 1 => simp [Finset.sum_range_succ]
  | (m + 2) =>
    have e1 : ∀ k ∈ range (m + 3),
        (k : ℝ) * ((k : ℝ) - 1) * ((m + 2).choose k : ℝ) * (p ^ k * (1 - p) ^ (m + 2 - k))
          = (k : ℝ) * ((m + 1 + 1).choose k : ℝ)
              * (((k : ℝ) - 1) * (p ^ k * (1 - p) ^ (m + 2 - k))) := by
      intro k _; ring
    rw [Finset.sum_congr rfl e1,
      choose_shift (m + 1) (fun k => ((k : ℝ) - 1) * (p ^ k * (1 - p) ^ (m + 2 - k)))]
    have e2 : ∀ j ∈ range (m + 2),
        ((m + 1).choose j : ℝ)
            * ((((j + 1 : ℕ) : ℝ) - 1) * (p ^ (j + 1) * (1 - p) ^ (m + 2 - (j + 1))))
          = (j : ℝ) * ((m + 1).choose j : ℝ) * (p ^ (j + 1) * (1 - p) ^ (m + 1 - j)) := by
      intro j _
      rw [Nat.succ_sub_succ]
      push_cast
      ring
    rw [Finset.sum_congr rfl e2,
      choose_shift m (fun j => p ^ (j + 1) * (1 - p) ^ (m + 1 - j))]
    have e3 : ∀ i ∈ range (m + 1),
        (m.choose i : ℝ) * (p ^ (i + 1 + 1) * (1 - p) ^ (m + 1 - (i + 1)))
          = p ^ 2 * ((m.choose i : ℝ) * (p ^ i * (1 - p) ^ (m - i))) := by
      intro i _
      rw [Nat.succ_sub_succ]
      ring
    rw [Finset.sum_congr rfl e3, ← Finset.mul_sum, binom_sum_one]
    push_cast
    ring

/-! ### P13(b): the Wright–Fisher sampling variance -/

/-- **P13(b).**  Under Wright–Fisher sampling of `n` gene copies from a pool at frequency
`p`, `E[p̂ q̂] = p q (1 - 1/n)`. -/
theorem binom_var_identity (n : ℕ) (hn : 1 ≤ n) (p : ℝ) :
    ∑ k ∈ range (n + 1),
        (n.choose k : ℝ) * (p ^ k * (1 - p) ^ (n - k)) * (((k : ℝ) / n) * (1 - (k : ℝ) / n))
      = p * (1 - p) * (1 - 1 / (n : ℝ)) := by
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have key : ∀ k ∈ range (n + 1),
      (n.choose k : ℝ) * (p ^ k * (1 - p) ^ (n - k)) * (((k : ℝ) / n) * (1 - (k : ℝ) / n))
        = 1 / (n : ℝ) ^ 2
            * (((n : ℝ) - 1) * ((k : ℝ) * (n.choose k : ℝ) * (p ^ k * (1 - p) ^ (n - k)))
              - (k : ℝ) * ((k : ℝ) - 1) * (n.choose k : ℝ) * (p ^ k * (1 - p) ^ (n - k))) := by
    intro k _
    field_simp
    ring
  rw [Finset.sum_congr rfl key, ← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.mul_sum,
    binom_moment1, binom_moment2]
  field_simp

/-! ### P13(c): the ratio of expectations -/

/-- The expected heterozygote fraction among `N` individuals each heterozygous with
probability `r` is exactly `r`; under random union of gametes from a pool at frequency `p`
this is `r = 2pq`. -/
theorem expected_het_fraction (N : ℕ) (hN : 1 ≤ N) (r : ℝ) :
    ∑ j ∈ range (N + 1), (N.choose j : ℝ) * (r ^ j * (1 - r) ^ (N - j)) * ((j : ℝ) / N) = r := by
  have hN0 : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have key : ∀ j ∈ range (N + 1),
      (N.choose j : ℝ) * (r ^ j * (1 - r) ^ (N - j)) * ((j : ℝ) / N)
        = 1 / (N : ℝ) * ((j : ℝ) * (N.choose j : ℝ) * (r ^ j * (1 - r) ^ (N - j))) := by
    intro j _; field_simp
  rw [Finset.sum_congr rfl key, ← Finset.mul_sum, binom_moment1]
  field_simp

/-- **P13(c)**, fully composed.  Numerator: the expected heterozygote fraction among `N`
individuals formed by random union of gametes from a pool at frequency `p`, each
heterozygous with probability `2pq`.  Denominator: `E[2 p̂ q̂]` under Wright–Fisher sampling
of `2N` gene copies.  The resulting coefficient is `-1/(2N-1)`, a constant independent of
the generation `t`, while Wright's `F_t` of P13(a) grows towards 1. -/
theorem fis_ratio_full (N : ℕ) (hN : 1 ≤ N) (p : ℝ) (hp0 : 0 < p) (hp1 : p < 1) :
    1 - (∑ j ∈ range (N + 1), (N.choose j : ℝ)
            * ((2 * p * (1 - p)) ^ j * (1 - 2 * p * (1 - p)) ^ (N - j)) * ((j : ℝ) / N))
        / (2 * ∑ k ∈ range (2 * N + 1), ((2 * N).choose k : ℝ)
            * (p ^ k * (1 - p) ^ (2 * N - k))
            * (((k : ℝ) / ((2 * N : ℕ) : ℝ)) * (1 - (k : ℝ) / ((2 * N : ℕ) : ℝ))))
      = -1 / (2 * (N : ℝ) - 1) := by
  rw [expected_het_fraction N hN (2 * p * (1 - p)),
    binom_var_identity (2 * N) (by omega) p]
  push_cast
  have hN1 : (1 : ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
  have hNne : (N : ℝ) ≠ 0 := by linarith
  have h2Npos : (0 : ℝ) < 2 * (N : ℝ) - 1 := by linarith
  have h2N : (2 : ℝ) * (N : ℝ) - 1 ≠ 0 := h2Npos.ne'
  have hple : (0 : ℝ) < 1 - p := by linarith
  have hp : p ≠ 0 := hp0.ne'
  have h1p : (1 : ℝ) - p ≠ 0 := hple.ne'
  have hfac : (1 : ℝ) - 1 / (2 * (N : ℝ)) ≠ 0 := by
    have hle : 1 / (2 * (N : ℝ)) ≤ 1 / 2 :=
      one_div_le_one_div_of_le (by norm_num) (by linarith)
    have hpos : (0 : ℝ) < 1 - 1 / (2 * (N : ℝ)) := by linarith
    exact hpos.ne'
  field_simp
  ring

end Drift
end Bastion
