/-
# Cross-sectional identifiability (P12)

Ground truth: `experiments/e01_theory.py`, functions `_obs_probs`, `fim`, `identifiability`
(experiment E02).  The reported fact is
`cross_sectional_rank_max = 2`: a single cross-sectional genotype sample can never
identify more than two of the four parameters `(p0, F, s, h)`.

P12 is the structural reason, with no reference to the particular model: the Fisher
information of a multinomial observation is `Jᵀ W J` where `J` is the Jacobian of the cell
probabilities.  Since the cell probabilities sum to one, every column of `J` sums to zero,
so `J` has rank at most `m - 1` and hence so does `Jᵀ W J`, whatever `W` and however many
parameters there are.
-/
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.Analysis.Calculus.Deriv.Add

namespace Bastion

open Matrix Module

/-- The linear functional "sum of coordinates" on `Fin m → ℝ`. -/
noncomputable def sumLM (m : ℕ) : (Fin m → ℝ) →ₗ[ℝ] ℝ where
  toFun x := ∑ r, x r
  map_add' := by intro x y; simp [Finset.sum_add_distrib]
  map_smul' := by intro c x; simp [Finset.mul_sum]

@[simp] theorem sumLM_apply {m : ℕ} (x : Fin m → ℝ) : sumLM m x = ∑ r, x r := rfl

/-- The hyperplane of zero-sum vectors has dimension `m - 1`. -/
theorem finrank_ker_sumLM_le (m : ℕ) : finrank ℝ (LinearMap.ker (sumLM m)) ≤ m - 1 := by
  rcases Nat.eq_zero_or_pos m with hm | hm
  · subst hm
    have : finrank ℝ (LinearMap.ker (sumLM 0)) ≤ finrank ℝ (Fin 0 → ℝ) :=
      Submodule.finrank_le _
    simpa using this
  · have hsurj : Function.Surjective (sumLM m) := by
      intro c
      refine ⟨fun r => if r = ⟨0, hm⟩ then c else 0, ?_⟩
      simp
    have hrange : LinearMap.range (sumLM m) = ⊤ := LinearMap.range_eq_top.2 hsurj
    have hadd := LinearMap.finrank_range_add_finrank_ker (sumLM m)
    rw [hrange] at hadd
    have h1 : finrank ℝ (⊤ : Submodule ℝ ℝ) = 1 := by
      rw [finrank_top]; exact Module.finrank_self ℝ
    rw [h1, Module.finrank_pi] at hadd
    simp only [Fintype.card_fin] at hadd
    omega

/-- A matrix whose columns sum to zero has rank at most `m - 1`. -/
theorem rank_le_of_col_sum_zero {m k : ℕ} (J : Matrix (Fin m) (Fin k) ℝ)
    (hJ : ∀ c, ∑ r, J r c = 0) : J.rank ≤ m - 1 := by
  have hsub : LinearMap.range J.mulVecLin ≤ LinearMap.ker (sumLM m) := by
    rintro y ⟨x, rfl⟩
    simp only [LinearMap.mem_ker, sumLM_apply, Matrix.mulVecLin_apply, Matrix.mulVec,
      dotProduct]
    rw [Finset.sum_comm]
    refine Finset.sum_eq_zero fun c _ => ?_
    rw [← Finset.sum_mul, hJ c, zero_mul]
  exact le_trans (Submodule.finrank_mono hsub) (finrank_ker_sumLM_le m)

/-- **P12.**  For any `W`, the quadratic form `Jᵀ W J` built from a Jacobian with zero
column sums has rank at most `m - 1`, no matter how many parameters there are. -/
theorem rank_quadratic_le {m k : ℕ} (J : Matrix (Fin m) (Fin k) ℝ)
    (hJ : ∀ c, ∑ r, J r c = 0) (W : Matrix (Fin m) (Fin m) ℝ) :
    (Jᵀ * W * J).rank ≤ m - 1 :=
  le_trans (Matrix.rank_mul_le_right _ _) (rank_le_of_col_sum_zero J hJ)

/-- **P12**, genotype corollary: one cross-sectional genotype sample (three cells) carries
Fisher information of rank at most two, however many parameters the model has. -/
theorem rank_fisher_genotype_le_two {k : ℕ} (J : Matrix (Fin 3) (Fin k) ℝ)
    (hJ : ∀ c, ∑ r, J r c = 0) (W : Matrix (Fin 3) (Fin 3) ℝ) :
    (Jᵀ * W * J).rank ≤ 2 :=
  rank_quadratic_le J hJ W

/-- **P12**, phenotype corollary: a dominant-phenotype sample (two cells) carries Fisher
information of rank at most one. -/
theorem rank_fisher_phenotype_le_one {k : ℕ} (J : Matrix (Fin 2) (Fin k) ℝ)
    (hJ : ∀ c, ∑ r, J r c = 0) (W : Matrix (Fin 2) (Fin 2) ℝ) :
    (Jᵀ * W * J).rank ≤ 1 :=
  rank_quadratic_le J hJ W

/-- **P12**, the hypothesis is automatic: the Jacobian of any differentiable family of
probability vectors has zero column sums. -/
theorem sum_deriv_eq_zero_of_probability_family {m : ℕ} (f : ℝ → Fin m → ℝ) (d : Fin m → ℝ)
    (t : ℝ) (hsum : ∀ t, ∑ r, f t r = 1) (hd : ∀ r, HasDerivAt (fun t => f t r) (d r) t) :
    ∑ r, d r = 0 := by
  have heq : (fun t => ∑ r, f t r) = ∑ r : Fin m, (fun t => f t r) := by
    funext t'; simp [Finset.sum_apply]
  have h1 : HasDerivAt (fun t => ∑ r, f t r) (∑ r, d r) t := by
    rw [heq]; exact HasDerivAt.sum fun r _ => hd r
  have h2 : HasDerivAt (fun t => ∑ r, f t r) 0 t := by
    simp only [hsum]
    exact hasDerivAt_const t 1
  exact h1.unique h2

end Bastion
