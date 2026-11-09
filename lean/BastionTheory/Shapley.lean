/-
# Shapley attribution (P9) and the two-player level split (P10)

Ground truth: `model.py`, functions `shapley_step` (the subset-weight formula
`|S|!(n-|S|-1)!/n!`, and the sequential life-cycle attribution `seq`) and `attribute_path`
(the two-player split of the level `D^a` into an inbreeding and a selection share).

The Shapley value is defined here *by the subset-weight formula*, i.e. exactly as
`shapley_step` computes it, and the three axioms are proved for that definition.  See the
README, section "Corrections", for the permutation-average characterisation, which is
stated but not machine-checked.
-/
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.LinearCombination
import BastionTheory.LifeCycle

namespace Bastion
namespace ShapleyGame

open Finset

variable {n : ℕ}

/-- The Shapley weight `|S|! (n - |S| - 1)! / n!` of a coalition of size `k`.
Python: `wgt = fact[k] * fact[n - k - 1] / fact[n]`. -/
noncomputable def wt (n k : ℕ) : ℝ :=
  (Nat.factorial k : ℝ) * (Nat.factorial (n - k - 1) : ℝ) / (Nat.factorial n : ℝ)

/-- The Shapley value of player `i` in the game `v`.  Python: `shapley_step`, array `phi`. -/
noncomputable def shapley (v : Finset (Fin n) → ℝ) (i : Fin n) : ℝ :=
  ∑ S ∈ (univ.erase i).powerset, wt n S.card * (v (insert i S) - v S)

private theorem mem_pow_erase {i : Fin n} {S : Finset (Fin n)} :
    S ∈ (univ.erase i).powerset ↔ i ∉ S := by
  rw [Finset.mem_powerset]
  constructor
  · intro h hiS; exact (Finset.mem_erase.1 (h hiS)).1 rfl
  · intro h x hx
    exact Finset.mem_erase.2 ⟨fun hxi => h (hxi ▸ hx), Finset.mem_univ x⟩

private theorem powerset_erase_eq (i : Fin n) :
    (univ.erase i).powerset = univ.filter (fun S : Finset (Fin n) => i ∉ S) := by
  ext S
  rw [mem_pow_erase, Finset.mem_filter]
  simp

/-! ### Factorial arithmetic -/

private theorem fact_mul_pred {t : ℕ} (ht : 1 ≤ t) :
    t * Nat.factorial (t - 1) = Nat.factorial t := by
  obtain ⟨m, rfl⟩ : ∃ m, t = m + 1 := ⟨t - 1, by omega⟩
  simp [Nat.factorial_succ]

private theorem cast_fact_mul_pred {t : ℕ} (ht : 1 ≤ t) :
    (t : ℝ) * (Nat.factorial (t - 1) : ℝ) = (Nat.factorial t : ℝ) := by
  exact_mod_cast congrArg (fun m : ℕ => (m : ℝ)) (fact_mul_pred ht)

private theorem cast_fact_pos (m : ℕ) : (0 : ℝ) < (Nat.factorial m : ℝ) := by
  exact_mod_cast Nat.factorial_pos m

/-- The coefficient that the player `i` sum attaches to `v T`. -/
private noncomputable def coeff (n t : ℕ) : ℝ := (t : ℝ) * wt n (t - 1) - ((n - t : ℕ) : ℝ) * wt n t

private theorem coeff_mid {t : ℕ} (h1 : 1 ≤ t) (h2 : t < n) : coeff n t = 0 := by
  have e1 : n - (t - 1) - 1 = n - t := by omega
  have hnt : 1 ≤ n - t := by omega
  have e2 := cast_fact_mul_pred h1
  have e3 := cast_fact_mul_pred hnt
  have hfac := (cast_fact_pos n).ne'
  simp only [coeff, wt, e1]
  field_simp
  linear_combination (Nat.factorial (n - t) : ℝ) * e2 - (Nat.factorial t : ℝ) * e3

private theorem coeff_zero (hn : 1 ≤ n) : coeff n 0 = -1 := by
  have e2 := cast_fact_mul_pred hn
  have hfac := (cast_fact_pos n).ne'
  simp only [coeff, wt, Nat.cast_zero, zero_mul, zero_sub, Nat.sub_zero, Nat.factorial_zero,
    Nat.cast_one, one_mul]
  field_simp
  linarith [e2]

private theorem coeff_full (hn : 1 ≤ n) : coeff n n = 1 := by
  have e2 := cast_fact_mul_pred hn
  have hfac := (cast_fact_pos n).ne'
  have e1 : n - (n - 1) - 1 = 0 := by omega
  simp only [coeff, wt, e1, Nat.sub_self, Nat.factorial_zero, Nat.cast_one, mul_one,
    Nat.cast_zero, zero_mul, sub_zero]
  field_simp
  linarith [e2]

/-! ### P9: efficiency -/

private theorem shapley_split (v : Finset (Fin n) → ℝ) (i : Fin n) :
    shapley v i
      = (∑ T ∈ univ.filter (fun T : Finset (Fin n) => i ∈ T), wt n (T.card - 1) * v T)
        - ∑ S ∈ univ.filter (fun S : Finset (Fin n) => i ∉ S), wt n S.card * v S := by
  rw [shapley]
  have hsub : ∀ S ∈ (univ.erase i).powerset,
      wt n S.card * (v (insert i S) - v S) = wt n S.card * v (insert i S) - wt n S.card * v S := by
    intro S _; ring
  rw [Finset.sum_congr rfl hsub, Finset.sum_sub_distrib]
  congr 1
  · refine Finset.sum_nbij' (fun S => insert i S) (fun T => T.erase i) ?_ ?_ ?_ ?_ ?_
    · intro S hS; simp [Finset.mem_filter]
    · intro T hT
      rw [mem_pow_erase]
      exact Finset.notMem_erase i T
    · intro S hS; exact Finset.erase_insert (mem_pow_erase.1 hS)
    · intro T hT
      exact Finset.insert_erase (by simpa using (Finset.mem_filter.1 hT).2)
    · intro S hS
      rw [Finset.card_insert_of_notMem (mem_pow_erase.1 hS)]
      simp
  · rw [powerset_erase_eq]

private theorem sum_mem_filter (f : Finset (Fin n) → ℝ) :
    (∑ i : Fin n, ∑ T ∈ univ.filter (fun T : Finset (Fin n) => i ∈ T), f T)
      = ∑ T ∈ (univ : Finset (Finset (Fin n))), (T.card : ℝ) * f T := by
  simp_rw [Finset.sum_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun T _ => ?_
  rw [Finset.sum_ite_mem, Finset.univ_inter, Finset.sum_const, nsmul_eq_mul]

private theorem sum_not_mem_filter (f : Finset (Fin n) → ℝ) :
    (∑ i : Fin n, ∑ S ∈ univ.filter (fun S : Finset (Fin n) => i ∉ S), f S)
      = ∑ S ∈ (univ : Finset (Finset (Fin n))), ((n - S.card : ℕ) : ℝ) * f S := by
  simp_rw [Finset.sum_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun S _ => ?_
  have : ∀ i : Fin n, (if i ∉ S then f S else 0) = (if i ∈ Sᶜ then f S else 0) := by
    intro i; simp [Finset.mem_compl]
  rw [Finset.sum_congr rfl fun i _ => this i, Finset.sum_ite_mem, Finset.univ_inter,
    Finset.sum_const, nsmul_eq_mul, Finset.card_compl, Fintype.card_fin]

/-- **P9**, efficiency: the Shapley values of all players sum to the total value of the
game.  This certifies the efficiency of `shapley_step`'s `phi`. -/
theorem sum_shapley (v : Finset (Fin n) → ℝ) : ∑ i, shapley v i = v univ - v ∅ := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp
  have hne : (∅ : Finset (Fin n)) ≠ univ := by
    intro h
    have := congrArg Finset.card h
    simp [Finset.card_univ] at this
    omega
  rw [Finset.sum_congr rfl fun i (_ : i ∈ univ) => shapley_split v i, Finset.sum_sub_distrib,
    sum_mem_filter, sum_not_mem_filter, ← Finset.sum_sub_distrib]
  have hcoeff : ∀ T ∈ (univ : Finset (Finset (Fin n))),
      (T.card : ℝ) * (wt n (T.card - 1) * v T) - ((n - T.card : ℕ) : ℝ) * (wt n T.card * v T)
        = coeff n T.card * v T := by
    intro T _; simp only [coeff]; ring
  rw [Finset.sum_congr rfl hcoeff]
  rw [← Finset.sum_subset (Finset.subset_univ ({∅, univ} : Finset (Finset (Fin n))))]
  · rw [Finset.sum_pair hne, Finset.card_empty, Finset.card_univ, Fintype.card_fin,
      coeff_zero hn, coeff_full hn]
    ring
  · intro T _ hT
    simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hT
    have hpos : 1 ≤ T.card := Finset.card_pos.2 (Finset.nonempty_iff_ne_empty.2 hT.1)
    have hlt : T.card < n := by
      have := Finset.card_lt_card (Finset.ssubset_univ_iff.2 hT.2)
      rwa [Finset.card_univ, Fintype.card_fin] at this
    rw [coeff_mid hpos hlt, zero_mul]

/-! ### P9: dummy and symmetry -/

/-- **P9**, dummy: a player who never changes the value gets zero. -/
theorem shapley_dummy (v : Finset (Fin n) → ℝ) {i : Fin n}
    (h : ∀ S : Finset (Fin n), i ∉ S → v (insert i S) = v S) : shapley v i = 0 := by
  rw [shapley]
  refine Finset.sum_eq_zero fun S hS => ?_
  rw [h S (mem_pow_erase.1 hS), sub_self, mul_zero]

/-- **P9**, symmetry: interchangeable players get equal Shapley values. -/
theorem shapley_symm (v : Finset (Fin n) → ℝ) {i j : Fin n} (hij : i ≠ j)
    (hsym : ∀ S : Finset (Fin n), i ∉ S → j ∉ S → v (insert i S) = v (insert j S)) :
    shapley v i = shapley v j := by
  rw [shapley, shapley]
  refine Finset.sum_nbij' (fun S => if j ∈ S then insert i (S.erase j) else S)
    (fun T => if i ∈ T then insert j (T.erase i) else T) ?_ ?_ ?_ ?_ ?_
  · intro S hS
    have hi : i ∉ S := mem_pow_erase.1 hS
    rw [mem_pow_erase]
    by_cases hj : j ∈ S
    · simp [hj, Finset.mem_insert, hij.symm, Finset.mem_erase]
    · simp [hj]
  · intro T hT
    have hj : j ∉ T := mem_pow_erase.1 hT
    rw [mem_pow_erase]
    by_cases hi : i ∈ T
    · simp [hi, Finset.mem_insert, hij, Finset.mem_erase]
    · simp [hi]
  · intro S hS
    have hi : i ∉ S := mem_pow_erase.1 hS
    by_cases hj : j ∈ S
    · have hiS : i ∉ S.erase j := fun hc => hi (Finset.mem_of_mem_erase hc)
      rw [if_pos hj, if_pos (Finset.mem_insert_self i (S.erase j)),
        Finset.erase_insert hiS, Finset.insert_erase hj]
    · simp [hj, hi]
  · intro T hT
    have hj : j ∉ T := mem_pow_erase.1 hT
    by_cases hi : i ∈ T
    · have hjT : j ∉ T.erase i := fun hc => hj (Finset.mem_of_mem_erase hc)
      rw [if_pos hi, if_pos (Finset.mem_insert_self j (T.erase i)),
        Finset.erase_insert hjT, Finset.insert_erase hi]
    · simp [hi, hj]
  · intro S hS
    have hi : i ∉ S := mem_pow_erase.1 hS
    by_cases hj : j ∈ S
    · have hiS : i ∉ S.erase j := fun hc => hi (Finset.mem_of_mem_erase hc)
      have hjS : j ∉ S.erase j := Finset.notMem_erase j S
      rw [if_pos hj]
      have hScard : 1 ≤ S.card := Finset.card_pos.2 ⟨j, hj⟩
      have hcard : (insert i (S.erase j)).card = S.card := by
        rw [Finset.card_insert_of_notMem hiS, Finset.card_erase_of_mem hj]
        omega
      have hset : insert j (insert i (S.erase j)) = insert i S := by
        rw [Finset.insert_comm, Finset.insert_erase hj]
      have hSeq : insert j (S.erase j) = S := Finset.insert_erase hj
      rw [hcard, hset, hsym (S.erase j) hiS hjS, hSeq]
    · rw [if_neg hj, hsym S hi hj]

/-! ### P9: sequential attribution along a fixed order -/

/-- **P9**, telescoping: the sequential attribution along *any* chain from `∅` to `univ`
sums to the total value of the game.  This certifies the `seq` array of `shapley_step`,
which is the life-cycle-order attribution of the paper. -/
theorem sum_seq_telescope (v : Finset (Fin n) → ℝ) (S : ℕ → Finset (Fin n))
    (h0 : S 0 = ∅) (hfull : S n = univ) :
    ∑ k ∈ Finset.range n, (v (S (k + 1)) - v (S k)) = v univ - v ∅ := by
  rw [Finset.sum_range_sub (fun k => v (S k)), h0, hfull]

end ShapleyGame

/-! ## P10: the two-player level split -/

open Geno

/-- Level deviation created by inbreeding alone.  Python: `vI` in `attribute_path`. -/
noncomputable def levelI (p F : ℝ) : ℝ := (zyg p F).D

/-- Level deviation created by selection alone, acting on Hardy–Weinberg zygotes.
Python: `vS` in `attribute_path`. -/
noncomputable def levelS (w : Fit) (p : ℝ) : ℝ := (sel w (zyg p 0)).D

/-- Level deviation created by inbreeding and selection together.  Python: `vIS` in
`attribute_path`. -/
noncomputable def levelIS (w : Fit) (p F : ℝ) : ℝ := (sel w (zyg p F)).D

/-- Inbreeding share of the observed adult deviation (two-player Shapley).
Python: `Dz` in `attribute_path`. -/
noncomputable def Dz (w : Fit) (p F : ℝ) : ℝ :=
  (levelI p F + levelIS w p F - levelS w p) / 2

/-- Selection share of the observed adult deviation (two-player Shapley).
Python: `DS` in `attribute_path`. -/
noncomputable def DS (w : Fit) (p F : ℝ) : ℝ :=
  (levelS w p + levelIS w p F - levelI p F) / 2

/-- **P10**, efficiency of the two-player split. -/
theorem Dz_add_DS (w : Fit) (p F : ℝ) : Dz w p F + DS w p F = levelIS w p F := by
  simp only [Dz, DS]; ring

/-- **P10**, the inbreeding share vanishes under random mating. -/
theorem Dz_eq_zero (w : Fit) (p : ℝ) : Dz w p 0 = 0 := by
  simp only [Dz, levelI, levelIS, levelS, D_zyg]
  ring

/-- **P10**, the selection share vanishes under neutrality. -/
theorem DS_eq_zero (p F : ℝ) : DS ⟨1, 1, 1⟩ p F = 0 := by
  have h1 : sel (⟨1, 1, 1⟩ : Fit) (zyg p 0) = zyg p 0 := sel_one (total_vec _ _)
  have h2 : sel (⟨1, 1, 1⟩ : Fit) (zyg p F) = zyg p F := sel_one (total_vec _ _)
  simp only [DS, levelS, levelIS, levelI, h1, h2, D_zyg]
  ring

/-- **P10**, the selection share vanishes when `s = 0` in the BASTION fitnesses. -/
theorem DS_eq_zero_bastion (h p F : ℝ) : DS (bastionFit 0 h) p F = 0 := by
  rw [bastionFit_zero]; exact DS_eq_zero p F

end Bastion
