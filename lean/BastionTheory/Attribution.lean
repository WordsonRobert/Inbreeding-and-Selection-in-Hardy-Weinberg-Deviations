/-
# The BASTION attribution game (P9, applied)

Ground truth: `model.py`, `shapley_step`, which builds the game
`v[mask] = hw_dev(operators in `mask`, applied in life-cycle order) - D0`
over the five operators `0 = M, 1 = Y0, 2 = I, 3 = Dr, 4 = S`, and takes Shapley values
of it.
-/
import BastionTheory.Shapley

namespace Bastion

open Geno Finset

/-- Apply `f` only if operator `i` is in the coalition `T`.
Python: `if mask & (1 << op): g = _apply(op, ...)`. -/
noncomputable def stepIf (T : Finset (Fin 5)) (i : Fin 5) (f : Geno → Geno) (x : Geno) : Geno :=
  if i ∈ T then f x else x

private theorem stepIf_pos {T : Finset (Fin 5)} {i : Fin 5} (h : i ∈ T) (f : Geno → Geno) (x : Geno) :
    stepIf T i f x = f x := by rw [stepIf, if_pos h]

private theorem stepIf_neg {T : Finset (Fin 5)} {i : Fin 5} (h : i ∉ T) (f : Geno → Geno) (x : Geno) :
    stepIf T i f x = x := by rw [stepIf, if_neg h]

private theorem stepIf_insert_other {i j : Fin 5} (hij : i ≠ j) (S : Finset (Fin 5))
    (f : Geno → Geno) (x : Geno) : stepIf (insert j S) i f x = stepIf S i f x := by
  by_cases h : i ∈ S
  · rw [stepIf_pos (Finset.mem_insert_of_mem h), stepIf_pos h]
  · have h' : i ∉ insert j S := by
      rw [Finset.mem_insert]; exact not_or.2 ⟨hij, h⟩
    rw [stepIf_neg h', stepIf_neg h]

private theorem total_stepIf {T : Finset (Fin 5)} {i : Fin 5} {f : Geno → Geno} {x : Geno}
    (hf : ∀ y : Geno, y.x1 + y.x2 + y.x3 = 1 → (f y).x1 + (f y).x2 + (f y).x3 = 1)
    (hx : x.x1 + x.x2 + x.x3 = 1) :
    (stepIf T i f x).x1 + (stepIf T i f x).x2 + (stepIf T i f x).x3 = 1 := by
  rw [stepIf]
  split_ifs with h
  · exact hf x hx
  · exact hx

private theorem p_stepIf {T : Finset (Fin 5)} {i : Fin 5} {f : Geno → Geno} {x : Geno}
    (hf : ∀ y : Geno, (f y).p = y.p) : (stepIf T i f x).p = x.p := by
  rw [stepIf]
  split_ifs with h
  · exact hf x
  · rfl

/-- The first four operators of the life cycle, applied in order.
Python: the `for op in perm` loop of `shapley_step` restricted to `op < 4`. -/
noncomputable def preSel (u v F δ : ℝ) (T : Finset (Fin 5)) (a : Geno) : Geno :=
  stepIf T 3 (drift δ) (stepIf T 2 (inbreed F) (stepIf T 1 reset (stepIf T 0 (mutate u v) a)))

/-- All five operators of the life cycle, applied in order.
Python: the `for op in range(n)` loop of `shapley_step`. -/
noncomputable def applySet (u v F δ : ℝ) (w : Fit) (T : Finset (Fin 5)) (a : Geno) : Geno :=
  stepIf T 4 (sel w) (preSel u v F δ T a)

/-- The attribution game.  Python: `v[mask] = hw_dev(...) - D0` in `shapley_step`. -/
noncomputable def game (u v F δ : ℝ) (w : Fit) (a : Geno) (T : Finset (Fin 5)) : ℝ :=
  (applySet u v F δ w T a).D - a.D

private theorem total_preSel (u v F δ : ℝ) (T : Finset (Fin 5)) {a : Geno}
    (ha : a.x1 + a.x2 + a.x3 = 1) :
    (preSel u v F δ T a).x1 + (preSel u v F δ T a).x2 + (preSel u v F δ T a).x3 = 1 := by
  refine total_stepIf (fun y _ => total_vec _ _) ?_
  refine total_stepIf (fun y _ => total_vec _ _) ?_
  refine total_stepIf (fun y _ => total_vec _ _) ?_
  exact total_stepIf (fun y hy => total_mutate _ _ _) ha

@[simp] private theorem preSel_empty (u v F δ : ℝ) (a : Geno) : preSel u v F δ ∅ a = a := by
  simp only [preSel, stepIf_neg (Finset.notMem_empty _)]

@[simp] private theorem applySet_empty (u v F δ : ℝ) (w : Fit) (a : Geno) :
    applySet u v F δ w ∅ a = a := by
  simp only [applySet, preSel_empty, stepIf_neg (Finset.notMem_empty _)]

private theorem preSel_univ (u v F δ : ℝ) (a : Geno) :
    preSel u v F δ univ a = drift δ (inbreed F (reset (mutate u v a))) := by
  simp only [preSel, stepIf_pos (Finset.mem_univ _)]

private theorem applySet_univ (u v F δ : ℝ) (w : Fit) (a : Geno) :
    applySet u v F δ w univ a = sel w (drift δ (inbreed F (reset (mutate u v a)))) := by
  simp only [applySet, preSel_univ, stepIf_pos (Finset.mem_univ _)]

/-- **P9**, efficiency applied: the Shapley shares of the five operators add up exactly to
the observed one-generation change of the HW deviation. -/
theorem sum_shapley_game (u v F δ : ℝ) (w : Fit) (a : Geno) :
    ∑ i, ShapleyGame.shapley (game u v F δ w a) i
      = (sel w (drift δ (inbreed F (reset (mutate u v a))))).D - a.D := by
  rw [ShapleyGame.sum_shapley]
  simp only [game, applySet_univ, applySet_empty, sub_self, sub_zero]

/-- **P9** + **P8**: the Shapley shares add up to the change in `D` between the adults of
one generation and the adults of the next. -/
theorem sum_shapley_game_lifecycle (u v F δ : ℝ) (w : Fit) {a : Geno}
    (ha : a.x1 + a.x2 + a.x3 = 1) (hm0 : 0 < mutAllele u v a.p) (hm1 : mutAllele u v a.p < 1) :
    ∑ i, ShapleyGame.shapley (game u v F δ w a) i
      = (sel w (zyg (mutAllele u v a.p + δ) F)).D - a.D := by
  rw [sum_shapley_game, lifecycle w u v F δ ha hm0 hm1]

/-! ### Null operators get a zero Shapley share -/

private theorem preSel_insert_four (u v F δ : ℝ) (S : Finset (Fin 5)) (a : Geno) :
    preSel u v F δ (insert 4 S) a = preSel u v F δ S a := by
  simp only [preSel, stepIf_insert_other (by decide : (0 : Fin 5) ≠ 4),
    stepIf_insert_other (by decide : (1 : Fin 5) ≠ 4),
    stepIf_insert_other (by decide : (2 : Fin 5) ≠ 4),
    stepIf_insert_other (by decide : (3 : Fin 5) ≠ 4)]

/-- **P9**, dummy applied: under neutrality (`s = 0`, i.e. `w = (1,1,1)`) the selection
operator is a dummy, so its Shapley share is exactly zero. -/
theorem shapley_selection_eq_zero (u v F δ : ℝ) {a : Geno} (ha : a.x1 + a.x2 + a.x3 = 1) :
    ShapleyGame.shapley (game u v F δ ⟨1, 1, 1⟩ a) 4 = 0 := by
  refine ShapleyGame.shapley_dummy _ fun S hS => ?_
  simp only [game, applySet, preSel_insert_four u v F δ S a,
    stepIf_pos (Finset.mem_insert_self (4 : Fin 5) S), stepIf_neg hS,
    sel_one (total_preSel u v F δ S ha)]

private theorem preSel_insert_zero (F δ : ℝ) {a : Geno} (ha : a.x1 + a.x2 + a.x3 = 1)
    {S : Finset (Fin 5)} (hS : (0 : Fin 5) ∉ S) :
    preSel 0 0 F δ (insert 0 S) a = preSel 0 0 F δ S a := by
  simp only [preSel, stepIf_insert_other (by decide : (1 : Fin 5) ≠ 0),
    stepIf_insert_other (by decide : (2 : Fin 5) ≠ 0),
    stepIf_insert_other (by decide : (3 : Fin 5) ≠ 0),
    stepIf_pos (Finset.mem_insert_self (0 : Fin 5) S), stepIf_neg hS, mutate_zero ha]

/-- **P9**, dummy applied: with zero mutation rates the mutation operator is a dummy, so
its Shapley share is exactly zero. -/
theorem shapley_mutation_eq_zero (F δ : ℝ) (w : Fit) {a : Geno}
    (ha : a.x1 + a.x2 + a.x3 = 1) :
    ShapleyGame.shapley (game 0 0 F δ w a) 0 = 0 := by
  refine ShapleyGame.shapley_dummy _ fun S hS => ?_
  simp only [game, applySet, stepIf_insert_other (by decide : (4 : Fin 5) ≠ 0),
    preSel_insert_zero F δ ha hS]

/-! ### The inbreeding and drift operators -/

/-- Unit total of the vector reaching the `I` step of `preSel`. -/
private theorem total_preI (u v : ℝ) (T : Finset (Fin 5)) {a : Geno}
    (ha : a.x1 + a.x2 + a.x3 = 1) :
    (stepIf T 1 reset (stepIf T 0 (mutate u v) a)).x1
      + (stepIf T 1 reset (stepIf T 0 (mutate u v) a)).x2
      + (stepIf T 1 reset (stepIf T 0 (mutate u v) a)).x3 = 1 := by
  refine total_stepIf (fun _ _ => total_vec _ _) ?_
  exact total_stepIf (fun _ _ => total_mutate _ _ _) ha

/-- Unit total of the vector reaching the `Dr` step of `preSel`. -/
private theorem total_preDr (u v F : ℝ) (T : Finset (Fin 5)) {a : Geno}
    (ha : a.x1 + a.x2 + a.x3 = 1) :
    (stepIf T 2 (inbreed F) (stepIf T 1 reset (stepIf T 0 (mutate u v) a))).x1
      + (stepIf T 2 (inbreed F) (stepIf T 1 reset (stepIf T 0 (mutate u v) a))).x2
      + (stepIf T 2 (inbreed F) (stepIf T 1 reset (stepIf T 0 (mutate u v) a))).x3 = 1 :=
  total_stepIf (fun _ _ => total_vec _ _) (total_preI u v T ha)

/-- The allele frequency reaching the `Dr` step is the one the `M` step produced: `Y0` and
`I` both preserve `p`. -/
private theorem p_preDr (u v F : ℝ) (T : Finset (Fin 5)) (a : Geno) :
    (stepIf T 2 (inbreed F) (stepIf T 1 reset (stepIf T 0 (mutate u v) a))).p
      = (stepIf T 0 (mutate u v) a).p := by
  rw [p_stepIf (fun y => p_inbreed F y), p_stepIf (fun y => p_reset y)]

/-- Whichever way the `M` step went, the frequency reaching `Dr` is interior as soon as
both `a.p` and the post-mutation frequency `m` are. -/
private theorem interior_preDr (u v F : ℝ) (T : Finset (Fin 5)) {a : Geno}
    (ha : a.x1 + a.x2 + a.x3 = 1) (hap0 : 0 < a.p) (hap1 : a.p < 1)
    (hm0 : 0 < mutAllele u v a.p) (hm1 : mutAllele u v a.p < 1) :
    0 < (stepIf T 2 (inbreed F) (stepIf T 1 reset (stepIf T 0 (mutate u v) a))).p
      ∧ (stepIf T 2 (inbreed F) (stepIf T 1 reset (stepIf T 0 (mutate u v) a))).p < 1 := by
  rw [p_preDr, stepIf]
  split_ifs with h
  · rw [p_mutate ha]; exact ⟨hm0, hm1⟩
  · exact ⟨hap0, hap1⟩

private theorem preSel_insert_two (u v δ : ℝ) {a : Geno} (ha : a.x1 + a.x2 + a.x3 = 1)
    {S : Finset (Fin 5)} (hS : (2 : Fin 5) ∉ S) :
    preSel u v 0 δ (insert 2 S) a = preSel u v 0 δ S a := by
  simp only [preSel, stepIf_insert_other (by decide : (0 : Fin 5) ≠ 2),
    stepIf_insert_other (by decide : (1 : Fin 5) ≠ 2),
    stepIf_insert_other (by decide : (3 : Fin 5) ≠ 2),
    stepIf_pos (Finset.mem_insert_self (2 : Fin 5) S), stepIf_neg hS,
    inbreed_zero (total_preI u v S ha)]

/-- **P9**, dummy applied: under random mating (`F = 0`) the inbreeding operator is a
dummy, so its Shapley share is exactly zero. -/
theorem shapley_inbreeding_eq_zero (u v δ : ℝ) (w : Fit) {a : Geno}
    (ha : a.x1 + a.x2 + a.x3 = 1) :
    ShapleyGame.shapley (game u v 0 δ w a) 2 = 0 := by
  refine ShapleyGame.shapley_dummy _ fun S hS => ?_
  simp only [game, applySet, stepIf_insert_other (by decide : (4 : Fin 5) ≠ 2),
    preSel_insert_two u v δ ha hS]

private theorem preSel_insert_three (u v F : ℝ) {a : Geno} (ha : a.x1 + a.x2 + a.x3 = 1)
    (hap0 : 0 < a.p) (hap1 : a.p < 1) (hm0 : 0 < mutAllele u v a.p)
    (hm1 : mutAllele u v a.p < 1) {S : Finset (Fin 5)} (hS : (3 : Fin 5) ∉ S) :
    preSel u v F 0 (insert 3 S) a = preSel u v F 0 S a := by
  obtain ⟨hi0, hi1⟩ := interior_preDr u v F S ha hap0 hap1 hm0 hm1
  simp only [preSel, stepIf_insert_other (by decide : (0 : Fin 5) ≠ 3),
    stepIf_insert_other (by decide : (1 : Fin 5) ≠ 3),
    stepIf_insert_other (by decide : (2 : Fin 5) ≠ 3),
    stepIf_pos (Finset.mem_insert_self (3 : Fin 5) S), stepIf_neg hS]
  exact drift_zero (total_preDr u v F S ha) hi0 hi1

/-- **P9**, dummy applied: with no drift step (`δ = 0`) the drift operator is a dummy, so
its Shapley share is exactly zero.  The extra hypotheses are the minimal ones that make
`Dr_0` the identity on every vector it can see: `drift_zero` needs an interior allele
frequency, and the frequency reaching the drift step is either `a.p` (when the mutation
operator is absent from the coalition) or the post-mutation frequency `m` (when it is
present), so both must be interior. -/
theorem shapley_drift_eq_zero (u v F : ℝ) (w : Fit) {a : Geno}
    (ha : a.x1 + a.x2 + a.x3 = 1) (hap0 : 0 < a.p) (hap1 : a.p < 1)
    (hm0 : 0 < mutAllele u v a.p) (hm1 : mutAllele u v a.p < 1) :
    ShapleyGame.shapley (game u v F 0 w a) 3 = 0 := by
  refine ShapleyGame.shapley_dummy _ fun S hS => ?_
  simp only [game, applySet, stepIf_insert_other (by decide : (4 : Fin 5) ≠ 3),
    preSel_insert_three u v F ha hap0 hap1 hm0 hm1 hS]

end Bastion
