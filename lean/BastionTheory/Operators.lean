/-
# The five life-cycle operators (P7)

Ground truth: `model.py`, function `_apply`, branches
`0 = M` (mutation, in `Mutation.lean`), `1 = Y0` (Mendelian reset), `2 = I` (inbreeding),
`3 = Dr` (drift), `4 = S` (selection, in `Selection.lean`).

Deviation from the Python: `_apply` calls `_vec_with`, which clips the deviation to the
admissible range, and guards the drift branch with `pq > 1e-15` (using `F_g = 0` below
that).  Here `vec` does not clip and the drift operator is stated for `p ∈ (0,1)`, so the
guard is never taken.
-/
import BastionTheory.Selection
import BastionTheory.Mutation

namespace Bastion

open Geno

/-- `Y0`: random union of gametes, i.e. Mendelian reset to Hardy–Weinberg at the current
allele frequency.  Python: `_apply`, branch `op == 1`. -/
noncomputable def reset (x : Geno) : Geno := vec x.p 0

/-- `I`: the mating system adds `F·p·q` of homozygosity.  Python: `_apply`, `op == 2`. -/
noncomputable def inbreed (F : ℝ) (x : Geno) : Geno := vec x.p (x.D + F * x.p * (1 - x.p))

/-- `Dr`: drift shifts the allele frequency by `δ`, keeping the genotypic `F_g = D/(pq)`
fixed.  Python: `_apply`, `op == 3`. -/
noncomputable def drift (δ : ℝ) (x : Geno) : Geno :=
  vec (x.p + δ) (x.D / (x.p * (1 - x.p)) * ((x.p + δ) * (1 - (x.p + δ))))

/-! ### P7: what each operator does to `(p, D)` -/

@[simp] theorem p_reset (x : Geno) : (reset x).p = x.p := p_vec _ _

/-- **P7.**  The Mendelian reset annihilates the HW deviation. -/
@[simp] theorem D_reset (x : Geno) : (reset x).D = 0 := D_vec _ _

@[simp] theorem p_inbreed (F : ℝ) (x : Geno) : (inbreed F x).p = x.p := p_vec _ _

/-- **P7.**  Inbreeding adds exactly `F·p·q`. -/
@[simp] theorem D_inbreed (F : ℝ) (x : Geno) :
    (inbreed F x).D = x.D + F * x.p * (1 - x.p) := D_vec _ _

@[simp] theorem p_drift (δ : ℝ) (x : Geno) : (drift δ x).p = x.p + δ := p_vec _ _

/-- **P7.**  Drift moves `p` and rescales `D` at fixed `F_g`. -/
@[simp] theorem D_drift (δ : ℝ) (x : Geno) :
    (drift δ x).D = x.D / (x.p * (1 - x.p)) * ((x.p + δ) * (1 - (x.p + δ))) := D_vec _ _

/-! ### P7: null operators are the identity -/

/-- **P7.**  `F = 0` makes inbreeding the identity. -/
theorem inbreed_zero {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1) : inbreed 0 x = x := by
  rw [inbreed]
  have : x.D + 0 * x.p * (1 - x.p) = x.D := by ring
  rw [this, vec_p_D_self hx]

/-- **P7.**  `δ = 0` makes drift the identity at an interior allele frequency. -/
theorem drift_zero {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1) (hp0 : 0 < x.p) (hp1 : x.p < 1) :
    drift 0 x = x := by
  have hp : x.p ≠ 0 := ne_of_gt hp0
  have hq : (1:ℝ) - x.p ≠ 0 := ne_of_gt (by linarith)
  have hpq : x.p * (1 - x.p) ≠ 0 := mul_ne_zero hp hq
  have key : x.D / (x.p * (1 - x.p)) * ((x.p + 0) * (1 - (x.p + 0))) = x.D := by
    rw [add_zero]; field_simp
  rw [drift, key, add_zero, vec_p_D_self hx]

/-- The neutral fitness scheme is `(1,1,1)`. -/
theorem bastionFit_zero (h : ℝ) : bastionFit 0 h = ⟨1, 1, 1⟩ := by
  simp only [bastionFit]; norm_num

/-- **P7.**  Neutral selection is the identity on vectors of unit total; with the BASTION
fitnesses this is `s = 0`. -/
theorem sel_one {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1) : sel ⟨1, 1, 1⟩ x = x := by
  have hb : (⟨1, 1, 1⟩ : Fit).bar x = 1 := by simp only [Fit.bar]; linarith
  ext <;> simp only [sel_x1, sel_x2, sel_x3, hb, div_one, one_mul]

/-- **P7.**  `s = 0` makes selection the identity. -/
theorem sel_bastion_zero {x : Geno} (h : ℝ) (hx : x.x1 + x.x2 + x.x3 = 1) :
    sel (bastionFit 0 h) x = x := by
  rw [bastionFit_zero]; exact sel_one hx

end Bastion
