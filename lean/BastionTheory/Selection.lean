/-
# Viability selection (P3, P4, P5)

Ground truth: `model.py`, functions `select`, `adults_at`, `adult_D_closed`.

`sel w x` is viability selection with fitnesses `w = (w1, w2, w3)` acting on the genotype
vector `x`; the BASTION fitnesses are `w = (1, 1 - h*s, 1 - s)`.  The quantity
`κ = w1*w3 - w2^2` is the selection distortion coefficient of the paper.
-/
import BastionTheory.Basic

namespace Bastion

open Geno

/-- Genotype fitnesses `(w_AA, w_Aa, w_aa)`.  Python: `w = (1, 1 - h*s, 1 - s)` in
`select` / `adult_D_closed`. -/
structure Fit where
  w1 : ℝ
  w2 : ℝ
  w3 : ℝ

/-- All three fitnesses are strictly positive. -/
structure Fit.Pos (w : Fit) : Prop where
  p1 : 0 < w.w1
  p2 : 0 < w.w2
  p3 : 0 < w.w3

namespace Fit

/-- Mean fitness.  Python: `wb` in `select` and `adult_D_closed`. -/
noncomputable def bar (w : Fit) (x : Geno) : ℝ := w.w1 * x.x1 + w.w2 * x.x2 + w.w3 * x.x3

/-- Selection distortion coefficient.  Python: `kap = w3 - w2 * w2` in
`e01_theory.identities` (with `w1 = 1`). -/
noncomputable def kappa (w : Fit) : ℝ := w.w1 * w.w3 - w.w2 ^ 2

end Fit

/-- Viability selection.  Python: `select`. -/
noncomputable def sel (w : Fit) (x : Geno) : Geno :=
  ⟨w.w1 * x.x1 / w.bar x, w.w2 * x.x2 / w.bar x, w.w3 * x.x3 / w.bar x⟩

/-- The BASTION fitness scheme.  Python: the `w2`, `w3` of `select`. -/
noncomputable def bastionFit (s h : ℝ) : Fit := ⟨1, 1 - h * s, 1 - s⟩

@[simp] theorem sel_x1 (w : Fit) (x : Geno) : (sel w x).x1 = w.w1 * x.x1 / w.bar x := rfl
@[simp] theorem sel_x2 (w : Fit) (x : Geno) : (sel w x).x2 = w.w2 * x.x2 / w.bar x := rfl
@[simp] theorem sel_x3 (w : Fit) (x : Geno) : (sel w x).x3 = w.w3 * x.x3 / w.bar x := rfl

/-- Mean fitness is positive on the simplex. -/
theorem bar_pos {w : Fit} {x : Geno} (hw : w.Pos) (hx : InSimplex x) : 0 < w.bar x := by
  have hc : 0 < min w.w1 (min w.w2 w.w3) := lt_min hw.p1 (lt_min hw.p2 hw.p3)
  have e1 : min w.w1 (min w.w2 w.w3) * x.x1 ≤ w.w1 * x.x1 :=
    mul_le_mul_of_nonneg_right (min_le_left _ _) hx.nn1
  have e2 : min w.w1 (min w.w2 w.w3) * x.x2 ≤ w.w2 * x.x2 :=
    mul_le_mul_of_nonneg_right ((min_le_right _ _).trans (min_le_left _ _)) hx.nn2
  have e3 : min w.w1 (min w.w2 w.w3) * x.x3 ≤ w.w3 * x.x3 :=
    mul_le_mul_of_nonneg_right ((min_le_right _ _).trans (min_le_right _ _)) hx.nn3
  have hsum : min w.w1 (min w.w2 w.w3) * (x.x1 + x.x2 + x.x3) ≤ w.bar x := by
    simp only [Fit.bar]; nlinarith
  rw [hx.total, mul_one] at hsum
  linarith

/-- Selection preserves unit total. -/
@[simp] theorem total_sel {w : Fit} {x : Geno} (hb : w.bar x ≠ 0) :
    (sel w x).x1 + (sel w x).x2 + (sel w x).x3 = 1 := by
  simp only [sel_x1, sel_x2, sel_x3, ← add_div]
  have h : w.w1 * x.x1 + w.w2 * x.x2 + w.w3 * x.x3 = w.bar x := rfl
  rw [h, div_self hb]

/-- Selection preserves the simplex. -/
theorem inSimplex_sel {w : Fit} {x : Geno} (hw : w.Pos) (hx : InSimplex x) :
    InSimplex (sel w x) := by
  have hb := bar_pos hw hx
  exact ⟨div_nonneg (mul_nonneg hw.p1.le hx.nn1) hb.le,
         div_nonneg (mul_nonneg hw.p2.le hx.nn2) hb.le,
         div_nonneg (mul_nonneg hw.p3.le hx.nn3) hb.le, total_sel hb.ne'⟩

/-- **P3.**  Closed form for the post-selection HW deviation.
This certifies `model.py : adult_D_closed`. -/
theorem D_sel_closed {w : Fit} {x : Geno} (hb : w.bar x ≠ 0) :
    (sel w x).D = (w.w1 * w.w3 * (x.x1 * x.x3) - w.w2 ^ 2 * x.x2 ^ 2 / 4) / (w.bar x) ^ 2 := by
  rw [D_eq_of_total (total_sel hb)]
  simp only [sel_x1, sel_x2, sel_x3]
  field_simp

/-- **P4.**  The `κ` decomposition: the post-selection deviation splits into a rescaled
inbreeding part and a selection distortion. -/
theorem D_sel_kappa {w : Fit} {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1) (hb : w.bar x ≠ 0) :
    (sel w x).D = (w.w1 * w.w3 * x.D + w.kappa * (x.x2 / 2) ^ 2) / (w.bar x) ^ 2 := by
  rw [D_sel_closed hb, D_eq_of_total hx, Fit.kappa]
  ring

/-- **P4**, BASTION fitnesses: `κ = s(2h − 1) − h²s²`. -/
theorem kappa_bastion (s h : ℝ) : (bastionFit s h).kappa = s * (2 * h - 1) - h ^ 2 * s ^ 2 := by
  simp only [Fit.kappa, bastionFit]; ring

/-- On a vector of unit total, `x2 = 2(pq − D)`; so `D < 0` is exactly heterozygote excess. -/
theorem x2_eq (x : Geno) : x.x2 = 2 * (x.p * (1 - x.p) - x.D) := by
  simp only [Geno.D, Geno.p]; ring

/-- **P5(a)**, the identity behind the sign rules: from Hardy–Weinberg zygotes the whole
post-selection deviation is the distortion term. -/
theorem D_sel_of_D_eq_zero {w : Fit} {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1)
    (hb : w.bar x ≠ 0) (h0 : x.D = 0) :
    (sel w x).D = w.kappa * ((x.x2 / 2) ^ 2 / (w.bar x) ^ 2) := by
  rw [D_sel_kappa hx hb, h0]; ring

section Sign

variable {w : Fit} {x : Geno}

/-- **P5(a)**, positive case. -/
theorem D_sel_pos_iff (hx : x.x1 + x.x2 + x.x3 = 1) (hb : 0 < w.bar x) (hx2 : 0 < x.x2)
    (h0 : x.D = 0) : 0 < (sel w x).D ↔ 0 < w.kappa := by
  have hc : 0 < (x.x2 / 2) ^ 2 / (w.bar x) ^ 2 := by positivity
  rw [D_sel_of_D_eq_zero hx hb.ne' h0]
  constructor <;> intro h <;> nlinarith

/-- **P5(a)**, zero case. -/
theorem D_sel_eq_zero_iff (hx : x.x1 + x.x2 + x.x3 = 1) (hb : 0 < w.bar x) (hx2 : 0 < x.x2)
    (h0 : x.D = 0) : (sel w x).D = 0 ↔ w.kappa = 0 := by
  have hc : 0 < (x.x2 / 2) ^ 2 / (w.bar x) ^ 2 := by positivity
  rw [D_sel_of_D_eq_zero hx hb.ne' h0]
  constructor
  · intro h; rcases mul_eq_zero.1 h with h | h
    · exact h
    · exact absurd h hc.ne'
  · intro h; rw [h, zero_mul]

/-- **P5(a)**, negative case. -/
theorem D_sel_neg_iff (hx : x.x1 + x.x2 + x.x3 = 1) (hb : 0 < w.bar x) (hx2 : 0 < x.x2)
    (h0 : x.D = 0) : (sel w x).D < 0 ↔ w.kappa < 0 := by
  have hc : 0 < (x.x2 / 2) ^ 2 / (w.bar x) ^ 2 := by positivity
  rw [D_sel_of_D_eq_zero hx hb.ne' h0]
  constructor <;> intro h <;> nlinarith

/-- **P5(b)**, overdominance makes `κ` negative. -/
theorem kappa_neg_of_overdominance (hw : w.Pos) (h12 : w.w1 < w.w2) (h32 : w.w3 < w.w2) :
    w.kappa < 0 := by
  have := hw.p1; have := hw.p3
  simp only [Fit.kappa]; nlinarith

/-- **P5(c)**, underdominance makes `κ` positive. -/
theorem kappa_pos_of_underdominance (hw : w.Pos) (h21 : w.w2 < w.w1) (h23 : w.w2 < w.w3) :
    0 < w.kappa := by
  have := hw.p2
  simp only [Fit.kappa]; nlinarith

/-- **P5(d)**, multiplicative fitness makes `κ` vanish. -/
theorem kappa_eq_zero_of_multiplicative (h : w.w2 ^ 2 = w.w1 * w.w3) : w.kappa = 0 := by
  simp only [Fit.kappa, h]; ring

/-- **P5(e)**, sign propagation. -/
theorem D_sel_nonneg (hw : w.Pos) (hx : InSimplex x) (hD : 0 ≤ x.D) (hk : 0 ≤ w.kappa) :
    0 ≤ (sel w x).D := by
  have hb := bar_pos hw hx
  rw [D_sel_kappa hx.total hb.ne']
  have h13 : 0 < w.w1 * w.w3 := mul_pos hw.p1 hw.p3
  have : 0 ≤ w.w1 * w.w3 * x.D + w.kappa * (x.x2 / 2) ^ 2 := by positivity
  positivity

/-- **P5(e)**, mirrored. -/
theorem D_sel_nonpos (hw : w.Pos) (hx : InSimplex x) (hD : x.D ≤ 0) (hk : w.kappa ≤ 0) :
    (sel w x).D ≤ 0 := by
  have hb := bar_pos hw hx
  rw [D_sel_kappa hx.total hb.ne']
  have h13 : 0 < w.w1 * w.w3 := mul_pos hw.p1 hw.p3
  have hnum : w.w1 * w.w3 * x.D + w.kappa * (x.x2 / 2) ^ 2 ≤ 0 := by nlinarith [sq_nonneg (x.x2 / 2)]
  exact div_nonpos_of_nonpos_of_nonneg hnum (sq_nonneg _)

end Sign

/-- Hardy–Weinberg zygotes are a genotype vector at any interior `p`. -/
theorem inSimplex_vec_zero {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) : InSimplex (vec p 0) := by
  rw [inSimplex_vec_iff hp0 hp1]
  constructor
  · simp only [neg_nonpos]; positivity
  · nlinarith

/-- **P5(b)**, corollary: overdominance creates heterozygote excess out of HW zygotes. -/
theorem D_sel_vec_zero_neg {w : Fit} {p : ℝ} (hw : w.Pos) (hp0 : 0 < p) (hp1 : p < 1)
    (hk : w.kappa < 0) : (sel w (vec p 0)).D < 0 := by
  have hx := inSimplex_vec_zero hp0.le hp1.le
  exact (D_sel_neg_iff hx.total (bar_pos hw hx) (by simp only [vec_x2]; nlinarith)
    (D_vec p 0)).2 hk

/-- **P5(c)**, corollary: underdominance creates homozygote excess out of HW zygotes. -/
theorem D_sel_vec_zero_pos {w : Fit} {p : ℝ} (hw : w.Pos) (hp0 : 0 < p) (hp1 : p < 1)
    (hk : 0 < w.kappa) : 0 < (sel w (vec p 0)).D := by
  have hx := inSimplex_vec_zero hp0.le hp1.le
  exact (D_sel_pos_iff hx.total (bar_pos hw hx) (by simp only [vec_x2]; nlinarith)
    (D_vec p 0)).2 hk

/-- **P5(d)**, corollary: multiplicative selection on HW zygotes leaves `D = 0`. -/
theorem D_sel_vec_zero_eq_zero {w : Fit} {p : ℝ} (hw : w.Pos) (hp0 : 0 < p) (hp1 : p < 1)
    (h : w.w2 ^ 2 = w.w1 * w.w3) : (sel w (vec p 0)).D = 0 := by
  have hx := inSimplex_vec_zero hp0.le hp1.le
  exact (D_sel_eq_zero_iff hx.total (bar_pos hw hx) (by simp only [vec_x2]; nlinarith)
    (D_vec p 0)).2 (kappa_eq_zero_of_multiplicative h)

end Bastion
