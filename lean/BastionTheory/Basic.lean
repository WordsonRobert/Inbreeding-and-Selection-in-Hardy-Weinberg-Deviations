/-
# Basic genotype algebra (P1, P2)

Ground truth: `bastion/model.py`, functions `hw_dev`, `_vec_with`, `zygotes`.

A genotype vector `x = (x1, x2, x3)` holds the frequencies of `AA`, `Aa`, `aa` at a single
biallelic locus.  `p x` is the frequency of allele `A`, `D x` the Hardy–Weinberg deviation.

Deviation from the Python: `_vec_with` and `zygotes` *clip* the deviation `D` to the
admissible interval `[-min(p,q)^2, p*q]`.  Here `vec` never clips; `P2`
(`inSimplex_vec_iff`) characterises exactly when the unclipped vector is a genotype
vector, i.e. exactly when the Python clip is inactive.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp

namespace Bastion

/-- A genotype frequency vector `(AA, Aa, aa)`. -/
structure Geno where
  x1 : ℝ
  x2 : ℝ
  x3 : ℝ

namespace Geno

@[ext] theorem ext {x y : Geno} (h1 : x.x1 = y.x1) (h2 : x.x2 = y.x2) (h3 : x.x3 = y.x3) :
    x = y := by
  cases x; cases y; simp_all

/-- Frequency of allele `A`.  Python: `p = g1 + 0.5 * g2`, as in `gamete_mean`. -/
noncomputable def p (x : Geno) : ℝ := x.x1 + x.x2 / 2

/-- Frequency of allele `a`, `q = 1 - p`. -/
noncomputable def q (x : Geno) : ℝ := 1 - x.p

/-- Hardy–Weinberg deviation, `D = f_AA - p^2`.  Python: `hw_dev`. -/
noncomputable def D (x : Geno) : ℝ := x.x1 - x.p ^ 2

/-- The genotype simplex: nonnegative frequencies summing to one.  Python: the
admissible range enforced by the clip inside `_vec_with`. -/
structure InSimplex (x : Geno) : Prop where
  nn1 : 0 ≤ x.x1
  nn2 : 0 ≤ x.x2
  nn3 : 0 ≤ x.x3
  total : x.x1 + x.x2 + x.x3 = 1

/-- **P1.**  On a vector with unit total, `D = x1 * x3 - x2^2 / 4`.
This certifies the identity asserted in the docstring of `model.py : hw_dev`. -/
theorem D_eq_of_total {x : Geno} (h : x.x1 + x.x2 + x.x3 = 1) :
    x.D = x.x1 * x.x3 - x.x2 ^ 2 / 4 := by
  have h3 : x.x3 = 1 - x.x1 - x.x2 := by linarith
  simp only [D, p, h3]; ring

/-- **P1** on the simplex. -/
theorem D_eq {x : Geno} (hx : InSimplex x) : x.D = x.x1 * x.x3 - x.x2 ^ 2 / 4 :=
  D_eq_of_total hx.total

end Geno

open Geno

/-- `vec p D` is the genotype vector with allele frequency `p` and HW deviation `D`.
Python: `_vec_with` (without its clip). -/
noncomputable def vec (p D : ℝ) : Geno := ⟨p ^ 2 + D, 2 * (p * (1 - p) - D), (1 - p) ^ 2 + D⟩

/-- Zygote frequencies under mating-system inbreeding `F`.  Python: `zygotes`. -/
noncomputable def zyg (p F : ℝ) : Geno := vec p (F * p * (1 - p))

@[simp] theorem vec_x1 (p D : ℝ) : (vec p D).x1 = p ^ 2 + D := rfl
@[simp] theorem vec_x2 (p D : ℝ) : (vec p D).x2 = 2 * (p * (1 - p) - D) := rfl
@[simp] theorem vec_x3 (p D : ℝ) : (vec p D).x3 = (1 - p) ^ 2 + D := rfl

/-- The entries of `vec p D` always sum to one. -/
@[simp] theorem total_vec (p D : ℝ) : (vec p D).x1 + (vec p D).x2 + (vec p D).x3 = 1 := by
  simp only [vec_x1, vec_x2, vec_x3]; ring

/-- **P2**, first half: `vec` has the allele frequency it was given. -/
@[simp] theorem p_vec (p D : ℝ) : (vec p D).p = p := by
  simp only [Geno.p, vec_x1, vec_x2]; ring

/-- **P2**, first half: `vec` has the HW deviation it was given. -/
@[simp] theorem D_vec (p D : ℝ) : (vec p D).D = D := by
  simp only [Geno.D, p_vec, vec_x1]; ring

@[simp] theorem p_zyg (p F : ℝ) : (zyg p F).p = p := p_vec _ _
@[simp] theorem D_zyg (p F : ℝ) : (zyg p F).D = F * p * (1 - p) := D_vec _ _

/-- A vector with unit total is recovered from its own `p` and `D`.
This is the statement that `(p, D)` are coordinates on the affine plane `x1+x2+x3 = 1`. -/
theorem vec_p_D_self {x : Geno} (h : x.x1 + x.x2 + x.x3 = 1) : vec x.p x.D = x := by
  have h3 : x.x3 = 1 - x.x1 - x.x2 := by linarith
  ext <;> simp only [vec_x1, vec_x2, vec_x3, Geno.D, Geno.p, h3] <;> ring

/-- **P2**, second half: admissibility of a `(p, D)` pair. -/
theorem inSimplex_vec_iff {p D : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    InSimplex (vec p D) ↔ -(min p (1 - p)) ^ 2 ≤ D ∧ D ≤ p * (1 - p) := by
  have hq0 : 0 ≤ 1 - p := by linarith
  have hm0 : 0 ≤ min p (1 - p) := le_min hp0 hq0
  have hmp : min p (1 - p) ≤ p := min_le_left _ _
  have hmq : min p (1 - p) ≤ 1 - p := min_le_right _ _
  constructor
  · rintro ⟨h1, h2, h3, -⟩
    simp only [vec_x1, vec_x2, vec_x3] at h1 h2 h3
    refine ⟨?_, by linarith⟩
    rcases le_total p (1 - p) with hle | hle
    · rw [min_eq_left hle]; linarith
    · rw [min_eq_right hle]; linarith
  · rintro ⟨hlo, hhi⟩
    refine ⟨?_, ?_, ?_, total_vec p D⟩ <;> simp only [vec_x1, vec_x2, vec_x3]
    · nlinarith
    · linarith
    · nlinarith

/-- **P2**, third part: admissibility of an inbreeding coefficient at an interior `p`.
`zyg p F` is a genotype vector iff `-min(p,q)/max(p,q) ≤ F ≤ 1`. -/
theorem inSimplex_zyg_iff {p F : ℝ} (hp0 : 0 < p) (hp1 : p < 1) :
    InSimplex (zyg p F) ↔ -(min p (1 - p)) / max p (1 - p) ≤ F ∧ F ≤ 1 := by
  have hq0 : (0:ℝ) < 1 - p := by linarith
  have hpq : (0:ℝ) < p * (1 - p) := mul_pos hp0 hq0
  rw [zyg, inSimplex_vec_iff hp0.le hp1.le]
  rcases le_total p (1 - p) with hle | hle
  · rw [min_eq_left hle, max_eq_right hle, div_le_iff₀ hq0]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨by nlinarith, by nlinarith⟩
    · rintro ⟨h1, h2⟩
      exact ⟨by nlinarith, by nlinarith⟩
  · rw [min_eq_right hle, max_eq_left hle, div_le_iff₀ hp0]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨by nlinarith, by nlinarith⟩
    · rintro ⟨h1, h2⟩
      exact ⟨by nlinarith, by nlinarith⟩

end Bastion
