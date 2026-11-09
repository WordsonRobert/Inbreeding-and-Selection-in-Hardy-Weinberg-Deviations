/-
# Germline mutation (P6)

Ground truth: `model.py`, `_apply` with `op == 0` (and `mutate_allele` for the allele-level
map).  Each allele of each parent mutates independently, `A → a` at rate `u`, `a → A` at
rate `v`.
-/
import BastionTheory.Basic

namespace Bastion

open Geno

/-- Genotype-level mutation.  Python: `_apply`, branch `op == 0`. -/
noncomputable def mutate (u v : ℝ) (x : Geno) : Geno :=
  ⟨x.x1 * (1 - u) ^ 2 + x.x2 * ((1 - u) * v) + x.x3 * v ^ 2,
   1 - (x.x1 * (1 - u) ^ 2 + x.x2 * ((1 - u) * v) + x.x3 * v ^ 2)
     - (x.x1 * u ^ 2 + x.x2 * (u * (1 - v)) + x.x3 * (1 - v) ^ 2),
   x.x1 * u ^ 2 + x.x2 * (u * (1 - v)) + x.x3 * (1 - v) ^ 2⟩

/-- Allele-level mutation.  Python: `mutate_allele`. -/
noncomputable def mutAllele (u v : ℝ) (p : ℝ) : ℝ := p * (1 - u) + (1 - p) * v

@[simp] theorem mutate_x1 (u v : ℝ) (x : Geno) :
    (mutate u v x).x1 = x.x1 * (1 - u) ^ 2 + x.x2 * ((1 - u) * v) + x.x3 * v ^ 2 := rfl
@[simp] theorem mutate_x3 (u v : ℝ) (x : Geno) :
    (mutate u v x).x3 = x.x1 * u ^ 2 + x.x2 * (u * (1 - v)) + x.x3 * (1 - v) ^ 2 := rfl
@[simp] theorem mutate_x2 (u v : ℝ) (x : Geno) :
    (mutate u v x).x2 = 1 - (mutate u v x).x1 - (mutate u v x).x3 := rfl

/-- Mutation preserves unit total (by construction). -/
@[simp] theorem total_mutate (u v : ℝ) (x : Geno) :
    (mutate u v x).x1 + (mutate u v x).x2 + (mutate u v x).x3 = 1 := by
  simp only [mutate_x1, mutate_x2, mutate_x3]; ring

/-- On the simplex the heterozygote output has the manifestly nonnegative form
`2u(1-u)·x1 + ((1-u)(1-v) + uv)·x2 + 2v(1-v)·x3`. -/
theorem mutate_x2_eq {u v : ℝ} {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1) :
    (mutate u v x).x2 =
      x.x1 * (2 * u * (1 - u)) + x.x2 * ((1 - u) * (1 - v) + u * v) + x.x3 * (2 * v * (1 - v)) := by
  have h3 : x.x3 = 1 - x.x1 - x.x2 := by linarith
  simp only [mutate_x2, mutate_x1, mutate_x3, h3]; ring

/-- **P6**, first part: mutation maps the simplex to itself. -/
theorem inSimplex_mutate {u v : ℝ} {x : Geno} (hu0 : 0 ≤ u) (hu1 : u ≤ 1) (hv0 : 0 ≤ v)
    (hv1 : v ≤ 1) (hx : InSimplex x) : InSimplex (mutate u v x) := by
  have h1 := hx.nn1; have h2 := hx.nn2; have h3 := hx.nn3
  refine ⟨?_, ?_, ?_, total_mutate u v x⟩
  · simp only [mutate_x1]
    have a1 : 0 ≤ x.x1 * (1 - u) ^ 2 := mul_nonneg h1 (sq_nonneg _)
    have a2 : 0 ≤ x.x2 * ((1 - u) * v) := mul_nonneg h2 (mul_nonneg (by linarith) hv0)
    have a3 : 0 ≤ x.x3 * v ^ 2 := mul_nonneg h3 (sq_nonneg _)
    linarith
  · rw [mutate_x2_eq hx.total]
    have a1 : 0 ≤ x.x1 * (2 * u * (1 - u)) := mul_nonneg h1 (by nlinarith)
    have a2 : 0 ≤ x.x2 * ((1 - u) * (1 - v) + u * v) := mul_nonneg h2 (by nlinarith)
    have a3 : 0 ≤ x.x3 * (2 * v * (1 - v)) := mul_nonneg h3 (by nlinarith)
    linarith
  · simp only [mutate_x3]
    have a1 : 0 ≤ x.x1 * u ^ 2 := mul_nonneg h1 (sq_nonneg _)
    have a2 : 0 ≤ x.x2 * (u * (1 - v)) := mul_nonneg h2 (mul_nonneg hu0 (by linarith))
    have a3 : 0 ≤ x.x3 * (1 - v) ^ 2 := mul_nonneg h3 (sq_nonneg _)
    linarith

/-- **P6**, second part: the genotype-level mutation operator projects onto the
allele-level map.  This certifies that `_apply(op=0)` is consistent with `mutate_allele`. -/
theorem p_mutate {u v : ℝ} {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1) :
    (mutate u v x).p = mutAllele u v x.p := by
  have h3 : x.x3 = 1 - x.x1 - x.x2 := by linarith
  simp only [Geno.p, mutate_x1, mutate_x2, mutate_x3, mutAllele, h3]; ring

/-- **P6**, third part: mutation contracts the HW deviation by exactly `(1 - u - v)^2`. -/
theorem D_mutate {u v : ℝ} {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1) :
    (mutate u v x).D = (1 - u - v) ^ 2 * x.D := by
  have h3 : x.x3 = 1 - x.x1 - x.x2 := by linarith
  simp only [Geno.D, Geno.p, mutate_x1, mutate_x2, mutate_x3, h3]; ring

/-- **P7**, null operator: zero mutation rates act as the identity on the simplex. -/
theorem mutate_zero {x : Geno} (hx : x.x1 + x.x2 + x.x3 = 1) : mutate 0 0 x = x := by
  ext
  · simp only [mutate_x1]; ring
  · simp only [mutate_x2, mutate_x1, mutate_x3]; ring_nf; linarith
  · simp only [mutate_x3]; ring

end Bastion
