/-
# Life-cycle consistency (P8)

Ground truth: `model.py`, the docstring life cycle
`gamete pool p → zygotes (inbreeding F) → viability selection → adults`, realised both by
`gamete_mean`/`adults_at` and, operator by operator, by `_apply` in the order
`M, Y0, I, Dr, S`.

P8 says the two agree: composing the five operators in life-cycle order reproduces
`sel w (zyg p' F)` with `p' = m + δ`, `m` the post-mutation gamete frequency.
-/
import BastionTheory.Operators

namespace Bastion

open Geno

/-- `Y0 ∘ M` sends an adult vector to Hardy–Weinberg zygotes at the post-mutation gamete
frequency.  Python: `_apply` branches `op = 0` then `op = 1`. -/
theorem reset_mutate (u v : ℝ) {a : Geno} (ha : a.x1 + a.x2 + a.x3 = 1) :
    reset (mutate u v a) = vec (mutAllele u v a.p) 0 := by
  rw [reset, p_mutate ha]

/-- `I ∘ Y0 ∘ M` sends an adult vector to the inbred zygotes at the post-mutation gamete
frequency.  Python: `_apply` branches `op = 0, 1, 2`. -/
theorem inbreed_reset_mutate (u v F : ℝ) {a : Geno} (ha : a.x1 + a.x2 + a.x3 = 1) :
    inbreed F (reset (mutate u v a)) = zyg (mutAllele u v a.p) F := by
  rw [reset_mutate u v ha, inbreed, p_vec, D_vec, zyg]
  congr 1
  ring

/-- **P8**, pre-selection half: `Dr ∘ I ∘ Y0 ∘ M` sends an adult vector to the zygote
vector at the new allele frequency `m + δ` with the same inbreeding coefficient `F`. -/
theorem drift_inbreed_reset_mutate (u v F δ : ℝ) {a : Geno} (ha : a.x1 + a.x2 + a.x3 = 1)
    (hm0 : 0 < mutAllele u v a.p) (hm1 : mutAllele u v a.p < 1) :
    drift δ (inbreed F (reset (mutate u v a))) = zyg (mutAllele u v a.p + δ) F := by
  have hmne : mutAllele u v a.p ≠ 0 := ne_of_gt hm0
  have hqne : (1 : ℝ) - mutAllele u v a.p ≠ 0 := ne_of_gt (by linarith)
  rw [inbreed_reset_mutate u v F ha, drift, p_zyg, D_zyg, zyg]
  congr 1
  field_simp

/-- **P8.**  The five operators applied in life-cycle order `M, Y0, I, Dr, S` reproduce the
generation map of the model: selection acting on the zygotes of the new generation. -/
theorem lifecycle (w : Fit) (u v F δ : ℝ) {a : Geno} (ha : a.x1 + a.x2 + a.x3 = 1)
    (hm0 : 0 < mutAllele u v a.p) (hm1 : mutAllele u v a.p < 1) :
    sel w (drift δ (inbreed F (reset (mutate u v a)))) =
      sel w (zyg (mutAllele u v a.p + δ) F) := by
  rw [drift_inbreed_reset_mutate u v F δ ha hm0 hm1]

/-- **P8**, admissibility of the output: when the new allele frequency is interior and `F`
is admissible there, the generation output is a genotype vector. -/
theorem inSimplex_lifecycle {w : Fit} {u v F δ : ℝ} {a : Geno} (hw : w.Pos)
    (ha : a.x1 + a.x2 + a.x3 = 1) (hm0 : 0 < mutAllele u v a.p) (hm1 : mutAllele u v a.p < 1)
    (hp0 : 0 < mutAllele u v a.p + δ) (hp1 : mutAllele u v a.p + δ < 1)
    (hF : -(min (mutAllele u v a.p + δ) (1 - (mutAllele u v a.p + δ)))
            / max (mutAllele u v a.p + δ) (1 - (mutAllele u v a.p + δ)) ≤ F)
    (hF1 : F ≤ 1) :
    InSimplex (sel w (drift δ (inbreed F (reset (mutate u v a))))) := by
  rw [lifecycle w u v F δ ha hm0 hm1]
  exact inSimplex_sel hw ((inSimplex_zyg_iff hp0 hp1).2 ⟨hF, hF1⟩)

/-- **P8**, agreement with the Python at *every* step.

`model.py` realises the `I` and `Dr` branches of `_apply` through `_vec_with`, which
*clips* the Hardy–Weinberg deviation to the admissible interval.  By P2
(`inSimplex_vec_iff`) a vector `vec p D` lies in the simplex exactly when that clip is
inactive, so proving that every intermediate vector of the life cycle is in the simplex is
exactly the statement that `model.py` computes the unclipped Lean maps at every step.

The hypotheses below are what that requires: `a` a genotype vector, mutation rates in
`[0,1]`, positive fitnesses, the post-mutation frequency `m` and the post-drift frequency
`p' = m + δ` both interior, and `F` admissible at **both** `m` (for the `I` step) and `p'`
(for the `Dr` step).  If `F` is inadmissible at `m`, the Python clips at the `I` step and
the subsequent `Dr` step propagates the clipped `F_g`, so its attribution can differ from
the Lean statement. -/
theorem lifecycle_python_agreement {w : Fit} {u v F δ : ℝ} {a : Geno}
    (hw : w.Pos) (ha : InSimplex a)
    (hu0 : 0 ≤ u) (hu1 : u ≤ 1) (hv0 : 0 ≤ v) (hv1 : v ≤ 1)
    (hm0 : 0 < mutAllele u v a.p) (hm1 : mutAllele u v a.p < 1)
    (hp0 : 0 < mutAllele u v a.p + δ) (hp1 : mutAllele u v a.p + δ < 1)
    (hFm : -(min (mutAllele u v a.p) (1 - mutAllele u v a.p))
             / max (mutAllele u v a.p) (1 - mutAllele u v a.p) ≤ F)
    (hFp : -(min (mutAllele u v a.p + δ) (1 - (mutAllele u v a.p + δ)))
             / max (mutAllele u v a.p + δ) (1 - (mutAllele u v a.p + δ)) ≤ F)
    (hF1 : F ≤ 1) :
    InSimplex (mutate u v a)
      ∧ reset (mutate u v a) = vec (mutAllele u v a.p) 0
      ∧ InSimplex (vec (mutAllele u v a.p) 0)
      ∧ inbreed F (reset (mutate u v a)) = zyg (mutAllele u v a.p) F
      ∧ InSimplex (zyg (mutAllele u v a.p) F)
      ∧ drift δ (inbreed F (reset (mutate u v a))) = zyg (mutAllele u v a.p + δ) F
      ∧ InSimplex (zyg (mutAllele u v a.p + δ) F)
      ∧ InSimplex (sel w (zyg (mutAllele u v a.p + δ) F)) := by
  have hzm : InSimplex (zyg (mutAllele u v a.p) F) :=
    (inSimplex_zyg_iff hm0 hm1).2 ⟨hFm, hF1⟩
  have hzp : InSimplex (zyg (mutAllele u v a.p + δ) F) :=
    (inSimplex_zyg_iff hp0 hp1).2 ⟨hFp, hF1⟩
  exact ⟨inSimplex_mutate hu0 hu1 hv0 hv1 ha,
    reset_mutate u v ha.total,
    inSimplex_vec_zero hm0.le hm1.le,
    inbreed_reset_mutate u v F ha.total,
    hzm,
    drift_inbreed_reset_mutate u v F δ ha.total hm0 hm1,
    hzp,
    inSimplex_sel hw hzp⟩

end Bastion
