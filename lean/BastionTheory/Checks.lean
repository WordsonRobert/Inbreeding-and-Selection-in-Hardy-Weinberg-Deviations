/-
# Axiom audit

Every main theorem named in `README.md` and `Statements.md` is listed here with
`#print axioms` — 81 of them, and no others.  A clean build must show
`[propext, Classical.choice, Quot.sound]` for each one; in particular no `sorryAx`.

`verify.sh` builds the project and checks this mechanically.

No `set_option` is used anywhere in this development: every proof elaborates inside Lean's
default limits.
-/
import BastionTheory.Basic
import BastionTheory.Selection
import BastionTheory.Mutation
import BastionTheory.Operators
import BastionTheory.LifeCycle
import BastionTheory.Shapley
import BastionTheory.Attribution
import BastionTheory.RareAllele
import BastionTheory.Identifiability
import BastionTheory.Drift

namespace Bastion

-- P1  — the two forms of the HW deviation
#print axioms Geno.D_eq_of_total
#print axioms Geno.D_eq

-- P2  — coordinates and admissibility
#print axioms p_vec
#print axioms D_vec
#print axioms vec_p_D_self
#print axioms inSimplex_vec_iff
#print axioms inSimplex_zyg_iff

-- P3  — closed form
#print axioms D_sel_closed

-- P4  — the kappa decomposition
#print axioms D_sel_kappa
#print axioms kappa_bastion

-- P5  — sign rules
#print axioms D_sel_of_D_eq_zero
#print axioms D_sel_pos_iff
#print axioms D_sel_eq_zero_iff
#print axioms D_sel_neg_iff
#print axioms x2_eq
#print axioms kappa_neg_of_overdominance
#print axioms D_sel_vec_zero_neg
#print axioms kappa_pos_of_underdominance
#print axioms D_sel_vec_zero_pos
#print axioms kappa_eq_zero_of_multiplicative
#print axioms D_sel_vec_zero_eq_zero
#print axioms D_sel_nonneg
#print axioms D_sel_nonpos

-- P6  — mutation
#print axioms inSimplex_mutate
#print axioms p_mutate
#print axioms D_mutate

-- P7  — the operators
#print axioms p_reset
#print axioms D_reset
#print axioms p_inbreed
#print axioms D_inbreed
#print axioms p_drift
#print axioms D_drift
#print axioms mutate_zero
#print axioms inbreed_zero
#print axioms sel_one
#print axioms sel_bastion_zero
#print axioms drift_zero

-- P8  — life-cycle consistency
#print axioms reset_mutate
#print axioms inbreed_reset_mutate
#print axioms drift_inbreed_reset_mutate
#print axioms lifecycle
#print axioms inSimplex_lifecycle
#print axioms lifecycle_python_agreement

-- P9  — Shapley attribution (generic)
#print axioms ShapleyGame.sum_shapley
#print axioms ShapleyGame.shapley_dummy
#print axioms ShapleyGame.shapley_symm
#print axioms ShapleyGame.sum_seq_telescope

-- P9  — Shapley attribution (applied to BASTION)
#print axioms sum_shapley_game
#print axioms sum_shapley_game_lifecycle
#print axioms shapley_selection_eq_zero
#print axioms shapley_mutation_eq_zero
#print axioms shapley_inbreeding_eq_zero
#print axioms shapley_drift_eq_zero

-- P10 — the two-player level split
#print axioms Dz_add_DS
#print axioms Dz_eq_zero
#print axioms DS_eq_zero
#print axioms DS_eq_zero_bastion

-- P11 — rare-allele selection
#print axioms qAdult_sub
#print axioms hasDerivAt_qAdult
#print axioms sEff_determines_linearisation
#print axioms sEff_zero
#print axioms tendsto_rareAdult_x2_div
#print axioms tendsto_rareAdult_x3_div
#print axioms tendsto_c2
#print axioms tendsto_c3
#print axioms first_order_hs_only
#print axioms first_order_not_identified
#print axioms first_order_injective
#print axioms first_order_injective_h
#print axioms first_order_limits_injective

-- P12 — cross-sectional identifiability
#print axioms rank_le_of_col_sum_zero
#print axioms rank_quadratic_le
#print axioms rank_fisher_genotype_le_two
#print axioms rank_fisher_phenotype_le_one
#print axioms sum_deriv_eq_zero_of_probability_family

-- P13 — drift versus the within-population F_IS
#print axioms Drift.wright_closed_form
#print axioms Drift.binom_moment1
#print axioms Drift.binom_moment2
#print axioms Drift.binom_var_identity
#print axioms Drift.expected_het_fraction
#print axioms Drift.fis_ratio_full

end Bastion
