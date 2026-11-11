# Inbreeding and selection in Hardy–Weinberg deviations

This repository holds the code, data and outputs for the manuscript

> **Inbreeding and selection in Hardy–Weinberg deviations: exact attribution, identifiability
> limits and calibrated inference from genotype time series**
> Wordson Robert, Gautam Menon, Amogh Pete Shenoy, Sahil Singh (IISER Kolkata)

The inference method is called **BASTION**. It does the following for a single biallelic
diploid locus:

* infers the mating-system inbreeding coefficient `F`, the selection coefficient `s`, the
  dominance `h` and (optionally) `N_e` jointly from genotype or phenotype counts sampled over
  generations;
* attributes each generation's change in the Hardy–Weinberg deviation to five operators:
  mutation, Mendelian reset, inbreeding, drift and selection.

The repository contains everything needed to rerun every simulation, fit, table and figure
from scratch, plus the outputs of the published run, so that any number can be checked
without rerunning anything. The algebraic results are machine-checked in Lean 4 (`lean/`).

---

## Contents

```
.
├── bastion/                  model, likelihood, sampler, simulators (the method)
│   ├── model.py              genotype-level life cycle, grid HMM likelihood, FFBS, Shapley attribution
│   ├── mcmc.py               adaptive Metropolis sampler, posterior trajectories, closed-model checks
│   ├── sims.py               data generators (model-exact and individual-based populations)
│   └── legacy_v2.py          earlier deterministic-trajectory model (used only for comparison, E10)
├── experiments/              one script per experiment (called by run_all.py)
│   ├── e01_theory.py         E01 drift vs F_IS, identity checks; E02 Fisher information; E06 operator order
│   ├── e03_calibration.py    E03 calibration, E04 robustness, E05 stress tests, E05b one-at-a-time sweeps
│   ├── e07_migration.py      E07 immigration pulses and closed-model checks
│   ├── e08_reconstruction.py E08 trajectory reconstruction at unsampled generations
│   ├── e09_empirical.py      E09 Panaxia, Biston, single samples, bootstrap closure tests
│   ├── e10_legacy.py         E10 deterministic-trajectory model vs BASTION on the same data
│   ├── e11_legacy_ode.py     E11 neural-ODE interpolator vs BASTION (optional, GPU)
│   ├── e12_allele_hmm_comparison.py  E12 allele-frequency HMM estimator of s vs BASTION (run on its own)
│   ├── worker.py             one fit per call (used by the process pool)
│   ├── common.py             replicate counts, MCMC settings, helpers
│   ├── summarize.py          writes results/SUMMARY.md from the summary JSON files
│   └── fix_cross_sectional.py  re-runs only the single-sample part of E09 on an existing results folder
├── legacy/                   original neural-ODE and simulator code (used only by E11)
├── data/                     input data, with sources and caveats in data/README.md
├── results/                  outputs of the published run (one folder per experiment)
├── figures/                  Figures 1–7 of the manuscript and the scripts that draw them
├── examples/                 fit_one_dataset.py: a minimal fit of one time series
├── lean/                     Lean 4 + Mathlib formalisation (81 theorems)
├── run_all.py                runs every experiment, writes results/ and results/SUMMARY.md
├── run.sh                    one-command reproduction of results/
├── slurm_job.sh              the same for a SLURM cluster
├── requirements.txt
└── CITATION.cff
```

---

## Quick start

```bash
git clone https://github.com/WordsonRobert/Inbreeding-and-Selection-in-Hardy-Weinberg-Deviations.git
cd Inbreeding-and-Selection-in-Hardy-Weinberg-Deviations
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, numba, pandas, matplotlib

# 1. smoke test: every experiment with very few replicates (~6-10 min on 2 cores)
python run_all.py --mode quick --no-ode --out results_quick

# 2. fit one data set (about a minute)
python examples/fit_one_dataset.py

# 3. reproduce results/ exactly (about 3.5 h on 16 threads; see "Reproducing the published run")
bash run.sh

# 4. redraw the figures of the manuscript from results/
python figures/make_figs.py

# 5. check the proofs (needs elan; see lean/README.md)
cd lean && lake exe cache get && ./verify.sh
```

Requirements:

* Python ≥ 3.10 (the published run used Python 3.13.11 with numba 0.68 on Linux).
* PyTorch is needed only for E11. Without it, E11 is skipped and everything else runs.

---

## The model in brief

One generation of a single biallelic locus with alleles `A` and `a`. Genotype vectors are
ordered `(AA, Aa, aa)`.

1. **Zygotes.** Zygotes form from the gamete pool at frequency `p_t` with mating-system
   inbreeding coefficient `F`:
   `z_t = (p² + Fpq, 2pq(1 − F), q² + Fpq)`.
2. **Selection.** Viability selection with fitnesses `w = (1, 1 − hs, 1 − s)` gives the adults
   `a_t = (w₁z₁, w₂z₂, w₃z₃) / w̄`. Samples are taken from the adults.
3. **Mutation.** Mutation `A → a` at rate `u` and `a → A` at rate `v` (both fixed at `1e-5`)
   gives the gamete frequency `m_t`.
4. **Drift.** Wright–Fisher sampling: `2N_e p_{t+1} ~ Binomial(2N_e, m_t)`.

**Observations.** Genotype counts are multinomial in `a_t`. When only a dominant phenotype is
scored, the counts are binomial in `a_t,3`.

**Parameters and priors** (`bastion/mcmc.py`, `DEFAULTS`):

| parameter | prior |
|---|---|
| `F` | Uniform(−0.2, 0.8) |
| `s` | Normal(0, 0.15²) on (−0.9, 0.9) |
| `h` | Normal(0.5, 0.5²) |
| fitness constraint | `1 − hs > 0.01`, `1 − s > 0.01` |
| `log10 N_e` (when inferred) | Uniform(1.5, 4.5) |
| initial `p` | Beta(2, 2) |

Reported derived quantities: `hs` and `s_eff = s[F + (1 − F)h]`.

**Likelihood** (`model.hmm_forward`):

* The hidden gamete frequency lives on a grid uniform on the arcsine scale, which includes
  the states 0 and 1 (`make_grid`).
* Grid size `K = min(400, max(120, ⌈2.6 √N_e⌉))` (`choose_K`). When `N_e` is inferred, `K` is
  taken from the levels {120, 160, 220, 300, 400}.
* The transitions are Gaussian approximations of the Wright–Fisher binomial, projected onto
  the piecewise-linear basis of the grid with a variance correction (`build_transition`).
* Posterior trajectories are drawn by forward filtering–backward sampling (`ffbs`).

**Sampler** (`mcmc.run_mcmc`):

* Adaptive random-walk Metropolis, with the proposal covariance re-estimated every 50
  iterations during warm-up.
* 4 dispersed chains of 4000 iterations, 1500 of them warm-up. When `N_e` is inferred,
  6000 iterations, 1995 of them warm-up. The empirical fits use 8 × 20 000 iterations,
  8000 of them warm-up.
* After warm-up, a chain whose mean log posterior is more than 5 units below the best chain's
  is restarted from the best chain.
* A run counts as converged when split-R̂ < 1.05 for every sampled parameter.

**Attribution** (`model.attribute_path`, `mcmc.attribution`):

* Along each posterior trajectory, the change `D(a_{t+1}) − D(a_t)` is split into the Shapley
  values of the five operators (mutation `M`, Mendelian reset `Y0`, inbreeding `I`, drift
  `Dr`, selection `S`), where `D = x₁ − p²`. Each coalition applies its operators in
  life-cycle order.
* The level of `D` in adults is split into an inbreeding part `D^z` and a selection part `D^S`.

**Closed-model checks** (`mcmc.discrepancy`):

* `score_pred`: the largest one-step-ahead predictive residual.
* `score_D`: the largest posterior-predictive z-score of the observed deviation.
* `score_comb`: their combination.
* For data sets, a parametric bootstrap (E09) gives the p-value.

---

## Experiments

`run_all.py` runs the stages below in order. Every replicate has its own seed, drawn from a
fixed master seed per stage. With the same `--scale`, the CPU stages therefore simulate the
same data sets and use the same sampler seeds on any machine and with any number of workers.

* **Replicates** are counts in the published run (`--scale 2.5`; `common.nrep` multiplies
  each base count by the scale).
* **Time** is wall time in the published run: 15 worker processes on a 16-thread laptop
  (`results/run_meta.json`).

| stage | what is done | design (see the script for every setting) | replicates | outputs in `results/` | time |
|---|---|---|---|---|---|
| E01 | drift vs within-population F_IS; numerical checks of the identities; adult F_IS from selection alone | random mating, N = 50, 60 generations; random parameter draws | 10 000 populations; 20 000 identity checks | `e01_theory/` | 16 s (E01, E02 and E06 together) |
| E02 | expected Fisher information and Cramér–Rao bounds of (p0, F, s, h) | 6 designs × 3 data types (adult genotypes, zygote genotypes, dominant phenotypes) × p0 ∈ {0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 0.95} × F ∈ {0, 0.1, 0.3} × s ∈ {−0.2, −0.05, 0.01, 0.05, 0.1, 0.2} × h ∈ {0, 0.25, 0.5, 0.75, 1} | 12 960 cases | `e01_theory/identifiability_fim.csv` | – |
| E06 | Shapley attribution vs sequential attribution under all 120 operator orders | model-exact, T = 30; p0 ~ U(0.05, 0.95), F ~ U(0, 0.4), s ~ ±U(0.01, 0.3), h ~ U(−0.2, 1.2), N_e ∈ {10², 10³, 10⁴} | 750 trajectories (22 500 generations) | `e01_theory/order_and_legacy_shares.csv` | – |
| E03 | simulation-based calibration | model-exact; p0, F, s, h from the priors; T ∈ {20, 25, …, 60}; n = 300 every 5 generations; log10 N_e ~ U(2, 4) given, or ~ U(1.5, 4.5) inferred | 2500 (N_e given) + 1500 (N_e inferred) | `e03_sbc/` | 35 min |
| E04 | individual-based populations | N ∈ {100, 500, 2000} × selfing σ ∈ {0, 0.2, 0.5}, T = 40; N_e given as N, or inferred, N/2 or 2N at N = 500 | 14 cells × 300 | `e04_robustness/` | 32 min |
| E05 | one violation at a time | baseline, changing s, bottleneck (N_e fixed / inferred), linked selection, Wahlund sample, genotyping error, zygote sampling, selfing starting at t = 0, individual-based N = 1000; T = 50, N_e = 1000 | 10 × 500 | `e05_violations/` | 29 min |
| E05b | one value changed at a time around a base case | base p0 = 0.4, F = 0.1, s = 0.1, h = 0.5, N_e = 1000, T = 30, n = 300; sweeps of p0, F, n, T, s and N_e (6 values each) | 36 × 100 | `e05_failure_map/` | 15 min |
| E07 | a single immigration pulse, and the same statistics on other violations | m ∈ {0, 0.02, 0.05, 0.1, 0.2} of the zygotes of one generation in [T/4, 3T/4), migrant frequency ~ U(0.05, 0.95), T ∈ {30, …, 60} | 5 × 1000 (N_e given); 3 × 375 (N_e inferred); 6 × 500 (other violations) | `e07_migration/` | 51 min |
| E08 | estimates of the adult allele frequency at unsampled generations | T = 99, 7 samples: a fixed irregular design, or random times and parameters | 2 × 500 | `e08_reconstruction/` | 8 min |
| E09 | empirical fits | Panaxia (census N_e; N_e inferred); Biston (h = 0 or free; n = 50 / 100 / 300; N_e inferred or 1000); 15 CCR5 samples, haemoglobin C, *Zoarces*, *Arabidopsis* | 7 long fits; 3 × 500 bootstrap refits | `e09_empirical/` | 33 min |
| E10 | the deterministic-trajectory model (`bastion/legacy_v2.py`) and BASTION fitted to the same populations | individual-based, N drawn from {200, 1000}, T = 40, random mating or σ = 0.3; Panaxia with 8 dispersed chains | 2 × 500 | `e10_legacy/` | 6 min |
| E11 | the original neural-ODE interpolator (`legacy/layer4_ode.py`) vs BASTION | the fixed design of E08 | 6 (GPU; runs in the background) | `e11_legacy_ode/` | 84 min |

Total: 12 500 s (3.5 h) of wall time.

**E12** is not part of `run_all.py`; it is a single script added for the manuscript
(Section 5.4, Table 3) and run on its own:

    python experiments/e12_allele_hmm_comparison.py --reps 100 --workers 2

| stage | what is done | design | replicates | outputs in `results/` | time |
|---|---|---|---|---|---|
| E12 | the standard allele-frequency HMM (random mating, additive selection, allele counts binomial in 2n; maximum likelihood with a 90% profile-likelihood interval, on BASTION's transition kernel) vs BASTION on the same data | model-exact, random mating, T = 40, n = 300 every 5 generations, N_e = 1000 known; p0 ~ U(0.2, 0.8), s ~ ±U(0.02, 0.15); h = 1/2 or h ~ U(0, 1) | 2 × 100 | `e12_allele_hmm_comparison/` | 197 s on 2 cores |

Result: with h = 1/2 the allele-frequency HMM is somewhat more precise (RMSE of s 0.020 vs
0.024; 90% coverage 0.91 vs 0.96); with h ~ U(0, 1) its 90% intervals cover s in 69% of data
sets and hs (as s/2) in 42%, against 96% and 99% for BASTION.

---

## Reproducing the published run

```bash
bash run.sh                                   # = python run_all.py --mode full --scale 2.5 --ode-reps 6 --resume --out results
```

Notes:

* **Existing outputs.** `run.sh` writes into `results/`, which already holds the published
  outputs. Because of `--resume`, it **skips every stage whose summary file already exists**.
  To recompute from scratch, use another folder (`OUT=results_rerun bash run.sh`), or move
  `results/` away first.
* **Smaller run.** `SCALE=1 bash run.sh` runs every experiment with 2.5× fewer replicates
  (about 1.4 h on 16 threads).
* **Workers.** `WORKERS=8 bash run.sh` limits the number of worker processes. The default is
  all cores minus one.
* **Selected stages.** `python run_all.py --mode full --scale 2.5 --only e03,e07 --out results_rerun`
  runs only the listed stages.
* **Machine check.** `python run_all.py --mode full --hours 2 --check` reports CPUs, speed and
  the estimated run time without running anything.
* **Where to run.** The CPU stages need about 50 CPU-hours at scale 2.5. A 16-thread laptop
  finishes in about 3.5 h. On a cluster, do not run on the login node; submit
  `slurm_job.sh` instead.
* **What is and is not exactly reproducible.**
  * The CPU stages are seeded. With the same `--scale` they simulate the same data sets and
    use the same sampler seeds. Fitted values can still differ in the last digits between
    machines and numpy/numba versions, because floating-point results differ slightly and
    Markov chains amplify the differences. Summary numbers agree to Monte Carlo error.
  * E11 trains neural networks on the GPU and is not bit-for-bit reproducible.
  * In the published run, E11 stopped after 6 replicates because of a time budget, so
    `run.sh` passes `--ode-reps 6`.
* **Single-sample results.** The single-sample part of E09 reports the effective inbreeding
  coefficient `F_eff = max(F, F_min(p))`, where `F_min(p) = −min(p,q)/max(p,q)` is the lower
  admissible bound. Below the bound `F` is not identified.
  * In the published run this part was recomputed after the main run with
    `python experiments/fix_cross_sectional.py results`, which takes about a minute.
  * The current `e09_empirical.py` already does the same, so a fresh run needs no extra step.
* **E02 rare/common split.** `results/e01_theory/theory_summary.json` summarises the Fisher
  information separately for rare and common alleles. The rare class is defined by
  `min(p0, 1 − p0) ≤ 0.05`.
  * In the published run this comparison was evaluated in floating point, where
    `1 − 0.95 > 0.05`, so p0 = 0.95 was put in the common class.
  * `e01_theory.py` now compares with a tolerance (`≤ 0.05 + 1e-9`).
  * The summary JSON and `SUMMARY.md` here were recomputed from the unchanged raw table
    `identifiability_fim.csv` with the corrected split.

---

## Outputs

`results/SUMMARY.md` collects the summary numbers of every stage in tables.

Each stage folder contains:

* a per-replicate CSV (one row per simulated data set or fit);
* a `*_summary.json` file (aggregated numbers, which `SUMMARY.md` is built from);
* diagnostic PNG figures drawn during the run.

Other files in `results/`:

* `run_meta.json`: machine, Python version, replicate scale and per-stage times of the
  published run;
* `run_full.log`: the console log of the published run (the output folder was called
  `results_v3` at the time, so the log names that folder);
* `e11_legacy_ode.log`: the console log of E11.

### Columns of the per-replicate CSVs (E03, E04, E05, E05b, E07)

| column | meaning |
|---|---|
| `id`, `cond`, `cond_val` | replicate number, condition label, value of the varied quantity |
| `T`, `n_obs` | series length (generations) and number of samples |
| `acc` | mean acceptance rate |
| `converged` | split-R̂ < 1.05 for all sampled parameters |
| `converged_F_hs` | split-R̂ < 1.05 for `F` and `hs` |
| `<par>_true` | true value; `<par>` is one of `F`, `s`, `h`, `hs`, `s_eff`, `log10Ne` |
| `<par>_med`, `_mean`, `_sd` | posterior median, mean and standard deviation |
| `<par>_q05`, `_q25`, `_q75`, `_q95` | posterior quantiles |
| `<par>_cov90`, `<par>_cov50` | 1 if the central 90% (50%) interval contains the true value |
| `<par>_rhat` | split-R̂ |
| `<par>_rank` | rank of the true value among 99 thinned posterior draws (E03) |
| `p0_true`, `p0_med`, `p0_cov90` | initial allele frequency |
| `path_cov90` | fraction of generations at which the 90% band of the hidden trajectory `p_t` contains the truth |
| `path_rmse` | root mean squared error of the posterior-mean trajectory |
| `Dinb_*`, `Dsel_*` | the level split `D^z` (inbreeding) and `D^S` (selection) at the last generation: true value, median, 90% coverage |
| `selshare_*` | `D^S / (|D^z| + |D^S|)` at the last generation |
| `cum_<operator>_true`, `_med`, `_cov90` | Shapley value summed over the series, for `mutation`, `mendelian_reset`, `inbreeding`, `drift`, `selection` |
| `score_pred`, `score_D`, `score_comb` | closed-model statistics (E05, E07) |
| `score_jump` | largest standardised drift per interval (E05, E07) |
| `pred_interval`, `jump_interval`, `loc_*` | interval flagged by each statistic, and its generation bounds |
| `mig_shift` | allele-frequency shift caused by the immigration pulse (E07) |
| `loc_hit`, `loc_err` | whether the flagged interval contains the pulse, and the distance in generations (E07) |

### Other outputs

| file | content |
|---|---|
| `e01_theory/theory_summary.json` | drift vs F_IS by generation (E01); identity residuals; Fisher-information summaries (E02); operator-order statistics (E06) |
| `e01_theory/identifiability_fim.csv` | one row per (design, data type, p0, F, s, h): rank, condition number, Cramér–Rao standard deviations of p0, F, s, h, hs, s_eff, and correlations |
| `e01_theory/order_and_legacy_shares.csv` | E06 per trajectory |
| `e08_reconstruction/reconstruction.csv` | RMSE at unsampled generations for the posterior mean (`HMM`), linear, cubic-spline and PCHIP interpolation and the deterministic model (`legacy_L0`); coverage of the 90% band |
| `e09_empirical/empirical_summary.json` | per data set and analysis: posterior summaries, discrepancy statistics, cumulative Shapley values, adult deviation by year, bootstrap closure test; and for single samples: n, p̂, observed F_IS, F_min, posterior of F_eff |
| `e10_legacy/legacy_vs_v3.csv` | per population: coverage and medians for the deterministic model (`legacy_*`) and BASTION (`v3_*`; "v3" is the name of the current method in the code) |
| `e10_legacy/legacy_summary.json` | the summaries, plus the eight dispersed chains of the deterministic model on Panaxia |
| `e11_legacy_ode/legacy_ode_summary.json`, `legacy_ode_rows.json` | E11 summary and per-replicate RMSEs |
| `e12_allele_hmm_comparison/comparison.csv` | E12 per data set: true values, both estimates and 90% intervals (`allele_*`, `bastion_*`), BASTION's F, convergence and time per fit |
| `e12_allele_hmm_comparison/comparison_summary.json`, `run.log` | E12 RMSE, bias, coverage and interval width per design and method; the console log |

---

## Figures

The figures of the manuscript are in `figures/` (PDF and PNG). `figures/make_figs.py` draws
them from `results/`:

```bash
python figures/make_figs.py           # all figures (about a minute)
python figures/make_figs.py 6 7       # selected figures
```

| figure | content | inputs |
|---|---|---|
| 1 | adult F_IS created by selection; adult F_IS against zygotic F | computed in the script (no inputs) |
| 2 | drift vs F_IS; Cramér–Rao standard deviations | `e01_theory/theory_summary.json`, `identifiability_fim.csv` |
| 3 | calibration rank histograms | `e03_sbc/sbc_Ne_known.csv`, `sbc_Ne_inferred.csv` |
| 4 | coverage under individual-based populations and violations | `e04_robustness/`, `e05_violations/` |
| 5 | immigration detection and flag rates | `e07_migration/` |
| 6 | Panaxia: trajectory, (s, h) posterior, cumulative Shapley values | `data/panaxia.csv`, `figures/refit/panaxia_Ne_FisherFord.npz` |
| 7 | Biston: trajectory, s under five analyses, posterior of h | `data/peppered_moth.csv`, `figures/refit/moth_*.npz`, `e09_empirical/empirical_summary.json` |

E09 stores only posterior summaries. Figures 6 and 7 need posterior samples, so
`figures/emp_refit.py` reruns the three fits concerned with exactly the E09 settings and
seeds and saves the samples to `figures/refit/`. The saved samples are included.

```bash
python figures/emp_refit.py           # about 10 min on 2 cores
```

Two further scripts compute numbers quoted in the text that are not part of `run_all.py`:

```bash
python figures/kernel_check.py        # accuracy of the transition kernel and grid likelihood (Section 4.1)
python figures/panaxia_profile.py     # Panaxia log-likelihood profiles (Section 6.1); output in figures/panaxia_profile.out
```

---

## Fitting your own data

`examples/fit_one_dataset.py` fits a CSV with columns `generation, N_AA, N_Aa, N_aa` and
(unless `--infer-ne`) `Ne`. It prints three things:

* the posterior of `F`, `s`, `h`, `hs` and `s_eff`;
* the cumulative Shapley values;
* the closed-model statistics.

The core calls are:

```python
from bastion import mcmc
data = mcmc.Data(gens, C, Ne_arr=Ne_per_generation)      # or mcmc.Data(gens, C) with infer_Ne=True
cfg  = mcmc.make_cfg(emission='geno_adult')              # 'geno_zygote', 'pheno_adult'; infer_Ne=True; infer_h=False, h_fixed=0.0
mc   = mcmc.run_mcmc(data, cfg, n_chains=4, n_steps=4000, n_warmup=1500, seed=0)
th, paths = mcmc.posterior_paths(mc, data, n_draws=200)  # posterior draws and hidden trajectories
att  = mcmc.attribution(mc, data, th, paths)             # Shapley values att['phi'], level split att['Dz'], att['DS']
disc = mcmc.discrepancy(mc, data, th, paths)             # closed-model statistics
```

---

## Lean formalisation

`lean/` proves Propositions 1–12 of the manuscript in 81 theorems, using only Lean's
standard axioms. `lean/README.md` lists every theorem with the Python function it covers and
says what is not formalised. In short, the likelihood, the sampler and all numerical results
are not formalised.

---

## Citation

If you use this code or data, please cite the manuscript (details in `CITATION.cff`; to be
updated on publication) and the original sources of the data (`data/README.md`).
