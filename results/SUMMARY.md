# BASTION — run summary

mode **full**, 15 workers, 12499.5 s total; started 2026-11-09 02:57:37

| stage | seconds |
|---|---|
| e01 | 16.2 |
| e03 | 2116.2 |
| e04 | 1902.5 |
| e05 | 1712.7 |
| e05b | 897.6 |
| e07 | 3051.9 |
| e08 | 468.1 |
| e09 | 1955.1 |
| e10 | 367.9 |

## E01 Theory checks

Random mating, N = 50 (10000 replicates): generation 60 — Wright's F 0.453, 1−H/H0 0.459, within-population F_IS -0.0111 (expected −1/2N = -0.0100).

| identity | max abs error |
|---|---|
| closed-form adult D | 8.3e-17 |
| D^a = (w3 D^z + κ (x2/2)^2)/w̄² | 5.6e-17 |
| genotype-level mutation map | 2.2e-16 |
| Shapley efficiency (Σφ = ΔD) | 1.8e-16 |
| dummy: s = 0 ⇒ φ_selection = 0 | 3.8e-17 |

Rare-allele effective selection s_eff = s[F + (1−F)h] (median relative error by q): q=0.0001: 0.00011, q=0.001: 0.0011, q=0.01: 0.011, q=0.05: 0.055, q=0.1: 0.11, q=0.2: 0.21

## E02 Identifiability (Fisher information)

Cross-sectional samples: max FIM rank 2 of 4. Temporal designs full rank in 0.785 of cases.

| data, T=30 n=300, s>0 (rare: min(p0, 1-p0) <= 0.05) | CRLB s | CRLB h | CRLB hs | CRLB s_eff | CRLB F | corr(s,h) |
|---|---|---|---|---|---|---|
| geno_rare | 1.067 | 5.572 | 0.348 | 0.128 | 0.108 | 0.985 |
| geno_common | 0.033 | 1.211 | 0.066 | 0.044 | 0.039 | -0.581 |
| pheno_rare | 19.688 | 92.177 | 1.353 | 1.200 | 12.841 | -1.000 |
| pheno_common | 0.522 | 5.610 | 0.359 | 0.552 | 8.944 | -0.995 |
| zygote_rare | 1.118 | 6.044 | 0.336 | 0.141 | 0.029 | 0.985 |
| zygote_common | 0.033 | 1.192 | 0.065 | 0.046 | 0.022 | -0.594 |

## E06 Attribution vs operator order

22500 generations. Lead force differs between life-cycle sequential attribution and Shapley in 0.056 of generations; differs for at least one of the 120 orderings in 0.711.
Median range of each operator's contribution across orderings (relative to Σ|φ|): mutation 0.0, mendelian_reset 0.5801, inbreeding 0.0359, drift 0.022, selection 0.0818

Selection share when F < 0.05 — legacy |α_s| 0.629 vs v3 0.352.

## E03 Simulation-based calibration (truth drawn from the priors)

### SBC

| condition | n | conv | cov90 F | cov90 s | cov90 h | cov90 hs | cov90 s_eff | cov90 log10Ne | bias s | contraction s | path cov90 | D^S cov90 | D^z cov90 | sel-share cov90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ne_known | 2500 | 0.97 | 0.89 | 0.90 | 0.89 | 0.90 | 0.90 | – | -0.0009 | 0.20 | 0.90 | 0.90 | 0.90 | 0.89 |
| Ne_inferred | 1500 | 0.96 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.88 | -0.0021 | 0.26 | 0.90 | 0.91 | 0.92 | 0.89 |

Ne_known: rank-uniformity χ² p-values F 0.829, s 0.640, h 0.320, hs 0.876
Ne_known: cumulative Shapley coverage mutation 0.83, mendelian_reset 0.89, inbreeding 0.88, drift 0.88, selection 0.89
Ne_inferred: rank-uniformity χ² p-values F 0.305, s 0.016, h 0.197, hs 0.077, log10Ne 0.446
Ne_inferred: cumulative Shapley coverage mutation 0.80, mendelian_reset 0.87, inbreeding 0.88, drift 0.88, selection 0.89

## E04 Robustness: individual-based populations (real finite-N drift, partial selfing)

### robustness

| condition | n | conv | cov90 F | cov90 s | cov90 h | cov90 hs | cov90 s_eff | bias s | contraction s | path cov90 | D^S cov90 | D^z cov90 | sel-share cov90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N100_sigma0.0_Neknown | 300 | 0.99 | 0.69 | 0.94 | 0.96 | 0.93 | 0.94 | -0.0015 | 0.41 | 0.88 | 0.93 | 0.71 | – |
| N100_sigma0.2_Neknown | 300 | 0.98 | 0.70 | 0.91 | 0.98 | 0.94 | 0.93 | -0.0058 | 0.41 | 0.88 | 0.96 | 0.76 | 0.96 |
| N100_sigma0.5_Neknown | 300 | 0.99 | 0.70 | 0.88 | 0.99 | 0.96 | 0.90 | -0.0056 | 0.32 | 0.86 | 0.95 | 0.83 | 0.91 |
| N2000_sigma0.0_Neknown | 300 | 0.92 | 0.89 | 0.95 | 0.96 | 0.94 | 0.94 | -0.0012 | 0.19 | 0.90 | 0.93 | 0.89 | – |
| N2000_sigma0.2_Neknown | 300 | 0.95 | 0.92 | 0.94 | 0.96 | 0.95 | 0.95 | -0.0012 | 0.15 | 0.89 | 0.95 | 0.89 | 0.96 |
| N2000_sigma0.5_Neknown | 300 | 0.98 | 0.89 | 0.91 | 0.96 | 0.95 | 0.95 | -0.0013 | 0.13 | 0.87 | 0.92 | 0.87 | 0.92 |
| N500_sigma0.0_Neinferred | 300 | 0.92 | 0.85 | 0.86 | 0.94 | 0.91 | 0.90 | 0.0030 | 0.25 | 0.84 | 0.90 | 0.84 | – |
| N500_sigma0.0_Neknown | 300 | 0.94 | 0.86 | 0.94 | 0.98 | 0.97 | 0.97 | 0.0010 | 0.25 | 0.90 | 0.95 | 0.83 | – |
| N500_sigma0.2_Nedouble | 300 | 0.97 | 0.84 | 0.84 | 0.95 | 0.89 | 0.87 | -0.0023 | 0.16 | 0.81 | 0.91 | 0.82 | 0.92 |
| N500_sigma0.2_Nehalf | 300 | 0.99 | 0.89 | 0.98 | 0.99 | 0.97 | 0.97 | -0.0015 | 0.25 | 0.93 | 0.97 | 0.87 | 0.97 |
| N500_sigma0.2_Neinferred | 300 | 0.95 | 0.82 | 0.90 | 0.94 | 0.94 | 0.94 | 0.0011 | 0.27 | 0.85 | 0.93 | 0.83 | 0.92 |
| N500_sigma0.2_Neknown | 300 | 0.96 | 0.85 | 0.93 | 0.96 | 0.95 | 0.94 | -0.0020 | 0.22 | 0.88 | 0.92 | 0.85 | 0.94 |
| N500_sigma0.5_Neinferred | 300 | 0.98 | 0.83 | 0.90 | 0.96 | 0.93 | 0.92 | 0.0009 | 0.20 | 0.87 | 0.94 | 0.84 | 0.95 |
| N500_sigma0.5_Neknown | 300 | 0.99 | 0.84 | 0.88 | 0.99 | 0.97 | 0.93 | 0.0001 | 0.17 | 0.86 | 0.93 | 0.85 | 0.95 |

## E05 Stress tests

### violations

| condition | n | conv | cov90 F | cov90 s | cov90 hs | cov90 s_eff | bias s | contraction s | path cov90 | D^S cov90 | D^z cov90 | sel-share cov90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 500 | 0.98 | 0.91 | 0.93 | 0.96 | 0.96 | -0.0026 | 0.17 | 0.89 | 0.95 | 0.90 | 0.94 |
| changing_s | 500 | 0.98 | 0.83 | 0.15 | 0.26 | 0.17 | 0.0282 | 0.13 | 0.70 | 0.50 | 0.82 | 0.51 |
| bottleneck_Ne_fixed | 500 | 0.95 | 0.83 | 0.69 | 0.72 | 0.68 | 0.0018 | 0.14 | 0.68 | 0.87 | 0.87 | 0.86 |
| bottleneck_Ne_inferred | 500 | 0.98 | 0.93 | 0.95 | 0.96 | 0.95 | 0.0037 | 0.32 | 0.89 | 0.96 | 0.92 | 0.96 |
| linked_selection | 500 | 0.98 | 0.88 | 0.52 | 0.87 | 0.80 | -0.0021 | 0.11 | 0.88 | 0.87 | 0.90 | 0.87 |
| wahlund_sampling | 500 | 1.00 | 0.00 | 0.92 | 0.72 | 0.71 | 0.0018 | 0.17 | 0.67 | 0.65 | 0.01 | 0.78 |
| genotyping_error | 500 | 0.98 | 0.63 | 0.95 | 0.96 | 0.96 | 0.0015 | 0.16 | 0.90 | 0.92 | 0.70 | 0.92 |
| zygote_sampled | 500 | 0.98 | 0.89 | 0.94 | 0.96 | 0.95 | -0.0003 | 0.15 | 0.85 | 0.94 | 0.87 | 0.94 |
| selfing_transient | 500 | 0.99 | 0.77 | 0.91 | 0.90 | 0.91 | -0.0004 | 0.15 | 0.88 | 0.92 | 0.80 | 0.93 |
| finite_pop | 500 | 0.96 | 0.87 | 0.92 | 0.96 | 0.96 | 0.0021 | 0.18 | 0.89 | 0.90 | 0.86 | – |

| condition | frac s-interval excludes 0 | frac hs-interval excludes 0 | median jump score | median predictive score | median D score |
|---|---|---|---|---|---|
| baseline | 0.96 | 0.36 | 1.10 | 1.64 | 1.62 |
| changing_s | 0.29 | 0.14 | 2.65 | 3.42 | 1.72 |
| bottleneck_Ne_fixed | 0.87 | 0.36 | 2.25 | 3.19 | 1.69 |
| bottleneck_Ne_inferred | 0.62 | 0.13 | 1.70 | 1.95 | 1.61 |
| linked_selection | 0.48 | 0.13 | 1.29 | 1.89 | 1.71 |
| wahlund_sampling | 0.95 | 0.31 | 1.11 | 1.71 | 1.70 |
| genotyping_error | 0.97 | 0.32 | 1.09 | 1.67 | 1.67 |
| zygote_sampled | 0.97 | 0.38 | 1.10 | 1.67 | 1.65 |
| selfing_transient | 0.98 | 0.38 | 1.13 | 1.75 | 2.66 |
| finite_pop | 0.92 | 0.42 | 1.15 | 1.71 | 1.88 |

## E05b Failure map (coverage s / F, convergence)

- **p0**: 0.02: s 0.97, F 0.90, conv 0.80; 0.05: s 1.00, F 0.99, conv 0.85; 0.1: s 1.00, F 0.92, conv 1.00; 0.2: s 0.95, F 0.93, conv 1.00; 0.4: s 0.93, F 0.90, conv 1.00; 0.5: s 0.97, F 0.90, conv 1.00
- **F**: 0.0: s 0.95, F 0.91, conv 1.00; 0.1: s 0.92, F 0.93, conv 1.00; 0.2: s 0.91, F 0.96, conv 1.00; 0.4: s 0.93, F 0.90, conv 1.00; 0.6: s 0.89, F 0.96, conv 1.00; 0.75: s 0.88, F 0.92, conv 1.00
- **n**: 10: s 0.89, F 0.91, conv 1.00; 20: s 0.94, F 0.94, conv 1.00; 50: s 0.98, F 0.90, conv 1.00; 100: s 0.91, F 0.94, conv 1.00; 300: s 0.91, F 0.93, conv 1.00; 1000: s 0.97, F 0.95, conv 1.00
- **T**: 5: s 0.92, F 0.93, conv 1.00; 10: s 0.92, F 0.93, conv 1.00; 15: s 0.87, F 0.96, conv 1.00; 20: s 0.93, F 0.96, conv 1.00; 30: s 0.93, F 0.98, conv 1.00; 60: s 0.96, F 0.95, conv 1.00
- **s**: 0.001: s 0.92, F 0.93, conv 1.00; 0.005: s 0.85, F 0.94, conv 1.00; 0.01: s 0.92, F 0.91, conv 1.00; 0.05: s 0.87, F 0.92, conv 1.00; 0.1: s 0.95, F 0.96, conv 1.00; 0.2: s 0.93, F 0.86, conv 1.00
- **Ne**: 20: s 0.83, F 0.95, conv 0.99; 50: s 0.90, F 0.98, conv 1.00; 100: s 0.93, F 1.00, conv 1.00; 250: s 0.91, F 0.94, conv 1.00; 1000: s 0.95, F 0.95, conv 1.00; 5000: s 0.93, F 0.91, conv 1.00

## E07 Closed-model discrepancy (immigration detection, data only)

### Ne_known (n closed = 1000)

| migration | AUC jump | AUC pred | AUC D | AUC comb | TPR@5%FPR comb | interval hit | hit if flagged | loc err (gens) |
|---|---|---|---|---|---|---|---|---|
| m=0.02 | 0.51 | 0.51 | 0.49 | 0.50 | 0.05 | 0.12 | 0.25 | 8.0 |
| m=0.05 | 0.53 | 0.54 | 0.51 | 0.54 | 0.07 | 0.18 | 0.32 | 7.0 |
| m=0.1 | 0.56 | 0.56 | 0.51 | 0.55 | 0.11 | 0.26 | 0.59 | 6.0 |
| m=0.2 | 0.66 | 0.65 | 0.54 | 0.65 | 0.22 | 0.42 | 0.86 | 3.0 |

Detection (comb score, 5% FPR) by the allele-frequency shift |Δp| the pulse caused: 0-0.01: 0.05 (n=1503), 0.01-0.02: 0.06 (n=787), 0.02-0.05: 0.08 (n=1010), 0.05-0.1: 0.25 (n=522), 0.1-1.0: 0.66 (n=178)

Flag rate at the 5%-FPR threshold for closed populations with other violations (specificity):

| violation | jump | pred | D | comb |
|---|---|---|---|---|
| finite_pop | 0.09 | 0.06 | 0.14 | 0.12 |
| changing_s | 0.78 | 0.64 | 0.05 | 0.55 |
| bottleneck | 0.72 | 0.66 | 0.04 | 0.57 |
| linked_selection | 0.15 | 0.09 | 0.04 | 0.09 |
| wahlund_sampling | 0.05 | 0.04 | 0.04 | 0.04 |
| genotyping_error | 0.06 | 0.04 | 0.06 | 0.05 |

### Ne_inferred (n closed = 375)

| migration | AUC jump | AUC pred | AUC D | AUC comb | TPR@5%FPR comb | interval hit | hit if flagged | loc err (gens) |
|---|---|---|---|---|---|---|---|---|
| m=0.05 | 0.50 | 0.48 | 0.48 | 0.47 | 0.05 | 0.17 | 0.50 | 7.0 |
| m=0.1 | 0.55 | 0.54 | 0.52 | 0.53 | 0.08 | 0.24 | 0.62 | 5.0 |

Detection (comb score, 5% FPR) by the allele-frequency shift |Δp| the pulse caused: 0-0.01: 0.00 (n=213), 0.01-0.02: 0.05 (n=175), 0.02-0.05: 0.07 (n=288), 0.05-0.1: 0.24 (n=74), 0.1-1.0: – (n=0)

## E08 Trajectory reconstruction (hidden generations, RMSE)

| design | n | HMM | linear | spline | pchip | legacy L0 | HMM better than linear | HMM 90% band coverage |
|---|---|---|---|---|---|---|---|---|
| paper_fig3 | 500 | 0.0172 | 0.0184 | 0.0221 | 0.0192 | 0.0216 | 0.65 | 0.90 |
| random | 500 | 0.0189 | 0.0221 | 0.0382 | 0.0228 | 0.0340 | 0.65 | 0.89 |

## E11 Original neural ODE (unchanged) vs HMM

6 reps on cuda: median RMSE ODE 0.0279, linear 0.0202, HMM 0.0192; HMM better than ODE in 1.00; ODE ensemble mean 90% width 0.0134.

## E09 Empirical

**Panaxia (Ne_FisherFord)**: F 0.006 [-0.022, 0.047] (R̂ 1.001), s 0.145 [0.043, 0.308] (R̂ 1.006), h 0.616 [0.139, 1.297] (R̂ 1.004), hs 0.091 [0.027, 0.149] (R̂ 1.000), s_eff 0.092 [0.029, 0.149] (R̂ 1.000); closure test p = 0.329
  cumulative Shapley ΔD (median): mutation 1.56e-07, mendelian_reset -2.69e-03, inbreeding 2.38e-03, drift -2.01e-05, selection 6.36e-04
**Panaxia (Ne_inferred)**: F 0.005 [-0.023, 0.046] (R̂ 1.001), s 0.142 [-0.048, 0.315] (R̂ 1.002), h 0.604 [-0.062, 1.289] (R̂ 1.006), hs 0.091 [-0.004, 0.158] (R̂ 1.002), s_eff 0.093 [-0.002, 0.159] (R̂ 1.002), log10Ne 3.054 [1.924, 4.337] (R̂ 1.003); closure test p = 0.495
  cumulative Shapley ΔD (median): mutation 1.32e-07, mendelian_reset -1.92e-03, inbreeding 1.28e-03, drift -7.41e-07, selection 7.03e-04
**Peppered moth (h0_n100_NeInf)**: F 0.006 [-0.179, 0.296] (R̂ 1.001), s -0.223 [-0.245, -0.198] (R̂ 1.001), s_eff -0.001 [-0.064, 0.039] (R̂ 1.001), log10Ne 3.758 [2.849, 4.425] (R̂ 1.002); closure test p = 1.000
**Peppered moth (hfree_n100_NeInf)**: F 0.085 [-0.166, 0.456] (R̂ 1.001), s -0.217 [-0.239, -0.191] (R̂ 1.002), h -0.124 [-0.385, 0.036] (R̂ 1.003), hs 0.027 [-0.008, 0.080] (R̂ 1.003), s_eff 0.002 [-0.066, 0.066] (R̂ 1.001), log10Ne 3.826 [2.882, 4.438] (R̂ 1.000)
**Peppered moth (h0_n100_Ne1000)**: F 0.056 [-0.171, 0.369] (R̂ 1.001), s -0.219 [-0.255, -0.186] (R̂ 1.001), s_eff -0.012 [-0.078, 0.037] (R̂ 1.001)
**Peppered moth (h0_n50_NeInf)**: F 0.021 [-0.176, 0.353] (R̂ 1.001), s -0.221 [-0.250, -0.189] (R̂ 1.001), s_eff -0.005 [-0.075, 0.039] (R̂ 1.001), log10Ne 3.680 [2.647, 4.424] (R̂ 1.001)
**Peppered moth (h0_n300_NeInf)**: F -0.016 [-0.178, 0.233] (R̂ 1.001), s -0.226 [-0.245, -0.204] (R̂ 1.001), s_eff 0.004 [-0.052, 0.040] (R̂ 1.001), log10Ne 3.616 [2.955, 4.379] (R̂ 1.001)

| cross-sectional | n | p̂ | F_IS observed | lower bound F_min(p̂) | no rare homozygotes | F_eff posterior median [90%] |
|---|---|---|---|---|---|---|
| ccr5_Ashkenazi | 43 | 0.791 | -0.1242 | -0.2647 |  | -0.0673 [-0.2501, 0.1837] |
| ccr5_Iceland | 75 | 0.873 | -0.0245 | -0.1450 |  | 0.0361 [-0.1188, 0.2576] |
| ccr5_Britain | 283 | 0.889 | -0.0181 | -0.1252 |  | 0.0030 [-0.0780, 0.1138] |
| ccr5_Spain_Basque | 29 | 0.914 | -0.0943 | -0.0943 | yes | -0.0677 [-0.1777, 0.4000] |
| ccr5_Spain_Catalonia | 41 | 0.902 | -0.1081 | -0.1081 | yes | -0.0973 [-0.1826, 0.2370] |
| ccr5_Italy | 44 | 0.955 | -0.0476 | -0.0476 | yes | -0.0297 [-0.1019, 0.4851] |
| ccr5_Ireland | 77 | 0.890 | -0.1241 | -0.1241 | yes | -0.1220 [-0.1811, 0.0386] |
| ccr5_Cyprus | 84 | 0.958 | -0.0435 | -0.0435 | yes | -0.0398 [-0.0820, 0.2978] |
| ccr5_Greece | 63 | 0.976 | -0.0244 | -0.0244 | yes | 0.0106 [-0.0608, 0.5890] |
| ccr5_Caucasus_Daghestan | 110 | 0.936 | -0.0680 | -0.0680 | yes | -0.0680 [-0.1056, 0.1203] |
| ccr5_Saudi_Arabia | 231 | 0.978 | -0.0221 | -0.0221 | yes | -0.0223 [-0.0384, 0.2304] |
| ccr5_Russia_Udmurtia | 46 | 0.902 | 0.1379 | -0.1084 |  | 0.2233 [-0.0316, 0.5351] |
| ccr5_Thailand | 101 | 0.995 | -0.0050 | -0.0050 | yes | 0.2068 [-0.0233, 0.7378] |
| ccr5_Nigeria | 111 | 0.995 | -0.0045 | -0.0045 | yes | 0.2069 [-0.0213, 0.7378] |
| ccr5_Nuu_Chah_Nulth | 38 | 0.987 | -0.0133 | -0.0133 | yes | 0.2006 [-0.0607, 0.7322] |
| hbc | 3154 | 0.861 | -0.0086 | -0.1619 |  | -0.0067 [-0.0350, 0.0224] |
| zoarces | 782 | 0.372 | -0.0125 | -0.5927 |  | -0.0093 [-0.0710, 0.0466] |
| arabidopsis | 50000 | 0.763 | 1.0000 | -0.3098 |  | 0.9999 [0.9997, 0.9999] |

F_eff = max(F, F_min(p)) = D^z/(pq): below F_min the zygote vector is clipped and F is not identified, so the effective coefficient is reported.

## E10 Legacy model

| data | n | legacy cov s | legacy cov F0 | legacy cov hs | v3 cov s | v3 cov F | v3 cov hs |
|---|---|---|---|---|---|---|---|
| random_mating | 500 | 0.68 | 0.63 | 0.67 | 0.93 | 0.84 | 0.95 |
| selfing_0.3 | 500 | 0.72 | 0.71 | 0.63 | 0.93 | 0.85 | 0.96 |

Original migration ROC reproduced from noise-free true trajectories only (no data, no MCMC): m=0.02 AUC 0.6966, m=0.05 AUC 0.8233, m=0.1 AUC 0.8941, m=0.2 AUC 0.9433

Panaxia, legacy model, 8 dispersed chains: s 0.160, h 0.752, hs 0.104; s 0.161, h 0.769, hs 0.105; s -0.220, h -0.608, hs 0.122; s 0.160, h 0.746, hs 0.103; s 0.159, h 0.751, hs 0.103; s 0.166, h 0.744, hs 0.104; s 0.162, h 0.744, hs 0.105; s 0.160, h 0.754, hs 0.104
