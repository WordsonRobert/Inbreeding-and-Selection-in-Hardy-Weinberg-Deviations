# Data

Every file was transcribed from a published source and is used as it is. Experiment E09
(`experiments/e09_empirical.py`) reads all six files, and E10 also reads `panaxia.csv`.

## Columns

Files with genotype counts (`panaxia`, `ccr5`, `hbc`, `zoarces`, `arabidopsis`):

| column | meaning |
|---|---|
| `generation` | year (Panaxia) or 0 for a single sample |
| `N_AA`, `N_Aa`, `N_aa` | genotype counts |
| `Ne` | effective size used in the analysis (time series only; ignored for single samples) |
| `dataset` | dataset name |
| `population` | sample name (CCR5 only) |
| `notes` | source table and genotype coding |

`peppered_moth.csv` has a different layout. See its section below.

## Files

### `panaxia.csv` — *Panaxia dominula*, Cothill, 1939–1946

* **Source:** Fisher, R. A. & Ford, E. B. (1947). The spread of a gene in natural conditions in
  a colony of the moth *Panaxia dominula* L. *Heredity* 1, 143–174, Table 1.
* **Coding:** `AA` = *dominula*, `Aa` = *medionigra*, `aa` = *bimacula*. There is one
  generation per year. The 8 samples have n = 117 to 986.
* **`Ne`:** for 1941–1946, the midpoint of the population-size range estimated by
  mark–recapture in Fisher & Ford's Table 15. For 1939 and 1940, which come before
  mark–recapture began, the value is 1000. These are census sizes. The analysis is also run
  with a constant Ne inferred from the data.

### `peppered_moth.csv` — *Biston betularia*, Caldy Common (Wirral), 1959–2001

* **Source:** annual frequency of the *carbonaria* phenotype, **digitised** from Cook, L. M.
  (2003). The rise and fall of the *carbonaria* form of the peppered moth. *Quarterly Review
  of Biology* 78, 399–417, Fig. 2.
* **Rows:** 34 years with data. The nine years without data are absent.
* **Columns:**
  * `year`, `generation` (years since 1959);
  * `f_carbonaria` (phenotype frequency; **this is the only column the analysis uses**);
  * `p_carbonaria_allele`, `q_typical_allele`;
  * `N_CC`, `N_Ct`, `N_tt` and `n_total`.
* **Approximations, which are not observed data:**
  * The annual sample sizes were not available. `n_total = 100` is a nominal value, and the
    analysis is repeated with n = 50, 100 and 300.
  * The allele frequencies and the genotype columns `N_CC`, `N_Ct`, `N_tt` were back-calculated
    under Hardy–Weinberg proportions from `f_carbonaria`, and **they are not used**.
* **Model:** the analysis fits the dominant-phenotype likelihood, with `A` = *carbonaria*:
  counts are `(round(f·n), 0, n − round(f·n))`.

### `ccr5.csv` — CCR5-Δ32, 15 human population samples

* **Source:** Martinson, J. J. et al. (1997). Global distribution of the CCR5 gene
  32-basepair deletion. *Nature Genetics* 16, 100–103, Table 1.
* **Coding:** `AA` = CCR5/CCR5, `Aa` = CCR5/Δ32, `aa` = Δ32/Δ32.
* **Use:** single-sample analysis only. The `Ne` column is not used.

### `hbc.csv` — haemoglobin C, Burkina Faso

* **Source:** Modiano, D. et al. (2001). Haemoglobin C protects against clinical *Plasmodium
  falciparum* malaria. *Nature* 414, 305–308, Table 2 (healthy subjects).
* **Coding:** `AA` = HbAA, `Aa` = HbAC, `aa` = HbCC.
* **Sample:** carriers of HbS (AS 335, SC 23, SS 1) are excluded, leaving n = 3154 of 3513.
* **Use:** single-sample analysis only.

### `zoarces.csv` — *Zoarces viviparus*, esterase III

* **Source:** Christiansen, F. B. & Frydenberg, O. (1973). Selection component analysis of
  natural polymorphisms using population samples including mother–offspring combinations.
  *Theoretical Population Biology* 4, 425–445, Table VIII (sterile females and males).
* **Coding:** `AA` = A1A1, `Aa` = A1A2, `aa` = A2A2. n = 782.
* **Use:** single-sample analysis only.

### `arabidopsis.csv` — *Arabidopsis thaliana*, 1001 Genomes

* **Source:** 1001 Genomes Consortium (2016). 1,135 genomes reveal the global pattern of
  polymorphism in *Arabidopsis thaliana*. *Cell* 166, 481–491. The data are binary haplotype
  calls (v3.1).
* **Counts:** pooled over 500 loci with minor-allele frequency > 0.05.
* **Heterozygotes:** the accessions are inbred lines called as haplotypes, so
  **`N_Aa = 0` by construction**. The file is used only as a check that a sample without
  heterozygotes returns F ≈ 1.
* **Downsampling:** the analysis scales the counts to n = 50 000.
