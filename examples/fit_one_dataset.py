"""Fit BASTION to one genotype time series and print the posterior, the attribution and
the closed-model statistics. A minimal template for your own data.

usage (from the repository root):
    python examples/fit_one_dataset.py                       # Panaxia dominula, census Ne
    python examples/fit_one_dataset.py --csv my_counts.csv   # your own file (same columns)
    python examples/fit_one_dataset.py --infer-ne            # treat Ne as unknown

Input columns: generation, N_AA, N_Aa, N_aa, and (unless --infer-ne) Ne.
Fitness parametrisation: w = (1, 1 - h s, 1 - s) for (AA, Aa, aa); s > 0 favours A.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bastion import mcmc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default=str(ROOT / 'data' / 'panaxia.csv'))
ap.add_argument('--infer-ne', action='store_true')
ap.add_argument('--chains', type=int, default=4)
ap.add_argument('--steps', type=int, default=4000, help='iterations per chain (the paper used 20000 for data)')
ap.add_argument('--warmup', type=int, default=1500)
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

df = pd.read_csv(args.csv).sort_values('generation')
g = df.generation.values.astype(int)
C = df[['N_AA', 'N_Aa', 'N_aa']].values.astype(float)
gens = g - g[0]
if args.infer_ne:
    data = mcmc.Data(gens, C)
else:
    # one Ne per generation between the first and the last sample (linear between samples)
    Ne_arr = np.interp(np.arange(g[0], g[-1] + 1), g, df.Ne.values.astype(float))
    data = mcmc.Data(gens, C, Ne_arr=Ne_arr)

cfg = mcmc.make_cfg(emission='geno_adult', infer_Ne=args.infer_ne)
mc = mcmc.run_mcmc(data, cfg, n_chains=args.chains, n_steps=args.steps, n_warmup=args.warmup, seed=args.seed)

K = mc['K'] if np.ndim(mc['K']) == 0 else ', '.join(str(k) for k in np.atleast_1d(mc['K']))
print(f'\nposterior (median [5%, 95%]), converged = {mc["converged"]}, grid K = {K}')
for name, S in mc['summary'].items():
    print(f'  {name:8s} {S["median"]:+.4f} [{S["q05"]:+.4f}, {S["q95"]:+.4f}]  Rhat {S["rhat"]:.3f}  ESS {S["ess"]:.0f}')

th, paths = mcmc.posterior_paths(mc, data, n_draws=200, seed=args.seed)
att = mcmc.attribution(mc, data, th, paths)
names = ['mutation', 'Mendelian reset', 'inbreeding', 'drift', 'selection']
cum = np.nansum(att['phi'], axis=1)                     # (draws, 5): summed over generations
print('\ncumulative Shapley values over the series (median [5%, 95%])')
for j, nm in enumerate(names):
    q5, q50, q95 = np.nanpercentile(cum[:, j], [5, 50, 95])
    print(f'  {nm:16s} {q50:+.2e} [{q5:+.2e}, {q95:+.2e}]')

disc = mcmc.discrepancy(mc, data, th, paths, seed=args.seed)
print('\nclosed-model statistics (compare with a parametric bootstrap, see experiments/e09_empirical.py)')
for k in ['score_pred', 'score_D', 'score_comb']:
    if k in disc:
        print(f'  {k:10s} {disc[k]:.3f}')
