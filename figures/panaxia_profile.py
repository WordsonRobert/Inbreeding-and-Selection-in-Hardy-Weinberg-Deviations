"""Log-likelihood profiles for Panaxia dominula, 1939-1946, census N_e (Section 6.1).
Differences are relative to the log-likelihood at the posterior medians
(F, s, h) = (0.006, 0.145, 0.62); F is maximised out where stated.
usage (from the repository root):  python figures/panaxia_profile.py   (~5 min)
Its output is saved in figures/panaxia_profile.out."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
V3 = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, V3)
from bastion import model  # noqa: E402

df = pd.read_csv(Path(V3) / 'data' / 'panaxia.csv')
g = df.generation.values; gens = g - g[0]
C = df[['N_AA', 'N_Aa', 'N_aa']].values.astype(float)
Ne_arr = np.interp(np.arange(g[0], g[-1] + 1), g, df.Ne.values.astype(float))
K = model.choose_K(Ne_arr.max()); grid, prior = model.make_grid(K)


def ll(F, s, h):
    return model.hmm_forward(grid, prior, F, s, h, 1e-5, 1e-5, Ne_arr, gens, C, 0, False)[0]


def prof(s, h):
    r = minimize_scalar(lambda F: -ll(F, s, h), bounds=(-0.05, 0.5), method='bounded', options=dict(xatol=1e-4))
    return -r.fun - ref, r.x


ref = ll(0.006, 0.145, 0.62)
print(f'grid K = {K}; log-likelihood at the posterior medians {ref:.3f}')
print('along hs = 0.091, F maximised:')
for h in [0.2, 0.4, 0.62, 1.0, 1.5]:
    v, F = prof(0.091 / h, h)
    print(f'  h = {h:4.2f}  s = {0.091 / h:.3f}  dloglik = {v:+.2f}  (F = {F:.3f})')
print('recessive models, h = 0, F maximised:')
for s in [0.1, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
    v, F = prof(s, 0.0)
    print(f'  s = {s:4.2f}  dloglik = {v:+.2f}  (F = {F:.3f}, s_eff = {s * F:.3f})')
v, F = prof(0.0, 0.5); print(f'neutral s = 0, F maximised: {v:+.2f}')
v, F = prof(-0.22, -0.61); print(f'second mode of the deterministic model (s, h) = (-0.22, -0.61), F maximised: {v:+.2f}')
n = C.sum(1); q = (C[:, 1] + 2 * C[:, 2]) / (2 * n)
print(f'bimacula adults: observed {int(C[:, 2].sum())}, expected under Hardy-Weinberg {np.sum(n * q * q):.2f}')
