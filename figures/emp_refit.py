"""Refit Panaxia (census Ne) and the two Biston n = 100 configurations (h = 0 and h free,
Ne inferred) with exactly the E09 settings and seeds, and keep the posterior samples,
trajectories and attributions that Figures 6 and 7 need (E09 itself stores only summaries).

usage (from the repository root):  python figures/emp_refit.py      (~10 min on 2 cores)
Writes figures/refit/*.npz."""
import os, sys, numpy as np, multiprocessing as mp
for v in ['OMP_NUM_THREADS','NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[v]='1'
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = str(HERE.parent)
(HERE / 'refit').mkdir(exist_ok=True)
sys.path.insert(0,R); sys.path.insert(0,R+'/experiments')
import e09_empirical as E
from worker import emp_fit
g,C,Ne_arr=E._panaxia_data()
specs=[dict(tag='panaxia_Ne_FisherFord', gens=g-g[0], C=C, Ne_arr=Ne_arr, inf=dict(infer_Ne=False), seed=11, mk=E.EMP_MK)]
for tag,inf,n in E.MOTH_CONFIGS:
    if tag in ('h0_n100_NeInf','hfree_n100_NeInf'):
        df,gm,Cm=E._moth_C(n)
        specs.append(dict(tag=f'moth_{tag}', gens=gm-gm[0], C=Cm, Ne=1000.0, inf=inf, seed=13, mk=E.EMP_MK, n_paths=300))
with mp.get_context('fork').Pool(2) as pool:
    fits=pool.map(emp_fit, specs)
for f in fits:
    if '_error' in f: print(f); continue
    np.savez(HERE / 'refit' / f'{f["tag"]}.npz',
             flat=f['flat_thin'], th=f['th'], paths=f['paths'], phi=f['att']['phi'], Dz=f['att']['Dz'], DS=f['att']['DS'])
    S=f['summary']; print(f['tag'], {k:(round(S[k]['median'],4),round(S[k]['q05'],4),round(S[k]['q95'],4)) for k in S})
