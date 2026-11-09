"""E11 (optional, GPU) — the ORIGINAL Layer 4 neural ODE, unchanged, on the E08
'paper_fig3' design, against the BASTION HMM smoother and linear interpolation.

The neural ODE is given legacy-model posterior means (as in the original pipeline,
ode_input_from_mcmc), not the true parameters. Runs in its own process so the GPU
stage overlaps the CPU experiments.

usage: python experiments/e11_legacy_ode.py --out results --reps 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'legacy')); sys.path.insert(0, str(ROOT / 'experiments'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results'); ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=2000); ap.add_argument('--ensemble', type=int, default=10)
    ap.add_argument('--max-minutes', type=float, default=None, help='stop starting new replicates after this')
    a = ap.parse_args()
    out = Path(a.out) / 'e11_legacy_ode'; out.mkdir(parents=True, exist_ok=True)
    import torch
    import layer4_ode as L4
    from bastion import mcmc, sims, model, legacy_v2 as L
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[E11] device={dev} reps={a.reps}', flush=True)
    rng = np.random.default_rng(111)
    rows = []
    t0 = time.time()
    for r in range(a.reps):
        seed = int(rng.integers(1, 2 ** 31 - 1))
        el = time.time() - t0
        if a.max_minutes is not None and rows and (el + el / len(rows)) / 60 > a.max_minutes:
            print(f'[E11] time budget reached after {len(rows)} replicates', flush=True)
            break
        g = dict(p0=0.35, F=0.05, s=0.08, h=0.5, T=99, Ne=1000, mode=0)
        P, X, A = sims.simulate(**g, seed=seed)
        gens = np.array([0, 6, 17, 31, 48, 70, 99])
        C = sims.observe(P, X, A, gens, 300, np.random.default_rng(seed))
        truth = A[:, 0] + A[:, 1] / 2
        hid = np.setdiff1d(np.arange(100), gens)
        lm = L.run_mcmc(gens, C[:, 0], C[:, 1], C[:, 2], np.full(100, 1000.0), n_chains=4, n_steps=20000,
                        n_warmup=8000, seed=seed % 1000, jacobian=True, init='dispersed')
        S = lm['summary']
        p0, F0 = S['p0']['mean'], S['F0']['mean']; q0 = 1 - p0
        inp = L4.ODEInput(p0=p0, F0=F0, s=S['s']['mean'], h=S['h']['mean'], mu1=1e-5, mu2=1e-5,
                          generations=gens, N_AA=C[:, 0].astype(int), N_Aa=C[:, 1].astype(int),
                          N_aa=C[:, 2].astype(int), Ne=np.full(100, 1000.0), changepoints=[],
                          F_lb=-min(p0, q0) / max(p0, q0))
        res = L4.run_ode_ensemble(inp, n_ensemble=a.ensemble, n_epochs=a.epochs, verbose=False)
        ode = np.interp(np.arange(100), res.t_grid, res.p_med)
        p_obs = (2 * C[:, 0] + C[:, 1]) / (2 * C.sum(1))
        lin = np.interp(np.arange(100), gens, p_obs)
        data = mcmc.Data(gens, C, Ne=1000.0)
        mc = mcmc.run_mcmc(data, mcmc.make_cfg(), seed=seed % 100000)
        th, paths = mcmc.posterior_paths(mc, data, n_draws=200, seed=seed)
        adult = np.array([[sum(model.adults_at(paths[i, t], th[i, 0], th[i, 1], th[i, 2])[:2] * np.array([1, 0.5]))
                           for t in range(100)] for i in range(len(paths))])
        hmm = adult.mean(0)
        row = {'rep': r, 'rmse_ode': float(np.sqrt(np.mean((ode[hid] - truth[hid]) ** 2))),
               'rmse_linear': float(np.sqrt(np.mean((lin[hid] - truth[hid]) ** 2))),
               'rmse_HMM': float(np.sqrt(np.mean((hmm[hid] - truth[hid]) ** 2))),
               'ode_ensemble_mean_width90': float(np.mean(res.p_hi - res.p_lo)),
               'ode_max_residual': float(res.max_residual)}
        rows.append(row)
        print(f'[E11] rep {r + 1}/{a.reps} ode={row["rmse_ode"]:.4f} lin={row["rmse_linear"]:.4f} '
              f'hmm={row["rmse_HMM"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        json.dump(rows, open(out / 'legacy_ode_rows.json', 'w'), indent=1)
    v = {k: [r[k] for r in rows] for k in rows[0]}
    summ = {'device': dev, 'reps': len(rows), 'epochs': a.epochs, 'ensemble': a.ensemble,
            'rmse_median': {k: float(np.median(v[f'rmse_{k}'])) for k in ['ode', 'linear', 'HMM']},
            'frac_HMM_better_than_ode': float(np.mean(np.array(v['rmse_HMM']) < np.array(v['rmse_ode']))),
            'ode_ensemble_mean_width90_median': float(np.median(v['ode_ensemble_mean_width90']))}
    json.dump(summ, open(out / 'legacy_ode_summary.json', 'w'), indent=2)
    print('[E11] done', summ, flush=True)


if __name__ == '__main__':
    main()
