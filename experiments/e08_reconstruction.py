"""E08 trajectory reconstruction from sparse samples.

Truth: the realised (stochastic) adult allele-frequency path. Methods:
  HMM   posterior mean of the BASTION smoother (FFBS paths), with 90% band
  linear / spline   interpolation of the observed sample allele frequencies
  legacy_L0         the legacy deterministic Layer 0 run from the legacy posterior
                    (old model, compiled; Jacobian on, Ne given) - the trajectory
                    the old Layer 4 physics would be anchored to
"""
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from common import outdir, save_json, write_csv, nrep, mcmc_kw, pmap, read_rows, safe, mcmc, sims, model


@safe
def recon_one(spec):
    rng = np.random.default_rng(spec['seed'])
    g = spec['gen']; T = g['T']
    P, X, A = sims.simulate(**g, seed=spec['seed'])
    gens = np.array(spec['gens'], int)
    C = sims.observe(P, X, A, gens, spec['n'], rng)
    truth = A[:, 0] + A[:, 1] / 2                      # adult allele frequency
    hid = np.setdiff1d(np.arange(T + 1), gens)
    p_obs = (2 * C[:, 0] + C[:, 1]) / (2 * C.sum(1))
    data = mcmc.Data(gens, C, Ne=g['Ne'])
    mc = mcmc.run_mcmc(data, mcmc.make_cfg(), seed=spec['seed'] % 100000, **spec['mk'])
    th, paths = mcmc.posterior_paths(mc, data, n_draws=200, seed=spec['seed'])
    adult = np.zeros_like(paths)
    for i in range(len(paths)):
        for t in range(T + 1):
            a = model.adults_at(paths[i, t], th[i, 0], th[i, 1], th[i, 2])
            adult[i, t] = a[0] + a[1] / 2
    est = {'HMM': np.nanmean(adult, 0),
           'linear': np.interp(np.arange(T + 1), gens, p_obs),
           'spline': CubicSpline(gens, p_obs)(np.arange(T + 1)),
           'pchip': PchipInterpolator(gens, p_obs)(np.arange(T + 1))}
    # legacy deterministic model
    from bastion import legacy_v2 as L
    try:
        lm = L.run_mcmc(gens, C[:, 0], C[:, 1], C[:, 2], np.full(T + 1, float(g['Ne'])), n_chains=4,
                        n_steps=20000, n_warmup=8000, seed=spec['seed'] % 1000, jacobian=True, init='dispersed')
        S = lm['summary']
        traj = L.simulate(S['p0']['q50'], S['F0']['q50'], S['s']['q50'], S['h']['q50'], 1e-5, 1e-5,
                          np.full(T + 1, float(g['Ne'])), T)[0]
        est['legacy_L0'] = traj
    except Exception:  # noqa: BLE001
        pass
    lo = np.nanpercentile(adult, 5, 0); hi = np.nanpercentile(adult, 95, 0)
    row = {'id': spec['id'], 'design': spec['design'],
           'HMM_band_cov90_hidden': float(np.mean((truth[hid] >= lo[hid]) & (truth[hid] <= hi[hid])))}
    for k, v in est.items():
        row[f'rmse_{k}'] = float(np.sqrt(np.mean((v[hid] - truth[hid]) ** 2)))
    return row


def run(cfg, pool):
    out = outdir(cfg, 'e08_reconstruction')
    rng = np.random.default_rng(81)
    specs = []; i = 0
    for _ in range(nrep(200, cfg, 6)):
        specs.append(dict(id=i, design='paper_fig3', seed=int(rng.integers(1, 2 ** 31 - 1)), n=300,
                          gen=dict(p0=0.35, F=0.05, s=0.08, h=0.5, T=99, Ne=1000, mode=0),
                          gens=[0, 6, 17, 31, 48, 70, 99], mk=mcmc_kw(cfg))); i += 1
    for _ in range(nrep(200, cfg, 6)):
        gens = sorted(set([0, 99] + list(rng.choice(np.arange(1, 99), 5, replace=False))))
        s = rng.choice([-1, 1]) * rng.uniform(0.02, 0.12)
        specs.append(dict(id=i, design='random', seed=int(rng.integers(1, 2 ** 31 - 1)), n=300,
                          gen=dict(p0=rng.uniform(0.15, 0.85), F=rng.uniform(0, 0.3), s=s, h=rng.uniform(0, 1),
                                   T=99, Ne=int(rng.choice([300, 1000, 3000])), mode=0),
                          gens=gens, mk=mcmc_kw(cfg))); i += 1
    rows = pmap(pool, recon_one, specs, 'reconstruction')
    write_csv(rows, out / 'reconstruction.csv')
    res = {}
    for dsn in ['paper_fig3', 'random']:
        R = [r for r in rows if r['design'] == dsn]
        d = {'n': len(R), 'HMM_band_cov90_hidden': float(np.nanmean(read_rows(R, 'HMM_band_cov90_hidden')))}
        hm = read_rows(R, 'rmse_HMM')
        for k in ['HMM', 'linear', 'spline', 'pchip', 'legacy_L0']:
            v = read_rows(R, f'rmse_{k}')
            d[f'rmse_{k}_median'] = float(np.nanmedian(v))
            if k != 'HMM':
                d[f'frac_HMM_better_than_{k}'] = float(np.nanmean(hm < v))
        res[dsn] = d
    save_json(res, out / 'reconstruction_summary.json')
    return res
