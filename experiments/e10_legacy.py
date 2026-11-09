"""E10 legacy comparison (the submitted model, compiled, with the Jacobian fixed).

(a) legacy vs v3 on the same realistic data (individual-based populations; random
    mating, or partial selfing) - coverage of s and of the inbreeding coefficient
(b) reproduction of the original migration ROC from noise-free true trajectories
    only (documents that the original score never used the data)
(c) Panaxia under the legacy model with dispersed chains: two (s, h) modes
"""
import numpy as np
import pandas as pd
from common import outdir, save_json, write_csv, nrep, mcmc_kw, pmap, read_rows, safe, ROOT, mcmc, sims, cover, auc
from bastion import legacy_v2 as L


@safe
def legacy_vs_v3(spec):
    rng = np.random.default_rng(spec['seed'])
    g = spec['gen']; T = g['T']
    P, X, A = sims.simulate(**g, seed=spec['seed'])
    gens = np.arange(0, T + 1, 5)
    C = sims.observe(P, X, A, gens, 300, rng)
    Ftrue = spec['F_true']
    row = {'id': spec['id'], 'cond': spec['cond']}
    lm = L.run_mcmc(gens, C[:, 0], C[:, 1], C[:, 2], np.full(T + 1, float(g['Ne'])), n_chains=4,
                    n_steps=20000, n_warmup=8000, seed=spec['seed'] % 1000, jacobian=True, init='dispersed')
    S = lm['summary']
    row.update(legacy_s_cov90=cover(S['s']['q05'], S['s']['q95'], g['s']),
               legacy_F0_cov90=cover(S['F0']['q05'], S['F0']['q95'], Ftrue),
               legacy_F0_med=S['F0']['q50'], legacy_s_med=S['s']['q50'],
               legacy_hs_cov90=cover(np.percentile(lm['flat'][:, 2] * lm['flat'][:, 3], 5),
                                     np.percentile(lm['flat'][:, 2] * lm['flat'][:, 3], 95), g['s'] * g['h']),
               legacy_converged=lm['converged'])
    mc = mcmc.run_mcmc(mcmc.Data(gens, C, Ne=g['Ne']), mcmc.make_cfg(), seed=spec['seed'] % 100000, **spec['mk'])
    V = mc['summary']
    row.update(v3_s_cov90=cover(V['s']['q05'], V['s']['q95'], g['s']),
               v3_F_cov90=cover(V['F']['q05'], V['F']['q95'], Ftrue),
               v3_hs_cov90=cover(V['hs']['q05'], V['hs']['q95'], g['s'] * g['h']),
               v3_F_med=V['F']['median'], v3_s_med=V['s']['median'], v3_converged=mc['converged'])
    return row


def oracle_roc(n=2000):
    """Original migration_roc.py scoring: max |R_true(t) - R_true(t-1)| of the noise-free
    trajectory with a one-generation blip (no data, no MCMC)."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        p0 = float(rng.uniform(0.2, 0.8)); q0 = 1 - p0; lb = -min(p0, q0) / max(p0, q0)
        F0 = float(rng.uniform(max(lb + 0.01, 0.0), 0.3)); s = float(rng.uniform(-0.15, 0.15))
        h = float(rng.uniform(0, 1)); Ne = int(rng.integers(500, 3000)); T = int(rng.integers(30, 60))
        T_mig = T // 2; p_mig = float(rng.uniform(0.1, 0.9)); _ = int(rng.integers(0, 1_000_000))
        p, F, R = L.simulate(p0, F0, s, h, 1e-5, 1e-5, np.full(T + 1, float(Ne)), T)[:3]
        for m in [0.0, 0.02, 0.05, 0.10, 0.20]:
            RR = R.copy()
            if m > 0:
                pa = (1 - m) * p[T_mig] + m * p_mig; Fa = (1 - m) * F[T_mig]
                qa = 1 - pa; lb2 = -min(pa, qa) / max(pa, qa); Fa = min(max(Fa, lb2 + 1e-6), 0.999)
                RR[T_mig] = Fa * pa * qa
            rows.append((m, float(np.abs(np.diff(RR)).max())))
    rows = np.array(rows)
    neg = rows[rows[:, 0] == 0, 1]
    return {f'm={m}': auc(neg, rows[rows[:, 0] == m, 1]) for m in [0.02, 0.05, 0.10, 0.20]}


def panaxia_legacy_modes():
    df = pd.read_csv(ROOT / 'data' / 'panaxia.csv')
    g = df.generation.values
    Ne = np.interp(np.arange(g[0], g[-1] + 1), g, df.Ne.values.astype(float))
    lm = L.run_mcmc(g, df.N_AA, df.N_Aa, df.N_aa, Ne, n_chains=8, n_steps=60000, n_warmup=20000, seed=7,
                    jacobian=True, init='dispersed')
    ch = lm['samples']
    return {'rhat': lm['r_hat'].tolist(),
            'chains': [{'s': float(ch[c, :, 2].mean()), 'h': float(ch[c, :, 3].mean()),
                        'hs': float((ch[c, :, 2] * ch[c, :, 3]).mean())} for c in range(ch.shape[0])]}


def run(cfg, pool):
    out = outdir(cfg, 'e10_legacy')
    rng = np.random.default_rng(101)
    specs = []; i = 0
    for cond, sig in [('random_mating', 0.0), ('selfing_0.3', 0.3)]:
        for _ in range(nrep(200, cfg, 6)):
            Feq = sig / (2 - sig)
            g = dict(p0=rng.uniform(0.2, 0.8), F=Feq, s=rng.choice([-1, 1]) * rng.uniform(0.02, 0.15),
                     h=rng.uniform(0, 1), T=40, Ne=int(rng.choice([200, 1000])), mode=1, sigma=sig)
            specs.append(dict(id=i, cond=cond, seed=int(rng.integers(1, 2 ** 31 - 1)), gen=g, F_true=Feq,
                              mk=mcmc_kw(cfg))); i += 1
    rows = pmap(pool, legacy_vs_v3, specs, 'legacy vs v3')
    write_csv(rows, out / 'legacy_vs_v3.csv')
    res = {'legacy_vs_v3': {}}
    for cond in ['random_mating', 'selfing_0.3']:
        R = [r for r in rows if r['cond'] == cond]
        res['legacy_vs_v3'][cond] = {k: float(np.nanmean(read_rows(R, k))) for k in
                                     ['legacy_s_cov90', 'legacy_F0_cov90', 'legacy_hs_cov90', 'legacy_converged',
                                      'v3_s_cov90', 'v3_F_cov90', 'v3_hs_cov90', 'v3_converged']}
        res['legacy_vs_v3'][cond]['n'] = len(R)
    print('  oracle ROC reproduction'); res['original_roc_from_true_trajectories_only'] = oracle_roc(
        2000 if cfg['mode'] != 'quick' else 200)
    print('  Panaxia legacy modes'); res['panaxia_legacy_dispersed_chains'] = panaxia_legacy_modes()
    save_json(res, out / 'legacy_summary.json')
    return res
