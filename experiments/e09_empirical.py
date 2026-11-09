"""E09 empirical datasets.

Temporal (full model):
  Panaxia dominula 1939-46  adult genotype counts (dominula / medionigra / bimacula)
  Biston betularia (Caldy)  adult PHENOTYPE counts (carbonaria dominant). The CSV
                            has frequencies x n = 100 per year; the true annual
                            sample sizes are not in the file, so n is a sensitivity
                            parameter. Genotypes are NOT reconstructed under HWE.
Cross-sectional (F only; s, h are not identifiable from one sample):
  CCR5-delta32 (15 populations), HbC, Zoarces EstIII, Arabidopsis
Closure test: parametric-bootstrap p-value of the discrepancy score against
closed populations simulated from the posterior with the same design.
The wolf / Isle Royale series are pedigree F, not genotype counts, so the
genotype-level model does not apply to them (see SUMMARY.md).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import outdir, save_json, write_csv, nrep, pmap, read_rows, COL, ROOT, mcmc, model
from worker import fit_one, emp_fit

EMP_MK = dict(n_chains=8, n_steps=20000, n_warmup=8000)


def _quick(cfg):
    return EMP_MK if cfg['mode'] != 'quick' else dict(n_chains=4, n_steps=4000, n_warmup=1500)


def _summ(fit):
    return {k: {kk: v[kk] for kk in ['median', 'mean', 'sd', 'q05', 'q95', 'hdi90', 'rhat', 'ess']}
            for k, v in fit['summary'].items()} | {'converged': fit['converged'], 'accept': list(fit['accept']),
                                                  'pruned_chains': fit['n_pruned'], 'grid_K': fit['K']}


def _disc(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def _closure(pool, fit, data, n_boot, seed, emission, cfg, label, ns, fix_h0=False):
    mk = dict(n_chains=4, n_steps=4000, n_warmup=1500) if cfg['mode'] != 'quick' else \
        dict(n_chains=2, n_steps=2500, n_warmup=1000)
    specs = _bootstrap_specs(fit['th'], fit['paths'], data, n_boot, np.random.default_rng(seed), emission, mk, ns)
    if fix_h0:
        for sp in specs:
            sp['inf'].update(infer_h=False, h_fixed=0.0)
            sp['gen']['h'] = 0.0
    rows = pmap(pool, fit_one, specs, label)
    null = read_rows(rows, 'score_comb')
    obs = fit['disc']['score_comb']
    return {'score_comb_obs': obs, 'n_boot': int(np.isfinite(null).sum()),
            'null_q50_q95': [float(np.nanpercentile(null, 50)), float(np.nanpercentile(null, 95))],
            'p_value': float((1 + np.sum(null >= obs)) / (1 + np.isfinite(null).sum()))}


def _attr_fig(att, years, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for key, col, lab in [('Dz', COL['inbreeding'], 'inbreeding share'), ('DS', COL['selection'], 'selection share')]:
        M = att[key]
        axes[0].plot(years, np.nanmedian(M, 0), color=col, lw=2, label=lab)
        axes[0].fill_between(years, np.nanpercentile(M, 5, 0), np.nanpercentile(M, 95, 0), color=col, alpha=0.2)
    axes[0].axhline(0, color='0.6', lw=0.6)
    axes[0].set_ylabel('adult HW deviation D'); axes[0].legend(frameon=False, fontsize=8)
    cum = np.nancumsum(att['phi'], 1)
    for k, nm in enumerate(model.OP_NAMES):
        axes[1].plot(years[1:], np.nanmedian(cum[:, :, k], 0), color=COL[nm], lw=1.8, label=nm.replace('_', ' '))
    axes[1].axhline(0, color='0.6', lw=0.6); axes[1].set_ylabel('cumulative Shapley contribution to ΔD')
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _bootstrap_specs(th, paths, data, n_boot, rng, emission, mk, ns):
    """Closed populations simulated from posterior draws with the observed design
    (same generations, same per-year sample sizes)."""
    specs = []
    idx = rng.choice(len(th), size=n_boot, replace=True)
    for b, i in enumerate(idx):
        F, s, h, lne = th[i][:4]
        Ne = 10 ** lne if np.isfinite(lne) else None
        gen = dict(p0=float(paths[i, 0]), F=float(F), s=float(s), h=float(h), T=int(data.T), mode=0)
        if Ne is None:
            gen['Ne_arr'] = np.concatenate([data.Ne_arr, data.Ne_arr[-1:]])
        else:
            gen['Ne'] = float(Ne)
        specs.append(dict(id=b, seed=int(rng.integers(1, 2 ** 31 - 1)), cond='boot', gen=gen,
                          obs=dict(gens=list(data.gens - data.gens[0]), n=list(ns), pheno=(emission == 'pheno_adult')),
                          inf=dict(emission=emission, infer_Ne=Ne is not None, Ne_fixed=1000.0,
                                   **({} if Ne is not None else {'Ne_arr': data.Ne_arr})),
                          truth=dict(F=F, s=s, h=h), mk=mk, n_paths=100, disc=True))
    return specs


def _panaxia_data():
    df = pd.read_csv(ROOT / 'data' / 'panaxia.csv')
    g = df.generation.values
    Ne_arr = np.interp(np.arange(g[0], g[-1] + 1), g, df.Ne.values.astype(float))
    C = df[['N_AA', 'N_Aa', 'N_aa']].values.astype(float)
    return g, C, Ne_arr


MOTH_CONFIGS = [('h0_n100_NeInf', dict(emission='pheno_adult', infer_h=False, h_fixed=0.0, infer_Ne=True), 100),
                ('hfree_n100_NeInf', dict(emission='pheno_adult', infer_h=True, infer_Ne=True), 100),
                ('h0_n100_Ne1000', dict(emission='pheno_adult', infer_h=False, h_fixed=0.0, infer_Ne=False), 100),
                ('h0_n50_NeInf', dict(emission='pheno_adult', infer_h=False, h_fixed=0.0, infer_Ne=True), 50),
                ('h0_n300_NeInf', dict(emission='pheno_adult', infer_h=False, h_fixed=0.0, infer_Ne=True), 300)]


def _moth_C(n):
    df = pd.read_csv(ROOT / 'data' / 'peppered_moth.csv')
    g = df.generation.values.astype(int)
    dom = np.round(df.f_carbonaria.values * n)
    return df, g, np.column_stack([dom, np.zeros_like(dom), n - dom])


def empirical_fits(cfg, pool):
    """All long empirical MCMC runs in parallel (one worker each)."""
    g, C, Ne_arr = _panaxia_data()
    specs = [dict(tag=f'panaxia_{t}', gens=g - g[0], C=C, Ne_arr=Ne_arr, inf=inf, seed=11, mk=_quick(cfg))
             for t, inf in [('Ne_FisherFord', dict(infer_Ne=False)), ('Ne_inferred', dict(infer_Ne=True))]]
    for tag, inf, n in MOTH_CONFIGS:
        df, gm, Cm = _moth_C(n)
        specs.append(dict(tag=f'moth_{tag}', gens=gm - gm[0], C=Cm, Ne=1000.0, inf=inf, seed=13,
                          mk=_quick(cfg), n_paths=300))
    fits = pmap(pool, emp_fit, specs, 'empirical MCMC')
    return {f['tag']: f for f in fits}


def panaxia(cfg, pool, out, fits):
    g, C, Ne_arr = _panaxia_data()
    res = {}
    for tag, inf in [('Ne_FisherFord', dict(infer_Ne=False)), ('Ne_inferred', dict(infer_Ne=True))]:
        fit = fits.get(f'panaxia_{tag}')
        if fit is None:
            continue
        data = mcmc.Data(g - g[0], C, Ne_arr=Ne_arr)
        att = fit['att']
        r = {'posterior': _summ(fit), 'discrepancy': _disc(fit['disc'])}
        cum = np.nansum(att['phi'], 1)
        r['cumulative_shapley_dD'] = {nm: [float(np.nanpercentile(cum[:, k], q)) for q in (5, 50, 95)]
                                      for k, nm in enumerate(model.OP_NAMES)}
        r['adult_D_by_year'] = {int(y): {'D_inbreeding': [float(np.nanpercentile(att['Dz'][:, t], q)) for q in (5, 50, 95)],
                                         'D_selection': [float(np.nanpercentile(att['DS'][:, t], q)) for q in (5, 50, 95)]}
                                for t, y in enumerate(range(g[0], g[-1] + 1))}
        _attr_fig(att, np.arange(g[0], g[-1] + 1), f'Panaxia dominula ({tag})', out / f'panaxia_{tag}.png')
        r['closure_test'] = _closure(pool, fit, data, nrep(200, cfg, 8), 5, 'geno_adult', cfg,
                                     f'Panaxia bootstrap {tag}', C.sum(1))
        res[tag] = r
        if tag == 'Ne_FisherFord':
            flat = fit['flat_thin']
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ax.scatter(flat[::2, 1], flat[::2, 2], s=2, alpha=0.3, color=COL['selection'])
            ss = np.linspace(-0.8, 0.8, 400); hsv = np.median(flat[:, 4])
            ok = np.abs(ss) > 0.02
            ax.plot(ss[ok & (ss > 0)], hsv / ss[ok & (ss > 0)], 'k--', lw=0.8)
            ax.plot(ss[ok & (ss < 0)], hsv / ss[ok & (ss < 0)], 'k--', lw=0.8, label=f'h·s = {hsv:.3f}')
            ax.set_xlim(-0.8, 0.8); ax.set_ylim(-2, 3); ax.set_xlabel('s'); ax.set_ylabel('h')
            ax.legend(frameon=False, fontsize=8); ax.set_title('Panaxia posterior (s, h): the h·s ridge', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout(); fig.savefig(out / 'panaxia_s_h_ridge.png', dpi=200); plt.close(fig)
    return res


def moth(cfg, pool, out, fits):
    res = {}
    for tag, inf, n in MOTH_CONFIGS:
        fit = fits.get(f'moth_{tag}')
        if fit is None:
            continue
        df, g, C = _moth_C(n)
        data = mcmc.Data(g - g[0], C, Ne=1000.0)
        paths = fit['paths']
        r = {'n_per_year_assumed': n, 'posterior': _summ(fit), 'discrepancy': _disc(fit['disc']),
             'carbonaria_allele_freq_first_last': [float(np.nanmedian(paths[:, 0])), float(np.nanmedian(paths[:, -1]))]}
        if tag == 'h0_n100_NeInf':
            _attr_fig(fit['att'], np.arange(g[0], g[-1] + 1) + int(df.year.values[0]),
                      'Biston betularia (phenotype data)', out / 'moth_attribution.png')
            r['closure_test'] = _closure(pool, fit, data, nrep(200, cfg, 8), 7, 'pheno_adult', cfg,
                                         'moth bootstrap', C.sum(1), fix_h0=True)
        res[tag] = r
    return res


def cross_sectional(cfg, out):
    """Single-sample F. Below F_min(p) = -min(p,q)/max(p,q) the zygote vector is clipped
    (no negative genotype frequencies), so every F below the bound gives the same
    likelihood: F itself is not identified there. The identified quantity is the
    effective inbreeding coefficient F_eff = max(F, F_min(p)) = D^z/(pq), reported
    per posterior draw (p from the posterior allele frequency). `at_bound` marks samples
    with no rare homozygotes, where F_IS sits on the bound."""
    res = {}
    for name in ['ccr5', 'hbc', 'zoarces', 'arabidopsis']:
        df = pd.read_csv(ROOT / 'data' / f'{name}.csv')
        for _, row in df.iterrows():
            C = np.array([[row.N_AA, row.N_Aa, row.N_aa]], float)
            if name == 'arabidopsis' and C.sum() > 50000:
                C = np.round(C * 50000 / C.sum())
            n = C.sum(); p = (2 * C[0, 0] + C[0, 1]) / (2 * n)
            fis_obs = 1 - C[0, 1] / (n * 2 * p * (1 - p)) if 0 < p < 1 else np.nan
            data = mcmc.Data([0], C, Ne=1000.0)
            Fhi = 0.9999 if name == 'arabidopsis' else 0.8
            cfgm = mcmc.make_cfg(infer_s=False, infer_h=False, s_fixed=0.0, F_hi=Fhi, F_lo=-0.5)
            mc = mcmc.run_mcmc(data, cfgm, seed=3, n_chains=4, n_steps=6000, n_warmup=2000)
            th, paths = mcmc.posterior_paths(mc, data, n_draws=2000, seed=3)
            pp = paths[:, 0]; qq = 1 - pp
            Fmin = -np.minimum(pp, qq) / np.maximum(np.maximum(pp, qq), 1e-12)
            Feff = np.clip(th[:, 0], Fmin, 1.0)
            key = name if name != 'ccr5' else f"ccr5_{row.population}"
            S = mc['summary']['F']
            res[key] = {'n': int(n), 'p_hat': float(p), 'F_IS_observed': float(fis_obs),
                        'F_min_at_p_hat': float(-min(p, 1 - p) / max(p, 1 - p)),
                        'at_bound': bool(min(C[0, 0], C[0, 2]) == 0),
                        'F_eff_median': float(np.median(Feff)), 'F_eff_q05': float(np.percentile(Feff, 5)),
                        'F_eff_q95': float(np.percentile(Feff, 95)),
                        'F_median_raw': S['median'], 'F_q05_raw': S['q05'], 'F_q95_raw': S['q95'], 'rhat': S['rhat'],
                        'frac_draws_below_bound': float(np.mean(th[:, 0] < Fmin))}
    return res


def run(cfg, pool):
    out = outdir(cfg, 'e09_empirical')
    res = {}
    print('  empirical fits (parallel)'); fits = empirical_fits(cfg, pool)
    print('  Panaxia'); res['panaxia'] = panaxia(cfg, pool, out, fits)
    print('  peppered moth'); res['peppered_moth'] = moth(cfg, pool, out, fits)
    print('  cross-sectional datasets'); res['cross_sectional'] = cross_sectional(cfg, out)
    save_json(res, out / 'empirical_summary.json')
    return res
