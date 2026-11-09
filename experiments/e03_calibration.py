"""E03 simulation-based calibration, E04 robustness (individual-based populations),
E05 stress tests, E05b failure map."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import outdir, save_json, write_csv, nrep, mcmc_kw, pmap, read_rows, prior_sd, COL, model
from worker import fit_one


def _prior_draw(rng, infer_Ne):
    while True:
        F = rng.uniform(-0.2, 0.8)
        s = rng.normal(0, 0.15)
        h = rng.normal(0.5, 0.5)
        if abs(s) < 0.9 and 1 - h * s > 0.01 and 1 - s > 0.01:
            break
    p0 = rng.beta(2, 2)
    lne = rng.uniform(1.5, 4.5) if infer_Ne else rng.uniform(2.0, 4.0)
    T = 5 * int(rng.integers(4, 13))
    return p0, F, s, h, lne, T


def _metrics(rows, keys=('F', 's', 'h', 'hs', 's_eff', 'log10Ne')):
    out = {'n': len(rows), 'converged': float(np.nanmean(read_rows(rows, 'converged'))),
           'converged_F_hs': float(np.nanmean(read_rows(rows, 'converged_F_hs')))}
    psd = prior_sd()
    for k in keys:
        cv = read_rows(rows, f'{k}_cov90')
        if np.all(np.isnan(cv)):
            continue
        tr = read_rows(rows, f'{k}_true'); md = read_rows(rows, f'{k}_med'); sd = read_rows(rows, f'{k}_sd')
        d = {'cov90': float(np.nanmean(cv)), 'cov50': float(np.nanmean(read_rows(rows, f'{k}_cov50'))),
             'bias_median': float(np.nanmedian(md - tr)), 'rmse': float(np.sqrt(np.nanmean((md - tr) ** 2))),
             'post_sd_median': float(np.nanmedian(sd))}
        if k in psd:
            d['contraction_median'] = float(np.nanmedian(sd) / psd[k])
        out[k] = d
    for k in ['p0', 'path', 'selshare', 'Dsel', 'Dinb'] + [f'cum_{n}' for n in model.OP_NAMES]:
        key = f'{k}_cov90'
        v = read_rows(rows, key)
        if not np.all(np.isnan(v)):
            out[key] = float(np.nanmean(v))
    return out


def sbc(cfg, pool):
    out = outdir(cfg, 'e03_sbc')
    res = {}
    for infer_Ne, n_full in [(False, 1000), (True, 600)]:
        rng = np.random.default_rng(31 + int(infer_Ne))
        specs = []
        for i in range(nrep(n_full, cfg, 12)):
            p0, F, s, h, lne, T = _prior_draw(rng, infer_Ne)
            Ne = 10 ** lne
            specs.append(dict(id=i, seed=int(rng.integers(1, 2 ** 31 - 1)),
                              gen=dict(p0=p0, F=F, s=s, h=h, T=T, Ne=Ne, mode=0),
                              obs=dict(gens=list(range(0, T + 1, 5)), n=300),
                              inf=dict(infer_Ne=infer_Ne, Ne_fixed=Ne),
                              truth=dict(F=F, s=s, h=h, **({'log10Ne': lne} if infer_Ne else {})),
                              mk=mcmc_kw(cfg), sbc_L=99, n_paths=150))
        tag = 'Ne_inferred' if infer_Ne else 'Ne_known'
        rows = pmap(pool, fit_one, specs, f'SBC {tag}')
        write_csv(rows, out / f'sbc_{tag}.csv')
        m = _metrics(rows)
        # rank-uniformity chi-square (10 bins over ranks 0..99)
        from scipy.stats import chisquare
        for k in ['F', 's', 'h', 'hs', 'log10Ne']:
            r = read_rows(rows, f'{k}_rank')
            r = r[np.isfinite(r)]
            if len(r) >= 20:
                cnt = np.histogram(r, bins=10, range=(0, 100))[0]
                m.setdefault(k, {})['rank_chisq_p'] = float(chisquare(cnt).pvalue)
        res[tag] = m
        # rank histograms
        ks = [k for k in ['F', 's', 'h', 'hs', 'log10Ne'] if np.isfinite(read_rows(rows, f'{k}_rank')).any()]
        fig, axes = plt.subplots(1, len(ks), figsize=(2.6 * len(ks), 2.6))
        for ax, k in zip(np.atleast_1d(axes), ks):
            r = read_rows(rows, f'{k}_rank'); r = r[np.isfinite(r)]
            ax.hist(r, bins=10, range=(0, 100), color='0.6', edgecolor='white')
            ax.axhline(len(r) / 10, color='k', lw=0.8)
            ax.set_title(k, fontsize=9); ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.suptitle(f'SBC rank histograms ({tag})', fontsize=10)
        fig.tight_layout(); fig.savefig(out / f'sbc_ranks_{tag}.png', dpi=200); plt.close(fig)
    save_json(res, out / 'sbc_summary.json')
    return res


def robustness(cfg, pool):
    """Individual-based populations: real finite-population drift, partial selfing."""
    out = outdir(cfg, 'e04_robustness')
    rng = np.random.default_rng(41)
    specs = []
    cells = [(N, sig, 'known') for N in [100, 500, 2000] for sig in [0.0, 0.2, 0.5]]
    cells += [(500, sig, 'inferred') for sig in [0.0, 0.2, 0.5]]
    cells += [(500, 0.2, 'half'), (500, 0.2, 'double')]
    i = 0
    for N, sig, neinf in cells:
        for _ in range(nrep(120, cfg, 6)):
            p0 = rng.uniform(0.15, 0.85); s = rng.choice([-1, 1]) * rng.uniform(0.02, 0.15); h = rng.uniform(0, 1)
            Feq = sig / (2 - sig)
            Ne_inf = {'known': N, 'inferred': N, 'half': N / 2, 'double': 2 * N}[neinf]
            specs.append(dict(id=i, seed=int(rng.integers(1, 2 ** 31 - 1)), cond=f'N{N}_sigma{sig}_Ne{neinf}',
                              gen=dict(p0=p0, F=Feq, s=s, h=h, T=40, Ne=N, mode=1, sigma=sig),
                              obs=dict(gens=list(range(0, 41, 5)), n=300),
                              inf=dict(infer_Ne=(neinf == 'inferred'), Ne_fixed=Ne_inf),
                              truth=dict(F=Feq, s=s, h=h), mk=mcmc_kw(cfg), n_paths=100))
            i += 1
    rows = pmap(pool, fit_one, specs, 'robustness')
    write_csv(rows, out / 'robustness.csv')
    res = {}
    for c in sorted(set(r['cond'] for r in rows)):
        res[c] = _metrics([r for r in rows if r['cond'] == c], keys=('F', 's', 'h', 'hs', 's_eff'))
    save_json(res, out / 'robustness_summary.json')
    return res


def violations(cfg, pool):
    out = outdir(cfg, 'e05_violations')
    rng = np.random.default_rng(51)
    T = 50; gens = list(range(0, T + 1, 5))
    conds = ['baseline', 'changing_s', 'bottleneck_Ne_fixed', 'bottleneck_Ne_inferred', 'linked_selection',
             'wahlund_sampling', 'genotyping_error', 'zygote_sampled', 'selfing_transient', 'finite_pop']
    specs = []; i = 0
    for c in conds:
        for _ in range(nrep(200, cfg, 6)):
            p0 = rng.uniform(0.2, 0.8); F = rng.uniform(0, 0.3); s = rng.choice([-1, 1]) * rng.uniform(0.03, 0.15)
            h = rng.uniform(0, 1)
            gen = dict(p0=p0, F=F, s=s, h=h, T=T, Ne=1000, mode=0)
            obs = dict(gens=gens, n=300)
            inf = dict(infer_Ne=False, Ne_fixed=1000.0)
            truth = dict(F=F, s=s, h=h)
            if c == 'changing_s':
                gen.update(s2=-s, T_switch=T // 2)
            elif c.startswith('bottleneck'):
                Ne_arr = np.full(T + 1, 1000.0); Ne_arr[T // 3: 2 * T // 3] = 50.0
                gen['Ne_arr'] = Ne_arr
                inf['infer_Ne'] = c.endswith('inferred')
            elif c == 'linked_selection':
                gen.update(s=0.0, hh_s=rng.choice([-1, 1]) * rng.uniform(0.1, 0.3), hh_D0=rng.uniform(0.05, 0.2),
                           hh_r=0.1, hh_p0=rng.uniform(0.3, 0.7))
                truth.update(s=0.0)
            elif c == 'wahlund_sampling':
                obs['wahlund_d'] = 0.15
            elif c == 'genotyping_error':
                obs['eps'] = 0.03
            elif c == 'zygote_sampled':
                obs['stage'] = 'zygote'
            elif c == 'selfing_transient':
                gen.update(F=0.0, mode=1, sigma=0.3, Ne=1000)
                truth.update(F=0.3 / 1.7)            # equilibrium F_IS; data start at F = 0
            elif c == 'finite_pop':
                gen.update(mode=1, sigma=0.0, F=0.0, Ne=1000)
                truth.update(F=0.0)
            specs.append(dict(id=i, seed=int(rng.integers(1, 2 ** 31 - 1)), cond=c, gen=gen, obs=obs, inf=inf,
                              truth=truth, mk=mcmc_kw(cfg), n_paths=100, disc=True,
                              s1=s, s2=gen.get('s2', s)))
            i += 1
    rows = pmap(pool, fit_one, specs, 'violations')
    write_csv(rows, out / 'violations.csv')
    res = {}
    for c in conds:
        R = [r for r in rows if r['cond'] == c]
        m = _metrics(R, keys=('F', 's', 'h', 'hs', 's_eff'))
        q05 = read_rows(R, 's_q05'); q95 = read_rows(R, 's_q95')
        m['frac_s_interval_excludes_0'] = float(np.nanmean((q05 > 0) | (q95 < 0)))
        q05 = read_rows(R, 'hs_q05'); q95 = read_rows(R, 'hs_q95')
        m['frac_hs_interval_excludes_0'] = float(np.nanmean((q05 > 0) | (q95 < 0)))
        m['median_score_jump'] = float(np.nanmedian(read_rows(R, 'score_jump')))
        m['median_score_pred'] = float(np.nanmedian(read_rows(R, 'score_pred')))
        m['median_score_D'] = float(np.nanmedian(read_rows(R, 'score_D')))
        res[c] = m
    save_json(res, out / 'violations_summary.json')
    # figure: coverage of s, hs, F by condition
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(conds)); w = 0.26
    for j, (k, col) in enumerate([('s', COL['selection']), ('hs', '#f4a582'), ('F', COL['inbreeding'])]):
        ax.bar(x + (j - 1) * w, [res[c].get(k, {}).get('cov90', np.nan) for c in conds], w, color=col, label=k)
    ax.axhline(0.9, color='k', lw=0.8, ls='--')
    ax.set_xticks(x); ax.set_xticklabels([c.replace('_', '\n') for c in conds], fontsize=7)
    ax.set_ylabel('90% interval coverage'); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'violations_coverage.png', dpi=200); plt.close(fig)
    return res


def failure_map(cfg, pool):
    out = outdir(cfg, 'e05_failure_map')
    rng = np.random.default_rng(55)
    base = dict(p0=0.4, F=0.1, s=0.1, h=0.5, Ne=1000, T=30, n=300)
    sweeps = {'p0': [0.02, 0.05, 0.1, 0.2, 0.4, 0.5], 'F': [0.0, 0.1, 0.2, 0.4, 0.6, 0.75],
              'n': [10, 20, 50, 100, 300, 1000], 'T': [5, 10, 15, 20, 30, 60],
              's': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2], 'Ne': [20, 50, 100, 250, 1000, 5000]}
    specs = []; i = 0
    for k, vals in sweeps.items():
        for v in vals:
            for _ in range(nrep(40, cfg, 4)):
                b = dict(base); b[k] = v
                specs.append(dict(id=i, seed=int(rng.integers(1, 2 ** 31 - 1)), cond=k, cond_val=v,
                                  gen=dict(p0=b['p0'], F=b['F'], s=b['s'], h=b['h'], T=b['T'], Ne=b['Ne'], mode=0),
                                  obs=dict(gens=list(range(0, b['T'] + 1, 5)), n=b['n']),
                                  inf=dict(infer_Ne=False, Ne_fixed=float(b['Ne'])),
                                  truth=dict(F=b['F'], s=b['s'], h=b['h']), mk=mcmc_kw(cfg), n_paths=60))
                i += 1
    rows = pmap(pool, fit_one, specs, 'failure map')
    write_csv(rows, out / 'failure_map.csv')
    res = {}
    for k, vals in sweeps.items():
        res[k] = {}
        for v in vals:
            R = [r for r in rows if r['cond'] == k and float(r['cond_val']) == float(v)]
            if R:
                m = _metrics(R, keys=('F', 's', 'hs'))
                res[k][str(v)] = {'cov_s': m.get('s', {}).get('cov90'), 'cov_hs': m.get('hs', {}).get('cov90'),
                                  'cov_F': m.get('F', {}).get('cov90'), 'converged': m['converged'],
                                  'sd_s': m.get('s', {}).get('post_sd_median'),
                                  'sd_F': m.get('F', {}).get('post_sd_median')}
    save_json(res, out / 'failure_map_summary.json')
    fig, axes = plt.subplots(3, len(sweeps), figsize=(15, 6.5), sharey='row')
    for j, (k, vals) in enumerate(sweeps.items()):
        for r_, (met, lab) in enumerate([('cov_s', 'coverage s'), ('cov_F', 'coverage F'), ('converged', 'converged')]):
            ax = axes[r_, j]
            y = [res[k].get(str(v), {}).get(met) for v in vals]
            y = [np.nan if q is None else q for q in y]
            ax.bar(range(len(vals)), y, color=[COL['selection'], COL['inbreeding'], '0.5'][r_])
            ax.axhline(0.9 if r_ < 2 else 1.0, color='k', lw=0.6, ls='--')
            ax.set_xticks(range(len(vals))); ax.set_xticklabels([str(v) for v in vals], fontsize=7, rotation=45)
            ax.set_ylim(0, 1.05)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if r_ == 0:
                ax.set_title(k, fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Failure map (one parameter varied at a time; base p0=0.4, F=0.1, s=0.1, h=0.5, Ne=1000, T=30, n=300)',
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'failure_map.png', dpi=200); plt.close(fig)
    return res


def run(cfg, pool, which=('sbc', 'robustness', 'violations', 'failure_map')):
    res = {}
    if 'sbc' in which:
        print('  E03 simulation-based calibration'); res['sbc'] = sbc(cfg, pool)
    if 'robustness' in which:
        print('  E04 robustness'); res['robustness'] = robustness(cfg, pool)
    if 'violations' in which:
        print('  E05 stress tests'); res['violations'] = violations(cfg, pool)
    if 'failure_map' in which:
        print('  E05b failure map'); res['failure_map'] = failure_map(cfg, pool)
    return res
