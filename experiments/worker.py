"""Generic simulate -> fit -> evaluate worker used by E03, E04, E05, E07."""
import numpy as np
from common import mcmc, sims, model, fit_summary, safe, cover

PARAMS = ['F', 's', 'h', 'hs', 's_eff', 'log10Ne']


def _truth(spec):
    t = dict(spec['truth'])
    if 'hs' not in t:
        t['hs'] = t['h'] * t['s']
    if 's_eff' not in t:
        t['s_eff'] = t['s'] * (t['F'] + (1 - t['F']) * t['h'])
    return t


@safe
def fit_one(spec):
    rng = np.random.default_rng(spec['seed'])
    g = spec['gen']
    P, X, A = sims.simulate(**g, seed=int(spec['seed']))
    o = spec['obs']
    gens = np.asarray(o['gens'], int)
    C = sims.observe(P, X, A, gens, o['n'], rng, stage=o.get('stage', 'adult'), pheno=o.get('pheno', False),
                     wahlund_d=o.get('wahlund_d', 0.0), F=g['F'], s=g['s'], h=g['h'], eps=o.get('eps', 0.0))
    inf = dict(spec['inf'])
    Ne_arr = inf.pop('Ne_arr', None)
    Ne_fixed = inf.pop('Ne_fixed', 1000.0)
    data = mcmc.Data(gens, C, Ne_arr=Ne_arr, Ne=Ne_fixed)
    cfg = mcmc.make_cfg(**inf)
    mk = dict(spec['mk'])
    if inf.get('infer_Ne') and mk.get('n_chains', 4) >= 4:
        # Ne-inferred posteriors are wide and mix more slowly: 50% longer chains
        mk.update(n_steps=int(mk['n_steps'] * 1.5), n_warmup=int(mk['n_warmup'] * 1.33))
    mc = mcmc.run_mcmc(data, cfg, seed=int(spec['seed']) % 100000, **mk)
    truth = _truth(spec)
    row = {'id': spec['id'], 'cond': spec.get('cond', ''), 'cond_val': spec.get('cond_val', ''),
           'T': int(gens[-1]), 'n_obs': len(gens), 'acc': float(np.mean(mc['accept']))}
    row.update(fit_summary(mc, truth))
    # SBC ranks against L thinned draws
    L = spec.get('sbc_L', 0)
    if L:
        flat = mc['samples'].reshape(-1, mc['samples'].shape[-1])
        idx = np.linspace(0, len(flat) - 1, L).astype(int)
        for k in ['F', 's', 'h', 'hs', 'log10Ne']:
            if k in truth and k in mc['summary']:
                i = mc['names'].index(k)
                row[f'{k}_rank'] = int((flat[idx, i] < truth[k]).sum())
    if spec.get('paths', True):
        th, paths = mcmc.posterior_paths(mc, data, n_draws=spec.get('n_paths', 150), seed=int(spec['seed']))
        p0d = paths[:, 0]
        row.update(p0_true=float(P[0]), p0_med=float(np.nanmedian(p0d)),
                   p0_cov90=cover(np.nanpercentile(p0d, 5), np.nanpercentile(p0d, 95), P[0]))
        # trajectory coverage of the true zygote allele frequency at every generation
        lo = np.nanpercentile(paths, 5, 0); hi = np.nanpercentile(paths, 95, 0)
        row['path_cov90'] = float(np.mean((P >= lo - 1e-9) & (P <= hi + 1e-9)))
        row['path_rmse'] = float(np.sqrt(np.mean((np.nanmean(paths, 0) - P) ** 2)))
        att = mcmc.attribution(mc, data, th, paths)
        Ttrue = truth.copy()
        Dz, Da, DS, phi, seq, tot = model.attribute_path(P[:data.T + 1], Ttrue['F'], g['s'], g['h'], 1e-5, 1e-5)
        cum_true = phi.sum(0)
        cum_post = np.nansum(att['phi'], 1)          # (draws, 5)
        for k, nm in enumerate(model.OP_NAMES):
            lo_, hi_ = np.nanpercentile(cum_post[:, k], [5, 95])
            row[f'cum_{nm}_true'] = float(cum_true[k]); row[f'cum_{nm}_med'] = float(np.nanmedian(cum_post[:, k]))
            row[f'cum_{nm}_cov90'] = cover(lo_, hi_, cum_true[k])
        tl = data.T
        sh_true = DS[tl] / (abs(Dz[tl]) + abs(DS[tl]) + 1e-300)
        shp = att['share_sel'][:, tl]
        lo_, hi_ = np.nanpercentile(shp, [5, 95])
        # the share is degenerate (+-1) when the true inbreeding part is ~0 (F = 0): no coverage then
        sh_ok = abs(Dz[tl]) > 1e-4
        row.update(selshare_true=float(sh_true), selshare_med=float(np.nanmedian(shp)),
                   selshare_cov90=cover(lo_, hi_, sh_true) if sh_ok else np.nan,
                   Dsel_true=float(DS[tl]), Dsel_med=float(np.nanmedian(att['DS'][:, tl])),
                   Dinb_true=float(Dz[tl]), Dinb_med=float(np.nanmedian(att['Dz'][:, tl])))
        for key, M, tv in [('Dsel', att['DS'][:, tl], DS[tl]), ('Dinb', att['Dz'][:, tl], Dz[tl])]:
            lo_, hi_ = np.nanpercentile(M, [5, 95])
            row[f'{key}_cov90'] = cover(lo_, hi_, tv)
        if spec.get('disc', False):
            d = mcmc.discrepancy(mc, data, th, paths, seed=int(spec['seed']))
            row.update(score_jump=d['score_jump'], score_pred=d['score_pred'], score_D=d['score_D'],
                       score_comb=d['score_comb'], jump_interval=d['jump_interval'], pred_interval=d['pred_interval'])
            Tm = g.get('T_mig', -1)
            if Tm >= 0 and g.get('m_mig', 0) > 0:
                mm = g['m_mig']
                row['mig_shift'] = float(abs(mm * (g['p_mig'] - P[Tm]) / (1 - mm)))   # |Δp| caused by the pulse
            # migrants enter as zygotes at T_mig: the shift appears between the last
            # observation before T_mig and the first at/after it
            for key, ik in [('', 'pred_interval'), ('_jump', 'jump_interval')]:
                ji = d[ik]
                if ji >= 0:
                    a, b = gens[ji], gens[ji + 1]
                    row[f'loc_lo{key}'] = int(a); row[f'loc_hi{key}'] = int(b)
                    if Tm >= 0:
                        row[f'loc_hit{key}'] = int(a < Tm <= b)
                        row[f'loc_err{key}'] = 0 if a < Tm <= b else int(min(abs(Tm - a), abs(Tm - b)))
    return row


@safe
def emp_fit(spec):
    """One empirical fit: MCMC, posterior paths, attribution, discrepancy."""
    data = mcmc.Data(spec['gens'], np.asarray(spec['C'], float), Ne_arr=spec.get('Ne_arr'),
                     Ne=spec.get('Ne', 1000.0))
    mc = mcmc.run_mcmc(data, mcmc.make_cfg(**spec['inf']), seed=spec['seed'], **spec['mk'])
    th, paths = mcmc.posterior_paths(mc, data, n_draws=spec.get('n_paths', 400), seed=spec['seed'])
    att = mcmc.attribution(mc, data, th, paths)
    disc = mcmc.discrepancy(mc, data, th, paths, seed=spec['seed'])
    flat = mc['samples'].reshape(-1, mc['samples'].shape[-1])
    return dict(tag=spec['tag'], summary=mc['summary'], converged=mc['converged'], accept=mc['accept'],
                n_pruned=mc['n_pruned'], K=mc['K'], flat_thin=flat[::10], th=th, paths=paths,
                att={k: att[k] for k in ['Dz', 'DS', 'Da', 'phi', 'share_sel']}, disc=disc)


def probe(seed=1):
    """Short fixed MCMC used to measure this machine's speed (single and in parallel)."""
    import time as _t
    t = _t.time()
    P, X, A = sims.simulate(0.4, 0.1, 0.06, 0.5, 40, Ne=1000, seed=3)
    gens = list(range(0, 41, 5))
    C = sims.observe(P, X, A, gens, 300, np.random.default_rng(3))
    d = mcmc.Data(gens, C, Ne=1000.0)
    mcmc.run_mcmc(d, mcmc.make_cfg(), n_chains=2, n_steps=800, n_warmup=300, seed=int(seed))
    return _t.time() - t


def cpu_burn(deadline):
    """Spin until the wall-clock deadline; return the CPU seconds this process received."""
    import time as _t
    c0 = _t.process_time()
    x = 0.0
    while _t.time() < deadline:
        for i in range(20000):
            x += i * 1e-9
    return _t.process_time() - c0
