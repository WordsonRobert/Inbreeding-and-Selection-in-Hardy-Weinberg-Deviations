"""E07 closed-model discrepancy: detection of immigration, specificity, localisation.

Scores are computed from the data only, never from the hidden trajectory:
  score_jump  standardised posterior (smoothed) drift per inter-observation interval
  score_pred  one-step-ahead predictive residual of each sample given the earlier ones
  score_D     posterior-predictive z-score of the observed HW deviation
  score_comb  hypot(score_pred, score_D)
The event is localised to the interval with the largest predictive residual.
Migration persists (migrants join the zygote pool; their alleles stay).
Negatives: closed populations under the model, closed individual-based populations,
and closed populations with other violations (specificity).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import outdir, save_json, write_csv, nrep, mcmc_kw, pmap, read_rows, auc, COL
from worker import fit_one


SCORES = ['score_jump', 'score_pred', 'score_D', 'score_comb']


def _base(rng):
    p0 = rng.uniform(0.2, 0.8); F = rng.uniform(0, 0.3)
    s = rng.choice([-1, 1]) * rng.uniform(0.0, 0.15); h = rng.uniform(0, 1)
    T = 5 * int(rng.integers(6, 13))
    return p0, F, s, h, T


def run(cfg, pool):
    out = outdir(cfg, 'e07_migration')
    rng = np.random.default_rng(71)
    specs = []; i = 0
    m_vals = [0.0, 0.02, 0.05, 0.10, 0.20]
    for ne_mode in ['known', 'inferred']:
        n_each = 400 if ne_mode == 'known' else 150
        for m in (m_vals if ne_mode == 'known' else [0.0, 0.05, 0.10]):
            for _ in range(nrep(n_each, cfg, 8)):
                p0, F, s, h, T = _base(rng)
                T_mig = int(rng.integers(T // 4, 3 * T // 4))
                p_mig = rng.uniform(0.05, 0.95)
                specs.append(dict(id=i, seed=int(rng.integers(1, 2 ** 31 - 1)), cond=f'm{m}_Ne{ne_mode}', cond_val=m,
                                  gen=dict(p0=p0, F=F, s=s, h=h, T=T, Ne=1000, mode=0, m_mig=m,
                                           T_mig=T_mig if m > 0 else -1, p_mig=p_mig),
                                  obs=dict(gens=list(range(0, T + 1, 5)), n=300),
                                  inf=dict(infer_Ne=(ne_mode == 'inferred'), Ne_fixed=1000.0),
                                  truth=dict(F=F, s=s, h=h), mk=mcmc_kw(cfg), n_paths=150, disc=True))
                i += 1
    # other closed-population violations (specificity), Ne known
    for c in ['finite_pop', 'changing_s', 'bottleneck', 'linked_selection', 'wahlund_sampling', 'genotyping_error']:
        for _ in range(nrep(200, cfg, 6)):
            p0, F, s, h, T = _base(rng)
            gen = dict(p0=p0, F=F, s=s, h=h, T=T, Ne=1000, mode=0)
            obs = dict(gens=list(range(0, T + 1, 5)), n=300)
            if c == 'finite_pop':
                gen.update(mode=1, sigma=0.0, F=0.0)
            elif c == 'changing_s':
                gen.update(s=np.sign(s) * max(abs(s), 0.05), s2=-np.sign(s) * max(abs(s), 0.05), T_switch=T // 2)
            elif c == 'bottleneck':
                Ne_arr = np.full(T + 1, 1000.0); Ne_arr[T // 3: 2 * T // 3] = 50.0; gen['Ne_arr'] = Ne_arr
            elif c == 'linked_selection':
                gen.update(s=0.0, hh_s=rng.choice([-1, 1]) * rng.uniform(0.1, 0.3), hh_D0=rng.uniform(0.05, 0.2),
                           hh_r=0.1, hh_p0=rng.uniform(0.3, 0.7))
            elif c == 'wahlund_sampling':
                obs['wahlund_d'] = 0.15
            elif c == 'genotyping_error':
                obs['eps'] = 0.03
            specs.append(dict(id=i, seed=int(rng.integers(1, 2 ** 31 - 1)), cond=f'neg_{c}', cond_val=0.0,
                              gen=gen, obs=obs, inf=dict(infer_Ne=False, Ne_fixed=1000.0),
                              truth=dict(F=gen['F'], s=gen['s'], h=h), mk=mcmc_kw(cfg), n_paths=150, disc=True))
            i += 1
    rows = pmap(pool, fit_one, specs, 'migration')
    write_csv(rows, out / 'migration.csv')
    res = {}
    for ne_mode in ['known', 'inferred']:
        neg = [r for r in rows if r['cond'] == f'm0.0_Ne{ne_mode}']
        if not neg:
            continue
        thr = {k: float(np.nanpercentile(read_rows(neg, k), 95)) for k in SCORES}
        R = {'n_closed': len(neg), 'threshold_5pct_FPR': thr}
        for m in m_vals[1:]:
            pos = [r for r in rows if r['cond'] == f'm{m}_Ne{ne_mode}']
            if not pos:
                continue
            d = {'n': len(pos)}
            for k in SCORES:
                d[f'AUC_{k}'] = auc(read_rows(neg, k), read_rows(pos, k))
                d[f'TPR_at_5pctFPR_{k}'] = float(np.nanmean(read_rows(pos, k) > thr[k]))
            hit = read_rows(pos, 'loc_hit'); err = read_rows(pos, 'loc_err')
            flagged = read_rows(pos, 'score_comb') > thr['score_comb']
            d['loc_interval_hit_rate'] = float(np.nanmean(hit))
            d['loc_interval_hit_rate_if_flagged'] = float(np.nanmean(hit[flagged])) if flagged.any() else None
            d['loc_err_median_gens'] = float(np.nanmedian(err))
            d['loc_interval_hit_rate_smoothed_jump'] = float(np.nanmean(read_rows(pos, 'loc_hit_jump')))
            R[f'm={m}'] = d
        # detection as a function of the size of the allele-frequency shift the pulse caused
        allpos = [r for r in rows if r['cond'].startswith('m') and r['cond'] != f'm0.0_Ne{ne_mode}'
                  and r['cond'].endswith(f'Ne{ne_mode}')]
        sh = read_rows(allpos, 'mig_shift'); sc = read_rows(allpos, 'score_comb')
        bins = [0, 0.01, 0.02, 0.05, 0.1, 1.0]
        R['TPR_at_5pctFPR_by_shift'] = {
            f'{bins[b]}-{bins[b + 1]}': {'n': int(((sh >= bins[b]) & (sh < bins[b + 1])).sum()),
                                         'TPR_comb': float(np.nanmean(sc[(sh >= bins[b]) & (sh < bins[b + 1])] > thr['score_comb']))
                                         if ((sh >= bins[b]) & (sh < bins[b + 1])).any() else None}
            for b in range(len(bins) - 1)}
        if ne_mode == 'known':
            spec_ = {}
            for c in ['finite_pop', 'changing_s', 'bottleneck', 'linked_selection', 'wahlund_sampling', 'genotyping_error']:
                Rc = [r for r in rows if r['cond'] == f'neg_{c}']
                if Rc:
                    spec_[c] = {k: float(np.nanmean(read_rows(Rc, k) > thr[k])) for k in thr}
            R['flag_rate_other_violations'] = spec_
        res[f'Ne_{ne_mode}'] = R
    save_json(res, out / 'migration_summary.json')
    # ROC figure (Ne known)
    neg = [r for r in rows if r['cond'] == 'm0.0_Neknown']
    if neg:
        fig, ax = plt.subplots(figsize=(4.2, 4))
        for m, col in zip(m_vals[1:], ['#fdae61', '#f46d43', '#d73027', '#a50026']):
            pos = [r for r in rows if r['cond'] == f'm{m}_Neknown']
            if not pos:
                continue
            sn = read_rows(neg, 'score_comb'); sp = read_rows(pos, 'score_comb')
            th = np.unique(np.concatenate([sn, sp]))[::-1]
            tpr = [np.mean(sp >= t) for t in th]; fpr = [np.mean(sn >= t) for t in th]
            ax.plot([0] + fpr + [1], [0] + tpr + [1], color=col, label=f'm = {m:.0%}  AUC {auc(sn, sp):.2f}')
        ax.plot([0, 1], [0, 1], color='0.7', lw=0.8, ls=':')
        ax.set_xlabel('false-positive rate'); ax.set_ylabel('true-positive rate')
        ax.set_title('Closed-model discrepancy (data only)', fontsize=9)
        ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(); fig.savefig(out / 'migration_roc.png', dpi=200); plt.close(fig)
    return res
