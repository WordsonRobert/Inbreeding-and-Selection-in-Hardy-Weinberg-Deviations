"""Collect every experiment's summary JSON into results/SUMMARY.md."""
import json
from pathlib import Path

import numpy as np


def _load(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def _f(x, d=3):
    if x is None:
        return '–'
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return f'{float(x):.{d}f}'
    except Exception:  # noqa: BLE001
        return str(x)


def write(out: Path):
    L = ['# BASTION — run summary', '']
    meta = _load(out / 'run_meta.json')
    if meta:
        m = meta['meta']
        L += [f"mode **{m['mode']}**, {m['workers']} workers, {m.get('total_s', '?')} s total; "
              f"started {m['start']}", '', '| stage | seconds |', '|---|---|']
        L += [f'| {k} | {v} |' for k, v in meta['timing_s'].items()] + ['']

    th = _load(out / 'e01_theory' / 'theory_summary.json')
    if th:
        d = th['drift_vs_FIS']
        L += ['## E01 Theory checks', '',
              f"Random mating, N = {d['N']} ({d['reps']} replicates): generation {d['gens'][-1]} — Wright's F "
              f"{_f(d['wright_F'][-1])}, 1−H/H0 {_f(d['one_minus_H_over_H0'][-1])}, within-population F_IS "
              f"{_f(d['F_IS_mean'][-1], 4)} (expected −1/2N = {_f(d['expected_F_IS'], 4)}).", '']
        i = th['identities']
        L += ['| identity | max abs error |', '|---|---|',
              f"| closed-form adult D | {i['closed_form_Da_max_err']:.1e} |",
              f"| D^a = (w3 D^z + κ (x2/2)^2)/w̄² | {i['kappa_identity_max_err']:.1e} |",
              f"| genotype-level mutation map | {i['mutation_map_max_err']:.1e} |",
              f"| Shapley efficiency (Σφ = ΔD) | {i['shapley_efficiency_max_err']:.1e} |",
              f"| dummy: s = 0 ⇒ φ_selection = 0 | {i['dummy_selection_max_abs']:.1e} |", '',
              'Rare-allele effective selection s_eff = s[F + (1−F)h] (median relative error by q): ' +
              ', '.join(f"q={r['q']}: {r['median_rel_err']:.2g}" for r in i['rare_allele_seff']), '']
        idf = th['identifiability']
        L += ['## E02 Identifiability (Fisher information)', '',
              f"Cross-sectional samples: max FIM rank {idf['cross_sectional_rank_max']} of 4. "
              f"Temporal designs full rank in {_f(idf['temporal_full_rank_fraction'])} of cases.", '',
              '| data, T=30 n=300, s>0 (rare: min(p0, 1-p0) <= 0.05) | CRLB s | CRLB h | CRLB hs | CRLB s_eff | CRLB F | corr(s,h) |',
              '|---|---|---|---|---|---|---|']
        for k, v in idf.items():
            if isinstance(v, dict):
                L.append(f"| {k} | {_f(v['median_crlb_s'])} | {_f(v['median_crlb_h'])} | {_f(v['median_crlb_hs'])} | "
                         f"{_f(v['median_crlb_seff'])} | {_f(v['median_crlb_F'])} | {_f(v['median_corr_s_h'])} |")
        o = th['order_dependence']
        L += ['', '## E06 Attribution vs operator order', '',
              f"{o['n_generations']} generations. Lead force differs between life-cycle sequential attribution and "
              f"Shapley in {_f(o['frac_lead_force_differs_lifecycle_vs_shapley'])} of generations; differs for at least "
              f"one of the 120 orderings in {_f(o['frac_lead_force_differs_some_ordering'])}.",
              'Median range of each operator\'s contribution across orderings (relative to Σ|φ|): ' +
              ', '.join(f'{k} {v}' for k, v in o['median_range_over_orderings_by_op'].items()), '',
              f"Selection share when F < 0.05 — legacy |α_s| "
              f"{_f(o['legacy_vs_v3_selection_share_when_F_lt_0.05']['legacy_mean_abs_alpha_s'])} vs v3 "
              f"{_f(o['legacy_vs_v3_selection_share_when_F_lt_0.05']['v3_selection_share'])}.", '']

    def met_table(title, summ, keys=('F', 's', 'h', 'hs', 's_eff', 'log10Ne')):
        R = [f'### {title}', '', '| condition | n | conv | ' + ' | '.join(f'cov90 {k}' for k in keys) +
             ' | bias s | contraction s | path cov90 | D^S cov90 | D^z cov90 | sel-share cov90 |',
             '|---|---|---|' + '---|' * len(keys) + '---|---|---|---|---|---|']
        for c, m in summ.items():
            if not isinstance(m, dict) or 'n' not in m:
                continue
            R.append(f"| {c} | {m['n']} | {_f(m.get('converged'), 2)} | " +
                     ' | '.join(_f(m.get(k, {}).get('cov90'), 2) for k in keys) +
                     f" | {_f(m.get('s', {}).get('bias_median'), 4)} | {_f(m.get('s', {}).get('contraction_median'), 2)}"
                     f" | {_f(m.get('path_cov90'), 2)} | {_f(m.get('Dsel_cov90'), 2)} | {_f(m.get('Dinb_cov90'), 2)}"
                     f" | {_f(m.get('selshare_cov90'), 2)} |")
        return R + ['']

    s = _load(out / 'e03_sbc' / 'sbc_summary.json')
    if s:
        L += ['## E03 Simulation-based calibration (truth drawn from the priors)', '']
        L += met_table('SBC', s)
        for tag, m in s.items():
            pv = {k: m[k].get('rank_chisq_p') for k in ['F', 's', 'h', 'hs', 'log10Ne'] if k in m and isinstance(m[k], dict)}
            L.append(f"{tag}: rank-uniformity χ² p-values " + ', '.join(f'{k} {_f(v, 3)}' for k, v in pv.items()))
            L.append(f"{tag}: cumulative Shapley coverage " + ', '.join(
                f"{k.replace('cum_', '').replace('_cov90', '')} {_f(v, 2)}" for k, v in m.items() if k.startswith('cum_')))
        L.append('')
    r = _load(out / 'e04_robustness' / 'robustness_summary.json')
    if r:
        L += ['## E04 Robustness: individual-based populations (real finite-N drift, partial selfing)', '']
        L += met_table('robustness', r, keys=('F', 's', 'h', 'hs', 's_eff'))
    v = _load(out / 'e05_violations' / 'violations_summary.json')
    if v:
        L += ['## E05 Stress tests', '']
        L += met_table('violations', v, keys=('F', 's', 'hs', 's_eff'))
        L += ['| condition | frac s-interval excludes 0 | frac hs-interval excludes 0 | median jump score | median predictive score | median D score |',
              '|---|---|---|---|---|---|']
        L += [f"| {c} | {_f(m['frac_s_interval_excludes_0'], 2)} | {_f(m['frac_hs_interval_excludes_0'], 2)} | "
              f"{_f(m['median_score_jump'], 2)} | {_f(m.get('median_score_pred'), 2)} | {_f(m['median_score_D'], 2)} |"
              for c, m in v.items()] + ['']
    fm = _load(out / 'e05_failure_map' / 'failure_map_summary.json')
    if fm:
        L += ['## E05b Failure map (coverage s / F, convergence)', '']
        for k, d in fm.items():
            L.append(f"- **{k}**: " + '; '.join(f"{val}: s {_f(x['cov_s'], 2)}, F {_f(x['cov_F'], 2)}, conv {_f(x['converged'], 2)}"
                                               for val, x in d.items()))
        L.append('')
    mg = _load(out / 'e07_migration' / 'migration_summary.json')
    if mg:
        L += ['## E07 Closed-model discrepancy (immigration detection, data only)', '']
        for tag, R in mg.items():
            L += [f'### {tag} (n closed = {R["n_closed"]})', '',
                  '| migration | AUC jump | AUC pred | AUC D | AUC comb | TPR@5%FPR comb | interval hit | hit if flagged | loc err (gens) |',
                  '|---|---|---|---|---|---|---|---|---|']
            for k, d in R.items():
                if k.startswith('m='):
                    L.append(f"| {k} | {_f(d['AUC_score_jump'], 2)} | {_f(d.get('AUC_score_pred'), 2)} | {_f(d['AUC_score_D'], 2)} | "
                             f"{_f(d['AUC_score_comb'], 2)} | "
                             f"{_f(d['TPR_at_5pctFPR_score_comb'], 2)} | {_f(d['loc_interval_hit_rate'], 2)} | "
                             f"{_f(d['loc_interval_hit_rate_if_flagged'], 2)} | {_f(d['loc_err_median_gens'], 1)} |")
            if 'TPR_at_5pctFPR_by_shift' in R:
                L += ['', 'Detection (comb score, 5% FPR) by the allele-frequency shift |Δp| the pulse caused: ' +
                      ', '.join(f"{k}: {_f(v['TPR_comb'], 2)} (n={v['n']})" for k, v in R['TPR_at_5pctFPR_by_shift'].items())]
            if 'flag_rate_other_violations' in R:
                L += ['', 'Flag rate at the 5%-FPR threshold for closed populations with other violations (specificity):', '',
                      '| violation | jump | pred | D | comb |', '|---|---|---|---|---|']
                L += [f"| {c} | {_f(x['score_jump'], 2)} | {_f(x.get('score_pred'), 2)} | {_f(x['score_D'], 2)} | {_f(x['score_comb'], 2)} |"
                      for c, x in R['flag_rate_other_violations'].items()]
            L.append('')
    rc = _load(out / 'e08_reconstruction' / 'reconstruction_summary.json')
    if rc:
        L += ['## E08 Trajectory reconstruction (hidden generations, RMSE)', '',
              '| design | n | HMM | linear | spline | pchip | legacy L0 | HMM better than linear | HMM 90% band coverage |',
              '|---|---|---|---|---|---|---|---|---|']
        for k, d in rc.items():
            L.append(f"| {k} | {d['n']} | {_f(d['rmse_HMM_median'], 4)} | {_f(d['rmse_linear_median'], 4)} | "
                     f"{_f(d['rmse_spline_median'], 4)} | {_f(d['rmse_pchip_median'], 4)} | {_f(d['rmse_legacy_L0_median'], 4)} | "
                     f"{_f(d['frac_HMM_better_than_linear'], 2)} | {_f(d['HMM_band_cov90_hidden'], 2)} |")
        L.append('')
    od = _load(out / 'e11_legacy_ode' / 'legacy_ode_summary.json')
    if od:
        L += ['## E11 Original neural ODE (unchanged) vs HMM', '',
              f"{od['reps']} reps on {od['device']}: median RMSE ODE {_f(od['rmse_median']['ode'], 4)}, linear "
              f"{_f(od['rmse_median']['linear'], 4)}, HMM {_f(od['rmse_median']['HMM'], 4)}; HMM better than ODE in "
              f"{_f(od['frac_HMM_better_than_ode'], 2)}; ODE ensemble mean 90% width {_f(od['ode_ensemble_mean_width90_median'], 4)}.", '']
    em = _load(out / 'e09_empirical' / 'empirical_summary.json')
    if em:
        L += ['## E09 Empirical', '']
        for tag, r_ in em.get('panaxia', {}).items():
            P = r_['posterior']
            L.append(f"**Panaxia ({tag})**: " + ', '.join(
                f"{k} {_f(P[k]['median'])} [{_f(P[k]['q05'])}, {_f(P[k]['q95'])}] (R̂ {_f(P[k]['rhat'], 3)})"
                for k in ['F', 's', 'h', 'hs', 's_eff', 'log10Ne'] if k in P) +
                f"; closure test p = {_f(r_.get('closure_test', {}).get('p_value'), 3)}")
            L.append('  cumulative Shapley ΔD (median): ' + ', '.join(
                f"{k} {v[1]:.2e}" for k, v in r_['cumulative_shapley_dD'].items()))
        for tag, r_ in em.get('peppered_moth', {}).items():
            P = r_['posterior']
            L.append(f"**Peppered moth ({tag})**: " + ', '.join(
                f"{k} {_f(P[k]['median'])} [{_f(P[k]['q05'])}, {_f(P[k]['q95'])}] (R̂ {_f(P[k]['rhat'], 3)})"
                for k in ['F', 's', 'h', 'hs', 's_eff', 'log10Ne'] if k in P and np.isfinite(P[k]['rhat'])) +
                (f"; closure test p = {_f(r_['closure_test']['p_value'], 3)}" if 'closure_test' in r_ else ''))
        L += ['', '| cross-sectional | n | p̂ | F_IS observed | lower bound F_min(p̂) | no rare homozygotes | '
              'F_eff posterior median [90%] |', '|---|---|---|---|---|---|---|']
        for k, d in em.get('cross_sectional', {}).items():
            if 'F_eff_median' in d:
                L.append(f"| {k} | {d['n']} | {_f(d['p_hat'])} | {_f(d['F_IS_observed'], 4)} | "
                         f"{_f(d['F_min_at_p_hat'], 4)} | {'yes' if d['at_bound'] else ''} | {_f(d['F_eff_median'], 4)} "
                         f"[{_f(d['F_eff_q05'], 4)}, {_f(d['F_eff_q95'], 4)}] |")
            else:
                L.append(f"| {k} | {d['n']} | {_f(d['p_hat'])} | {_f(d['F_IS_observed'], 4)} | – | – | "
                         f"{_f(d['F_median'], 4)} [{_f(d['F_q05'], 4)}, {_f(d['F_q95'], 4)}] |")
        L += ['', 'F_eff = max(F, F_min(p)) = D^z/(pq): below F_min the zygote vector is clipped and F is not '
              'identified, so the effective coefficient is reported.']
        L.append('')
    lg = _load(out / 'e10_legacy' / 'legacy_summary.json')
    if lg:
        L += ['## E10 Legacy model', '', '| data | n | legacy cov s | legacy cov F0 | legacy cov hs | v3 cov s | v3 cov F | v3 cov hs |',
              '|---|---|---|---|---|---|---|---|']
        for c, d in lg['legacy_vs_v3'].items():
            L.append(f"| {c} | {d['n']} | {_f(d['legacy_s_cov90'], 2)} | {_f(d['legacy_F0_cov90'], 2)} | {_f(d['legacy_hs_cov90'], 2)} | "
                     f"{_f(d['v3_s_cov90'], 2)} | {_f(d['v3_F_cov90'], 2)} | {_f(d['v3_hs_cov90'], 2)} |")
        L += ['', 'Original migration ROC reproduced from noise-free true trajectories only (no data, no MCMC): ' +
              ', '.join(f'{k} AUC {_f(v, 4)}' for k, v in lg['original_roc_from_true_trajectories_only'].items()), '',
              'Panaxia, legacy model, 8 dispersed chains: ' + '; '.join(
                  f"s {_f(c['s'])}, h {_f(c['h'])}, hs {_f(c['hs'])}" for c in lg['panaxia_legacy_dispersed_chains']['chains']), '']
    (out / 'SUMMARY.md').write_text('\n'.join(L))
    return out / 'SUMMARY.md'


if __name__ == '__main__':
    import sys
    print(write(Path(sys.argv[1] if len(sys.argv) > 1 else 'results')))
