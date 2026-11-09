"""Figures 1-7 of the manuscript, drawn from results/ (and, for Figures 6 and 7, from the
posterior samples in figures/refit/, written by emp_refit.py with the E09 settings and seeds).

usage (from the repository root):
    python figures/make_figs.py            # all figures
    python figures/make_figs.py 6 7        # selected figures
Writes figures/figN_*.pdf and .png."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import os
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = Path(os.environ.get('BASTION_RESULTS', ROOT / 'results'))   # outputs of run_all.py
DATA = ROOT / 'data'
OUT = HERE
REFIT = HERE / 'refit'             # written by emp_refit.py
OUT.mkdir(parents=True, exist_ok=True)

C_INB, C_SEL, C_MUT, C_DR, C_RES = '#2166ac', '#d6604d', '#4dac26', '#7f7f7f', '#9970ab'
mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.2,
    'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'figure.dpi': 150})
W = 6.5   # text width in inches


def tag(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── closed forms (Proposition 3 / 4) ─────────────────────────────────────────
def zyg(p, F):
    q = 1 - p
    return np.array([p * p + F * p * q, 2 * p * q * (1 - F), q * q + F * p * q])


def adult_fis(p, F, s, h):
    x = zyg(p, F)
    w = np.array([1.0, 1 - h * s, 1 - s])
    a = w[:, None] * x if x.ndim > 1 else w * x
    a = a / a.sum(0)
    pa = a[0] + a[1] / 2
    D = a[0] - pa ** 2
    return D / (pa * (1 - pa))


# ── Figure 1b,c: selection-induced F_IS ──────────────────────────────────────
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={'width_ratios': [1.15, 1]})
    hh = np.linspace(-0.5, 1.5, 301); ss = np.linspace(-0.5, 0.5, 301)
    H, S = np.meshgrid(hh, ss)
    M = np.full(H.shape, np.nan)
    ok = (1 - H * S > 0.02) & (1 - S > 0.02)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if ok[i, j]:
                M[i, j] = adult_fis(0.5, 0.0, S[i, j], H[i, j])
    vm = np.nanmax(np.abs(M))
    ax = axes[0]
    im = ax.imshow(M, origin='lower', extent=[hh[0], hh[-1], ss[0], ss[-1]], aspect='auto',
                   cmap='RdBu_r', vmin=-vm, vmax=vm, interpolation='nearest')
    kap = S * (2 * H - 1) - H ** 2 * S ** 2
    ax.contour(hh, ss, np.where(ok, kap, np.nan), levels=[0], colors='k', linewidths=0.7)
    ax.set_xlabel('dominance $h$'); ax.set_ylabel('selection coefficient $s$')
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label('adult $F_{IS}$ from selection alone', fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.text(1.12, 0.36, r'$\kappa>0$', fontsize=7); ax.text(-0.42, 0.36, r'$\kappa<0$', fontsize=7)
    ax.text(1.12, -0.42, r'$\kappa<0$', fontsize=7); ax.text(-0.42, -0.42, r'$\kappa>0$', fontsize=7)
    tag(ax, 'b', x=-0.2)
    ax = axes[1]
    Fz = np.linspace(-0.15, 0.6, 200)
    cases = [('overdominant  ($s=-0.3,\\ h=1.5$)', -0.3, 1.5, C_SEL, '-'),
             ('underdominant ($s=0.3,\\ h=1.5$)', 0.3, 1.5, C_INB, '-'),
             ('additive ($s=0.3,\\ h=0.5$)', 0.3, 0.5, '0.25', '--'),
             ('neutral ($s=0$)', 0.0, 0.5, '0.6', ':')]
    for lab, s, h, col, ls in cases:
        ax.plot(Fz, [adult_fis(0.5, f, s, h) for f in Fz], color=col, ls=ls, label=lab)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('zygotic inbreeding coefficient $F$ ($p=0.5$)'); ax.set_ylabel('adult $F_{IS}$')
    ax.legend(loc='upper left', fontsize=6.3, handlelength=2.2)
    tag(ax, 'c', x=-0.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig1_selection_fis')


# ── Figure 2: drift vs F_IS, identifiability ────────────────────────────────
def fig2():
    th = json.load(open(RES / 'e01_theory' / 'theory_summary.json'))
    d = th['drift_vs_FIS']; N = d['N']
    fim = pd.read_csv(RES / 'e01_theory' / 'identifiability_fim.csv')
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.2))
    ax = axes[0]
    t = np.arange(0, 61)
    ax.plot(t, 1 - (1 - 1 / (2 * N)) ** t, color=C_INB, label="Wright's $F_t$")
    ax.plot(d['gens'], d['one_minus_H_over_H0'], 'o', ms=3, mfc='white', mec='k', mew=0.7,
            label=r'$1-\bar H_t/H_0$ (simulated)')
    ax.plot(d['gens'], d['F_IS_mean'], 's', ms=3, color=C_SEL, label=r'within-population $F_{IS}$')
    ax.axhline(-1 / (2 * N - 1), color=C_SEL, lw=0.6, ls=':')
    ax.set_xlabel('generation'); ax.set_ylabel('coefficient')
    ax.legend(loc='upper left', fontsize=6.3)
    tag(ax, 'a')
    sub = fim[(fim.design == 'T30_n300') & (fim.s == 0.1)]
    p0s = sorted(sub.p0.unique())
    ax = axes[1]
    for key, lab, col, ls in [('crlb_s', '$s$', C_SEL, '-'), ('crlb_h', '$h$', '0.3', '--'),
                              ('crlb_hs', '$hs$', '#f4a582', '-'), ('crlb_seff', r'$s_{\rm eff}$', C_RES, '-.')]:
        y = [np.nanmedian(sub[(sub.kind == 'geno') & (sub.p0 == p)][key]) for p in p0s]
        ax.plot(p0s, y, ls, color=col, marker='o', ms=2.5, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('initial frequency $p_0$'); ax.set_ylabel('Cramér–Rao s.d. (genotypes)')
    ax.legend(fontsize=6.5, ncol=2, loc='upper right')
    tag(ax, 'b')
    ax = axes[2]
    for kind, lab, col, ls in [('geno', 'adult genotypes', C_INB, '-'), ('zygote', 'zygote genotypes', '0.3', '--'),
                               ('pheno', 'dominant phenotypes', C_SEL, '-')]:
        y = [np.nanmedian(sub[(sub.kind == kind) & (sub.p0 == p)]['crlb_F']) for p in p0s]
        ax.plot(p0s, y, ls, color=col, marker='o', ms=2.5, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('initial frequency $p_0$'); ax.set_ylabel('Cramér–Rao s.d. of $F$')
    ax.legend(fontsize=6.5, loc='center right')
    tag(ax, 'c')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig2_drift_identifiability')


# ── Figure 3: simulation-based calibration ───────────────────────────────────
def fig3():
    fig, axes = plt.subplots(2, 5, figsize=(W, 2.9))
    for r, (tagname, keys) in enumerate([('Ne_known', ['F', 's', 'h', 'hs']),
                                        ('Ne_inferred', ['F', 's', 'h', 'hs', 'log10Ne'])]):
        df = pd.read_csv(RES / 'e03_sbc' / f'sbc_{tagname}.csv')
        n = len(df); nb = 20
        lo, hi = stats.binom.ppf([0.005, 0.995], n, 1 / nb)
        for c in range(5):
            ax = axes[r, c]
            if c >= len(keys):
                ax.axis('off'); continue
            k = keys[c]
            rk = df[f'{k}_rank'].dropna().values
            ax.fill_between([0, 100], lo, hi, color='0.88', lw=0)
            ax.hist(rk, bins=np.linspace(0, 100, nb + 1), color=C_INB if k == 'F' else
                    (C_RES if k == 'log10Ne' else C_SEL), alpha=0.85, edgecolor='white', lw=0.3)
            ax.axhline(n / nb, color='k', lw=0.5)
            lab = {'F': '$F$', 's': '$s$', 'h': '$h$', 'hs': '$hs$', 'log10Ne': r'$\log_{10}N_e$'}[k]
            ax.set_title(lab, fontsize=8, pad=2)
            ax.set_yticks([]); ax.spines['left'].set_visible(False)
            ax.set_xticks([0, 50, 100]); ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(['$N_e$ known\n(2500 data sets)', '$N_e$ inferred\n(1500 data sets)'][r], fontsize=7)
                ax.spines['left'].set_visible(True); ax.set_yticks([])
        axes[1, 2].set_xlabel('rank of the true value among 99 posterior draws')
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, 'fig3_sbc_ranks')


# ── Figure 4: robustness and stress tests ────────────────────────────────────
def fig4():
    rob = json.load(open(RES / 'e04_robustness' / 'robustness_summary.json'))
    vio = json.load(open(RES / 'e05_violations' / 'violations_summary.json'))
    order_r = ['N100_sigma0.0_Neknown', 'N100_sigma0.2_Neknown', 'N100_sigma0.5_Neknown',
               'N500_sigma0.0_Neknown', 'N500_sigma0.2_Neknown', 'N500_sigma0.5_Neknown',
               'N2000_sigma0.0_Neknown', 'N2000_sigma0.2_Neknown', 'N2000_sigma0.5_Neknown',
               'N500_sigma0.0_Neinferred', 'N500_sigma0.2_Neinferred', 'N500_sigma0.5_Neinferred',
               'N500_sigma0.2_Nehalf', 'N500_sigma0.2_Nedouble']

    def rlab(c):
        N = c.split('_')[0][1:]; sg = c.split('_')[1][5:]; ne = c.split('_')[2][2:]
        ne = {'known': '', 'inferred': r', $N_e$ inferred', 'half': r', $N_e$ set to $N/2$',
              'double': r', $N_e$ set to $2N$'}[ne]
        return f'$N={N}$, $\\sigma={sg}${ne}'
    order_v = ['baseline', 'finite_pop', 'zygote_sampled', 'genotyping_error', 'wahlund_sampling',
               'selfing_transient', 'linked_selection', 'bottleneck_Ne_inferred', 'bottleneck_Ne_fixed',
               'changing_s']
    vlab = {'baseline': 'baseline (model-exact)', 'finite_pop': 'finite population ($N=1000$)',
            'zygote_sampled': 'zygotes sampled', 'genotyping_error': 'genotyping error (3%)',
            'wahlund_sampling': 'Wahlund sample ($p\\pm0.15$)', 'selfing_transient': 'selfing starts at $t=0$',
            'linked_selection': 'linked selection ($s=0$)', 'bottleneck_Ne_inferred': r'bottleneck, $N_e$ inferred',
            'bottleneck_Ne_fixed': r'bottleneck, $N_e$ fixed', 'changing_s': '$s$ reverses at $T/2$'}
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), gridspec_kw={'width_ratios': [1, 1]})
    for ax, D, order, labf, nrep, tg in [(axes[0], rob, order_r, rlab, 300, 'a'),
                                         (axes[1], vio, order_v, lambda c: vlab[c], 500, 'b')]:
        y = np.arange(len(order))[::-1]
        half = 1.96 * np.sqrt(0.9 * 0.1 / nrep)
        ax.axvspan(0.9 - half, 0.9 + half, color='0.9', lw=0)
        ax.axvline(0.9, color='k', lw=0.5)
        for k, col, mk, dy in [('F', C_INB, 'o', 0.2), ('s', C_SEL, 's', 0.0), ('hs', '#f4a582', 'D', -0.2)]:
            x = [D[c].get(k, {}).get('cov90', np.nan) for c in order]
            ax.plot(x, y + dy, mk, color=col, ms=3.2, label={'F': '$F$', 's': '$s$', 'hs': '$hs$'}[k])
        ax.set_yticks(y); ax.set_yticklabels([labf(c) for c in order], fontsize=6.5)
        ax.set_xlim(-0.03, 1.03); ax.set_xlabel('coverage of central 90% interval')
        tag(ax, tg, x=-0.02 if tg == 'a' else -0.02, y=1.01)
    axes[0].legend(loc='lower left', fontsize=6.5, handletextpad=0.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig4_robustness')


# ── Figure 5: closed-model adequacy and immigration ──────────────────────────
def fig5():
    mg = json.load(open(RES / 'e07_migration' / 'migration_summary.json'))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={'width_ratios': [1, 1.2]})
    ax = axes[0]
    for tagname, col, mk in [('Ne_known', C_SEL, 'o'), ('Ne_inferred', C_RES, 's')]:
        B = mg[tagname]['TPR_at_5pctFPR_by_shift']
        keys = list(B.keys())
        x = np.arange(len(keys)); yv = [B[k]['TPR_comb'] for k in keys]
        nn = [B[k]['n'] for k in keys]
        yv = [np.nan if v is None else v for v in yv]
        err = [1.96 * np.sqrt(v * (1 - v) / max(n, 1)) if np.isfinite(v) and n > 0 else np.nan for v, n in zip(yv, nn)]
        ax.errorbar(x + (0.08 if tagname == 'Ne_inferred' else -0.08), yv, yerr=err, fmt=mk, color=col, ms=3.5,
                    capsize=1.5, lw=0.8, label={'Ne_known': '$N_e$ known', 'Ne_inferred': '$N_e$ inferred'}[tagname])
    ax.axhline(0.05, color='k', lw=0.5, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(['<0.01', '0.01–0.02', '0.02–0.05', '0.05–0.1', '>0.1'], fontsize=6.5)
    ax.set_xlabel(r'allele-frequency shift caused by the pulse, $|\Delta p|$'); ax.set_ylabel('detection rate at 5% FPR')
    ax.legend(loc='upper left'); tag(ax, 'a')
    ax = axes[1]
    fl = mg['Ne_known']['flag_rate_other_violations']
    names = ['finite_pop', 'wahlund_sampling', 'genotyping_error', 'linked_selection', 'bottleneck', 'changing_s']
    lab = ['finite\npop.', 'Wahlund\nsample', 'genotype\nerror', 'linked\nselection', 'bottle-\nneck', '$s$\nreverses']
    x = np.arange(len(names)); wd = 0.27
    for j, (k, col, l) in enumerate([('score_pred', C_SEL, 'predictive residual'), ('score_D', C_INB, 'HW deviation'),
                                     ('score_comb', '0.35', 'combined')]):
        ax.bar(x + (j - 1) * wd, [fl[n][k] for n in names], wd, color=col, label=l)
    ax.axhline(0.05, color='k', lw=0.5, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.3)
    ax.set_ylabel('flag rate at 5% FPR'); ax.legend(loc='upper left', fontsize=6.5)
    tag(ax, 'b')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig5_adequacy')



# ── empirical helpers ────────────────────────────────────────────────────────
def _adult(p, F, s, h):
    q = 1 - p
    z = np.array([p * p + F * p * q, 2 * p * q * (1 - F), q * q + F * p * q])
    z = np.clip(z, 0, None); z = z / z.sum(0)
    w = np.array([1.0, 1 - h * s, 1 - s])[:, None] if np.ndim(p) else np.array([1.0, 1 - h * s, 1 - s])
    a = z * w
    return a / a.sum(0)


def _band(Y, q=(5, 50, 95)):
    return [np.nanpercentile(Y, k, 0) for k in q]


# ── Figure 6: Panaxia dominula ───────────────────────────────────────────────
def fig6():
    df = pd.read_csv(DATA / 'panaxia.csv')
    z = np.load(REFIT / 'panaxia_Ne_FisherFord.npz')
    th, paths, flat, phi = z['th'], z['paths'], z['flat'], z['phi']
    yrs = df.generation.values
    n = (df.N_AA + df.N_Aa + df.N_aa).values
    qobs = (df.N_Aa + 2 * df.N_aa).values / (2 * n)
    lo, hi = stats.beta.ppf([0.025, 0.975], (df.N_Aa + 2 * df.N_aa).values[:, None] + 0.5,
                            (2 * n - df.N_Aa - 2 * df.N_aa).values[:, None] + 0.5).T
    Q = np.array([[1 - (lambda a: a[0] + a[1] / 2)(_adult(paths[i, t], th[i, 0], th[i, 1], th[i, 2]))
                   for t in range(paths.shape[1])] for i in range(len(th))])
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15), gridspec_kw={'width_ratios': [1.15, 1, 1.05]})
    ax = axes[0]
    b5, b50, b95 = _band(Q)
    ax.fill_between(yrs, b5, b95, color=C_SEL, alpha=0.22, lw=0)
    ax.plot(yrs, b50, color=C_SEL, lw=1)
    ax.errorbar(yrs, qobs, yerr=[qobs - lo, hi - qobs], fmt='o', color='k', ms=2.6, lw=0.7, capsize=1.2)
    ax.set_xlabel('year'); ax.set_ylabel('frequency of $medionigra$')
    ax.set_xticks([1939, 1942, 1945]); ax.set_ylim(0, 0.16)
    tag(ax, 'a')
    ax = axes[1]
    sel = np.random.default_rng(1).choice(len(flat), 3000, replace=False)
    ax.scatter(flat[sel, 1], flat[sel, 2], s=2.2, color=C_SEL, alpha=0.35, lw=0, rasterized=True)
    hs_med = float(np.median(flat[:, 4]))
    for branch in (np.linspace(0.04, 0.62, 300), np.linspace(-0.45, -0.04, 300)):
        for v, ls in [(hs_med, '-'), (np.percentile(flat[:, 4], 5), ':'), (np.percentile(flat[:, 4], 95), ':')]:
            ax.plot(branch, v / branch, color='k', lw=0.7, ls=ls)
    ax.axhline(0, color='0.6', lw=0.4); ax.axvline(0, color='0.6', lw=0.4)
    ax.set_xlim(-0.45, 0.62); ax.set_ylim(-1.6, 2.0)
    ax.set_xlabel('$s$'); ax.set_ylabel('$h$')
    ax.text(0.30, 0.62, r'$hs=%.3f$' % hs_med, fontsize=6.5)
    tag(ax, 'b')
    ax = axes[2]
    names = ['mutation', 'reset', 'inbreeding', 'drift', 'selection']
    cols = [C_MUT, C_RES, C_INB, C_DR, C_SEL]
    cum = np.nansum(phi, 1) * 1e3
    y = np.arange(5)[::-1]
    for j in range(5):
        l5, l50, l95 = np.nanpercentile(cum[:, j], [5, 50, 95])
        ax.plot([l5, l95], [y[j], y[j]], color=cols[j], lw=1.6)
        ax.plot(l50, y[j], 'o', color=cols[j], ms=3.2)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel(r'$\sum_t\varphi_{i,t}$, 1939–1946 ($\times10^{-3}$)')
    tag(ax, 'c', x=-0.42)
    fig.tight_layout(w_pad=0.9)
    save(fig, 'fig6_panaxia')
    print('panaxia hs med', hs_med, 'Q band last', b5[-1], b95[-1])


# ── Figure 7: Biston betularia ───────────────────────────────────────────────
def fig7():
    df = pd.read_csv(DATA / 'peppered_moth.csv')
    z = np.load(REFIT / 'moth_h0_n100_NeInf.npz')
    th, paths = z['th'], z['paths']
    zf = np.load(REFIT / 'moth_hfree_n100_NeInf.npz')
    em = json.load(open(RES / 'e09_empirical' / 'empirical_summary.json'))['peppered_moth']
    yr0 = int(df.year.values[0])
    gens = df.generation.values; f = df.f_carbonaria.values
    n = 100
    lo, hi = stats.beta.ppf([0.025, 0.975], (f * n)[:, None] + 0.5, ((1 - f) * n)[:, None] + 0.5).T
    Pd = np.array([[(lambda a: a[0] + a[1])(_adult(paths[i, t], th[i, 0], th[i, 1], th[i, 2]))
                    for t in range(paths.shape[1])] for i in range(len(th))])
    fig = plt.figure(figsize=(W, 2.2))
    axes = [fig.add_axes([0.075, 0.19, 0.28, 0.74]), fig.add_axes([0.575, 0.19, 0.17, 0.74]),
            fig.add_axes([0.80, 0.19, 0.185, 0.74])]
    ax = axes[0]
    tt = np.arange(paths.shape[1]) + yr0
    b5, b50, b95 = _band(Pd)
    ax.fill_between(tt, b5, b95, color=C_SEL, alpha=0.22, lw=0)
    ax.plot(tt, b50, color=C_SEL, lw=1)
    ax.errorbar(gens + yr0, f, yerr=[f - lo, hi - f], fmt='o', color='k', ms=2.2, lw=0.6, capsize=0)
    ax.set_xlabel('year'); ax.set_ylabel('frequency of $carbonaria$ phenotype'); ax.set_ylim(0, 1.02)
    tag(ax, 'a')
    ax = axes[1]
    cfgs = [('h0_n50_NeInf', r'$h=0$, $n=50$, $N_e$ inferred'), ('h0_n100_NeInf', r'$h=0$, $n=100$, $N_e$ inferred'),
            ('h0_n300_NeInf', r'$h=0$, $n=300$, $N_e$ inferred'), ('h0_n100_Ne1000', r'$h=0$, $n=100$, $N_e=1000$'),
            ('hfree_n100_NeInf', r'$h$ free, $n=100$, $N_e$ inferred')]
    for j, (k, lab) in enumerate(cfgs):
        S = em[k]['posterior']['s']
        yy = len(cfgs) - 1 - j
        ax.plot([S['q05'], S['q95']], [yy, yy], color=C_SEL, lw=1.6)
        ax.plot(S['median'], yy, 'o', color=C_SEL, ms=3.2)
    ax.set_yticks(range(len(cfgs))[::-1]); ax.set_yticklabels([c[1] for c in cfgs], fontsize=6.3)
    ax.set_xlim(-0.27, -0.17); ax.set_xticks([-0.26, -0.22, -0.18])
    ax.set_xlabel('$s$')
    tag(ax, 'b', x=-1.2)
    ax = axes[2]
    hh = zf['flat'][:, 2]
    ax.hist(hh, bins=np.linspace(-1.0, 1.0, 41), density=True, color=C_SEL, alpha=0.7, edgecolor='white', lw=0.3)
    xx = np.linspace(-1.0, 1.0, 200)
    ax.plot(xx, stats.norm.pdf(xx, 0.5, 0.5), color='k', lw=0.7, ls='--')
    ax.set_xlabel('$h$'); ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    tag(ax, 'c', x=-0.08)
    save(fig, 'fig7_biston')
    print('moth band at 1959', b5[0], b50[0], b95[0], 'at 2001', b5[-1], b50[-1], b95[-1])

if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7']
    for w in which:
        globals()[f'fig{w}']()
        print('fig', w, 'done')
