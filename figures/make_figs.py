"""Figures 1-7 of the manuscript.

Inputs: the outputs of run_all.py (results/), the data files (data/) and, for Figures 6
and 7, the posterior samples written by emp_refit.py (figures/refit/).

usage (from the repository root):
    python figures/make_figs.py            # all figures
    python figures/make_figs.py 6 7        # selected figures
Writes figures/figN_*.pdf (vector, fonts embedded) and figures/figN_*.png (300 dpi).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = Path(os.environ.get('BASTION_RESULTS', ROOT / 'results'))
DATA = Path(os.environ.get('BASTION_DATA', ROOT / 'data'))
REFIT = Path(os.environ.get('BASTION_REFIT', HERE / 'refit'))
OUT = Path(os.environ.get('BASTION_FIGS', HERE))
OUT.mkdir(parents=True, exist_ok=True)

# ── design tokens ────────────────────────────────────────────────────────────
INK = '#1d1d1b'        # text, data points
INK2 = '#55544f'       # secondary text, axes
MUTED = '#8f8e88'      # tertiary marks
RULE = '#c9c8c2'       # reference lines
BAND = '#e7e6e1'       # neutral bands
# categorical slots (validated light-mode order; CVD-safe for the first three all-pairs)
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
C_F, C_S, C_HS, C_H = BLUE, ORANGE, AQUA, VIOLET     # colour follows the parameter
BLUE_LIGHT, ORANGE_LIGHT = '#cde2fb', '#fbd9ca'
RED = '#e34948'
DIVERGING = LinearSegmentedColormap.from_list(
    'bluegrayred', ['#104281', BLUE, '#9ec5f4', '#f0efec', '#f3b1b0', RED, '#8f1f1e'])

mpl.rcParams.update({
    'font.family': 'Liberation Sans',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold',
    'mathtext.sf': 'Liberation Sans', 'mathtext.fallback': 'stixsans',
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'legend.fontsize': 7,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'lines.linewidth': 1.3, 'lines.solid_capstyle': 'round',
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.2,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 150})
FULL = 5.4          # text width of the manuscript, inches (vector output scales to the journal column)


def tag(ax, s, dx=0.0, dy=0.0):
    """Panel letter at the top-left corner of the axes' tight bounding box."""
    ax.annotate(s, xy=(0, 1), xycoords='axes fraction', xytext=(-22 + dx, 6 + dy),
                textcoords='offset points', fontsize=9, fontweight='bold', va='bottom', ha='left',
                color=INK, annotation_clip=False)


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def endlabel(ax, x, y, text, color, dx=3, dy=0, **kw):
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords='offset points', va='center',
                ha='left', color=color, fontsize=6.8, annotation_clip=False, **kw)


# ── closed forms (Propositions 3 and 4) ─────────────────────────────────────
def zyg(p, F):
    q = 1 - p
    return np.array([p * p + F * p * q, 2 * p * q * (1 - F), q * q + F * p * q])


def adult_fis(p, F, s, h):
    x = zyg(p, F)
    w = np.array([1.0, 1 - h * s, 1 - s])
    a = w * x
    a = a / a.sum()
    pa = a[0] + a[1] / 2
    return (a[0] - pa ** 2) / (pa * (1 - pa))


# ── Figure 1 ────────────────────────────────────────────────────────────────
def _schematic(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis('off')
    boxes = [(9, 'adults', '$a_t$'), (29.5, 'gametes', '$m_t$'), (50, 'Hardy–Weinberg', r'$\mathrm{vec}(m_t,0)$'),
             (70.5, 'zygotes', r'$\mathrm{zyg}(m_t,F)$'), (91, 'zygotes', r'$\mathrm{zyg}(p_{t+1},F)$')]
    bw, bh, yc = 13.0, 8.0, 12.5
    for x, top, bot in boxes:
        ax.add_patch(FancyBboxPatch((x - bw / 2, yc - bh / 2), bw, bh, boxstyle='round,pad=0,rounding_size=1.6',
                                    fc='white', ec=INK2, lw=0.7))
        ax.text(x, yc + 1.6, top, ha='center', va='center', fontsize=6.2, color=INK2)
        ax.text(x, yc - 1.9, bot, ha='center', va='center', fontsize=6.9, color=INK)
    ops = [('M', 'mutation'), ('Y_0', 'reset'), ('I_F', 'inbreeding'), (r'\mathrm{Dr}_\delta', 'drift')]
    for (x0, *_), (x1, *_), (sym, name) in zip(boxes[:-1], boxes[1:], ops):
        ax.add_patch(FancyArrowPatch((x0 + bw / 2 + 0.4, yc), (x1 - bw / 2 - 0.4, yc), arrowstyle='-|>',
                                     mutation_scale=6, lw=0.8, color=INK))
        xm = (x0 + x1) / 2
        ax.text(xm, yc + 2.4, f'${sym}$', ha='center', va='bottom', fontsize=7.3, color=INK)
        ax.text(xm, yc - 1.6, name, ha='center', va='top', fontsize=5.6, color=INK2)
    # selection closes the loop back to the adults of the next generation
    x_last = boxes[-1][0]
    ax.add_patch(FancyArrowPatch((x_last, yc - bh / 2 - 0.3), (boxes[0][0], yc - bh / 2 - 0.3),
                                 connectionstyle='arc3,rad=-0.12', arrowstyle='-|>', mutation_scale=6,
                                 lw=0.8, color=INK, shrinkA=0, shrinkB=0))
    ax.text(50, 1.0, r'$S$  selection:  $a_{t+1}=\mathrm{sel}_w(\mathrm{zyg}(p_{t+1},F))$', ha='center', va='center',
            fontsize=7.0, color=INK)


def fig1():
    fig = plt.figure(figsize=(FULL, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.55, left=0.085, right=0.86, top=0.60, bottom=0.12)
    ax = fig.add_axes([0.0, 0.70, 1.0, 0.28]); _schematic(ax); tag(ax, 'a', dx=22, dy=-4)
    # b: adult F_IS created by selection alone at p = 0.5
    ax = fig.add_subplot(gs[0])
    hh = np.linspace(-0.5, 1.5, 401); ss = np.linspace(-0.5, 0.5, 401)
    H, S = np.meshgrid(hh, ss)
    ok = (1 - H * S > 0.02) & (1 - S > 0.02)
    M = np.full(H.shape, np.nan)
    it = np.nditer(H, flags=['multi_index'])
    for _ in it:
        i, j = it.multi_index
        if ok[i, j]:
            M[i, j] = adult_fis(0.5, 0.0, S[i, j], H[i, j])
    vm = 0.45
    im = ax.imshow(M, origin='lower', extent=[hh[0], hh[-1], ss[0], ss[-1]], aspect='auto', cmap=DIVERGING,
                   vmin=-vm, vmax=vm, interpolation='bilinear', rasterized=True)
    kap = S * (2 * H - 1) - H ** 2 * S ** 2
    ax.contour(hh, ss, np.where(ok, kap, np.nan), levels=[0], colors=INK, linewidths=0.7)
    ax.set_xlabel('dominance $h$'); ax.set_ylabel('selection coefficient $s$')
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5]); ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for x, y, txt in [(-0.42, 0.40, r'$\kappa<0$'), (1.42, 0.40, r'$\kappa>0$'), (-0.42, -0.43, r'$\kappa>0$'),
                      (1.42, -0.43, r'$\kappa<0$')]:
        ax.text(x, y, txt, fontsize=6.8, color=INK, ha='left' if x < 0 else 'right', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
    cax = ax.inset_axes([1.035, 0.0, 0.035, 1.0])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.4, -0.2, 0, 0.2, 0.4])
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=6.3)
    cax.set_title('$F_{IS}$', fontsize=6.6, pad=3, color=INK)
    tag(ax, 'b', dx=-10)
    # c: adult F_IS against zygotic F
    ax = fig.add_subplot(gs[1])
    Fz = np.linspace(-0.15, 0.6, 200)
    cases = [('underdominant', 0.3, 1.5, RED, '-'), ('additive', 0.3, 0.5, INK2, '-'),
             ('overdominant', -0.3, 1.5, BLUE, '-')]
    ax.axhline(0, color=RULE, lw=0.6, zorder=0); ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    ax.plot(Fz, Fz, color=MUTED, lw=0.8, ls=(0, (1.2, 1.4)), zorder=0)
    ax.text(-0.13, 0.68, 'dotted line:\nno selection', fontsize=6.2, color=INK2, va='top', linespacing=1.0)
    for lab, s, h, col, ls in cases:
        y = np.array([adult_fis(0.5, f, s, h) for f in Fz])
        ax.plot(Fz, y, color=col, ls=ls, lw=1.4)
        endlabel(ax, Fz[-1], y[-1], lab, col)
    ax.set_xlim(-0.15, 0.6); ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6])
    ax.set_xlabel('zygotic inbreeding coefficient $F$'); ax.set_ylabel('adult $F_{IS}$')
    tag(ax, 'c', dx=-10)
    save(fig, 'fig1_selection_fis')


# ── Figure 2 ────────────────────────────────────────────────────────────────
def fig2():
    th = json.load(open(RES / 'e01_theory' / 'theory_summary.json'))
    d = th['drift_vs_FIS']; N = d['N']
    fim = pd.read_csv(RES / 'e01_theory' / 'identifiability_fim.csv')
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05), gridspec_kw={'width_ratios': [1, 1.12, 1.12]})
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.2, top=0.9, wspace=0.62)
    ax = axes[0]
    t = np.arange(0, 61)
    ax.axhline(0, color=RULE, lw=0.6, zorder=0)
    ax.plot(t, 1 - (1 - 1 / (2 * N)) ** t, color=INK, lw=1.3)
    ax.plot(d['gens'], d['one_minus_H_over_H0'], 'o', ms=4.2, mfc='white', mec=INK, mew=0.9)
    ax.axhline(-1 / (2 * N - 1), color=C_F, lw=0.7, ls=(0, (1.5, 1.5)))
    ax.plot(d['gens'], d['F_IS_mean'], 's', ms=3.6, color=C_F, mec='white', mew=0.5)
    ax.text(0, 0.5, "Wright's $F_t$ (line),\n$1-\\bar H_t/H_0$ (circles)", fontsize=6.4, color=INK, va='top', ha='left')
    ax.text(60, 0.035, 'within-population $F_{IS}$', color=C_F, fontsize=6.6, ha='right', va='bottom')
    ax.set_xlim(-2, 62); ax.set_ylim(-0.04, 0.5)
    ax.set_xlabel('generation'); ax.set_ylabel('coefficient')
    tag(ax, 'a')
    sub = fim[(fim.design == 'T30_n300') & (fim.s == 0.1)]
    p0s = np.array(sorted(sub.p0.unique()))
    ax = axes[1]
    series = [('crlb_h', '$h$', C_H), ('crlb_hs', '$hs$', C_HS), ('crlb_s', '$s$', C_S),
              ('crlb_seff', r'$s_{\mathrm{eff}}$', INK2)]
    ends = {}
    for key, lab, col in series:
        y = np.array([np.nanmedian(sub[(sub.kind == 'geno') & (sub.p0 == p)][key]) for p in p0s])
        ax.plot(p0s, y, '-', color=col, lw=1.2, marker='o', ms=2.8, mec='white', mew=0.4)
        ends[lab] = (p0s[-1], y[-1], col)
    offs = {'$h$': 0, '$hs$': 3, '$s$': 0, r'$s_{\mathrm{eff}}$': -3}
    for lab, (x, y, col) in ends.items():
        endlabel(ax, x, y, lab, col, dy=offs[lab])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('initial frequency of $A$, $p_0$'); ax.set_ylabel('Cramér–Rao s.d., adult genotypes')
    tag(ax, 'b')
    ax = axes[2]
    for kind, lab, col, ls in [('pheno', 'dominant\nphenotypes', MUTED, '-'), ('geno', 'adult\ngenotypes', C_F, '-'),
                               ('zygote', 'zygote\ngenotypes', INK, (0, (3, 1.5)))]:
        y = np.array([np.nanmedian(sub[(sub.kind == kind) & (sub.p0 == p)]['crlb_F']) for p in p0s])
        ax.plot(p0s, y, color=col, ls=ls, lw=1.2, marker='o', ms=2.8, mec='white', mew=0.4)
        ok = np.isfinite(y)
        dy = {'pheno': 0, 'geno': 4, 'zygote': -5}[kind]
        endlabel(ax, p0s[ok][-1], y[ok][-1], lab, col if col != MUTED else INK2, dy=dy, linespacing=0.95)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('initial frequency of $A$, $p_0$'); ax.set_ylabel('Cramér–Rao s.d. of $F$')
    tag(ax, 'c')
    save(fig, 'fig2_drift_identifiability')


# ── Figure 3 ────────────────────────────────────────────────────────────────
def fig3():
    summ = json.load(open(RES / 'e03_sbc' / 'sbc_summary.json'))
    cols = {'F': C_F, 's': C_S, 'h': C_H, 'hs': C_HS, 'log10Ne': INK2}
    labs = {'F': '$F$', 's': '$s$', 'h': '$h$', 'hs': '$hs$', 'log10Ne': r'$\log_{10}N_e$'}
    fig, axes = plt.subplots(2, 5, figsize=(FULL, 2.45), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.9, wspace=0.3, hspace=0.45)
    for r, (tg, keys, name) in enumerate([('Ne_known', ['F', 's', 'h', 'hs'], '$N_e$ known'),
                                          ('Ne_inferred', ['F', 's', 'h', 'hs', 'log10Ne'], '$N_e$ inferred')]):
        df = pd.read_csv(RES / 'e03_sbc' / f'sbc_{tg}.csv')
        n = len(df); nb = 20
        lo, hi = stats.binom.ppf([0.005, 0.995], n, 1 / nb)
        ymax = stats.binom.ppf(0.9995, n, 1 / nb) * 1.25
        for c in range(5):
            ax = axes[r, c]
            if c >= len(keys):
                ax.axis('off'); continue
            k = keys[c]
            rk = df[f'{k}_rank'].dropna().values
            ax.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
            cnt, edges = np.histogram(rk, bins=np.linspace(0, 100, nb + 1))
            ax.bar(edges[:-1], cnt, width=np.diff(edges) * 0.82, align='edge', color=cols[k], lw=0, zorder=2)
            ax.axhline(n / nb, color=INK, lw=0.6, zorder=3)
            ax.set_ylim(0, ymax); ax.set_xlim(-1, 101)
            ax.set_yticks([]); ax.spines['left'].set_visible(False)
            ax.set_xticks([0, 50, 100])
            pv = summ[tg][k].get('rank_chisq_p')
            ax.set_title(labs[k], fontsize=7.5, pad=2, color=INK)
            if pv is not None:
                ax.text(0.98, 0.97, f'$p$ = {pv:.2f}' if pv >= 0.05 else f'$p$ = {pv:.3f}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=6.2, color=INK2)
            if c == 0:
                ax.text(-0.12, 0.5, f'{name}\n({n} data sets)', transform=ax.transAxes, rotation=90,
                        ha='right', va='center', fontsize=7, color=INK)
    fig.text(0.54, 0.02, 'rank of the true value among 99 posterior draws', ha='center', fontsize=7.5)
    save(fig, 'fig3_sbc_ranks')


# ── Figure 4 ────────────────────────────────────────────────────────────────
def _coverage_panel(ax, rows, D, nrep, tg):
    """rows: list of (label, key) or ('#', group heading)."""
    y = 0; ys, labels, heads = [], [], []
    for lab, key in rows:
        if lab == '#':
            y -= 0.35 if ys else 0
            heads.append((y, key)); y -= 0.95
        else:
            ys.append((y, key)); labels.append(lab); y -= 1
    half = 1.96 * np.sqrt(0.9 * 0.1 / nrep)
    ax.axvspan(0.9 - half, 0.9 + half, color=BAND, lw=0, zorder=0)
    ax.axvline(0.9, color=INK2, lw=0.6, zorder=1)
    for xv in (0, 0.25, 0.5, 0.75, 1.0):
        ax.axvline(xv, color='#efeee9', lw=0.5, zorder=0)
    mk = {'F': ('o', C_F, 3.6), 's': ('s', C_S, 3.4), 'hs': ('D', C_HS, 3.0)}
    for yy, key in ys:
        vals = [D[key].get(k, {}).get('cov90', np.nan) for k in ('F', 's', 'hs')]
        ax.plot([np.nanmin(vals), np.nanmax(vals)], [yy, yy], color=RULE, lw=0.8, zorder=1)
        for k, v in zip(('F', 's', 'hs'), vals):
            m, c, ms = mk[k]
            ax.plot(v, yy, m, color=c, ms=ms, mec='white', mew=0.5, zorder=3)
    ax.set_yticks([v for v, _ in ys]); ax.set_yticklabels(labels, fontsize=6.4)
    ax.tick_params(axis='y', length=0, pad=4)
    for yy, txt in heads:
        ax.text(-0.02, yy + 0.05, txt, transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=6.5, fontweight='bold', color=INK)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-0.02, 1.02); ax.set_xticks([0, 0.5, 0.9, 1.0]); ax.set_xticklabels(['0', '0.5', '0.9', ''])
    ax.set_ylim(min(v for v, _ in ys) - 0.7, 0.6)
    ax.set_xlabel('coverage of 90% interval')


def fig4():
    rob = json.load(open(RES / 'e04_robustness' / 'robustness_summary.json'))
    vio = json.load(open(RES / 'e05_violations' / 'violations_summary.json'))
    rows_a = [('#', 'Individual-based, $N_e=N$')]
    for N in (100, 500, 2000):
        for sg in ('0.0', '0.2', '0.5'):
            rows_a.append((f'$N={N}$, $\\sigma={float(sg):g}$', f'N{N}_sigma{sg}_Neknown'))
    rows_a.append(('#', '$N=500$, $N_e$ inferred'))
    for sg in ('0.0', '0.2', '0.5'):
        rows_a.append((f'$\\sigma={float(sg):g}$', f'N500_sigma{sg}_Neinferred'))
    rows_a += [('#', '$N=500$, $\\sigma=0.2$, $N_e$ wrong'), ('$N_e$ set to $N/2$', 'N500_sigma0.2_Nehalf'),
               ('$N_e$ set to $2N$', 'N500_sigma0.2_Nedouble')]
    rows_b = [('#', 'Little effect'), ('baseline', 'baseline'), ('finite $N=1000$', 'finite_pop'),
              ('zygotes sampled', 'zygote_sampled'), ('bottleneck, $N_e$ inferred', 'bottleneck_Ne_inferred'),
              ('#', 'Absorbed by $F$'), ('genotyping error', 'genotyping_error'),
              ('Wahlund sample', 'wahlund_sampling'), ('selfing starts at $t=0$', 'selfing_transient'),
              ('#', 'Bias in $s$'), ('linked selection', 'linked_selection'),
              ('bottleneck, $N_e$ fixed', 'bottleneck_Ne_fixed'), ('$s$ reverses at $T/2$', 'changing_s')]
    fig = plt.figure(figsize=(FULL, 3.45))
    axes = [fig.add_axes([0.215, 0.12, 0.27, 0.80]), fig.add_axes([0.725, 0.12, 0.27, 0.80])]
    _coverage_panel(axes[0], rows_a, rob, 300, 'a'); tag(axes[0], 'a', dx=-95)
    _coverage_panel(axes[1], rows_b, vio, 500, 'b'); tag(axes[1], 'b', dx=-85)
    hs = [plt.Line2D([], [], marker=m, color=c, ls='none', ms=ms, mec='white', mew=0.5)
          for m, c, ms in [('o', C_F, 3.8), ('s', C_S, 3.6), ('D', C_HS, 3.2)]]
    fig.legend(hs, ['$F$', '$s$', '$hs$'], loc='upper right', ncol=3, bbox_to_anchor=(0.995, 1.01),
               handletextpad=0.2, columnspacing=1.4, fontsize=7)
    save(fig, 'fig4_robustness')


# ── Figure 5 ────────────────────────────────────────────────────────────────
def fig5():
    mg = json.load(open(RES / 'e07_migration' / 'migration_summary.json'))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.15), gridspec_kw={'width_ratios': [1, 1.05]})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=0.92, wspace=0.55)
    ax = axes[0]
    keys = ['0-0.01', '0.01-0.02', '0.02-0.05', '0.05-0.1', '0.1-1.0']
    ax.axhline(0.05, color=INK2, lw=0.6, ls=(0, (1.5, 1.5)))
    ax.text(4.45, 0.065, '5% false positives', fontsize=6.2, color=INK2, ha='right', va='bottom')
    for tg, col, mk, dx, mfc in [('Ne_known', INK, 'o', -0.09, INK), ('Ne_inferred', MUTED, 's', 0.09, 'white')]:
        B = mg[tg]['TPR_at_5pctFPR_by_shift']
        x, yv, lo, hi = [], [], [], []
        for i, k in enumerate(keys):
            v, n = B[k]['TPR_comb'], B[k]['n']
            if v is None or n == 0:
                continue
            ci = stats.beta.ppf([0.025, 0.975], v * n + 0.5, (1 - v) * n + 0.5)
            x.append(i + dx); yv.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
        ax.errorbar(x, yv, yerr=[lo, hi], fmt=mk, color=col, mfc=mfc, mec=col, ms=3.8, mew=0.9, lw=0.8, capsize=0)
    ax.text(1.0, 0.36, '$N_e$ known', color=INK, fontsize=6.8)
    ax.text(1.0, 0.29, '$N_e$ inferred', color=INK2, fontsize=6.8)
    ax.plot([0.75], [0.372], 'o', color=INK, ms=3.8); ax.plot([0.75], [0.302], 's', mfc='white', mec=MUTED, ms=3.8, mew=0.9)
    ax.set_xticks(range(5)); ax.set_xticklabels(['<0.01', '0.01–\n0.02', '0.02–\n0.05', '0.05–\n0.1', '>0.1'], linespacing=0.95)
    ax.set_xlim(-0.4, 4.4); ax.set_ylim(-0.02, 0.75)
    ax.set_xlabel(r'allele-frequency shift caused by the pulse, $|\Delta p|$'); ax.set_ylabel('detection rate')
    tag(ax, 'a')
    ax = axes[1]
    fl = mg['Ne_known']['flag_rate_other_violations']
    names = ['changing_s', 'bottleneck', 'finite_pop', 'linked_selection', 'genotyping_error', 'wahlund_sampling']
    lab = {'changing_s': '$s$ reverses at $T/2$', 'bottleneck': 'bottleneck', 'finite_pop': 'individual-based drift',
           'linked_selection': 'linked selection', 'genotyping_error': 'genotyping error', 'wahlund_sampling': 'Wahlund sample'}
    y = np.arange(len(names))[::-1]
    ax.axvline(0.05, color=INK2, lw=0.6, ls=(0, (1.5, 1.5)))
    for xv in (0.2, 0.4, 0.6):
        ax.axvline(xv, color='#efeee9', lw=0.5, zorder=0)
    for yy, n in zip(y, names):
        v = [fl[n]['score_pred'], fl[n]['score_D'], fl[n]['score_comb']]
        ax.plot([min(v), max(v)], [yy, yy], color=RULE, lw=0.8, zorder=1)
        ax.plot(v[0], yy, 'o', color=C_S, ms=3.8, mec='white', mew=0.5, zorder=3)
        ax.plot(v[1], yy, 's', color=C_F, ms=3.5, mec='white', mew=0.5, zorder=3)
        ax.plot(v[2], yy, 'D', color=INK, ms=3.4, mec='white', mew=0.5, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([lab[n] for n in names]); ax.tick_params(axis='y', length=0, pad=4)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 0.72); ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel('flag rate at the 5% false-positive threshold')
    hs = [plt.Line2D([], [], marker=m, color=c, ls='none', ms=ms, mec='white', mew=0.5)
          for m, c, ms in [('o', C_S, 3.8), ('s', C_F, 3.5), ('D', INK, 3.4)]]
    ax.legend(hs, ['predictive residual', 'HW deviation', 'combined'], loc='lower right', handletextpad=0.2,
              fontsize=6.6, borderaxespad=0.1)
    tag(ax, 'b', dx=-62)
    save(fig, 'fig5_adequacy')


# ── empirical helpers ───────────────────────────────────────────────────────
def _adult(p, F, s, h):
    q = 1 - p
    z = np.array([p * p + F * p * q, 2 * p * q * (1 - F), q * q + F * p * q])
    z = np.clip(z, 0, None); z = z / z.sum(0)
    a = z * np.array([1.0, 1 - h * s, 1 - s])
    return a / a.sum(0)


def _band(Y, q=(5, 50, 95)):
    return [np.nanpercentile(Y, k, 0) for k in q]


def _hpd_levels(Z, masses):
    zs = np.sort(Z.ravel())[::-1]; c = np.cumsum(zs); c /= c[-1]
    return [zs[np.searchsorted(c, m)] for m in masses]


# ── Figure 6 ────────────────────────────────────────────────────────────────
def fig6():
    df = pd.read_csv(DATA / 'panaxia.csv')
    z = np.load(REFIT / 'panaxia_Ne_FisherFord.npz')
    th, paths, flat, phi = z['th'], z['paths'], z['flat'], z['phi']
    yrs = df.generation.values
    n = (df.N_AA + df.N_Aa + df.N_aa).values
    k = (df.N_Aa + 2 * df.N_aa).values
    qobs = k / (2 * n)
    lo, hi = stats.beta.ppf([[0.025], [0.975]], k + 0.5, 2 * n - k + 0.5)
    Q = np.array([[1 - (lambda a: a[0] + a[1] / 2)(_adult(paths[i, t], th[i, 0], th[i, 1], th[i, 2]))
                   for t in range(paths.shape[1])] for i in range(len(th))])
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.1), gridspec_kw={'width_ratios': [1.05, 1, 1.05]})
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.2, top=0.9, wspace=0.5)
    ax = axes[0]
    b5, b50, b95 = _band(Q)
    ax.fill_between(yrs, b5, b95, color=BLUE_LIGHT, lw=0)
    ax.plot(yrs, b50, color=C_F, lw=1.3)
    ax.errorbar(yrs, qobs, yerr=[qobs - lo, hi - qobs], fmt='o', color=INK, ms=3.2, lw=0.8, capsize=0,
                mec='white', mew=0.5)
    for x, nn in zip(yrs, n):
        pass
    ax.set_xlabel('year'); ax.set_ylabel('frequency of $medionigra$')
    ax.set_xticks([1939, 1941, 1943, 1945]); ax.set_ylim(0, 0.165); ax.set_xlim(1938.5, 1946.5)
    ax.text(1938.8, 0.018, 'samples, 95% CI', ha='left', fontsize=6.2, color=INK)
    ax.text(1938.8, 0.004, 'posterior median, 90% band', ha='left', fontsize=6.2, color=C_F)
    tag(ax, 'a')
    ax = axes[1]
    s_, h_ = flat[:, 1], flat[:, 2]
    kde = stats.gaussian_kde(np.vstack([s_, h_]), bw_method=0.18)
    gx, gy = np.meshgrid(np.linspace(-0.45, 0.62, 220), np.linspace(-1.6, 2.0, 220))
    Z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    lv = _hpd_levels(Z, [0.95, 0.5])
    ax.contourf(gx, gy, Z, levels=[lv[0], lv[1], Z.max() * 1.01], colors=[ORANGE_LIGHT, '#f4a283'], zorder=1)
    ax.contour(gx, gy, Z, levels=[lv[0], lv[1]], colors=[C_S], linewidths=[0.5, 0.7], zorder=2)
    sel = np.random.default_rng(1).choice(len(flat), 1500, replace=False)
    ax.scatter(s_[sel], h_[sel], s=1.0, color=C_S, alpha=0.35, lw=0, rasterized=True, zorder=2)
    hs_med = float(np.median(flat[:, 4]))
    for branch in (np.linspace(0.045, 0.62, 300), np.linspace(-0.45, -0.045, 300)):
        ax.plot(branch, hs_med / branch, color=INK, lw=0.8, zorder=3)
        for v in np.percentile(flat[:, 4], [5, 95]):
            ax.plot(branch, v / branch, color=INK, lw=0.5, ls=(0, (1.5, 1.5)), zorder=3)
    ax.axhline(0, color=RULE, lw=0.5, zorder=0); ax.axvline(0, color=RULE, lw=0.5, zorder=0)
    ax.set_xlim(-0.45, 0.62); ax.set_ylim(-1.6, 2.0)
    ax.set_xlabel('$s$'); ax.set_ylabel('$h$')
    ax.text(0.6, -0.3, f'$hs={hs_med:.3f}$', fontsize=6.4, ha='right', va='top', color=INK)
    ax.annotate('minor mode\n(2.3%)', xy=(-0.17, -0.6), xytext=(-0.43, 1.75), fontsize=6.0, color=INK2,
                va='top', ha='left', arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5, shrinkA=1, shrinkB=2))
    tag(ax, 'b')
    ax = axes[2]
    names = ['mutation', 'reset', 'inbreeding', 'drift', 'selection']
    cum = np.nansum(phi, 1) * 1e3
    y = np.arange(5)[::-1]
    ax.axvline(0, color=INK2, lw=0.6)
    for j in range(5):
        l5, l50, l95 = np.nanpercentile(cum[:, j], [5, 50, 95])
        ax.plot([l5, l95], [y[j], y[j]], color=INK, lw=1.4, solid_capstyle='round')
        ax.plot(l50, y[j], 'o', color='white', mec=INK, ms=4.2, mew=1.0)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.tick_params(axis='y', length=0, pad=4)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-0.6, 4.6); ax.set_xticks([-10, 0, 10, 20])
    ax.set_xlabel(r'cumulative Shapley value ($\times10^{-3}$)')
    tag(ax, 'c', dx=-14)
    save(fig, 'fig6_panaxia')


# ── Figure 7 ────────────────────────────────────────────────────────────────
def fig7():
    df = pd.read_csv(DATA / 'peppered_moth.csv')
    z = np.load(REFIT / 'moth_h0_n100_NeInf.npz')
    th, paths = z['th'], z['paths']
    zf = np.load(REFIT / 'moth_hfree_n100_NeInf.npz')
    em = json.load(open(RES / 'e09_empirical' / 'empirical_summary.json'))['peppered_moth']
    yr0 = int(df.year.values[0])
    gens = df.generation.values; f = df.f_carbonaria.values
    n = 100
    lo, hi = stats.beta.ppf([[0.025], [0.975]], f * n + 0.5, (1 - f) * n + 0.5)
    Pd = np.array([[(lambda a: a[0] + a[1])(_adult(paths[i, t], th[i, 0], th[i, 1], th[i, 2]))
                    for t in range(paths.shape[1])] for i in range(len(th))])
    fig = plt.figure(figsize=(FULL, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.95, 0.8], wspace=0.48,
                          left=0.07, right=0.985, bottom=0.2, top=0.9)
    ax = fig.add_subplot(gs[0])
    tt = np.arange(paths.shape[1]) + yr0
    b5, b50, b95 = _band(Pd)
    ax.fill_between(tt, b5, b95, color=BLUE_LIGHT, lw=0)
    ax.plot(tt, b50, color=C_F, lw=1.3)
    ax.errorbar(gens + yr0, f, yerr=[f - lo, hi - f], fmt='o', color=INK, ms=2.6, lw=0.6, capsize=0,
                mec='white', mew=0.4)
    ax.set_xlabel('year'); ax.set_ylabel('frequency of $carbonaria$ phenotype'); ax.set_ylim(0, 1.03)
    ax.set_xlim(1957, 2003)
    ax.text(1958.5, 0.36, 'samples, 95% CI\nat $n=100$', fontsize=6.2, color=INK, ha='left', va='top', linespacing=1.0)
    ax.text(1958.5, 0.17, 'posterior median,\n90% band', fontsize=6.2, color=C_F, ha='left', va='top', linespacing=1.0)
    tag(ax, 'a')
    ax = fig.add_subplot(gs[1])
    cfgs = [('h0_n100_NeInf', 'main analysis'), ('h0_n50_NeInf', r'$n=50$'),
            ('h0_n300_NeInf', r'$n=300$'), ('h0_n100_Ne1000', r'$N_e=1000$'),
            ('hfree_n100_NeInf', r'$h$ free')]
    ref = em['h0_n100_NeInf']['posterior']['s']['median']
    ax.axvline(ref, color=RULE, lw=0.6, zorder=0)
    for j, (k, lab) in enumerate(cfgs):
        S = em[k]['posterior']['s']
        yy = len(cfgs) - 1 - j
        col = C_S if j == 0 else INK
        ax.plot([S['q05'], S['q95']], [yy, yy], color=col, lw=1.4)
        ax.plot(S['median'], yy, 'o', color='white', mec=col, ms=4.0, mew=1.0)
    ax.set_yticks(range(len(cfgs))[::-1]); ax.set_yticklabels([c[1] for c in cfgs])
    ax.tick_params(axis='y', length=0, pad=4); ax.spines['left'].set_visible(False)
    ax.set_xlim(-0.265, -0.18); ax.set_xticks([-0.26, -0.24, -0.22, -0.20, -0.18])
    ax.set_xticklabels(['−0.26', '', '−0.22', '', '−0.18'])
    ax.set_ylim(-0.6, len(cfgs) - 0.4)
    ax.set_xlabel('selection coefficient $s$')
    tag(ax, 'b', dx=-42)
    ax = fig.add_subplot(gs[2])
    hh = zf['flat'][:, 2]
    cnt, edges = np.histogram(hh, bins=np.linspace(-1.0, 1.0, 41), density=True)
    ax.bar(edges[:-1], cnt, width=np.diff(edges) * 0.82, align='edge', color=C_H, lw=0)
    xx = np.linspace(-1.0, 1.0, 200)
    ax.plot(xx, stats.norm.pdf(xx, 0.5, 0.5), color=INK, lw=0.8, ls=(0, (2, 1.5)))
    ax.text(0.98, 0.95, 'prior', transform=ax.transAxes, ha='right', fontsize=6.4, color=INK)
    ax.text(0.03, 0.95, 'posterior', transform=ax.transAxes, ha='left', fontsize=6.4, color=C_H)
    ax.set_xlabel('dominance $h$'); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlim(-1.0, 1.0)
    tag(ax, 'c', dx=-8)
    save(fig, 'fig7_biston')


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7']
    for w in which:
        globals()[f'fig{w}']()
        print('figure', w, 'done')
