"""E01 theory checks, E02 identifiability (Fisher information), E06 operator order.

No MCMC. Fast. Produces the analytical backbone of the paper:
  E01a  drift raises Wright's F but not F_IS (why the legacy Layer 0 was wrong)
  E01b  identities: closed-form adult D, Shapley efficiency / dummy, mutation map
  E01c  selection-induced HW deviation kappa = w3 - w2^2 = s(2h-1) - h^2 s^2
  E01d  rare-allele effective selection s_eff = s[F + (1-F) h]
  E02   Fisher-information identifiability over parameter space and designs
  E06   attribution vs operator order (Shapley vs all 120 life-cycle orderings)
"""
import itertools
import numpy as np
import numba as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import outdir, save_json, write_csv, nrep, COL, model, sims


# ── E01a ─────────────────────────────────────────────────────────────────────
def drift_vs_fis(cfg):
    out = outdir(cfg, 'e01_theory')
    N, T = 50, 60
    reps = nrep(4000, cfg, 200)
    fis = np.full((reps, T + 1), np.nan); hrel = np.zeros((reps, T + 1))
    for r in range(reps):
        P, X, A = sims.simulate(0.5, 0.0, 0.0, 0.5, T, Ne=N, seed=r + 1, mode=1, sigma=0.0, mu1=0, mu2=0)
        pq = P * (1 - P)
        ok = pq > 0
        fis[r, ok] = 1 - X[ok, 1] / (2 * pq[ok])
        hrel[r] = X[:, 1] / 0.5
    wright = 1 - (1 - 1 / (2 * N)) ** np.arange(T + 1)
    res = {'N': N, 'reps': reps,
           'gens': list(range(0, T + 1, 10)),
           'wright_F': [float(wright[t]) for t in range(0, T + 1, 10)],
           'one_minus_H_over_H0': [float(1 - hrel[:, t].mean()) for t in range(0, T + 1, 10)],
           'F_IS_mean': [float(np.nanmean(fis[:, t])) for t in range(0, T + 1, 10)],
           'expected_F_IS': -1 / (2 * N)}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    t = np.arange(T + 1)
    ax.plot(t, wright, color=COL['inbreeding'], lw=2, label="Wright's F (legacy Layer 0 F$_{IS}$)")
    ax.plot(t, 1 - hrel.mean(0), color='k', ls='--', lw=1.2, label='1 − H$_t$/H$_0$ (simulated)')
    ax.plot(t, np.nanmean(fis, 0), color=COL['selection'], lw=2, label='within-population F$_{IS}$ (simulated)')
    ax.axhline(-1 / (2 * N), color='0.5', lw=0.8, ls=':')
    ax.set_xlabel('generation'); ax.set_ylabel('coefficient')
    ax.set_title(f'Random mating, N = {N}: drift lowers H but not F$_{{IS}}$', fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'drift_vs_FIS.png', dpi=200); plt.close(fig)
    return res


# ── E01b-d ───────────────────────────────────────────────────────────────────
def identities(cfg):
    rng = np.random.default_rng(0)
    e_closed = e_mut = e_shap = e_dummy = 0.0
    e_shap_neg = 0.0
    n = 0
    for _ in range(20000):
        p = rng.uniform(0.02, 0.98); s = rng.uniform(-0.6, 0.6); h = rng.uniform(-0.5, 1.5)
        F = rng.uniform(0.0, 0.7)
        if 1 - h * s <= 0.05 or 1 - s <= 0.05:
            continue
        n += 1
        a = model.adults_at(p, F, s, h)
        e_closed = max(e_closed, abs(model.adult_D_closed(p, F, s, h) - model.hw_dev(*a)))
        g = model._apply(0, a[0], a[1], a[2], F, s, h, 1e-3, 2e-3, 0.0)
        e_mut = max(e_mut, abs((g[0] + g[1] / 2) - model.mutate_allele(a[0] + a[1] / 2, 1e-3, 2e-3)))
        m = model.mutate_allele(a[0] + a[1] / 2, 1e-5, 1e-5)
        pn = float(np.clip(m + rng.normal(0, 0.01), 0.01, 0.99))
        phi, seq, tot = model.shapley_step(a[0], a[1], a[2], F, s, h, 1e-5, 1e-5, pn - m)
        target = model.hw_dev(*model.adults_at(pn, F, s, h)) - model.hw_dev(*a)
        e_shap = max(e_shap, abs(phi.sum() - target), abs(seq.sum() - target))
        x = model.zygotes(p, F)
        phi0, _, _ = model.shapley_step(x[0], x[1], x[2], F, 0.0, h, 1e-5, 1e-5, 0.0)
        e_dummy = max(e_dummy, abs(phi0[4]))
    # kappa identity: D^a = (w3 D^z + kappa (x2/2)^2) / wbar^2
    e_kappa = 0.0
    for _ in range(5000):
        p = rng.uniform(0.02, 0.98); s = rng.uniform(-0.6, 0.6); h = rng.uniform(-0.5, 1.5)
        F = rng.uniform(0, 0.7)
        if 1 - h * s <= 0.05:
            continue
        x1, x2, x3 = model.zygotes(p, F)
        w2 = 1 - h * s; w3 = 1 - s; wb = x1 + x2 * w2 + x3 * w3
        kap = w3 - w2 * w2
        e_kappa = max(e_kappa, abs(model.adult_D_closed(p, F, s, h) - (w3 * model.hw_dev(x1, x2, x3) + kap * (x2 / 2) ** 2) / wb ** 2))
    # rare-allele effective selection
    rows = []
    for q in [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2]:
        errs = []
        for _ in range(2000):
            F = rng.uniform(0, 0.6); s = rng.uniform(-0.3, 0.3); h = rng.uniform(0, 1)
            if abs(s) < 0.01:
                continue
            a = model.adults_at(1 - q, F, s, h)
            qn = a[2] + a[1] / 2
            errs.append(abs((qn - q) / q + s * (F + (1 - F) * h)) / abs(s * (F + (1 - F) * h) + 1e-12))
        rows.append({'q': q, 'median_rel_err': float(np.median(errs)), 'p90_rel_err': float(np.percentile(errs, 90))})
    return {'n_checked': n, 'closed_form_Da_max_err': e_closed, 'kappa_identity_max_err': e_kappa,
            'mutation_map_max_err': e_mut, 'shapley_efficiency_max_err': e_shap,
            'dummy_selection_max_abs': e_dummy, 'rare_allele_seff': rows}


def kappa_map(cfg):
    out = outdir(cfg, 'e01_theory')
    hs_ = np.linspace(-0.5, 1.5, 201); ss = np.linspace(-0.5, 0.5, 201)
    M = np.full((len(ss), len(hs_)), np.nan)
    for i, s in enumerate(ss):
        for j, h in enumerate(hs_):
            if 1 - h * s <= 0.02 or 1 - s <= 0.02:
                continue
            M[i, j] = model.adult_D_closed(0.5, 0.0, s, h) / 0.25   # adult F_IS created by selection alone
    fig, ax = plt.subplots(figsize=(5.2, 4))
    vm = np.nanmax(np.abs(M))
    im = ax.imshow(M, origin='lower', extent=[hs_[0], hs_[-1], ss[0], ss[-1]], aspect='auto',
                   cmap='RdBu_r', vmin=-vm, vmax=vm)
    ax.contour(hs_, ss, M, levels=[0], colors='k', linewidths=0.8)
    ax.set_xlabel('dominance h'); ax.set_ylabel('selection s')
    ax.set_title('Adult F$_{IS}$ created by viability selection alone (p = 0.5)', fontsize=9)
    fig.colorbar(im, ax=ax, label='D$^a$ / pq')
    fig.tight_layout(); fig.savefig(out / 'selection_induced_FIS.png', dpi=200); plt.close(fig)


# ── E02 Fisher information ───────────────────────────────────────────────────
@nb.njit(cache=True)
def _obs_probs(theta, gens, stage):
    p0, F, s, h = theta[0], theta[1], theta[2], theta[3]
    k = len(gens); out = np.zeros((k, 3))
    p = p0; j = 0
    for t in range(gens[-1] + 1):
        if j < k and gens[j] == t:
            if stage == 1:
                x = model.zygotes(p, F)
            else:
                x = model.adults_at(p, F, s, h)
            out[j, 0] = x[0]; out[j, 1] = x[1]; out[j, 2] = x[2]
            j += 1
        p = model.gamete_mean(p, F, s, h, 1e-5, 1e-5)
    return out


def fim(theta, gens, n, kind):
    """kind: 'geno' (adult genotypes), 'pheno' (adult, A dominant), 'zygote' (pre-selection)."""
    stage = 1 if kind == 'zygote' else 0
    eps = np.array([1e-5, 1e-5, 1e-5, 1e-5])
    J = []
    for i in range(4):
        tp = theta.copy(); tm = theta.copy(); tp[i] += eps[i]; tm[i] -= eps[i]
        J.append((_obs_probs(tp, gens, stage) - _obs_probs(tm, gens, stage)) / (2 * eps[i]))
    J = np.stack(J, -1)                     # (k, 3, 4)
    P = _obs_probs(theta, gens, stage)
    I = np.zeros((4, 4))
    for t in range(len(gens)):
        if kind == 'pheno':
            a3 = P[t, 2]; g = J[t, 2]
            I += n * np.outer(g, g) / (a3 * (1 - a3) + 1e-300)
        else:
            for c in range(3):
                if P[t, c] > 1e-12:
                    I += n * np.outer(J[t, c], J[t, c]) / P[t, c]
    return I


def identifiability(cfg):
    out = outdir(cfg, 'e01_theory')
    rows = []
    designs = [('T10_n300', 10, 300), ('T30_n300', 30, 300), ('T60_n300', 60, 300),
               ('T30_n50', 30, 50), ('T30_n1000', 30, 1000), ('cross_n300', 0, 300)]
    p0s = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 0.95]
    Fs = [0.0, 0.1, 0.3]
    ss = [-0.2, -0.05, 0.01, 0.05, 0.1, 0.2]
    hs_ = [0.0, 0.25, 0.5, 0.75, 1.0]
    kinds = ['geno', 'pheno', 'zygote']
    for (dn, T, n), kind, p0, F, s, h in itertools.product(designs, kinds, p0s, Fs, ss, hs_):
        gens = np.arange(0, T + 1, 5) if T > 0 else np.array([0])
        th = np.array([p0, F, s, h], float)
        I = fim(th, gens.astype(np.int64), n, kind)
        ev, evec = np.linalg.eigh(I)
        rank = int((ev > ev.max() * 1e-10).sum()) if ev.max() > 0 else 0
        row = {'design': dn, 'kind': kind, 'p0': p0, 'F': F, 's': s, 'h': h, 'rank': rank,
               'cond': float(ev.max() / max(ev.min(), 1e-300)),
               'weak_p0': abs(evec[0, 0]), 'weak_F': abs(evec[1, 0]), 'weak_s': abs(evec[2, 0]), 'weak_h': abs(evec[3, 0])}
        if rank == 4:
            Ci = np.linalg.inv(I)
            sd = np.sqrt(np.clip(np.diag(Ci), 0, None))
            g_hs = np.array([0, 0, h, s])
            g_se = np.array([0, s * (1 - h), F + (1 - F) * h, s * (1 - F)])
            row.update(crlb_p0=sd[0], crlb_F=sd[1], crlb_s=sd[2], crlb_h=sd[3],
                       crlb_hs=float(np.sqrt(max(g_hs @ Ci @ g_hs, 0))),
                       crlb_seff=float(np.sqrt(max(g_se @ Ci @ g_se, 0))),
                       corr_s_h=float(Ci[2, 3] / (sd[2] * sd[3] + 1e-300)),
                       corr_F_h=float(Ci[1, 3] / (sd[1] * sd[3] + 1e-300)),
                       corr_F_s=float(Ci[1, 2] / (sd[1] * sd[2] + 1e-300)))
        rows.append(row)
    write_csv(rows, out / 'identifiability_fim.csv')
    # summaries
    R = rows
    def med(sel, key):
        v = [r[key] for r in R if sel(r) and key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else None
    summ = {'n_cases': len(R),
            'cross_sectional_rank_max': max(r['rank'] for r in R if r['design'] == 'cross_n300'),
            'temporal_full_rank_fraction': float(np.mean([r['rank'] == 4 for r in R if r['design'] != 'cross_n300']))}
    for kind in kinds:
        for rare in [True, False]:
            sel = lambda r, k=kind, rr=rare: (r['kind'] == k and r['design'] == 'T30_n300' and
                                                ((min(r['p0'], 1 - r['p0']) <= 0.05 + 1e-9) == rr) and r['s'] > 0)
            tag = f"{kind}_{'rare' if rare else 'common'}"
            summ[tag] = {'median_crlb_s': med(sel, 'crlb_s'), 'median_crlb_h': med(sel, 'crlb_h'),
                         'median_crlb_hs': med(sel, 'crlb_hs'), 'median_crlb_seff': med(sel, 'crlb_seff'),
                         'median_crlb_F': med(sel, 'crlb_F'), 'median_corr_s_h':
                         med(lambda r, f=sel: f(r) and 'corr_s_h' in r, 'corr_s_h')}
    # figure: CRLB(h) and CRLB(hs) vs p0 for geno, T30
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, lab in zip(axes, ['crlb_s', 'crlb_h', 'crlb_F'], ['s', 'h', 'F']):
        for kind, ls in zip(kinds, ['-', '--', ':']):
            y = [med(lambda r, p=p, k=kind: r['kind'] == k and r['design'] == 'T30_n300' and r['p0'] == p and r['s'] == 0.1, key)
                 for p in p0s]
            ax.plot(p0s, y, ls, marker='o', ms=3, label=kind)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('initial frequency p$_0$')
        ax.set_ylabel(f'Cramér–Rao sd({lab})'); ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Identifiability (T = 30, n = 300, s = 0.1; median over F, h)', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'identifiability_crlb.png', dpi=200); plt.close(fig)
    return summ


# ── E06 operator order ───────────────────────────────────────────────────────
def order_dependence(cfg):
    out = outdir(cfg, 'e01_theory')
    rng = np.random.default_rng(6)
    perms = list(itertools.permutations(range(5)))
    n_paths = nrep(300, cfg, 20)
    rows = []
    flips_life = flips_any = n_gen = 0
    rng_rel = []
    old_sel_share = []; new_sel_share = []
    for r in range(n_paths):
        p0 = rng.uniform(0.05, 0.95); F = rng.uniform(0, 0.4); s = rng.choice([-1, 1]) * rng.uniform(0.01, 0.3)
        h = rng.uniform(-0.2, 1.2); Ne = rng.choice([100, 1000, 10000]); T = 30
        if 1 - h * s <= 0.05:
            continue
        P, X, A = sims.simulate(p0, F, s, h, T, Ne=Ne, seed=r + 11, mode=0)
        for t in range(T):
            a = A[t]
            m = model.mutate_allele(a[0] + a[1] / 2, 1e-5, 1e-5)
            delta = P[t + 1] - m
            phi, seq, tot = model.shapley_step(a[0], a[1], a[2], F, s, h, 1e-5, 1e-5, delta)
            scale = np.abs(phi).sum()
            if scale < 1e-12:
                continue
            contrib = np.zeros((len(perms), 5))
            for pi, perm in enumerate(perms):
                g = (a[0], a[1], a[2]); D0 = model.hw_dev(*g)
                for op in perm:
                    g2 = model._apply(op, g[0], g[1], g[2], F, s, h, 1e-5, 1e-5, delta)
                    contrib[pi, op] = model.hw_dev(*g2) - model.hw_dev(*g)
                    g = g2
            n_gen += 1
            rng_rel.append((contrib.max(0) - contrib.min(0)) / scale)
            lead = np.argmax(np.abs(phi))
            flips_life += int(np.argmax(np.abs(seq)) != lead)
            flips_any += int(np.any(np.argmax(np.abs(contrib), 1) != lead))
        # legacy vs v3 share of selection (same parameters, deterministic legacy recursion)
        from bastion import legacy_v2 as L
        Ne_arr = np.full(T + 1, float(Ne))
        _, _, _, dRF, dRs, dRmu, aF, a_s, amu = L.simulate(p0, F, s, h, 1e-5, 1e-5, Ne_arr, T)
        Dz, Da, DS, ph, sq, tt = model.attribute_path(P, F, s, h, 1e-5, 1e-5)
        old_sel_share.append(float(np.nanmean(np.abs(a_s[1:]))))
        new_sel_share.append(float(np.nanmean(np.abs(DS) / (np.abs(Dz) + np.abs(DS) + 1e-300))))
        rows.append({'p0': p0, 'F': F, 's': s, 'h': h, 'Ne': int(Ne),
                     'legacy_mean_abs_alpha_s': old_sel_share[-1], 'v3_mean_selection_share': new_sel_share[-1]})
    rng_rel = np.array(rng_rel)
    write_csv(rows, out / 'order_and_legacy_shares.csv')
    F_small = [r for r in rows if r['F'] < 0.05]
    return {'n_generations': n_gen,
            'median_range_over_orderings_by_op': dict(zip(model.OP_NAMES, np.median(rng_rel, 0).round(4).tolist())),
            'p90_range_over_orderings_by_op': dict(zip(model.OP_NAMES, np.percentile(rng_rel, 90, 0).round(4).tolist())),
            'frac_lead_force_differs_lifecycle_vs_shapley': flips_life / max(n_gen, 1),
            'frac_lead_force_differs_some_ordering': flips_any / max(n_gen, 1),
            'legacy_vs_v3_selection_share_when_F_lt_0.05': {
                'legacy_mean_abs_alpha_s': float(np.mean([r['legacy_mean_abs_alpha_s'] for r in F_small])) if F_small else None,
                'v3_selection_share': float(np.mean([r['v3_mean_selection_share'] for r in F_small])) if F_small else None}}


def run(cfg, pool):
    res = {}
    print('  E01a drift vs F_IS'); res['drift_vs_FIS'] = drift_vs_fis(cfg)
    print('  E01b identities'); res['identities'] = identities(cfg)
    print('  E01c selection-induced F_IS map'); kappa_map(cfg)
    print('  E02 Fisher-information identifiability'); res['identifiability'] = identifiability(cfg)
    print('  E06 operator order'); res['order_dependence'] = order_dependence(cfg)
    save_json(res, outdir(cfg, 'e01_theory') / 'theory_summary.json')
    return res
