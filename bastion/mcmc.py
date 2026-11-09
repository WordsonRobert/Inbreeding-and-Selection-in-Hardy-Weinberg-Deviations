"""
BASTION — adaptive Metropolis over (F, s, h, [log10 Ne]) with the HMM likelihood.

Priors (all with the change-of-variables Jacobian included):
  F        ~ Uniform(F_lo, F_hi)                    default (-0.2, 0.8)
  s        ~ Normal(0, s_sd^2) truncated to (-0.9, 0.9), default s_sd = 0.15
  h        ~ Normal(h_mu, h_sd^2),                   default (0.5, 0.5)
  log10 Ne ~ Uniform(lNe_lo, lNe_hi)  (only if Ne is inferred), default (1.5, 4.5)
  p0       ~ Beta(2, 2), integrated out by the HMM (posterior via FFBS)
Fitness constraint: 1-hs > 0.01 and 1-s > 0.01.
"""
import math
import numpy as np
import numba as nb
from .model import hmm_forward, predictive_residuals, make_grid, make_levels, K_LEVELS, choose_K, ffbs, gamete_mean, adults_at, \
    attribute_path, hw_dev

# fvals indices
F_LO, F_HI, S_LO, S_HI, S_SD, H_MU, H_SD, LNE_LO, LNE_HI, S_FIX, H_FIX, MU1, MU2 = range(13)


@nb.njit(cache=True)
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


@nb.njit(cache=True)
def unpack(u, iflags, fv):
    """u -> F, s, h, log10Ne (nan if not inferred), log-Jacobian."""
    has_s = iflags[1]; has_h = iflags[2]; has_Ne = iflags[3]
    i = 0
    z = _sig(u[i]); i += 1
    F = fv[F_LO] + z * (fv[F_HI] - fv[F_LO])
    lj = math.log(max(z, 1e-300)) + math.log(max(1 - z, 1e-300)) + math.log(fv[F_HI] - fv[F_LO])
    if has_s:
        z = _sig(u[i]); i += 1
        s = fv[S_LO] + z * (fv[S_HI] - fv[S_LO])
        lj += math.log(max(z, 1e-300)) + math.log(max(1 - z, 1e-300)) + math.log(fv[S_HI] - fv[S_LO])
    else:
        s = fv[S_FIX]
    if has_h:
        h = u[i]; i += 1
    else:
        h = fv[H_FIX]
    lne = np.nan
    if has_Ne:
        z = _sig(u[i]); i += 1
        lne = fv[LNE_LO] + z * (fv[LNE_HI] - fv[LNE_LO])
        lj += math.log(max(z, 1e-300)) + math.log(max(1 - z, 1e-300)) + math.log(fv[LNE_HI] - fv[LNE_LO])
    return F, s, h, lne, lj


@nb.njit(cache=True)
def _level(Ne, NEL):
    for i in range(len(NEL)):
        if Ne <= NEL[i]:
            return i
    return len(NEL) - 1


@nb.njit(cache=True)
def logpost(u, iflags, fv, GR, PR, KL, NEL, Ne_arr, gens, C):
    """Log posterior. With Ne inferred the HMM grid size follows the current Ne (a fixed
    function of theta, so the Metropolis target is well defined); otherwise level 0."""
    F, s, h, lne, lj = unpack(u, iflags, fv)
    if 1.0 - h * s <= 0.01 or 1.0 - s <= 0.01:
        return -np.inf
    lp = lj
    if iflags[1]:
        lp += -0.5 * (s / fv[S_SD]) ** 2
    if iflags[2]:
        lp += -0.5 * ((h - fv[H_MU]) / fv[H_SD]) ** 2
    if iflags[3]:
        Ne_use = np.array([10.0 ** lne])
        lev = _level(10.0 ** lne, NEL)
    else:
        Ne_use = Ne_arr
        lev = 0
    K = KL[lev]
    ll, _ = hmm_forward(GR[lev, :K], PR[lev, :K], F, s, h, fv[MU1], fv[MU2], Ne_use, gens, C, iflags[0], False)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


@nb.njit(cache=True)
def _chol(A):
    d = A.shape[0]
    L = np.zeros_like(A)
    for i in range(d):
        for j in range(i + 1):
            sm = A[i, j]
            for k in range(j):
                sm -= L[i, k] * L[j, k]
            if i == j:
                if sm <= 0.0:
                    return L, False
                L[i, i] = math.sqrt(sm)
            else:
                L[i, j] = sm / L[j, j]
    return L, True


@nb.njit(cache=True)
def mh_run(u0, L0, n_steps, adapt, seed, iflags, fv, GR, PR, KL, NEL, Ne_arr, gens, C):
    """Random-walk Metropolis from u0 with proposal Cholesky L0. If adapt, the proposal
    covariance is re-estimated every 50 steps (Haario-style, window restarted half-way).
    Returns states (n_steps, d), log-posteriors, acceptance rate, final Cholesky factor."""
    np.random.seed(seed)
    d = len(u0)
    u = u0.copy()
    lp = logpost(u, iflags, fv, GR, PR, KL, NEL, Ne_arr, gens, C)
    scale = 2.38 ** 2 / d
    L = L0.copy()
    S1 = np.zeros(d); S2 = np.zeros((d, d)); nw = 0
    out = np.empty((n_steps, d)); lps = np.empty(n_steps); nacc = 0
    for st in range(n_steps):
        z = np.random.standard_normal(d)
        up = u + L @ z
        lpp = logpost(up, iflags, fv, GR, PR, KL, NEL, Ne_arr, gens, C)
        if math.log(np.random.random()) < lpp - lp:
            u = up; lp = lpp; nacc += 1
        if adapt:
            nw += 1; S1 += u; S2 += np.outer(u, u)
            if nw > 2 * d + 10 and st % 50 == 0:
                m = S1 / nw
                Cm = (S2 - nw * np.outer(m, m)) / (nw - 1)
                for i in range(d):
                    Cm[i, i] += 1e-6
                L2, ok2 = _chol(scale * Cm)
                if ok2:
                    L = L2
            if st == n_steps // 2:
                S1[:] = 0.0; S2[:, :] = 0.0; nw = 0
        out[st] = u; lps[st] = lp
    return out, lps, nacc / n_steps, L


@nb.njit(cache=True)
def unpack_many(U, iflags, fv):
    n = U.shape[0]
    out = np.empty((n, 4))
    for i in range(n):
        F, s, h, lne, _ = unpack(U[i], iflags, fv)
        out[i, 0] = F; out[i, 1] = s; out[i, 2] = h; out[i, 3] = lne
    return out


DEFAULTS = dict(F_lo=-0.2, F_hi=0.8, s_lo=-0.9, s_hi=0.9, s_sd=0.15, h_mu=0.5, h_sd=0.5,
                lNe_lo=1.5, lNe_hi=4.5, s_fixed=0.0, h_fixed=0.5, mu1=1e-5, mu2=1e-5)


def make_cfg(emission='geno_adult', infer_s=True, infer_h=True, infer_Ne=False, **kw):
    em = {'geno_adult': 0, 'geno_zygote': 1, 'pheno_adult': 2}[emission]
    c = dict(DEFAULTS); c.update(kw)
    fv = np.array([c['F_lo'], c['F_hi'], c['s_lo'], c['s_hi'], c['s_sd'], c['h_mu'], c['h_sd'],
                   c['lNe_lo'], c['lNe_hi'], c['s_fixed'], c['h_fixed'], c['mu1'], c['mu2']], float)
    iflags = np.array([em, int(infer_s), int(infer_h), int(infer_Ne)], np.int64)
    return iflags, fv


def _logit(z):
    z = min(max(z, 1e-9), 1 - 1e-9)
    return math.log(z / (1 - z))


def rhat_split(ch):
    n = ch.shape[1]; half = n // 2
    sc = np.concatenate([ch[:, :half], ch[:, half:2 * half]], axis=0)
    M, N = sc.shape[0], sc.shape[1]
    cm = sc.mean(1); gm = cm.mean(0)
    B = N / (M - 1) * ((cm - gm) ** 2).sum(0)
    W = sc.var(1, ddof=1).mean(0)
    return np.sqrt(((N - 1) / N * W + B / N) / (W + 1e-30))


def ess(x):
    x = np.asarray(x, float); n = len(x)
    x = x - x.mean(); v = x.var()
    if v < 1e-30:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (v * n)
    rho = 1.0
    for t in range(1, min(n - 1, 4000), 2):
        pr = acf[t] + acf[t + 1]
        if pr < 0:
            break
        rho += 2 * pr
    return n / max(rho, 1.0)


def hdi(x, mass=0.9):
    x = np.sort(np.asarray(x)); n = len(x); k = int(np.floor(mass * n))
    if k < 1:
        return (np.nan, np.nan)
    w = x[k:] - x[:n - k]; i = int(np.argmin(w))
    return (float(x[i]), float(x[i + k]))


class Data:
    """gens: observed generations (sorted ints); C: (n_obs, 3) counts; Ne_arr per generation."""
    def __init__(self, gens, C, Ne_arr=None, Ne=1000.0):
        self.gens = np.asarray(gens, np.int64)
        self.C = np.asarray(C, float)
        T = int(self.gens[-1] - self.gens[0])
        if Ne_arr is None:
            Ne_arr = np.full(max(T, 1), float(Ne))
        self.Ne_arr = np.asarray(Ne_arr, float)
        self.T = T


def run_mcmc(data, cfg, n_chains=4, n_steps=4000, n_warmup=1500, seed=0, K=None, prune_dlp=5.0):
    """n_chains dispersed chains: adaptive warm-up, then fixed-proposal sampling.

    Stuck-chain pruning: after warm-up, a chain whose mean log-posterior (second half of
    warm-up) is more than prune_dlp below the best chain's sits in a mode carrying a
    negligible share of the posterior (< ~e^-5); it is restarted from a state of the
    best chain and re-adapted for n_warmup/3 steps. Modes of comparable mass are kept
    and are then flagged by split R-hat."""
    iflags, fv = cfg
    if iflags[3]:
        K = K_LEVELS if K is None else (K,)
    else:
        K = (choose_K(float(np.max(data.Ne_arr))),) if K is None else (K,)
    GR, PR, KL, NEL = make_levels(K)
    d = 1 + int(iflags[1]) + int(iflags[2]) + int(iflags[3])
    init_sd = np.array([0.3] * d)
    L0 = np.diag(np.sqrt(2.38 ** 2 / d) * init_sd)
    args = (iflags, fv, GR, PR, KL, NEL, data.Ne_arr, data.gens, data.C)
    starts, rngs = [], []
    for c in range(n_chains):
        r = np.random.default_rng(seed * 7919 + c * 104729 + 1)
        u0 = []
        u0.append(_logit((r.uniform(-0.1, 0.5) - fv[F_LO]) / (fv[F_HI] - fv[F_LO])))
        if iflags[1]:
            u0.append(_logit((r.uniform(-0.3, 0.3) - fv[S_LO]) / (fv[S_HI] - fv[S_LO])))
        if iflags[2]:
            u0.append(r.uniform(-0.5, 1.5))
        if iflags[3]:
            u0.append(_logit(r.uniform(0.15, 0.85)))
        u0 = np.array(u0)
        lp0 = logpost(u0, *args)
        tries = 0
        while not np.isfinite(lp0) and tries < 100:
            u0 = u0 + r.normal(0, 0.3, d)
            if iflags[1] and iflags[2]:
                u0[2] = r.uniform(-0.2, 1.2)
            lp0 = logpost(u0, *args)
            tries += 1
        if np.isfinite(lp0):
            starts.append(u0); rngs.append(r)
    if not starts:
        raise RuntimeError('no chain could start')
    # phase 1: adaptive warm-up
    W = []
    for u0, r in zip(starts, rngs):
        U, lps, acc, L = mh_run(u0, L0, n_warmup, True, int(r.integers(1, 2 ** 31 - 1)), *args)
        W.append([U, lps, L])
    mlp = np.array([w[1][n_warmup // 2:].mean() for w in W])
    best = int(np.argmax(mlp))
    n_pruned = 0
    for c in range(len(W)):
        if mlp[c] < mlp[best] - prune_dlp:
            n_pruned += 1
            Ub = W[best][0][n_warmup // 2:]
            u_new = Ub[int(rngs[c].integers(len(Ub)))]
            U, lps, acc, L = mh_run(u_new, W[best][2], max(n_warmup // 3, 100), True,
                                    int(rngs[c].integers(1, 2 ** 31 - 1)), *args)
            W[c] = [U, lps, L]
    # phase 2: sampling with the adapted proposal
    chains, accs = [], []
    for c in range(len(W)):
        U, lps, acc, L = mh_run(W[c][0][-1], W[c][2], n_steps - n_warmup, False,
                                int(rngs[c].integers(1, 2 ** 31 - 1)), *args)
        chains.append(unpack_many(U, iflags, fv)); accs.append(acc)
    ch = np.stack(chains)                       # (chains, n, 4): F, s, h, log10Ne
    names = ['F', 's', 'h', 'log10Ne']
    active = [True, bool(iflags[1]), bool(iflags[2]), bool(iflags[3])]
    # derived: heterozygote selection hs and rare-allele effective selection s[F+(1-F)h]
    hs = ch[..., 2] * ch[..., 1]
    seff = ch[..., 1] * (ch[..., 0] + (1 - ch[..., 0]) * ch[..., 2])
    full = np.concatenate([ch, hs[..., None], seff[..., None]], axis=-1)
    names_full = names + ['hs', 's_eff']
    act_full = active + [active[1], active[1]]
    rh = np.full(len(names_full), np.nan)
    es = np.full(len(names_full), np.nan)
    flat = full.reshape(-1, full.shape[-1])
    for i, a in enumerate(act_full):
        if a:
            rh[i] = rhat_split(full[:, :, i:i + 1])[0] if full.shape[0] > 1 else np.nan
            es[i] = ess(flat[:, i])
    summ = {}
    for i, nm in enumerate(names_full):
        if not act_full[i]:
            continue
        v = flat[:, i]
        if v.std() == 0:
            rh[i] = np.nan                      # a fixed quantity (e.g. hs with h fixed at 0)
        summ[nm] = dict(mean=float(v.mean()), sd=float(v.std()), median=float(np.median(v)),
                        q05=float(np.percentile(v, 5)), q25=float(np.percentile(v, 25)),
                        q75=float(np.percentile(v, 75)), q95=float(np.percentile(v, 95)),
                        hdi90=hdi(v), rhat=float(rh[i]), ess=float(es[i]))
    core_idx = [i for i in range(4) if active[i]]
    conv = bool(np.all(rh[core_idx] < 1.05)) if full.shape[0] > 1 else True
    conv_ident = bool(np.all(rh[[0] + ([4] if active[1] else [])] < 1.05)) if full.shape[0] > 1 else True
    return dict(samples=full, names=names_full, active=act_full, summary=summ, rhat=rh, ess=es,
                converged=conv, converged_F_hs=conv_ident, accept=np.array(accs),
                cfg=cfg, K=K, n_pruned=n_pruned, warmup_mean_logpost=mlp.tolist())


def posterior_paths(mc, data, n_draws=200, seed=0):
    """Thinned posterior draws of theta, each with one FFBS allele-frequency path."""
    iflags, fv = mc['cfg']
    GR, PR, KL, NEL = make_levels(mc['K'])
    flat = mc['samples'].reshape(-1, mc['samples'].shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    th = flat[idx]
    T = data.T
    paths = np.zeros((len(idx), T + 1))
    for i in range(len(idx)):
        F, s, h, lne = th[i, 0], th[i, 1], th[i, 2], th[i, 3]
        Ne_use = np.array([10 ** lne]) if iflags[3] else data.Ne_arr
        lev = _level(10.0 ** lne, NEL) if iflags[3] else 0
        grid, prior = GR[lev, :KL[lev]], PR[lev, :KL[lev]]
        p = ffbs(grid, prior, F, s, h, fv[MU1], fv[MU2], Ne_use, data.gens, data.C,
                 int(iflags[0]), 1, int(rng.integers(1, 2 ** 31 - 1)))
        paths[i] = _dediscretise(p, grid, rng)[0]
    return th, paths


def _dediscretise(paths, grid, rng):
    """Map grid-valued FFBS paths back to continuous allele frequencies: a weight on node
    k of the hat basis stands for mass spread as the hat function on [g_{k-1}, g_{k+1}],
    so each interior value is drawn from that triangular density. The boundary states
    (loss / fixation) stay exact."""
    out = paths.copy()
    ok = np.isfinite(paths)
    k = np.searchsorted(grid, np.where(ok, paths, 0.0))
    k = np.clip(k, 0, len(grid) - 1)
    inner = ok & (k > 0) & (k < len(grid) - 1)
    if inner.any():
        kk = k[inner]
        out[inner] = rng.triangular(grid[kk - 1], grid[kk], grid[kk + 1])
    return out


def attribution(mc, data, th, paths):
    """Posterior level + Shapley change attribution along sampled paths."""
    iflags, fv = mc['cfg']
    n, T1 = paths.shape
    Dz = np.zeros((n, T1)); Da = np.zeros((n, T1)); DS = np.zeros((n, T1))
    phi = np.zeros((n, T1 - 1, 5)); seq = np.zeros((n, T1 - 1, 5))
    for i in range(n):
        if not np.all(np.isfinite(paths[i])):
            Dz[i] = Da[i] = DS[i] = np.nan; phi[i] = seq[i] = np.nan
            continue
        a, b, c, d, e, _ = attribute_path(paths[i], th[i, 0], th[i, 1], th[i, 2], fv[MU1], fv[MU2])
        Dz[i], Da[i], DS[i], phi[i], seq[i] = a, b, c, d, e
    share_inb = Dz / (np.abs(Dz) + np.abs(DS) + 1e-300)
    share_sel = DS / (np.abs(Dz) + np.abs(DS) + 1e-300)
    return dict(Dz=Dz, Da=Da, DS=DS, phi=phi, seq=seq, share_inb=share_inb, share_sel=share_sel)


def discrepancy(mc, data, th, paths, seed=0):
    """Data-only closed-model discrepancy.
    (1) standardised posterior drift per inter-observation interval (allele-frequency
        jumps that drift at the assumed/inferred Ne cannot explain);
    (2) posterior-predictive z-score of the observed HW deviation at each observed time;
    (3) one-step-ahead predictive residual of each observation given the earlier ones.
    score_comb = hypot(score_pred, score_D)."""
    iflags, fv = mc['cfg']
    rng = np.random.default_rng(seed)
    g = data.gens - data.gens[0]
    n = paths.shape[0]
    nint = len(g) - 1
    Z = np.full((n, nint), np.nan)
    Drep = np.full((n, len(g)), np.nan)
    ns = data.C.sum(1)
    for i in range(n):
        if not np.all(np.isfinite(paths[i])):
            continue
        F, s, h, lne = th[i, 0], th[i, 1], th[i, 2], th[i, 3]
        for k in range(nint):
            num = 0.0; den = 0.0
            for t in range(g[k], g[k + 1]):
                m = gamete_mean(paths[i, t], F, s, h, fv[MU1], fv[MU2])
                Ne = 10 ** lne if iflags[3] else data.Ne_arr[min(t, len(data.Ne_arr) - 1)]
                num += paths[i, t + 1] - m
                den += m * (1 - m) / (2 * Ne)
            Z[i, k] = num / np.sqrt(den + 1e-300)
        if iflags[0] != 2:
            for k in range(len(g)):
                a1, a2, a3 = adults_at(paths[i, g[k]], F, s, h) if iflags[0] == 0 else \
                    adults_at(paths[i, g[k]], F, 0.0, 0.5)
                pr = np.clip([a1, a2, a3], 0, None); pr = pr / pr.sum()
                c = rng.multinomial(int(ns[k]), pr)
                pa = (2 * c[0] + c[1]) / (2 * ns[k])
                Drep[i, k] = c[0] / ns[k] - pa * pa
    Zbar = np.nanmean(Z, 0)
    out = dict(Z=Zbar, score_jump=float(np.nanmax(np.abs(Zbar))) if nint else 0.0,
               jump_interval=int(np.nanargmax(np.abs(Zbar))) if nint else -1)
    # (3) one-step-ahead predictive residuals from the forward filter (uses only the past
    #     of each observation, so a pulse is not smoothed away by later data)
    GR, PR, KL, NEL = make_levels(mc['K'])
    R = np.full((n, len(g)), np.nan)
    for i in range(min(n, 100)):
        F, s, h, lne = th[i, 0], th[i, 1], th[i, 2], th[i, 3]
        Ne_use = np.array([10 ** lne]) if iflags[3] else data.Ne_arr
        lev = _level(10.0 ** lne, NEL) if iflags[3] else 0
        R[i] = predictive_residuals(GR[lev, :KL[lev]], PR[lev, :KL[lev]], F, s, h, fv[MU1], fv[MU2], Ne_use,
                                    data.gens, data.C, int(iflags[0]))
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            Rbar = np.nanmean(R, 0)
    if len(g) > 1 and np.isfinite(Rbar[1:]).any():
        kmax = int(np.nanargmax(np.abs(Rbar[1:]))) + 1
        out.update(R=Rbar, score_pred=float(np.abs(Rbar[kmax])), pred_interval=kmax - 1)
    else:
        out.update(R=Rbar, score_pred=0.0, pred_interval=-1)
    if iflags[0] != 2:
        p_obs = (2 * data.C[:, 0] + data.C[:, 1]) / (2 * ns)
        D_obs = data.C[:, 0] / ns - p_obs ** 2
        zD = (D_obs - np.nanmean(Drep, 0)) / (np.nanstd(Drep, 0) + 1e-12)
        out.update(zD=zD, score_D=float(np.nanmax(np.abs(zD))), D_obs=D_obs)
    else:
        out.update(zD=None, score_D=0.0)
    out['score_comb'] = float(np.hypot(out['score_pred'], out['score_D']))
    return out
