"""
BASTION v2 core — compiled (numba) port of Layers 0, 2 and 3.

Faithful to the original code (simulation_wf.py, layer2_mcmc.py,
layer3_decomposition.py): same recursion, same likelihood, same priors
(s ~ N(0, 0.15^2), h ~ N(0.5, 0.5^2), p0 ~ Beta(2,2), F0 uniform on its
admissible range), same transforms and same adaptive Metropolis scheme.

Changes, each behind a switch so the original behaviour can be reproduced:
  jacobian   : add log|d theta / d u| so the priors are the stated priors
               (original: False — priors were silently changed by the transforms)
  init       : 'original' (all chains start near the data estimate) or
               'dispersed' (chains start spread over parameter space, so
               R-hat means something)
Summaries report both the equal-tailed 5-95% interval ('eti90', what the
original code called 'hdi90') and the true 90% highest-density interval.
"""
import numpy as np
import numba as nb

LOG2 = np.log(2.0)

# ─────────────────────────────────────────────────────────────────────────────
# Layer 0: deterministic recursion (identical to simulation_wf.step_wf)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def step(p, F, Ne, s, h, mu1, mu2):
    q = 1.0 - p
    wAa = 1.0 - h * s
    waa = 1.0 - s
    wbar = p * p + 2.0 * p * q * wAa + q * q * waa
    if wbar <= 0.0:
        dps = 0.0
    else:
        w1 = p + q * wAa
        dps = p * (w1 / wbar - 1.0)
    ps = p + dps
    dpm = mu2 * (1.0 - ps) - mu1 * ps
    pn = ps + dpm
    if pn < 1e-12:
        pn = 1e-12
    elif pn > 1.0 - 1e-12:
        pn = 1.0 - 1e-12
    Ft = 1.0 / (2.0 * Ne) + (1.0 - 1.0 / (2.0 * Ne)) * F
    Fn = Ft * (1.0 - mu1 - mu2) ** 2
    qn = 1.0 - pn
    mx = max(pn, qn)
    lb = -min(pn, qn) / mx if mx > 0 else -1.0
    if Fn < lb + 1e-12:
        Fn = lb + 1e-12
    elif Fn > 1.0 - 1e-12:
        Fn = 1.0 - 1e-12
    return pn, Fn, dps, dpm, Fn - F


@nb.njit(cache=True)
def simulate(p0, F0, s, h, mu1, mu2, Ne_arr, T):
    """Full trajectory + true decomposition (matches simulation_wf.simulate)."""
    p = np.zeros(T + 1); F = np.zeros(T + 1); R = np.zeros(T + 1)
    aF = np.full(T + 1, np.nan); a_s = np.full(T + 1, np.nan)
    amu = np.full(T + 1, np.nan)
    dRF = np.full(T + 1, np.nan); dRs = np.full(T + 1, np.nan)
    dRmu = np.full(T + 1, np.nan)
    p[0] = p0; F[0] = F0; R[0] = F0 * p0 * (1 - p0)
    for t in range(1, T + 1):
        pp = p[t - 1]; Fp = F[t - 1]
        Ne = Ne_arr[min(t - 1, len(Ne_arr) - 1)]
        pn, Fn, dps, dpm, dF = step(pp, Fp, Ne, s, h, mu1, mu2)
        p[t] = pn; F[t] = Fn; R[t] = Fn * pn * (1 - pn)
        a = pp * (1 - pp) * dF
        b = Fp * (1 - 2 * pp) * dps
        c = Fp * (1 - 2 * pp) * dpm
        dRF[t] = a; dRs[t] = b; dRmu[t] = c
        Z = abs(a) + abs(b) + abs(c)
        if Z < 1e-30:
            aF[t] = 0.0; a_s[t] = 0.0; amu[t] = 0.0
        else:
            aF[t] = a / Z; a_s[t] = b / Z; amu[t] = c / Z
    return p, F, R, dRF, dRs, dRmu, aF, a_s, amu


@nb.njit(cache=True)
def genotype_probs(p, F):
    q = 1.0 - p
    return p * p + F * p * q, 2 * p * q * (1 - F), q * q + F * p * q


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: likelihood, priors, transforms
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def loglik_temporal(p0, F0, s, h, mu1, mu2, Ne_arr, gens, NAA, NAa, Naa):
    g0 = gens[0]
    T = gens[-1] - g0
    nobs = len(gens)
    p = p0; F = F0; ll = 0.0; k = 0
    for st in range(T + 1):
        g = g0 + st
        if k < nobs and gens[k] == g:
            q = 1.0 - p
            PAA = max(p * p + F * p * q, 1e-300)
            PAa = max(2 * p * q * (1 - F), 1e-300)
            Paa = max(q * q + F * p * q, 1e-300)
            ll += NAA[k] * np.log(PAA) + NAa[k] * np.log(PAa) + Naa[k] * np.log(Paa)
            k += 1
            if k == nobs:
                break
        Ne = Ne_arr[min(st, len(Ne_arr) - 1)]
        p, F, _a, _b, _c = step(p, F, Ne, s, h, mu1, mu2)
        if not (p > 0.0 and p < 1.0) or not np.isfinite(F):
            return -np.inf
    return ll


@nb.njit(cache=True)
def loglik_pheno(p0, F0, s, h, mu1, mu2, Ne_arr, gens, Ndom, Nrec):
    """Phenotype likelihood: A dominant, so only aa is distinguishable.
    P(recessive phenotype) = q^2 + F p q. Ndom = AA+Aa count, Nrec = aa count."""
    g0 = gens[0]
    T = gens[-1] - g0
    nobs = len(gens)
    p = p0; F = F0; ll = 0.0; k = 0
    for st in range(T + 1):
        g = g0 + st
        if k < nobs and gens[k] == g:
            q = 1.0 - p
            Prec = min(max(q * q + F * p * q, 1e-300), 1.0 - 1e-16)
            ll += Ndom[k] * np.log(1.0 - Prec) + Nrec[k] * np.log(Prec)
            k += 1
            if k == nobs:
                break
        Ne = Ne_arr[min(st, len(Ne_arr) - 1)]
        p, F, _a, _b, _c = step(p, F, Ne, s, h, mu1, mu2)
        if not (p > 0.0 and p < 1.0) or not np.isfinite(F):
            return -np.inf
    return ll


@nb.njit(cache=True)
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@nb.njit(cache=True)
def unpack(u, model):
    """u -> (p0, F0, s, h, lb_clamped). model 0 = temporal, 1 = cross-sectional."""
    p0 = _sig(u[0])
    q0 = 1.0 - p0
    lb = -min(p0, q0) / max(p0, q0)
    lbc = max(lb, -0.9999)
    F0 = lbc + _sig(u[1]) * (1.0 - lbc)
    if model == 0 or model == 2:
        s = _sig(u[2]) * 2.0 - 1.0
        h = u[3]
    elif model == 3:
        s = _sig(u[2]) * 2.0 - 1.0
        h = 0.0          # phenotype selection with complete dominance (w_Aa = w_AA)
    else:
        s = 0.0; h = 0.5
    return p0, F0, s, h, lbc


@nb.njit(cache=True)
def logpost(u, model, Ne_arr, gens, NAA, NAa, Naa, mu1, mu2, jac, s_sd, h_sd):
    p0, F0, s, h, lbc = unpack(u, model)
    q0 = 1.0 - p0
    lb_actual = -min(p0, q0) / max(p0, q0)
    if F0 <= lb_actual or F0 >= 1.0 or p0 <= 0.0 or p0 >= 1.0:
        return -np.inf
    lp = np.log(p0) + np.log(1.0 - p0)
    # (fixed-mu prior terms in the original are constants; omitted)
    if model == 0 or model == 2:
        lp += -0.5 * (s / s_sd) ** 2 - 0.5 * ((h - 0.5) / h_sd) ** 2
    elif model == 3:
        lp += -0.5 * (s / s_sd) ** 2
    if jac:
        lp += np.log(p0) + np.log(1.0 - p0)
        lp += np.log(F0 - lbc) + np.log(1.0 - F0) - np.log(1.0 - lbc)
        if model != 1:
            lp += np.log(1.0 - s * s) - LOG2
    if model == 0:
        ll = loglik_temporal(p0, F0, s, h, mu1, mu2, Ne_arr, gens, NAA, NAa, Naa)
    elif model == 2 or model == 3:
        ll = loglik_pheno(p0, F0, s, h, mu1, mu2, Ne_arr, gens, NAA, Naa)
    else:
        q = 1.0 - p0
        PAA = max(p0 * p0 + F0 * p0 * q, 1e-300)
        PAa = max(2 * p0 * q * (1 - F0), 1e-300)
        Paa = max(q * q + F0 * p0 * q, 1e-300)
        ll = NAA[0] * np.log(PAA) + NAa[0] * np.log(PAa) + Naa[0] * np.log(Paa)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Metropolis (same scheme as layer2_mcmc.adaptive_mh_chain)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _chol_or_none(A):
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
                L[i, i] = np.sqrt(sm)
            else:
                L[i, j] = sm / L[j, j]
    return L, True


@nb.njit(cache=True)
def mh_chain(u0, n_steps, n_warmup, seed, init_cov, adapt_interval,
             model, Ne_arr, gens, NAA, NAa, Naa, mu1, mu2, jac, s_sd, h_sd):
    np.random.seed(seed)
    d = len(u0)
    u = u0.copy()
    lp = logpost(u, model, Ne_arr, gens, NAA, NAa, Naa, mu1, mu2, jac, s_sd, h_sd)
    scale = 2.38 ** 2 / d
    L, ok = _chol_or_none(scale * init_cov)
    S1 = np.zeros(d); S2 = np.zeros((d, d)); nw = 0
    npost = n_steps - n_warmup
    out = np.empty((npost, d)); lps = np.empty(npost)
    nacc = 0; j = 0
    for st in range(n_steps):
        z = np.random.standard_normal(d)
        if ok:
            up = u + L @ z
        else:
            up = u + 0.01 * z
        lpp = logpost(up, model, Ne_arr, gens, NAA, NAa, Naa, mu1, mu2, jac, s_sd, h_sd)
        if np.log(np.random.random()) < lpp - lp:
            u = up; lp = lpp; nacc += 1
        if st < n_warmup:
            nw += 1
            S1 += u
            S2 += np.outer(u, u)
            if nw > d + 1 and st % adapt_interval == 0:
                m = S1 / nw
                C = (S2 - nw * np.outer(m, m)) / (nw - 1)
                for i in range(d):
                    C[i, i] += 1e-6
                L2, ok2 = _chol_or_none(scale * C)
                if ok2:
                    L = L2; ok = True
                else:
                    L, ok = _chol_or_none(scale * np.eye(d) * 0.05 ** 2)
        else:
            out[j] = u; lps[j] = lp; j += 1
    return out, lps, nacc / n_steps


@nb.njit(cache=True)
def unpack_many(U, model):
    n = U.shape[0]
    out = np.empty((n, 4))
    for i in range(n):
        p0, F0, s, h, _ = unpack(U[i], model)
        out[i, 0] = p0; out[i, 1] = F0; out[i, 2] = s; out[i, 3] = h
    return out


def _to_u(p0, F0, s, h, lb_for_F, model):
    lbc = max(lb_for_F, -0.9999)
    Fs = np.clip((F0 - lbc) / (1 - lbc), 1e-9, 1 - 1e-9)
    u = [np.log(p0 / (1 - p0)), np.log(Fs / (1 - Fs))]
    if model != 1:
        ss = np.clip((s + 1) / 2, 1e-9, 1 - 1e-9)
        u += [np.log(ss / (1 - ss)), float(h)]
    return np.array(u)


def rhat_split(ch):
    """ch: (n_chains, n, d) -> split R-hat (same formula as the original)."""
    n = ch.shape[1]; half = n // 2
    sc = np.concatenate([ch[:, :half], ch[:, half:2 * half]], axis=0)
    M, N = sc.shape[0], sc.shape[1]
    cm = sc.mean(1); gm = cm.mean(0)
    B = N / (M - 1) * ((cm - gm) ** 2).sum(0)
    W = sc.var(1, ddof=1).mean(0)
    var_hat = (N - 1) / N * W + B / N
    return np.sqrt(var_hat / (W + 1e-30))


def ess_geyer(x):
    """ESS of a 1-D series via FFT autocorrelation + Geyer initial sequence."""
    x = np.asarray(x, float); n = len(x)
    x = x - x.mean(); v = x.var()
    if v < 1e-30:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (v * n)
    rho = 1.0
    for t in range(1, min(n - 1, 2000), 2):
        pair = acf[t] + acf[t + 1]
        if pair < 0:
            break
        rho += 2 * pair
    return n / max(rho, 1.0)


def hdi(x, mass=0.9):
    x = np.sort(np.asarray(x)); n = len(x)
    k = int(np.floor(mass * n))
    if k < 1 or n == 0:
        return (np.nan, np.nan)
    w = x[k:] - x[:n - k]
    i = int(np.argmin(w))
    return (float(x[i]), float(x[i + k]))


def summarize(samples, names):
    out = {}
    for i, nm in enumerate(names):
        v = samples[:, i]
        out[nm] = {'mean': float(v.mean()), 'sd': float(v.std()),
                   'q05': float(np.percentile(v, 5)), 'q50': float(np.percentile(v, 50)),
                   'q95': float(np.percentile(v, 95)),
                   'eti90': (float(np.percentile(v, 5)), float(np.percentile(v, 95))),
                   'hdi90': hdi(v)}
    return out


def run_mcmc(gens, NAA, NAa, Naa, Ne_arr, mu1=1e-5, mu2=1e-5, model='temporal',
             n_chains=4, n_steps=30000, n_warmup=10000, seed=42,
             jacobian=True, init='dispersed', s_sd=0.15, h_sd=0.5,
             adapt_interval=50):
    """Returns dict with samples (chains, n, 4 constrained), r_hat, ess, summary, ..."""
    m = {'temporal': 0, 'cross_sectional': 1, 'pheno': 2, 'pheno_h0': 3}[model]
    gens = np.asarray(gens, np.int64)
    NAA = np.asarray(NAA, float); NAa = np.asarray(NAa, float); Naa = np.asarray(Naa, float)
    Ne_arr = np.asarray(Ne_arr, float)
    Ntot = NAA + NAa + Naa
    if m in (2, 3):   # NAA = dominant-phenotype count, Naa = recessive count
        p_init = float(np.clip((1 - np.sqrt(Naa / Ntot)).mean(), 0.05, 0.95))
    else:
        p_init = float(np.clip(((2 * NAA + NAa) / (2 * Ntot)).mean(), 0.05, 0.95))
    lb_init = -min(p_init, 1 - p_init) / max(p_init, 1 - p_init)
    names = {0: ['p0', 'F0', 's', 'h'], 1: ['p0', 'F0'], 2: ['p0', 'F0', 's', 'h'],
             3: ['p0', 'F0', 's']}[m]
    scales = {'p0': 0.02, 'F0': 0.02, 's': 0.03, 'h': 0.10}
    init_cov = np.diag(np.array([scales[k] for k in names]) ** 2)
    chains_u, accs = [], []
    for c in range(n_chains):
        r = np.random.default_rng(seed + c * 1000)
        if init == 'original':
            p0c = np.clip(p_init + r.normal(0, 0.04), 0.05, 0.95)
            F0c = np.clip(0.05 + r.normal(0, 0.02), lb_init + 0.001, 0.5)
            sc = r.normal(0, 0.02); hc = r.normal(0.5, 0.1)
            lbF = lb_init
        else:
            p0c = np.clip(p_init + r.normal(0, 0.15), 0.03, 0.97)
            lbc = -min(p0c, 1 - p0c) / max(p0c, 1 - p0c)
            F0c = r.uniform(max(lbc, -0.5) + 0.01, 0.6)
            sc = r.uniform(-0.3, 0.3); hc = r.uniform(-0.5, 1.5)
            lbF = lbc
        u0 = _to_u(p0c, F0c, sc, hc, lbF, m)
        if m == 3:
            u0 = u0[:3]
        lp0 = logpost(u0, m, Ne_arr, gens, NAA, NAa, Naa, mu1, mu2, jacobian, s_sd, h_sd)
        tries = 0
        while not np.isfinite(lp0) and tries < 50:
            u0 = u0 + r.normal(0, 0.1, len(u0))
            lp0 = logpost(u0, m, Ne_arr, gens, NAA, NAa, Naa, mu1, mu2, jacobian, s_sd, h_sd)
            tries += 1
        if not np.isfinite(lp0):
            continue
        cs = int(r.integers(0, 2 ** 31 - 1))
        U, lps, acc = mh_chain(u0, n_steps, n_warmup, cs, init_cov, adapt_interval,
                               m, Ne_arr, gens, NAA, NAa, Naa, mu1, mu2, jacobian, s_sd, h_sd)
        chains_u.append(U); accs.append(acc)
    if not chains_u:
        raise RuntimeError('no chain started')
    ch = np.stack([unpack_many(U, m)[:, :len(names)] for U in chains_u])
    rh = rhat_split(ch)
    flat = ch.reshape(-1, len(names))
    ess = np.array([ess_geyer(flat[:, i]) for i in range(len(names))])
    return {'samples': ch, 'names': names, 'r_hat': rh, 'ess': ess,
            'converged': bool(np.all(rh < 1.05)), 'accept': np.array(accs),
            'summary': summarize(flat, names), 'flat': flat}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: posterior decomposition (matches layer3_decomposition.decompose_posterior)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def decompose_draws(draws, Ne_arr, T, mu1, mu2):
    n = draws.shape[0]
    P = np.empty((T + 1, n)); Fm = np.empty((T + 1, n)); R = np.empty((T + 1, n))
    dRF = np.full((T + 1, n), np.nan); dRs = np.full((T + 1, n), np.nan)
    dRmu = np.full((T + 1, n), np.nan)
    aF = np.full((T + 1, n), np.nan); a_s = np.full((T + 1, n), np.nan)
    amu = np.full((T + 1, n), np.nan)
    for i in range(n):
        p = draws[i, 0]; F = draws[i, 1]; s = draws[i, 2]; h = draws[i, 3]
        P[0, i] = p; Fm[0, i] = F; R[0, i] = F * p * (1 - p)
        for t in range(1, T + 1):
            Ne = Ne_arr[min(t - 1, len(Ne_arr) - 1)]
            pn, Fn, dps, dpm, dF = step(p, F, Ne, s, h, mu1, mu2)
            a = p * (1 - p) * dF
            b = F * (1 - 2 * p) * dps
            c = F * (1 - 2 * p) * dpm
            dRF[t, i] = a; dRs[t, i] = b; dRmu[t, i] = c
            Z = abs(a) + abs(b) + abs(c)
            if Z > 1e-30:
                aF[t, i] = a / Z; a_s[t, i] = b / Z; amu[t, i] = c / Z
            p = pn; F = Fn
            P[t, i] = p; Fm[t, i] = F; R[t, i] = F * p * (1 - p)
    return P, Fm, R, dRF, dRs, dRmu, aF, a_s, amu


def _band(M):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return (np.nanmedian(M, 1), np.nanpercentile(M, 5, 1), np.nanpercentile(M, 95, 1))


def decompose(mc, Ne_arr, T, mu1=1e-5, mu2=1e-5, thin=10):
    flat = mc['samples'].reshape(-1, mc['samples'].shape[-1])[::thin]
    if flat.shape[1] == 2:
        flat = np.column_stack([flat, np.zeros(len(flat)), np.full(len(flat), 0.5)])
    elif flat.shape[1] == 3:
        flat = np.column_stack([flat, np.zeros(len(flat))])
    P, Fm, R, dRF, dRs, dRmu, aF, a_s, amu = decompose_draws(
        np.ascontiguousarray(flat), np.asarray(Ne_arr, float), int(T), mu1, mu2)
    out = {'n_samples': flat.shape[0]}
    for k, M in [('p', P), ('F', Fm), ('R', R), ('dR_F', dRF), ('dR_s', dRs),
                 ('dR_mu', dRmu), ('alpha_F', aF), ('alpha_s', a_s), ('alpha_mu', amu)]:
        out[k + '_med'], out[k + '_lo'], out[k + '_hi'] = _band(M)
    out['_mats'] = {'p': P, 'F': Fm, 'R': R}
    return out
