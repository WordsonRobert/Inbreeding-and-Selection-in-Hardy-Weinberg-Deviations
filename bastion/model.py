"""
BASTION — genotype-level model, drift HMM likelihood, attribution.

Life cycle of one generation t (single biallelic locus, alleles A/a):
  gamete pool p_t  --mating (inbreeding F)-->  zygotes x_t = (p^2+Fpq, 2pq(1-F), q^2+Fpq)
  zygotes  --viability selection w=(1, 1-hs, 1-s)-->  adults a_t = w*x_t / wbar   [sampled]
  adults   --gametes + mutation-->  m_t = p^a (1-mu1) + (1-p^a) mu2
  m_t      --drift (2Ne gene copies)-->  p_{t+1} ~ Bin(2Ne, m_t)/(2Ne)

F is the mating-system inbreeding coefficient of zygotes (F_IS), not Wright's
drift recursion: in a randomly mating finite population drift moves p but leaves
F_IS ~ -1/(2N).  Selection acts on the actual (inbred) genotype frequencies.

The likelihood integrates over the hidden allele-frequency path with a grid HMM
(K grid points uniform on the arcsine scale, including the loss/fixation states
p = 0 and p = 1; transitions use a mean-preserving 'hat-function' projection of the
Gaussian WF kernel with the interpolation variance removed; K grows with Ne so the
kernel variance is matched within a few percent), so p0 is integrated out under a
Beta(2,2) prior.

Exact attribution of the observed adult HW deviation D = a_AA - (p^a)^2:
  level:  D^a_t = D^z_t (inbreeding, = F p q) + D^S_t (selection distortion)
          with D^a = (w1 w3 x1 x3 - w2^2 x2^2 / 4) / wbar^2          (closed form)
  change: D^a_{t+1} - D^a_t split over the generation's operators
          {mutation M, Mendelian reset Y0, inbreeding I, drift Dr, selection S}
          by Shapley values (exact, order-free) and by the life-cycle sequence.
"""
import math
import numpy as np
import numba as nb

# emission / model codes
EM_GENO_ADULT = 0
EM_GENO_ZYGOTE = 1
EM_PHENO_ADULT = 2       # A dominant: only aa distinguishable


# ─────────────────────────────────────────────────────────────────────────────
# Genotype-level operators
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def zygotes(p, F):
    """Zygote genotype frequencies at allele frequency p with inbreeding F.
    D = F p q is clipped to its admissible range [-min(p,q)^2, p q]."""
    q = 1.0 - p
    D = F * p * q
    lo = -min(p, q) ** 2
    if D < lo:
        D = lo
    if D > p * q:
        D = p * q
    return p * p + D, 2.0 * (p * q - D), q * q + D


@nb.njit(cache=True)
def select(x1, x2, x3, s, h):
    w2 = 1.0 - h * s
    w3 = 1.0 - s
    wb = x1 + x2 * w2 + x3 * w3
    return x1 / wb, x2 * w2 / wb, x3 * w3 / wb


@nb.njit(cache=True)
def mutate_allele(p, mu1, mu2):
    return p * (1.0 - mu1) + (1.0 - p) * mu2


@nb.njit(cache=True)
def adults_at(p, F, s, h):
    x1, x2, x3 = zygotes(p, F)
    return select(x1, x2, x3, s, h)


@nb.njit(cache=True)
def gamete_mean(p, F, s, h, mu1, mu2):
    a1, a2, a3 = adults_at(p, F, s, h)
    return mutate_allele(a1 + 0.5 * a2, mu1, mu2)


@nb.njit(cache=True)
def hw_dev(g1, g2, g3):
    """D = f_AA - p^2 = f_AA f_aa - f_Aa^2 / 4 (for a normalised vector)."""
    return g1 * g3 - 0.25 * g2 * g2


@nb.njit(cache=True)
def adult_D_closed(p, F, s, h):
    """Closed form D^a = (w1 w3 x1 x3 - w2^2 x2^2/4) / wbar^2."""
    x1, x2, x3 = zygotes(p, F)
    w2 = 1.0 - h * s
    w3 = 1.0 - s
    wb = x1 + x2 * w2 + x3 * w3
    return (w3 * x1 * x3 - 0.25 * w2 * w2 * x2 * x2) / (wb * wb)


# ─────────────────────────────────────────────────────────────────────────────
# HMM transition (mean-preserving hat-function discretisation)
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / 1.4142135623730951))


@nb.njit(cache=True)
def _phi(z):
    return 0.3989422804014327 * math.exp(-0.5 * z * z)


@nb.njit(cache=True)
def _cell(grid, x):
    """Index i with grid[i] <= x < grid[i+1] (clipped to 0..K-2)."""
    K = len(grid)
    i = np.searchsorted(grid, x, side='right') - 1
    if i < 0:
        i = 0
    if i > K - 2:
        i = K - 2
    return i


@nb.njit(cache=True)
def build_transition(grid, F, s, h, mu1, mu2, Ne, A, lo, hi):
    """Fill banded A[j, k] = P(p_{t+1} = grid[k] | p_t = grid[j]).

    The grid is non-uniform (uniform on the arcsine scale, with the exact boundary
    states p = 0 and p = 1). The Gaussian WF kernel N(m, m(1-m)/2Ne) is projected on
    piecewise-linear 'hat' basis functions, which preserves the mean exactly; Gaussian
    mass beyond 0 or 1 goes to the boundary state (loss / fixation). Interpolation
    adds ~c^2/6 of variance (c = local cell width), which is subtracted; when the drift
    variance is below that floor the kernel is the mean-preserving two-point split."""
    K = len(grid)
    for j in range(K):
        m = gamete_mean(grid[j], F, s, h, mu1, mu2)
        if m < 0.0:
            m = 0.0
        if m > 1.0:
            m = 1.0
        for k in range(lo[j], hi[j] + 1):
            A[j, k] = 0.0
        i0 = _cell(grid, m)
        c = grid[i0 + 1] - grid[i0]
        var = m * (1.0 - m) / (2.0 * Ne) - c * c / 6.0
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd < 0.05 * c:
            w = (m - grid[i0]) / c
            if w < 0.0:
                w = 0.0
            if w > 1.0:
                w = 1.0
            lo[j] = i0; hi[j] = i0 + 1
            A[j, i0] = 1.0 - w; A[j, i0 + 1] = w
            continue
        a = _cell(grid, m - 6.0 * sd) - 1
        b = _cell(grid, m + 6.0 * sd) + 2
        if a < 0:
            a = 0
        if b > K - 1:
            b = K - 1
        lo[j] = a; hi[j] = b
        # Phi and phi at every node of the band (+1 on each side), each computed once
        a0 = a - 1 if a > 0 else 0
        b0 = b + 1 if b < K - 1 else K - 1
        nb_ = b0 - a0 + 1
        Pz = np.empty(nb_); pz = np.empty(nb_)
        for k in range(a0, b0 + 1):
            z = (grid[k] - m) / sd
            Pz[k - a0] = _Phi(z); pz[k - a0] = _phi(z)
        tot = 0.0
        for k in range(a, b + 1):
            pk = grid[k]
            i = k - a0
            val = 0.0
            if k > 0:
                L = grid[k - 1]
                val += ((m - L) * (Pz[i] - Pz[i - 1]) + sd * (pz[i - 1] - pz[i])) / (pk - L)
            else:
                val += Pz[i]                            # Gaussian mass below 0 -> loss state
            if k < K - 1:
                R = grid[k + 1]
                val += ((R - m) * (Pz[i + 1] - Pz[i]) + sd * (pz[i + 1] - pz[i])) / (R - pk)
            else:
                val += 1.0 - Pz[i]                      # mass above 1 -> fixation state
            if val < 0.0:
                val = 0.0
            A[j, k] = val
            tot += val
        if tot > 0:
            for k in range(a, b + 1):
                A[j, k] /= tot


@nb.njit(cache=True)
def _emission_logs(grid, F, s, h, em):
    K = len(grid)
    L = np.empty((K, 3))
    for k in range(K):
        if em == EM_GENO_ZYGOTE:
            g1, g2, g3 = zygotes(grid[k], F)
        else:
            g1, g2, g3 = adults_at(grid[k], F, s, h)
        L[k, 0] = math.log(max(g1, 1e-300))
        L[k, 1] = math.log(max(g2, 1e-300))
        L[k, 2] = math.log(max(g3, 1e-300))
        if em == EM_PHENO_ADULT:
            L[k, 0] = math.log(max(1.0 - g3, 1e-300))   # dominant phenotype
    return L


@nb.njit(cache=True)
def hmm_forward(grid, prior, F, s, h, mu1, mu2, Ne_arr, gens, C, em, store):
    """Log-likelihood of counts C (n_obs x 3) at generations gens.
    C columns: (AA, Aa, aa) for genotype data; (dominant, 0, recessive) for phenotype.
    If store: also returns filtered distributions alpha[t] for t = 0..T (for FFBS)."""
    K = len(grid)
    T = gens[-1] - gens[0]
    L = _emission_logs(grid, F, s, h, em)
    A = np.zeros((K, K)); lo = np.zeros(K, np.int64); hi = np.zeros(K, np.int64)
    for j in range(K):
        lo[j] = 0; hi[j] = K - 1
    alpha = prior.copy()
    ALPHA = np.zeros((T + 1 if store else 1, K))
    ll = 0.0; k_obs = 0; nobs = len(gens)
    last_Ne = -1.0
    for st in range(T + 1):
        if k_obs < nobs and gens[k_obs] == gens[0] + st:
            c1 = C[k_obs, 0]; c2 = C[k_obs, 1]; c3 = C[k_obs, 2]
            emax = -1e300
            e = np.empty(K)
            for k in range(K):
                if em == EM_PHENO_ADULT:
                    e[k] = c1 * L[k, 0] + c3 * L[k, 2]
                else:
                    e[k] = c1 * L[k, 0] + c2 * L[k, 1] + c3 * L[k, 2]
                if alpha[k] > 0 and e[k] > emax:
                    emax = e[k]
            tot = 0.0
            for k in range(K):
                alpha[k] *= math.exp(e[k] - emax)
                tot += alpha[k]
            if tot <= 0 or not np.isfinite(tot):
                return -np.inf, ALPHA
            ll += math.log(tot) + emax
            for k in range(K):
                alpha[k] /= tot
            k_obs += 1
        if store:
            for k in range(K):
                ALPHA[st, k] = alpha[k]
        if st == T:
            break
        Ne = Ne_arr[min(st, len(Ne_arr) - 1)]
        if Ne != last_Ne:
            build_transition(grid, F, s, h, mu1, mu2, Ne, A, lo, hi)
            last_Ne = Ne
        new = np.zeros(K)
        for j in range(K):
            aj = alpha[j]
            if aj < 1e-300:
                continue
            for k in range(lo[j], hi[j] + 1):
                new[k] += aj * A[j, k]
        alpha = new
    return ll, ALPHA


@nb.njit(cache=True)
def ffbs(grid, prior, F, s, h, mu1, mu2, Ne_arr, gens, C, em, n_paths, seed):
    """Forward-filter backward-sample allele-frequency paths p_0..p_T (grid values)."""
    np.random.seed(seed)
    K = len(grid)
    T = gens[-1] - gens[0]
    ll, ALPHA = hmm_forward(grid, prior, F, s, h, mu1, mu2, Ne_arr, gens, C, em, True)
    paths = np.zeros((n_paths, T + 1))
    if not np.isfinite(ll):
        paths[:, :] = np.nan
        return paths
    A = np.zeros((K, K)); lo = np.zeros(K, np.int64); hi = np.zeros(K, np.int64)
    for j in range(K):
        lo[j] = 0; hi[j] = K - 1
    for r in range(n_paths):
        # sample p_T
        u = np.random.random(); c = 0.0; kk = K - 1
        for k in range(K):
            c += ALPHA[T, k]
            if u <= c:
                kk = k; break
        paths[r, T] = grid[kk]
        last_Ne = -1.0
        for t in range(T - 1, -1, -1):
            Ne = Ne_arr[min(t, len(Ne_arr) - 1)]
            if Ne != last_Ne:
                build_transition(grid, F, s, h, mu1, mu2, Ne, A, lo, hi)
                last_Ne = Ne
            w = np.empty(K); tot = 0.0
            for j in range(K):
                if lo[j] <= kk and kk <= hi[j]:
                    w[j] = ALPHA[t, j] * A[j, kk]
                else:
                    w[j] = 0.0
                tot += w[j]
            if tot <= 0:
                # numerical underflow: fall back to filtered distribution
                for j in range(K):
                    w[j] = ALPHA[t, j]
                tot = 1.0
            u = np.random.random() * tot; c = 0.0; jj = K - 1
            for j in range(K):
                c += w[j]
                if u <= c:
                    jj = j; break
            kk = jj
            paths[r, t] = grid[kk]
    return paths


def make_grid(K=150):
    """Grid uniform on the arcsine scale, p_k = sin^2(pi u_k / 2), u_k = k/(K-1).
    Includes p = 0 and p = 1 exactly; spacing is proportional to sqrt(pq), i.e. to the
    drift standard deviation, so the WF kernel is resolved equally well at every p.
    Prior weights: Beta(2,2) density times cell width (zero at the two boundaries)."""
    u = np.linspace(0.0, 1.0, K)
    grid = np.sin(0.5 * np.pi * u) ** 2
    grid[0] = 0.0; grid[-1] = 1.0
    width = np.empty(K)
    width[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    width[0] = 0.5 * grid[1]; width[-1] = 0.5 * (1 - grid[-2])
    prior = grid * (1 - grid) * width
    return grid, prior / prior.sum()


K_LEVELS = (120, 160, 220, 300, 400)


def make_levels(Ks):
    """Padded stack of grids for several sizes; level i serves Ne <= NEL[i]."""
    Ks = list(Ks)
    Kmax = max(Ks)
    GR = np.zeros((len(Ks), Kmax)); PR = np.zeros((len(Ks), Kmax))
    for i, K in enumerate(Ks):
        g, pr = make_grid(K)
        GR[i, :K] = g; PR[i, :K] = pr
    NEL = np.array([(K / 2.6) ** 2 for K in Ks]); NEL[-1] = np.inf
    return GR, PR, np.array(Ks, np.int64), NEL


def choose_K(Ne_max):
    """Grid size so that the interpolation floor c^2/6 stays below ~half the drift
    variance for every Ne up to Ne_max (c^2/6 / Var = (pi^2/3) Ne / K^2)."""
    return int(min(400, max(120, math.ceil(2.6 * math.sqrt(max(Ne_max, 1.0))))))


# ─────────────────────────────────────────────────────────────────────────────
# Attribution
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _vec_with(p, D):
    """Genotype vector with allele freq p and HW deviation D (clipped to admissible)."""
    q = 1.0 - p
    lo = -min(p, q) ** 2
    hi = p * q
    if D < lo:
        D = lo
    if D > hi:
        D = hi
    return p * p + D, 2.0 * (p * q - D), q * q + D


@nb.njit(cache=True)
def _apply(op, g1, g2, g3, F, s, h, mu1, mu2, delta):
    p = g1 + 0.5 * g2
    if op == 0:      # M: germline mutation of each allele (exact genotype-level map)
        # each allele of each parent mutates independently (A->a: mu1, a->A: mu2)
        u = mu1; v = mu2
        n1 = g1 * (1 - u) ** 2 + g2 * (1 - u) * v + g3 * v * v
        n3 = g1 * u * u + g2 * u * (1 - v) + g3 * (1 - v) ** 2
        n2 = 1.0 - n1 - n3
        return n1, n2, n3
    elif op == 1:    # Y0: random union of gametes (Mendelian reset to HW at current p)
        q = 1 - p
        return p * p, 2 * p * q, q * q
    elif op == 2:    # I: inbreeding adds F p q of homozygosity
        D = hw_dev(g1, g2, g3)
        return _vec_with(p, D + F * p * (1 - p))
    elif op == 3:    # Dr: drift shifts p by delta, keeping F_g = D/(pq)
        D = hw_dev(g1, g2, g3)
        pq = p * (1 - p)
        Fg = D / pq if pq > 1e-15 else 0.0
        pn = min(max(p + delta, 1e-12), 1 - 1e-12)
        return _vec_with(pn, Fg * pn * (1 - pn))
    else:            # S: viability selection
        return select(g1, g2, g3, s, h)


@nb.njit(cache=True)
def shapley_step(a1, a2, a3, F, s, h, mu1, mu2, delta):
    """Shapley attribution of D(a_{t+1}) - D(a_t) over operators
    0=M, 1=Y0, 2=I, 3=Dr, 4=S applied in life-cycle order (M,Y0,I,Dr,S).
    Returns (phi[5], seq[5], total)."""
    n = 5
    D0 = hw_dev(a1, a2, a3)
    v = np.zeros(1 << n)
    for mask in range(1 << n):
        g1 = a1; g2 = a2; g3 = a3
        for op in range(n):
            if mask & (1 << op):
                g1, g2, g3 = _apply(op, g1, g2, g3, F, s, h, mu1, mu2, delta)
        v[mask] = hw_dev(g1, g2, g3) - D0
    fact = np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0])
    phi = np.zeros(n)
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            k = 0
            for j in range(n):
                if mask & (1 << j):
                    k += 1
            wgt = fact[k] * fact[n - k - 1] / fact[n]
            phi[i] += wgt * (v[mask | (1 << i)] - v[mask])
    seq = np.zeros(n)
    mask = 0
    for op in range(n):
        seq[op] = v[mask | (1 << op)] - v[mask]
        mask |= (1 << op)
    return phi, seq, v[(1 << n) - 1]


@nb.njit(cache=True)
def attribute_path(path, F, s, h, mu1, mu2):
    """For one allele-frequency path p_0..p_T: level and change attribution.
    Level: D^a_t = Dz_t (inbreeding share) + DS_t (selection share), a two-player
    Shapley split of the adult deviation starting from HW zygotes (the interaction of
    selection with inbred genotype frequencies is shared equally).
    Returns Dz, Da, DS (T+1), phi (T, 5), seq (T, 5), total (T)."""
    T = len(path) - 1
    Dz = np.zeros(T + 1); Da = np.zeros(T + 1); DS = np.zeros(T + 1)
    phi = np.zeros((T, 5)); seq = np.zeros((T, 5)); tot = np.zeros(T)
    for t in range(T + 1):
        x1, x2, x3 = zygotes(path[t], F)
        a1, a2, a3 = select(x1, x2, x3, s, h)
        pz = path[t]; qz = 1.0 - pz
        b1, b2, b3 = select(pz * pz, 2 * pz * qz, qz * qz, s, h)   # selection on HW zygotes
        vI = hw_dev(x1, x2, x3); vS = hw_dev(b1, b2, b3); vIS = hw_dev(a1, a2, a3)
        Da[t] = vIS
        Dz[t] = 0.5 * (vI + vIS - vS)        # Shapley share of inbreeding
        DS[t] = 0.5 * (vS + vIS - vI)        # Shapley share of selection
        if t < T:
            pa = a1 + 0.5 * a2
            m = mutate_allele(pa, mu1, mu2)
            delta = path[t + 1] - m
            ph, sq, tt = shapley_step(a1, a2, a3, F, s, h, mu1, mu2, delta)
            for i in range(5):
                phi[t, i] = ph[i]; seq[t, i] = sq[i]
            tot[t] = tt
    return Dz, Da, DS, phi, seq, tot


OP_NAMES = ['mutation', 'mendelian_reset', 'inbreeding', 'drift', 'selection']


@nb.njit(cache=True)
def predictive_residuals(grid, prior, F, s, h, mu1, mu2, Ne_arr, gens, C, em):
    """One-step-ahead (prequential) standardised residuals of each observation given all
    earlier ones, from the forward filter: r_k = (y_k - E[y_k | y_<k]) / SD[y_k | y_<k],
    with y = sample allele frequency (genotype data) or recessive-phenotype frequency
    (phenotype data). Sampling variance is included. r_0 is not defined (nan)."""
    K = len(grid)
    T = gens[-1] - gens[0]
    L = _emission_logs(grid, F, s, h, em)
    Y = np.empty(K); V = np.empty(K)
    for k in range(K):
        if em == EM_GENO_ZYGOTE:
            g1, g2, g3 = zygotes(grid[k], F)
        else:
            g1, g2, g3 = adults_at(grid[k], F, s, h)
        if em == EM_PHENO_ADULT:
            Y[k] = g3; V[k] = g3 * (1.0 - g3)
        else:
            Y[k] = g1 + 0.5 * g2; V[k] = g1 + 0.25 * g2 - Y[k] * Y[k]
    A = np.zeros((K, K)); lo = np.zeros(K, np.int64); hi = np.zeros(K, np.int64)
    for j in range(K):
        lo[j] = 0; hi[j] = K - 1
    alpha = prior.copy()
    nobs = len(gens)
    r = np.full(nobs, np.nan)
    k_obs = 0
    last_Ne = -1.0
    for st in range(T + 1):
        if k_obs < nobs and gens[k_obs] == gens[0] + st:
            c1 = C[k_obs, 0]; c2 = C[k_obs, 1]; c3 = C[k_obs, 2]
            n = c1 + c2 + c3
            if k_obs > 0 and n > 0:
                ey = 0.0; ey2 = 0.0
                for k in range(K):
                    ey += alpha[k] * Y[k]
                    ey2 += alpha[k] * (Y[k] * Y[k] + V[k] / n)
                vy = ey2 - ey * ey
                if em == EM_PHENO_ADULT:
                    yo = c3 / n
                else:
                    yo = (c1 + 0.5 * c2) / n
                r[k_obs] = (yo - ey) / math.sqrt(max(vy, 1e-300))
            emax = -1e300
            e = np.empty(K)
            for k in range(K):
                if em == EM_PHENO_ADULT:
                    e[k] = c1 * L[k, 0] + c3 * L[k, 2]
                else:
                    e[k] = c1 * L[k, 0] + c2 * L[k, 1] + c3 * L[k, 2]
                if alpha[k] > 0 and e[k] > emax:
                    emax = e[k]
            tot = 0.0
            for k in range(K):
                alpha[k] *= math.exp(e[k] - emax)
                tot += alpha[k]
            if tot <= 0 or not np.isfinite(tot):
                return r
            for k in range(K):
                alpha[k] /= tot
            k_obs += 1
        if st == T:
            break
        Ne = Ne_arr[min(st, len(Ne_arr) - 1)]
        if Ne != last_Ne:
            build_transition(grid, F, s, h, mu1, mu2, Ne, A, lo, hi)
            last_Ne = Ne
        new = np.zeros(K)
        for j in range(K):
            aj = alpha[j]
            if aj < 1e-300:
                continue
            for k in range(lo[j], hi[j] + 1):
                new[k] += aj * A[j, k]
        alpha = new
    return r
