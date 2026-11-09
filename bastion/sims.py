"""
BASTION — data generators.

mode 0  model-exact: gamete-pool drift p_{t+1} ~ Bin(2Ne, m_t)/(2Ne), zygotes at
        constant mating-system F, viability selection, adults sampled.
mode 1  individual-based: N zygotes per generation drawn from the expected zygote
        distribution under partial selfing (rate sigma) + random outcrossing,
        survival is binomial with probability w_i / max(w); next gametes come from
        the surviving adults (with mutation). F_IS emerges (sigma/(2-sigma)); drift
        variance is the genuine finite-population one, not the model's.

Events / violations (any mode):
  s2, T_switch        selection coefficient switches to s2 at generation T_switch
  Ne_arr              per-generation Ne (bottlenecks)
  m_mig, T_mig, p_mig migrants (HW at p_mig) form a fraction m_mig of zygotes at T_mig;
                      the allele-frequency shift persists
  hh_*                linked selection: the focal locus is pushed by hitchhiking on a
                      linked additive locus: dp = (D_l / (p_l q_l)) dp_l, D_l decays (1-r)
Sampling options: genotype or phenotype, adults or zygotes, Wahlund pooling of two
subpopulations at p +/- d, heterozygote miscalling rate eps.
"""
import math
import numpy as np
import numba as nb
from .model import zygotes, select, mutate_allele


@nb.njit(cache=True)
def generate(p0, F, s, h, mu1, mu2, Ne_arr, T, seed, mode, sigma,
             s2, T_switch, m_mig, T_mig, p_mig, hh_s, hh_D0, hh_r, hh_p0):
    np.random.seed(seed)
    P = np.zeros(T + 1)                 # zygote allele frequency (gamete pool)
    X = np.zeros((T + 1, 3))            # zygote genotype frequencies
    Aad = np.zeros((T + 1, 3))          # adult genotype frequencies
    p = p0
    pl = hh_p0; Dl = hh_D0
    x1, x2, x3 = zygotes(p0, F)
    if mode == 1:
        N = int(Ne_arr[0])
        c = np.random.multinomial(N, np.array([x1, x2, x3]))
        x1 = c[0] / N; x2 = c[1] / N; x3 = c[2] / N
    for t in range(T + 1):
        s_t = s2 if (T_switch >= 0 and t >= T_switch) else s
        if m_mig > 0 and t == T_mig:
            qm = 1 - p_mig
            x1 = (1 - m_mig) * x1 + m_mig * p_mig * p_mig
            x2 = (1 - m_mig) * x2 + m_mig * 2 * p_mig * qm
            x3 = (1 - m_mig) * x3 + m_mig * qm * qm
        X[t, 0] = x1; X[t, 1] = x2; X[t, 2] = x3
        P[t] = x1 + 0.5 * x2
        Ne = Ne_arr[min(t, len(Ne_arr) - 1)]
        if mode == 1:
            N = int(Ne)
            w2 = 1 - h * s_t; w3 = 1 - s_t
            wm = max(1.0, max(w2, w3))
            n1 = int(round(x1 * N)); n2 = int(round(x2 * N)); n3 = max(N - n1 - n2, 0)
            k1 = np.random.binomial(n1, 1.0 / wm) if n1 > 0 else 0
            k2 = np.random.binomial(n2, w2 / wm) if n2 > 0 else 0
            k3 = np.random.binomial(n3, w3 / wm) if n3 > 0 else 0
            tot = k1 + k2 + k3
            if tot == 0:
                a1, a2, a3 = select(x1, x2, x3, s_t, h)
            else:
                a1 = k1 / tot; a2 = k2 / tot; a3 = k3 / tot
        else:
            a1, a2, a3 = select(x1, x2, x3, s_t, h)
        Aad[t, 0] = a1; Aad[t, 1] = a2; Aad[t, 2] = a3
        if t == T:
            break
        pa = a1 + 0.5 * a2
        m = mutate_allele(pa, mu1, mu2)
        if hh_D0 > 0:
            dpl = hh_s * pl * (1 - pl) / 2.0
            m = m + (Dl / max(pl * (1 - pl), 1e-9)) * dpl
            pl = pl + dpl; Dl = Dl * (1 - hh_r)
            m = min(max(m, 1e-9), 1 - 1e-9)
        Ne_next = Ne_arr[min(t + 1, len(Ne_arr) - 1)]
        if mode == 0:
            N2 = max(int(round(2 * Ne)), 2)
            p = np.random.binomial(N2, min(max(m, 0.0), 1.0)) / N2
            x1, x2, x3 = zygotes(p, F)
        else:
            # partial selfing: selfed offspring of adults + outcrossed HW at gamete freq m
            q = 1 - m
            s1 = a1 + 0.25 * a2; s2_ = 0.5 * a2; s3 = a3 + 0.25 * a2
            e1 = sigma * s1 + (1 - sigma) * m * m
            e2 = sigma * s2_ + (1 - sigma) * 2 * m * q
            e3 = sigma * s3 + (1 - sigma) * q * q
            et = e1 + e2 + e3
            N = int(Ne_next)
            c = np.random.multinomial(N, np.array([e1 / et, e2 / et, e3 / et]))
            x1 = c[0] / N; x2 = c[1] / N; x3 = c[2] / N
            p = x1 + 0.5 * x2
    return P, X, Aad


def simulate(p0, F, s, h, T, Ne=1000, Ne_arr=None, seed=0, mode=0, sigma=0.0,
             mu1=1e-5, mu2=1e-5, s2=0.0, T_switch=-1, m_mig=0.0, T_mig=-1, p_mig=0.5,
             hh_s=0.0, hh_D0=0.0, hh_r=0.1, hh_p0=0.5):
    if Ne_arr is None:
        Ne_arr = np.full(T + 1, float(Ne))
    return generate(float(p0), float(F), float(s), float(h), float(mu1), float(mu2),
                    np.asarray(Ne_arr, float), int(T), int(seed), int(mode), float(sigma),
                    float(s2), int(T_switch), float(m_mig), int(T_mig), float(p_mig),
                    float(hh_s), float(hh_D0), float(hh_r), float(hh_p0))


def observe(P, X, Aad, gens, n, rng, stage='adult', pheno=False, wahlund_d=0.0,
            F=0.0, s=0.0, h=0.5, eps=0.0):
    """Counts (n_obs, 3) at generations gens; n is a scalar or one sample size per
    observation. Phenotype: (dominant, 0, recessive)."""
    from .model import adults_at
    C = np.zeros((len(gens), 3))
    nn = np.broadcast_to(np.asarray(n, int), (len(gens),))
    for i, g in enumerate(gens):
        if wahlund_d > 0:
            pg = P[g]
            v1 = np.array(adults_at(min(pg + wahlund_d, 0.999), F, s, h)) if stage == 'adult' \
                else np.array(zygotes(min(pg + wahlund_d, 0.999), F))
            v2 = np.array(adults_at(max(pg - wahlund_d, 0.001), F, s, h)) if stage == 'adult' \
                else np.array(zygotes(max(pg - wahlund_d, 0.001), F))
            pr = 0.5 * (v1 + v2)
        else:
            pr = Aad[g].copy() if stage == 'adult' else X[g].copy()
        pr = np.clip(pr, 0, None); pr = pr / pr.sum()
        c = rng.multinomial(int(nn[i]), pr).astype(float)
        if eps > 0 and c[1] > 0:
            k = rng.binomial(int(c[1]), eps)
            k1 = rng.binomial(k, 0.5)
            c[1] -= k; c[0] += k1; c[2] += k - k1
        if pheno:
            c = np.array([c[0] + c[1], 0.0, c[2]])
        C[i] = c
    return C
