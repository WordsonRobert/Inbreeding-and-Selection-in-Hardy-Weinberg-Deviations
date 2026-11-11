"""E12 comparison with a standard allele-frequency estimator of s (random mating).

The comparison method is the usual hidden Markov model for allele-frequency time series
under Wright-Fisher drift with additive selection and random mating (as in Bollback et al.
2008 and Mathieson & McVean 2013). It is re-implemented here on the same grid transition
kernel as BASTION, with F = 0 and h = 1/2 fixed: each sample is reduced to its allele
count (binomial in 2n), N_e is known, and s is estimated by maximum likelihood with a
profile-likelihood 90% interval.

Two random-mating designs, model-exact simulation, T = 40 generations, n = 300 adults
sampled every 5 generations, N_e = 1000, p0 ~ U(0.2, 0.8), s ~ +-U(0.02, 0.15):
  additive    h = 1/2   (the allele-frequency model is correctly specified)
  dominance   h ~ U(0, 1)
BASTION is fitted to the genotype counts of the same data sets with its default settings
(F, s and h inferred; 4 chains of 4000 iterations, 1500 warm-up).

usage:   python <this file> [--reps 100] [--workers N] [--out DIR]
The script imports the `bastion` package from the folder one level above it. Output goes to
--out; by default to results/e12_allele_hmm_comparison/ when the script is in experiments/
(the repository layout), and otherwise to a results/ folder next to the script.
Every data set has its own seed drawn from master seed 121, so the CSV and the summary do not
depend on the number of workers. Run time: 197 s on 2 cores for 2 x 100 data sets.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_v] = '1'
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from bastion import mcmc, model, sims  # noqa: E402

MU = 1e-5
T, STEP, N_SAMPLE, NE = 40, 5, 300, 1000.0
CHI2_90 = 2.705543454095404          # 90% quantile of chi-square(1)


# ── the allele-frequency HMM (random mating, additive selection) ─────────────
class AlleleHMM:
    def __init__(self, gens, k, n2, Ne):
        self.gens = np.asarray(gens, int); self.k = np.asarray(k, float); self.n2 = np.asarray(n2, float)
        self.Ne = float(Ne)
        K = model.choose_K(self.Ne)
        self.grid, self.prior = model.make_grid(K)
        self.A = np.zeros((K, K)); self.lo = np.zeros(K, np.int64); self.hi = np.full(K, K - 1, np.int64)

    def loglik(self, s):
        if 1 - s <= 0.01 or 1 - 0.5 * s <= 0.01:
            return -np.inf
        model.build_transition(self.grid, 0.0, s, 0.5, MU, MU, self.Ne, self.A, self.lo, self.hi)
        pa = np.array([(lambda a: a[0] + 0.5 * a[1])(model.adults_at(g, 0.0, s, 0.5)) for g in self.grid])
        pa = np.clip(pa, 1e-12, 1 - 1e-12)
        alpha = self.prior.copy(); ll = 0.0; j = 0
        for t in range(self.gens[-1] + 1):
            if j < len(self.gens) and self.gens[j] == t:
                e = binom.logpmf(self.k[j], self.n2[j], pa)
                m = e.max(); w = alpha * np.exp(e - m); tot = w.sum()
                if tot <= 0:
                    return -np.inf
                ll += np.log(tot) + m; alpha = w / tot; j += 1
            if t < self.gens[-1]:
                alpha = alpha @ self.A
        return ll

    def fit(self):
        sg = np.linspace(-0.6, 0.6, 241)
        lls = np.array([self.loglik(s) for s in sg])
        i = int(np.nanargmax(lls))
        r = minimize_scalar(lambda s: -self.loglik(s), bounds=(sg[max(i - 1, 0)], sg[min(i + 1, len(sg) - 1)]),
                            method='bounded', options=dict(xatol=1e-5))
        s_hat, ll_max = float(r.x), -float(r.fun)
        cut = ll_max - CHI2_90 / 2
        f = lambda s: self.loglik(s) - cut
        below = np.where(lls < cut)[0]
        left = below[below < i]; right = below[below > i]
        lo = brentq(f, sg[left[-1]], s_hat) if len(left) else sg[0]
        hi = brentq(f, s_hat, sg[right[0]]) if len(right) else sg[-1]
        return s_hat, float(lo), float(hi)


# ── one data set ────────────────────────────────────────────────────────────
def one(spec):
    rng = np.random.default_rng(spec['seed'])
    P, X, A = sims.simulate(spec['p0'], 0.0, spec['s'], spec['h'], T, Ne=NE, seed=spec['seed'], mode=0)
    gens = np.arange(0, T + 1, STEP)
    C = sims.observe(P, X, A, gens, N_SAMPLE, rng)
    t0 = time.time()
    hmm = AlleleHMM(gens, 2 * C[:, 0] + C[:, 1], 2 * C.sum(1), NE)
    s_hat, s_lo, s_hi = hmm.fit()
    t_allele = time.time() - t0
    t0 = time.time()
    mc = mcmc.run_mcmc(mcmc.Data(gens, C, Ne=NE), mcmc.make_cfg(), n_chains=4, n_steps=4000, n_warmup=1500,
                       seed=spec['seed'] % 100000)
    t_bastion = time.time() - t0
    S, HS, Fp = mc['summary']['s'], mc['summary']['hs'], mc['summary']['F']
    return dict(design=spec['design'], id=spec['id'], seed=spec['seed'], p0=spec['p0'], s_true=spec['s'],
                h_true=spec['h'], hs_true=spec['h'] * spec['s'],
                allele_s=s_hat, allele_s_lo=s_lo, allele_s_hi=s_hi, allele_seconds=t_allele,
                bastion_s=S['median'], bastion_s_lo=S['q05'], bastion_s_hi=S['q95'],
                bastion_hs=HS['median'], bastion_hs_lo=HS['q05'], bastion_hs_hi=HS['q95'],
                bastion_F=Fp['median'], bastion_F_lo=Fp['q05'], bastion_F_hi=Fp['q95'],
                bastion_converged=bool(mc['converged']), bastion_seconds=t_bastion)


def summarise(rows):
    out = {}
    for d in sorted(set(r['design'] for r in rows)):
        R = [r for r in rows if r['design'] == d]
        a = lambda k: np.array([r[k] for r in R], float)
        s, hs = a('s_true'), a('hs_true')
        res = {'n': len(R), 'bastion_converged': float(a('bastion_converged').mean())}
        for m, est, lo, hi in [('allele_HMM', a('allele_s'), a('allele_s_lo'), a('allele_s_hi')),
                               ('BASTION', a('bastion_s'), a('bastion_s_lo'), a('bastion_s_hi'))]:
            res[m] = {'s_bias_median': float(np.median(est - s)), 's_rmse': float(np.sqrt(np.mean((est - s) ** 2))),
                      's_cov90': float(np.mean((lo <= s) & (s <= hi))), 's_width90_median': float(np.median(hi - lo))}
        # heterozygote effect: hs for BASTION; s/2 for the additive allele-frequency model
        for m, est, lo, hi in [('allele_HMM', a('allele_s') / 2, a('allele_s_lo') / 2, a('allele_s_hi') / 2),
                               ('BASTION', a('bastion_hs'), a('bastion_hs_lo'), a('bastion_hs_hi'))]:
            res[m].update({'hs_rmse': float(np.sqrt(np.mean((est - hs) ** 2))),
                           'hs_cov90': float(np.mean((lo <= hs) & (hs <= hi)))})
        res['BASTION']['F_cov90'] = float(np.mean((a('bastion_F_lo') <= 0) & (0 <= a('bastion_F_hi'))))
        res['seconds_per_fit_median'] = {'allele_HMM': float(np.median(a('allele_seconds'))),
                                         'BASTION': float(np.median(a('bastion_seconds')))}
        out[d] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    default_out = (ROOT / 'results' / 'e12_allele_hmm_comparison') if HERE.name == 'experiments' else (HERE / 'results')
    ap.add_argument('--out', default=str(default_out))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(121)
    specs = []
    for design in ['additive', 'dominance']:
        for i in range(args.reps):
            specs.append(dict(design=design, id=i, seed=int(rng.integers(1, 2 ** 31 - 1)),
                              p0=float(rng.uniform(0.2, 0.8)), s=float(rng.choice([-1, 1]) * rng.uniform(0.02, 0.15)),
                              h=0.5 if design == 'additive' else float(rng.uniform(0, 1))))
    t0 = time.time()
    import multiprocessing as mp
    rows = []
    with mp.get_context('fork').Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, specs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(specs)} done ({time.time() - t0:.0f} s)', flush=True)
    rows.sort(key=lambda r: (r['design'], r['id']))
    import csv
    with open(out / 'comparison.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = summarise(rows)
    summ['meta'] = {'reps_per_design': args.reps, 'workers': args.workers, 'wall_seconds': time.time() - t0,
                    'design': dict(T=T, step=STEP, n=N_SAMPLE, Ne=NE, p0='U(0.2,0.8)', s='+-U(0.02,0.15)',
                                   h={'additive': 0.5, 'dominance': 'U(0,1)'}, F=0.0, mu=MU, master_seed=121)}
    json.dump(summ, open(out / 'comparison_summary.json', 'w'), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
