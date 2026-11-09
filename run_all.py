#!/usr/bin/env python3
"""BASTION — run every experiment top to bottom.

  python run_all.py --mode full --scale 2.5      # reproduces results/ exactly (the published run)
  python run_all.py --mode quick                 # smoke test, ~6-10 min on 2 cores
  python run_all.py --mode full --hours 2        # the paper run, shrunk to fit 2 h if needed
  python run_all.py --mode full --hours 2 --fill # ... or grown (up to 2.5x) to use the 2 h
  python run_all.py --mode full                  # the paper run at full size, however long
  python run_all.py --mode full --only e03,e07   # selected stages
  python run_all.py --mode full --no-ode         # skip the GPU neural-ODE stage (E11)

Outputs go to results/ (one folder per experiment, CSV + JSON + PNG) and
results/SUMMARY.md collects the headline numbers. Everything is seeded: with the same
--scale the CPU stages simulate the same data sets with the same seeds on any machine
and with any worker count.
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# every worker is single-threaded by design; inherited thread settings would oversubscribe
for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[v] = '1'
# compiled-kernel cache on local disk (a network filesystem makes numba's cache very slow)
import tempfile  # noqa: E402
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(tempfile.gettempdir(), f'bastion_numba_{os.getuid()}'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), f'bastion_mpl_{os.getuid()}'))

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))

STAGES = ['e01', 'e03', 'e04', 'e05', 'e05b', 'e07', 'e08', 'e09', 'e10']
DONE_FILE = {'e01': 'e01_theory/theory_summary.json', 'e03': 'e03_sbc/sbc_summary.json',
             'e04': 'e04_robustness/robustness_summary.json', 'e05': 'e05_violations/violations_summary.json',
             'e05b': 'e05_failure_map/failure_map_summary.json', 'e07': 'e07_migration/migration_summary.json',
             'e08': 'e08_reconstruction/reconstruction_summary.json', 'e09': 'e09_empirical/empirical_summary.json',
             'e10': 'e10_legacy/legacy_summary.json'}


def warmup():
    """Compile all numba kernels once in the parent so forked workers reuse them."""
    import numpy as np
    from bastion import mcmc, sims, model, legacy_v2 as L
    P, X, A = sims.simulate(0.4, 0.1, 0.05, 0.5, 10, Ne=500, seed=1)
    sims.simulate(0.4, 0.1, 0.05, 0.5, 10, Ne=500, seed=1, mode=1, sigma=0.2)
    C = sims.observe(P, X, A, [0, 5, 10], 100, np.random.default_rng(0))
    d = mcmc.Data([0, 5, 10], C, Ne=500)
    for kw in [dict(), dict(infer_Ne=True), dict(emission='pheno_adult', infer_h=False, h_fixed=0.0),
               dict(infer_s=False, infer_h=False)]:
        mc = mcmc.run_mcmc(d, mcmc.make_cfg(**kw), n_chains=2, n_steps=60, n_warmup=20)
    th, paths = mcmc.posterior_paths(mc, d, n_draws=4)
    mc = mcmc.run_mcmc(d, mcmc.make_cfg(), n_chains=2, n_steps=60, n_warmup=20)
    th, paths = mcmc.posterior_paths(mc, d, n_draws=4)
    mcmc.attribution(mc, d, th, paths); mcmc.discrepancy(mc, d, th, paths)
    L.run_mcmc([0, 5, 10], C[:, 0], C[:, 1], C[:, 2], np.full(11, 500.0), n_chains=2, n_steps=60, n_warmup=20)
    L.simulate(0.4, 0.1, 0.05, 0.5, 1e-5, 1e-5, np.full(11, 500.0), 10)


# cost of each stage at full size: (standard-fit units, Ne-inferred-fit units, serial seconds).
# One standard fit = Ne known, T = 40, 4 chains x 4000 steps, posterior paths + attribution;
# the units fold in each stage's replicate count, series length and chain length.
COST = {'e01': (0, 0, 120), 'e03': (1000, 900, 0), 'e04': (1320, 540, 0), 'e05': (2250, 375, 0),
        'e05b': (1000, 0, 0), 'e07': (3520, 740, 0), 'e08': (1040, 0, 0), 'e09': (340, 0, 150),
        'e10': (460, 0, 30)}


def benchmark():
    """Seconds for one standard fit and one Ne-inferred fit (single core)."""
    import numpy as np
    from bastion import mcmc, sims
    P, X, A = sims.simulate(0.4, 0.1, 0.06, 0.5, 40, Ne=1000, seed=3)
    gens = list(range(0, 41, 5))
    C = sims.observe(P, X, A, gens, 300, np.random.default_rng(3))
    d = mcmc.Data(gens, C, Ne=1000.0)
    out = []
    for kw in [dict(), dict(infer_Ne=True)]:
        t = time.time()
        mc = mcmc.run_mcmc(d, mcmc.make_cfg(**kw), n_chains=4, n_steps=4000, n_warmup=1500, seed=1)
        th, paths = mcmc.posterior_paths(mc, d, n_draws=150)
        mcmc.attribution(mc, d, th, paths); mcmc.discrepancy(mc, d, th, paths)
        out.append(time.time() - t)
    return out


# speed probe (experiments/worker.py:probe) on the reference core, and how a standard /
# Ne-inferred fit relates to it there
REF_PROBE_S, STD_PER_PROBE, INF_PER_PROBE = 0.18, 10.7, 22.0


def usable_cpus():
    """CPUs this process may really use: affinity mask, cgroup CPU quota (v1/v2, every
    level of this process's cgroup), and the batch scheduler's allocation."""
    info = {'os_cpu_count': os.cpu_count()}
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        n = os.cpu_count() or 1
    info['affinity'] = n
    quota = None
    try:
        rel = [l.split(':', 2)[2].strip() for l in open('/proc/self/cgroup') if l.startswith('0::')]
        if rel:
            parts = Path(rel[0]).parts
            for k in range(len(parts), 0, -1):
                f = Path('/sys/fs/cgroup', *parts[1:k]) / 'cpu.max' if len(parts) > 1 else Path('/sys/fs/cgroup/cpu.max')
                if f.exists():
                    q, per = f.read_text().split()[:2]
                    if q != 'max':
                        quota = min(quota or 1e9, float(q) / float(per))
        f = Path('/sys/fs/cgroup/cpu.max')
        if f.exists():
            q, per = f.read_text().split()[:2]
            if q != 'max':
                quota = min(quota or 1e9, float(q) / float(per))
    except Exception:  # noqa: BLE001
        pass
    try:
        q = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
        per = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
        if q > 0:
            quota = min(quota or 1e9, q / per)
    except Exception:  # noqa: BLE001
        pass
    if quota is not None:
        info['cgroup_quota_cpus'] = round(quota, 2)
        n = min(n, max(1, int(quota)))
    for v in ['SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'PBS_NP', 'NSLOTS', 'LSB_DJOB_NUMPROC']:
        if os.environ.get(v, '').isdigit():
            info[v] = int(os.environ[v]); n = min(n, int(os.environ[v])); break
    try:
        info['load_avg_1min'] = round(os.getloadavg()[0], 1)
    except Exception:  # noqa: BLE001
        pass
    info['usable'] = n
    return n, info


def machine_check(pool, workers, window=8.0):
    """Single-core speed (a short fixed MCMC, idle machine-wide pool) and the CPU time the
    whole pool really receives when every worker is busy (bounded: `window` seconds)."""
    from worker import probe, cpu_burn
    probe(0)
    t1 = min(probe(i) for i in range(1, 4))
    pool.map(cpu_burn, [time.time() + 0.5] * workers, chunksize=1)        # wake every worker
    t = time.time()
    cpu = pool.map(cpu_burn, [t + window] * workers, chunksize=1)
    wall = time.time() - t
    eff = min(float(sum(cpu)) / wall, workers)                             # cores' worth of CPU
    return t1, t1 / REF_PROBE_S, eff


def plan_scale(args, stages, eff_cores, t1):
    t_std, t_inf = STD_PER_PROBE * t1, INF_PER_PROBE * t1
    par = sum(COST[s][0] * t_std + COST[s][1] * t_inf for s in stages if s in COST)
    ser = sum(COST[s][2] for s in stages if s in COST) * t1 / REF_PROBE_S
    wall_full = (1.3 * par / max(eff_cores, 0.1) + ser) / 3600
    if args.mode != 'full' or args.hours is None:
        return None, wall_full, t_std, t_inf
    cap = 2.5 if args.fill else 1.0
    scale = min(cap, max(0.0, (args.hours - ser / 3600) / max(wall_full - ser / 3600, 1e-9)))
    return scale, wall_full, t_std, t_inf


def maybe_start_ode(args):
    if args.no_ode:
        return None
    try:
        import torch  # noqa: F401
        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        print('[run_all] torch not installed: skipping E11 (legacy neural ODE)')
        return None
    reps = args.ode_reps if args.ode_reps is not None else (20 if args.mode == 'full' else (6 if args.mode == 'medium' else 2))
    epochs = 2000 if args.mode != 'quick' else 200
    if not cuda and args.mode == 'full' and args.ode_reps is None:
        reps = 12                               # ~5 min per replicate on one CPU core
    log = open(Path(args.out) / 'e11_legacy_ode.log', 'w')
    print(f'[run_all] starting E11 in background (cuda={cuda}, reps={reps}, epochs={epochs}) -> e11_legacy_ode.log')
    cmd = [sys.executable, str(ROOT / 'experiments' / 'e11_legacy_ode.py'), '--out', args.out,
           '--reps', str(reps), '--epochs', str(epochs)]
    if args.hours is not None:
        # E11 must not outlast the CPU stages: stop starting replicates at ~80% of the budget
        cmd += ['--max-minutes', str(0.8 * 60 * args.hours)]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['quick', 'medium', 'full'], default='quick')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--out', default=str(ROOT / 'results'))
    ap.add_argument('--only', default='')
    ap.add_argument('--no-ode', action='store_true')
    ap.add_argument('--ode-reps', type=int, default=None)
    ap.add_argument('--resume', action='store_true', help='skip stages whose summary file already exists')
    ap.add_argument('--check', action='store_true', help='only report CPUs, speed and estimated run time')
    ap.add_argument('--force', action='store_true', help='run even if the machine is too slow for the budget')
    ap.add_argument('--fill', action='store_true',
                    help='with --hours: also scale replicate counts UP (to at most 2.5x) to use the whole budget')
    ap.add_argument('--scale', type=float, default=None,
                    help='full mode only: multiply every replicate count by this factor (the published run used 2.5); '
                         'overrides --hours/--fill')
    ap.add_argument('--hours', type=float, default=None,
                    help='full mode only: shrink replicate counts (never grow them) to fit this wall time')
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.only.split(',') if s.strip()] or STAGES
    if args.resume:
        done = [s for s in stages if s in DONE_FILE and (Path(args.out) / DONE_FILE[s]).exists()]
        if done:
            print(f'[run_all] resume: skipping finished stages {done}')
        stages = [s for s in stages if s not in done]
    ncpu, cpuinfo = usable_cpus()
    print(f'[run_all] CPUs: {cpuinfo}', flush=True)
    workers = args.workers or max(1, ncpu - (0 if args.no_ode else 1))
    if args.workers and args.workers > ncpu:
        print(f'[run_all] WARNING: --workers {args.workers} > {ncpu} usable CPUs; extra workers only add overhead')
    cfg = {'mode': args.mode, 'out': args.out}
    meta = {'mode': args.mode, 'workers': workers, 'python': sys.version, 'platform': platform.platform(),
            'start': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': stages}
    print(f'[run_all] mode={args.mode} workers={workers} out={args.out}')
    t0 = time.time()
    print('[run_all] compiling kernels (normally 20-90 s) ...', flush=True); warmup()
    tc = time.time() - t0
    print(f'[run_all] compiled in {tc:.0f}s', flush=True)
    import e01_theory, e03_calibration, e07_migration, e08_reconstruction, e09_empirical, e10_legacy  # noqa: F401
    import worker  # noqa: F401  (imported before forking so every worker inherits it)
    ctx = mp.get_context('fork')
    pool = ctx.Pool(workers)
    print('[run_all] measuring single-core speed and parallel throughput ...', flush=True)
    t1, slow, eff = machine_check(pool, workers)
    scale, wall_full, t_std, t_inf = plan_scale(args, stages, eff, t1)
    meta.update(cpus=cpuinfo, probe_s=round(t1, 3), slowdown_vs_reference=round(slow, 1),
                effective_parallel_cores=round(eff, 1), est_full_size_hours=round(wall_full, 2))
    print(f'[run_all] one core here is {slow:.1f}x the reference time (standard fit ~{t_std:.1f}s); '
          f'{workers} workers deliver ~{eff:.1f} cores of throughput; full size ~{wall_full:.2f} h', flush=True)
    problems = []
    if slow > 5:
        problems.append(f'a single core runs {slow:.0f}x slower than a normal machine (other jobs on it, '
                        f'a CPU limit for your user, or a login node)')
    if eff < 0.5 * workers:
        problems.append(f'{workers} worker processes only get ~{eff:.1f} cores of CPU between them')
    for p_ in problems:
        print(f'[run_all] WARNING: {p_}', flush=True)
    if eff < 0.5 * workers and not args.workers:
        new_w = max(1, int(round(eff * 1.25)))
        print(f'[run_all] reducing workers {workers} -> {new_w}', flush=True)
        pool.terminate(); workers = new_w; pool = ctx.Pool(workers); meta['workers'] = workers
    if args.check:
        print('[run_all] --check: machine report only, nothing run.'); pool.terminate(); return
    if args.scale is not None and args.mode == 'full':
        scale = args.scale
        print(f'[run_all] replicate scale {scale:.2f} (fixed by --scale)', flush=True)
        cfg['scale'] = scale; meta['replicate_scale'] = scale
    elif scale is not None:
        if scale < 0.25 and not args.force:
            print(f'\n[run_all] STOP: in {args.hours} h this machine can only do {scale:.0%} of the replicates, too '
                  f'few to be useful (full size would take ~{wall_full:.1f} h here). Run it on a machine or batch job '
                  f'with more free cores (see README, "Where to run"), raise --hours, or pass --force.', flush=True)
            pool.terminate(); sys.exit(2)
        scale = max(scale, 0.05)
        print(f'[run_all] replicate scale {scale:.2f} for a {args.hours} h budget', flush=True)
        cfg['scale'] = scale; meta['replicate_scale'] = scale
    ode_done = (Path(args.out) / 'e11_legacy_ode' / 'legacy_ode_summary.json').exists() and args.resume
    ode = maybe_start_ode(args) if (('e11' in stages or not args.only) and not ode_done and not args.check) else None
    results = {}
    timing = {}
    if args.resume and (Path(args.out) / 'run_meta.json').exists():
        try:
            timing = json.load(open(Path(args.out) / 'run_meta.json')).get('timing_s', {})
        except Exception:  # noqa: BLE001
            timing = {}
    with pool:
        for st in stages:
            if st == 'e11':
                continue
            ts = time.time()
            print(f'\n=== {st} ===', flush=True)
            if st == 'e01':
                results['e01'] = e01_theory.run(cfg, pool)
            elif st == 'e03':
                results['e03'] = e03_calibration.run(cfg, pool, which=('sbc',))
            elif st == 'e04':
                results['e04'] = e03_calibration.run(cfg, pool, which=('robustness',))
            elif st == 'e05':
                results['e05'] = e03_calibration.run(cfg, pool, which=('violations',))
            elif st == 'e05b':
                results['e05b'] = e03_calibration.run(cfg, pool, which=('failure_map',))
            elif st == 'e07':
                results['e07'] = e07_migration.run(cfg, pool)
            elif st == 'e08':
                results['e08'] = e08_reconstruction.run(cfg, pool)
            elif st == 'e09':
                results['e09'] = e09_empirical.run(cfg, pool)
            elif st == 'e10':
                results['e10'] = e10_legacy.run(cfg, pool)
            timing[st] = round(time.time() - ts, 1)
            print(f'=== {st} done in {timing[st]:.0f}s (total {time.time() - t0:.0f}s) ===', flush=True)
            json.dump({'meta': meta, 'timing_s': timing}, open(Path(args.out) / 'run_meta.json', 'w'), indent=2)
    if ode is not None:
        print('[run_all] waiting for E11 (neural ODE) ...', flush=True)
        ode.wait()
    meta['end'] = time.strftime('%Y-%m-%d %H:%M:%S'); meta['total_s'] = round(time.time() - t0, 1)
    json.dump({'meta': meta, 'timing_s': timing}, open(Path(args.out) / 'run_meta.json', 'w'), indent=2)
    if ode is not None and ode.returncode != 0:
        print(f'[run_all] E11 exited with code {ode.returncode}; see e11_legacy_ode.log', flush=True)
    import summarize
    summarize.write(Path(args.out))
    print(f'\n[run_all] finished in {time.time() - t0:.0f}s -> {args.out}/SUMMARY.md')


if __name__ == '__main__':
    main()
