"""Shared helpers for the BASTION experiment suite."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bastion import mcmc, sims, model  # noqa: E402

COL = {'inbreeding': '#2166ac', 'selection': '#d6604d', 'mutation': '#4dac26',
       'drift': '#7f7f7f', 'mendelian_reset': '#9970ab'}

# run-size presets: scale factors applied to every experiment's replicate counts
MODES = {'quick': 0.04, 'medium': 0.3, 'full': 1.0}


def nrep(n_full, cfg, minimum=4):
    scale = cfg.get('scale', MODES[cfg['mode']])
    return max(minimum, int(round(n_full * scale)))


def mcmc_kw(cfg):
    if cfg['mode'] == 'quick':
        return dict(n_chains=2, n_steps=2500, n_warmup=1000)
    return dict(n_chains=4, n_steps=4000, n_warmup=1500)


def outdir(cfg, name):
    d = Path(cfg['out']) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=conv)


def safe(fn):
    """Wrap a worker so one failing replicate does not kill a run (picklable:
    functools.wraps copies __module__/__qualname__ so the pool finds it by name)."""
    import functools

    @functools.wraps(fn)
    def wrapped(args):
        try:
            return fn(args)
        except Exception as e:  # noqa: BLE001
            return {'_error': repr(e), '_tb': traceback.format_exc()[-800:]}
    return wrapped


def pmap(pool, fn, args, label='', heartbeat=600):
    """Parallel map with progress every 10% and a heartbeat line at least every
    `heartbeat` seconds, so a slow stage is never silent."""
    import multiprocessing as _mp
    t = time.time()
    out = []
    n = len(args)
    it = pool.imap_unordered(fn, args, chunksize=1)
    i = 0
    while i < n:
        try:
            r = it.next(timeout=heartbeat)
        except _mp.TimeoutError:
            el = time.time() - t
            eta = f', ETA ~{el / i * (n - i) / 60:.0f} min' if i else ''
            print(f"    [{label}] still running: {i}/{n} done after {el / 60:.0f} min{eta}", flush=True)
            continue
        out.append(r); i += 1
        if i % max(1, n // 10) == 0 or i == n:
            el = time.time() - t
            print(f"    [{label}] {i}/{n}  ({el:.0f}s, ETA ~{el / i * (n - i) / 60:.1f} min)", flush=True)
    errs = [r for r in out if isinstance(r, dict) and '_error' in r]
    if errs:
        print(f"    [{label}] {len(errs)} replicate(s) failed, e.g. {errs[0]['_error']}", flush=True)
    return [r for r in out if not (isinstance(r, dict) and '_error' in r)]


def cover(lo, hi, v):
    return int(lo <= v <= hi)


def fit_summary(mc, truth):
    """Flatten an MCMC result against a truth dict {name: value}."""
    row = {'converged': mc['converged'], 'converged_F_hs': mc['converged_F_hs']}
    for k, v in truth.items():
        if k not in mc['summary']:
            continue
        S = mc['summary'][k]
        row.update({f'{k}_true': v, f'{k}_med': S['median'], f'{k}_mean': S['mean'], f'{k}_sd': S['sd'],
                    f'{k}_q05': S['q05'], f'{k}_q95': S['q95'], f'{k}_q25': S['q25'], f'{k}_q75': S['q75'],
                    f'{k}_cov90': cover(S['q05'], S['q95'], v), f'{k}_cov50': cover(S['q25'], S['q75'], v),
                    f'{k}_rhat': S['rhat']})
    return row


def write_csv(rows, path):
    import csv
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})


def read_rows(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], float)


def auc(neg, pos):
    neg = np.asarray(neg, float); pos = np.asarray(pos, float)
    neg = neg[np.isfinite(neg)]; pos = pos[np.isfinite(pos)]
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    allv = np.concatenate([neg, pos])
    r = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def prior_sd():
    d = mcmc.DEFAULTS
    return {'F': (d['F_hi'] - d['F_lo']) / np.sqrt(12), 's': d['s_sd'], 'h': d['h_sd'],
            'log10Ne': (d['lNe_hi'] - d['lNe_lo']) / np.sqrt(12)}
