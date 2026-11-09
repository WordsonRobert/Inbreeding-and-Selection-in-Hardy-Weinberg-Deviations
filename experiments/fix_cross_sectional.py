"""Re-run only the cross-sectional part of E09 with the F_eff reporting and rebuild
SUMMARY.md. Takes about a minute.   usage: python experiments/fix_cross_sectional.py results"""
import json
import os
import sys
from pathlib import Path

for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[v] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import e09_empirical  # noqa: E402
import summarize  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'results')
f = out / 'e09_empirical' / 'empirical_summary.json'
res = json.load(open(f))
res['cross_sectional'] = e09_empirical.cross_sectional({'mode': 'full', 'out': str(out)}, out / 'e09_empirical')
json.dump(res, open(f, 'w'), indent=2, default=float)
summarize.write(out)
for k, d in res['cross_sectional'].items():
    print(f"{k:26s} n={d['n']:6d} p={d['p_hat']:.3f} F_IS={d['F_IS_observed']:+.4f} F_min={d['F_min_at_p_hat']:+.4f} "
          f"{'bound' if d['at_bound'] else '     '} F_eff={d['F_eff_median']:+.4f} "
          f"[{d['F_eff_q05']:+.4f}, {d['F_eff_q95']:+.4f}]")
print('updated', f, 'and', out / 'SUMMARY.md')
