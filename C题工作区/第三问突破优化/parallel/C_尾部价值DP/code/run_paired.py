"""Paired protocol runner for the Role C DP tail policies.

Protocol is copied verbatim from ``confirm_lookahead.py``:
  3 cases x 30 public validation seeds x 5 duration mappings, paired per
  (case, seed, mapping); hidden durations by ``random.Random(seed)`` walking
  ``jobs_for(tasks, case)`` in order and sampling the validation&eligible pool of
  the job family; ``d_hat = (completion - service_start - beta) / alpha`` fed
  back per family; official ``Engine`` plus ``independent_metrics`` cross-check.

Usage
-----
    python run_paired.py --methods pressure_gated,v2_s20,dp_quant_dt05 --mappings all
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parents[3] / "第三问改进正式方案" / "code"
sys.path.insert(0, str(FORMAL))
sys.path.insert(0, str(HERE))

from support import load, jobs_for                      # noqa: E402
from engine import Engine, independent_metrics          # noqa: E402
from deadline_policy import choose, pressure            # noqa: E402
from baseline import action                             # noqa: E402
import dp_tail_policy as dp                             # noqa: E402

ROOT = HERE.parents[3]
MAPPINGS = [(1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.)]
ORDERS = ('edf', 'wdd', 'slack')


def policy(method):
    """Return a callable ``(o, pools, history) -> action`` for a method name."""
    if method == 'pressure_gated':
        return lambda o, pools, hist, med=None: choose(o, pools, hist) if pressure(o, pools, hist) else action(o, med, 'density', 'minimum')
    if method == 'v2_s20':
        return lambda o, pools, hist, med=None: dp.select_v2(o, pools, hist, strength=20, feedback_strength=3)
    if method.startswith('dp_quant'):
        dt = 1.0 if method.endswith('dt10') else .5
        return lambda o, pools, hist, med=None: dp.select(o, pools, hist, mode='plan', timeline='quantile',
                                                          scenarios=(.2, .5, .8), dt=dt, orders=ORDERS)
    if method.startswith('dp_mean'):
        dt = 1.0 if method.endswith('dt10') else .5
        return lambda o, pools, hist, med=None: dp.select(o, pools, hist, mode='plan_mean', timeline='mean',
                                                          dt=dt, orders=ORDERS)
    if method.startswith('dp_mc'):
        # dp_mc_t4_s64_dt05 / dp_mcg_... (greedy plans only) / dp_mcd_... (DP plans only)
        # dp_mc2_... additionally builds a forced-first-job plan for each candidate
        seg = method.split('_')
        use_dp = seg[1] != 'mcg'
        use_greedy = seg[1] != 'mcd'
        force_first = seg[1] == 'mc2'
        k = int(seg[2][1:])
        s = int(seg[3][1:])
        dt = 1.0 if method.endswith('dt10') else .5
        return lambda o, pools, hist, med=None: dp.select(o, pools, hist, mode='mc', topk=k, mc_samples=s,
                                                          timeline='quantile', scenarios=(.2, .5, .8),
                                                          tail_quantiles=3, dt=dt, orders=ORDERS,
                                                          use_dp=use_dp, use_greedy=use_greedy,
                                                          force_first=force_first)
    if method.startswith('dp_hybrid'):
        k = int(method.split('k')[1].split('_')[0])
        dt = 1.0 if method.endswith('dt10') else .5
        return lambda o, pools, hist, med=None: dp.select(o, pools, hist, mode='hybrid', topk=k,
                                                          timeline='quantile', scenarios=(.2, .5, .8),
                                                          tail_quantiles=3, dt=dt, orders=ORDERS)
    if method.startswith('dp_onestep'):
        return lambda o, pools, hist, med=None: dp.select(o, pools, hist, mode='onestep', timeline='quantile',
                                                          scenarios=(.2, .5, .8), tail_quantiles=3,
                                                          dt=.5, orders=ORDERS)
    if method.startswith('dp_quant5'):
        return lambda o, pools, hist, med=None: dp.select(o, pools, hist, mode='plan', timeline='quantile',
                                                          scenarios=(.1, .3, .5, .7, .9), dt=.5, orders=ORDERS)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default='pressure_gated,v2_s20,dp_quant_dt05')
    ap.add_argument('--mappings', default='all', choices=['all', 'nominal'])
    ap.add_argument('--tag', default='main')
    ap.add_argument('--seeds', type=int, default=30)
    args = ap.parse_args()

    methods = [m for m in args.methods.split(',') if m]
    assert methods[0] == 'pressure_gated', 'baseline must be included for pairing'
    maps = MAPPINGS if args.mappings == 'all' else [(1., .5)]
    out = HERE.parent / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=True)

    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / '处理结果' / 'v1' / 'segments_private.csv')
    pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby('family_id')}
    med = {f: float(np.median(v)) for f, v in pools.items()}
    vpool = seg[(seg.split == 'validation') & (seg.eligible == 1)]
    seeds = cond['validation_seeds'][:args.seeds]

    def hidden_for(jobs, seed):
        rng = random.Random(seed)
        out = {}
        for j in jobs:
            c = vpool[vpool.family_id == j['family_id']]
            out[j['job_id']] = float(c.iloc[rng.randrange(len(c))].duration_s)
        return out

    rows, max_diff = [], 0.0
    for case in cond['cases']:
        jobs = jobs_for(tasks, case['case_id'])
        for seed in seeds:
            dur = hidden_for(jobs, seed)
            hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)}) for k, v in dur.items()}
            for alpha, beta in maps:
                for method in methods:
                    fn = policy(method)
                    e = Engine(jobs, hidden, zones, case['horizon_s'], case['cost_budget'],
                               case['observation_limit'], alpha, beta)
                    history = {}
                    t_pol, ndec = 0.0, 0
                    while not e.terminal:
                        o = e.observe()
                        t0 = time.perf_counter()
                        a = fn(o, pools, history, med)
                        t_pol += time.perf_counter() - t0
                        ndec += 1
                        ev = e.step(a)
                        if a['type'] == 'execute' and ev.get('completion') is not None:
                            f = next(j['family_id'] for j in jobs if j['job_id'] == a['job_id'])
                            history.setdefault(f, []).append((ev['completion'] - ev['service_start'] - beta) / alpha)
                    m = e.metrics()
                    ind = independent_metrics(jobs, e.events, e.H)
                    diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                    max_diff = max(max_diff, diff)
                    assert not m['invalid_actions'] and diff < 1e-9, (case['case_id'], seed, method, diff)
                    rows.append(dict(case=case['case_id'], seed=seed, alpha=alpha, beta=beta, method=method,
                                     weighted_on_time_rate=float(m['weighted_on_time_rate']),
                                     weighted_completion_rate=float(m['weighted_completion_rate']),
                                     cost=float(m['cost']), observations=int(m['observations']),
                                     decisions=ndec, policy_seconds=t_pol))
        print('finished', case['case_id'], flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(out / 'trials.csv', index=False)
    summary = build_summary(d, methods, maps, max_diff, dict(tag=args.tag, methods=methods,
                                                             mappings=[list(m) for m in maps],
                                                             orders=list(ORDERS)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(summary['headline'], indent=2, ensure_ascii=False))


def build_summary(d, methods, maps, max_diff, config):
    nominal = d[(d.alpha == 1.0) & (d.beta == .5)]
    piv = nominal.pivot_table(index='method', columns='case', values='weighted_on_time_rate', aggfunc='mean')
    base = nominal[nominal.method == 'pressure_gated'].set_index(['case', 'seed']).weighted_on_time_rate
    rng = np.random.default_rng(20261122)
    out = {'config': config, 'max_independent_metric_diff': max_diff, 'methods': {}}
    for m in methods:
        if m == 'pressure_gated':
            continue
        cand = nominal[nominal.method == m].set_index(['case', 'seed']).weighted_on_time_rate
        delta = cand - base
        by_seed = delta.groupby(level=1).mean()
        boot = [float(by_seed.sample(frac=1, replace=True, random_state=int(s)).mean())
                for s in rng.integers(0, 10 ** 6, 4000)]
        by_map = {}
        for a, b in maps:
            x = d[(d.method == m) & (d.alpha == a) & (d.beta == b)].set_index(['case', 'seed']).weighted_on_time_rate
            y = d[(d.method == 'pressure_gated') & (d.alpha == a) & (d.beta == b)].set_index(['case', 'seed']).weighted_on_time_rate
            by_map[f'{a},{b}'] = dict(cand=float(x.mean()), base=float(y.mean()), delta_pp=float((x - y).mean() * 100),
                                      n_below=int(((x - y) < 0).sum()))
        out['methods'][m] = {
            'nominal_avg': float(piv.loc[m].mean()),
            'nominal_by_case': {c: float(piv.loc[m, c]) for c in piv.columns},
            'all_mapping_avg': float(d[d.method == m].weighted_on_time_rate.mean()),
            'by_mapping': by_map,
            'delta_nominal_pp': float(delta.mean() * 100),
            'delta_nominal_by_case_pp': {c: float(delta[delta.index.get_level_values(0) == c].mean() * 100)
                                         for c in ('S1', 'S2', 'S3')},
            'delta_ci95_pp': [float(np.percentile(boot, 2.5) * 100), float(np.percentile(boot, 97.5) * 100)],
            'seeds_positive': int((by_seed > 0).sum()),
            'n_seeds': int(len(by_seed)),
            'nominal_cost': float(nominal[nominal.method == m].cost.mean()),
            'nominal_completion': float(nominal[nominal.method == m].weighted_completion_rate.mean()),
            'nominal_observations': float(nominal[nominal.method == m].observations.mean()),
            'policy_seconds_per_decision': float(
                (nominal[nominal.method == m].policy_seconds / nominal[nominal.method == m].decisions).mean()),
            'policy_seconds_per_trial': float(nominal[nominal.method == m].policy_seconds.mean()),
        }
    bpiv = piv.loc['pressure_gated'] if 'pressure_gated' in piv.index else None
    out['headline'] = {m: {k: v for k, v in out['methods'][m].items() if k in
                           ('nominal_avg', 'delta_nominal_pp', 'delta_ci95_pp', 'seeds_positive', 'all_mapping_avg')}
                       for m in out['methods']}
    if bpiv is not None:
        out['baseline'] = {'nominal_avg': float(bpiv.mean()),
                           'nominal_by_case': {c: float(bpiv[c]) for c in bpiv.index}}
    return out


if __name__ == '__main__':
    main()
