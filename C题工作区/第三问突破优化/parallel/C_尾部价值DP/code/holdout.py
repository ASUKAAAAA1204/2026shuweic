"""Environment-group holdout for the Role C MC/DP tail policies.

Follows ``第三问突破优化/code/holdout_lookahead.py`` exactly:
  * two disjoint halves of the 16 training environments (``registration.json``),
  * for each fold the duration pools are fitted on the *other* half,
  * held durations are drawn from the held half's eligible training segments,
  * 20 fresh seeds (2026120100..2026120119), nominal mapping (alpha=1, beta=0.5).

This is an independent environment-level stress test, not the official validation
split; it is never used for parameter selection.
"""
import json
import random
import sys
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
OUT = HERE.parent / "results" / "holdout"
OUT.mkdir(parents=True, exist_ok=True)
ORDERS = ('edf', 'wdd', 'slack')
METHODS = ('pressure_gated', 'v2_s20', 'dp_mc_t4_s256_dt10')


def policy(method):
    if method == 'pressure_gated':
        return lambda o, pools, hist, med=None: choose(o, pools, hist) if pressure(o, pools, hist) else action(o, med, 'density', 'minimum')
    if method == 'v2_s20':
        return lambda o, pools, hist, med=None: dp.select_v2(o, pools, hist, strength=20, feedback_strength=3)
    if method == 'dp_mc_t4_s256_dt10':
        return lambda o, pools, hist, med=None: dp.select(o, pools, hist, mode='mc', topk=4, mc_samples=256,
                                                          timeline='quantile', scenarios=(.2, .5, .8),
                                                          tail_quantiles=3, dt=1.0, orders=ORDERS)
    raise ValueError(method)


def main():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    reg = ROOT / "第三问突破优化" / "results" / "grouped_training" / "registration.json"
    groups = json.loads(reg.read_text())["groups"]
    rows = []
    for fold, held in enumerate(groups):
        fit = seg[(seg.split == "train") & (seg.eligible == 1) & (~seg.environment_id.isin(held))]
        evp = seg[(seg.split == "train") & (seg.eligible == 1) & (seg.environment_id.isin(held))]
        pools = {f: g.duration_s.to_numpy(float) for f, g in fit.groupby("family_id")}
        med = {f: float(np.median(v)) for f, v in pools.items()}
        for case in cond["cases"]:
            jobs = jobs_for(tasks, case["case_id"])
            for seed in range(2026120100, 2026120120):
                rng = random.Random(seed)
                dur = {}
                for j in jobs:
                    c = evp[evp.family_id == j["family_id"]]
                    dur[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
                hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)})
                          for k, v in dur.items()}
                for method in METHODS:
                    fn = policy(method)
                    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                               case["observation_limit"], 1.0, .5)
                    history = {}
                    while not e.terminal:
                        o = e.observe()
                        a = fn(o, pools, history, med)
                        ev = e.step(a)
                        if a["type"] == "execute" and ev.get("completion") is not None:
                            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                            history.setdefault(f, []).append(ev["completion"] - ev["service_start"] - .5)
                    m = e.metrics()
                    ind = independent_metrics(jobs, e.events, e.H)
                    diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                    assert not m["invalid_actions"] and diff < 1e-9, (fold, case["case_id"], seed, method, diff)
                    rows.append(dict(fold=fold, case=case["case_id"], seed=seed, method=method,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                                     weighted_completion_rate=float(m["weighted_completion_rate"]),
                                     cost=float(m["cost"])))
        print("finished fold", fold, flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "trials.csv", index=False)
    piv = d.pivot_table(index="method", columns="case", values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    base = d[d.method == "pressure_gated"].set_index(["fold", "seed", "case"]).weighted_on_time_rate
    summary = {"by_case_pct": (piv * 100).round(4).to_dict(), "by_fold": {}}
    for m in METHODS[1:]:
        x = d[d.method == m].set_index(["fold", "seed", "case"]).weighted_on_time_rate
        delta = (x - base) * 100
        summary["by_fold"][m] = {
            "mean_pp": float(delta.mean()),
            "fold0_pp": float(delta[delta.index.get_level_values(0) == 0].mean()),
            "fold1_pp": float(delta[delta.index.get_level_values(0) == 1].mean()),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
