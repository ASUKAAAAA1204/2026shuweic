"""Paired evaluation of the lookahead policy against pressure_gated on the official validation split.

Protocol: 3 cases x 30 public validation seeds x 5 duration mappings x 2 policies.
Fitting uses only the train split (family duration pools).  The policy never sees
hidden durations, validation outcomes, case ids or target values.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parents[1] / "第三问改进正式方案" / "code"
sys.path.insert(0, str(FORMAL))
sys.path.insert(0, str(HERE))

from support import load, jobs_for
from engine import Engine, independent_metrics
from deadline_policy import choose, pressure
from baseline import action
from lookahead_policy import select as lookahead_select

ROOT = HERE.parents[1]
OUT = HERE.parent / "results" / "lookahead_validation"
OUT.mkdir(parents=True, exist_ok=True)
MAPPINGS = [(1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.)]
CONFIG = dict(strength=40, feedback_strength=3, quantiles=5)


def main():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
    med = {f: float(np.median(v)) for f, v in pools.items()}
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]

    def hidden_for(jobs, seed):
        rng = random.Random(seed)
        out = {}
        for j in jobs:
            c = vpool[vpool.family_id == j["family_id"]]
            out[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
        return out

    rows = []
    max_diff = 0.0
    for case in cond["cases"]:
        jobs = jobs_for(tasks, case["case_id"])
        for seed in cond["validation_seeds"]:
            dur = hidden_for(jobs, seed)
            hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)}) for k, v in dur.items()}
            for alpha, beta in MAPPINGS:
                for method in ("pressure_gated", "lookahead"):
                    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                               case["observation_limit"], alpha, beta)
                    history = {}
                    while not e.terminal:
                        o = e.observe()
                        if method == "pressure_gated":
                            a = choose(o, pools, history) if pressure(o, pools, history) else action(o, med, "density", "minimum")
                        else:
                            a = lookahead_select(o, pools, history, **CONFIG)
                        ev = e.step(a)
                        if a["type"] == "execute" and ev.get("completion") is not None:
                            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                            history.setdefault(f, []).append((ev["completion"] - ev["service_start"] - beta) / alpha)
                    m = e.metrics()
                    ind = independent_metrics(jobs, e.events, e.H)
                    diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                    max_diff = max(max_diff, diff)
                    assert not m["invalid_actions"] and diff < 1e-9, (case["case_id"], seed, method, diff)
                    rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                                     method=method, **{k: float(m[k]) for k in m}))
        print("finished", case["case_id"], flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "trials.csv", index=False)

    nominal = d[(d.alpha == 1.0) & (d.beta == .5)]
    piv = nominal.pivot_table(index="method", columns="case", values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    base = nominal[nominal.method == "pressure_gated"].set_index(["case", "seed"]).weighted_on_time_rate
    cand = nominal[nominal.method == "lookahead"].set_index(["case", "seed"]).weighted_on_time_rate
    delta = (cand - base)
    rng = np.random.default_rng(20261122)
    by_seed = delta.groupby(level=1).mean()
    boot = [float(by_seed.sample(frac=1, replace=True, random_state=int(s)).mean())
            for s in rng.integers(0, 10 ** 6, 4000)]
    summary = {
        "config": CONFIG,
        "max_independent_metric_diff": max_diff,
        "nominal_avg": {m: float(piv.loc[m, "avg3"]) for m in piv.index},
        "nominal_by_case": {m: {c: float(piv.loc[m, c]) for c in ("S1", "S2", "S3")} for m in piv.index},
        "delta_nominal_pp": float(delta.mean() * 100),
        "delta_nominal_by_case_pp": {c: float(delta[delta.index.get_level_values(0) == c].mean() * 100) for c in ("S1", "S2", "S3")},
        "delta_ci95_pp": [float(np.percentile(boot, 2.5) * 100), float(np.percentile(boot, 97.5) * 100)],
        "seeds_with_positive_delta": int((by_seed > 0).sum()),
        "n_seeds": int(len(by_seed)),
        "all_mapping_avg": {m: float(d[d.method == m].weighted_on_time_rate.mean()) for m in ("pressure_gated", "lookahead")},
        "all_mapping_delta_pp": float((d[d.method == "lookahead"].weighted_on_time_rate.mean()
                                       - d[d.method == "pressure_gated"].weighted_on_time_rate.mean()) * 100),
        "nominal_cost": {m: float(nominal[nominal.method == m].cost.mean()) for m in ("pressure_gated", "lookahead")},
        "nominal_completion": {m: float(nominal[nominal.method == m].weighted_completion_rate.mean()) for m in ("pressure_gated", "lookahead")},
        "nominal_observations": {m: float(nominal[nominal.method == m].observations.mean()) for m in ("pressure_gated", "lookahead")},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
