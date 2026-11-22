"""Environment-group holdout for role F (mirrors 第三问突破优化/code/holdout_lookahead.py).

Two disjoint halves of the 16 training environments.  For each fold the duration
pools are fitted on the other half; every method is then replayed on the held
half with fresh seeds and the nominal mapping (alpha=1, beta=0.5).  Reported
methods: pressure_gated (legal baseline), v2_s20 (legal online reference),
oracle (perfect-information upper bound, replayed through the engine),
ub_relax_knapsack (provable relaxation upper bound, no simulation).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parents[3] / "第三问改进正式方案" / "code"
BREAK = HERE.parents[3] / "第三问突破优化" / "code"
sys.path.insert(0, str(FORMAL))
sys.path.insert(0, str(BREAK))
sys.path.insert(0, str(HERE))

from support import load, jobs_for
from engine import Engine, independent_metrics
from deadline_policy import choose, pressure
from baseline import action
import lookahead_policy as LP
import oracle as O
import reference_policy

ROOT = HERE.parents[3]
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
V2 = dict(depth=1, strength=20, feedback_strength=3, nq_root=5, nq=2, beam=3)
S40 = dict(strength=40, feedback_strength=3, quantiles=5)
ALPHA, BETA = 1.0, 0.5
SEEDS = list(range(2026120100, 2026120120))


def main():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    groups = json.loads((ROOT / "第三问突破优化" / "results" / "grouped_training"
                         / "registration.json").read_text())["groups"]
    rows = []
    max_diff = 0.0
    for fold, held in enumerate(groups):
        fit = seg[(seg.split == "train") & (seg.eligible == 1)
                  & (~seg.environment_id.isin(held))]
        evp = seg[(seg.split == "train") & (seg.eligible == 1)
                  & (seg.environment_id.isin(held))]
        pools = {f: g.duration_s.to_numpy(float) for f, g in fit.groupby("family_id")}
        med = {f: float(np.median(v)) for f, v in pools.items()}
        for case in cond["cases"]:
            jobs = jobs_for(tasks, case["case_id"])
            for seed in SEEDS:
                rng = random.Random(seed)
                dur = {}
                for j in jobs:
                    c = evp[evp.family_id == j["family_id"]]
                    dur[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
                hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)})
                          for k, v in dur.items()}
                for method in ("pressure_gated", "v2_s20", "lookahead_s40"):
                    e = Engine(jobs, hidden, zones, case["horizon_s"],
                               case["cost_budget"], case["observation_limit"], ALPHA, BETA)
                    history = {}
                    while not e.terminal:
                        o = e.observe()
                        if method == "pressure_gated":
                            a = choose(o, pools, history) if pressure(o, pools, history) \
                                else action(o, med, "density", "minimum")
                        elif method == "v2_s20":
                            a = reference_policy.select(o, pools, history, **V2)
                        else:
                            a = LP.select(o, pools, history, **S40)
                        ev = e.step(a)
                        if a["type"] == "execute" and ev.get("completion") is not None:
                            f = next(j["family_id"] for j in jobs
                                     if j["job_id"] == a["job_id"])
                            history.setdefault(f, []).append(
                                (ev["completion"] - ev["service_start"] - BETA) / ALPHA)
                    m = e.metrics()
                    ind = independent_metrics(jobs, e.events, e.H)
                    diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                    max_diff = max(max_diff, diff)
                    assert m["invalid_actions"] == 0 and diff < 1e-9
                    rows.append(dict(fold=fold, case=case["case_id"], seed=seed,
                                     method=method,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"])))
                p, xy, z, D, D0 = O.build_arrays(jobs, dur, ALPHA, BETA, zones)
                dp = O.solve_oracle_dp(jobs, p, zones, case["horizon_s"],
                                       case["cost_budget"], D, D0)
                m, ind, diff, _ = O.replay_plan(
                    jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                    case["observation_limit"], ALPHA, BETA, dp["order"], return_engine=True)
                diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                max_diff = max(max_diff, diff)
                assert abs(m["weighted_on_time_rate"] - dp["value"]) < 1e-9
                rows.append(dict(fold=fold, case=case["case_id"], seed=seed, method="oracle",
                                 weighted_on_time_rate=float(m["weighted_on_time_rate"])))
                ub, _ = O.relaxation_ub(jobs, p, zones, case["horizon_s"],
                                        case["cost_budget"], D0)
                assert ub >= dp["value"] - 1e-9
                rows.append(dict(fold=fold, case=case["case_id"], seed=seed,
                                 method="ub_relax_knapsack",
                                 weighted_on_time_rate=float(ub)))
        print("finished fold", fold, flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "holdout_trials.csv", index=False)
    piv = d.pivot_table(index="method", columns="case",
                        values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    base = d[d.method == "pressure_gated"].set_index(["fold", "seed", "case"]) \
        .weighted_on_time_rate
    summary = {"by_case_pct": (piv * 100).round(4).to_dict(), "by_fold": {},
               "max_independent_metric_diff": max_diff, "seeds_per_fold": len(SEEDS)}
    for m in ("v2_s20", "lookahead_s40", "oracle"):
        x = d[d.method == m].set_index(["fold", "seed", "case"]).weighted_on_time_rate
        delta = (x - base) * 100
        summary["by_fold"][m] = dict(
            mean_pp=float(delta.mean()),
            fold0_pp=float(delta[delta.index.get_level_values(0) == 0].mean()),
            fold1_pp=float(delta[delta.index.get_level_values(0) == 1].mean()),
        )
    (OUT / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
