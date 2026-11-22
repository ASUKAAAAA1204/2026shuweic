"""Environment-group holdout check for the joint-scenario policy.

Two disjoint halves of the 16 training environments (same registration as
``holdout_lookahead.py``).  For each fold the duration pools are fitted on the
other half and every policy is replayed on the held half with fresh seeds and
the nominal mapping.  This is an independent environment-level stress test; it
is not the official validation split and the test split is never read.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "第三问改进正式方案" / "code"))
sys.path.insert(0, str(HERE))

from support import load, jobs_for                       # noqa: E402
from engine import Engine, independent_metrics           # noqa: E402
from deadline_policy import choose, pressure              # noqa: E402
from baseline import action                              # noqa: E402
import v2_reference                                      # noqa: E402
from joint_policy import JointScenarioPolicy             # noqa: E402

OUT = HERE.parent / "results" / "holdout_environment"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["pressure_gated", "v2", "joint_survival_k16", "joint_survival_k32",
           "joint_survival_k64", "joint_survival_k128"]
JOINT = {"joint_survival_k16": 16, "joint_survival_k32": 32,
         "joint_survival_k64": 64, "joint_survival_k128": 128}


def bootstrap_ci(values, rng, draws=5000):
    v = np.asarray(values, float)
    i = rng.integers(0, len(v), size=(draws, len(v)))
    m = v[i].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    groups = json.loads((ROOT / "第三问突破优化" / "results" / "grouped_training"
                         / "registration.json").read_text())["groups"]
    rows = []
    for fold, held in enumerate(groups):
        fit = seg[(seg.split == "train") & (seg.eligible == 1)
                  & (~seg.environment_id.isin(held))]
        evp = seg[(seg.split == "train") & (seg.eligible == 1)
                  & (seg.environment_id.isin(held))]
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
                    policy = JointScenarioPolicy(K=JOINT[method], base="survival",
                                                 stratified=True) if method in JOINT else None
                    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                               case["observation_limit"], 1.0, .5)
                    history = {}
                    while not e.terminal:
                        o = e.observe()
                        if method == "pressure_gated":
                            a = choose(o, pools, history) if pressure(o, pools, history) \
                                else action(o, med, "density", "minimum")
                        elif method == "v2":
                            a = v2_reference.select(o, pools, history, strength=20,
                                                    feedback_strength=3, depth=1,
                                                    nq_root=5, nq=2, beam=3)
                        else:
                            a = policy.select(o, pools, history)
                        ev = e.step(a)
                        if a["type"] == "execute" and ev.get("completion") is not None:
                            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                            history.setdefault(f, []).append(
                                ev["completion"] - ev["service_start"] - .5)
                    m = e.metrics()
                    ind = independent_metrics(jobs, e.events, e.H)
                    diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                    assert not m["invalid_actions"] and diff < 1e-9, (fold, case, seed, method, diff)
                    rows.append(dict(fold=fold, case=case["case_id"], seed=seed, method=method,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                                     weighted_completion_rate=float(m["weighted_completion_rate"]),
                                     cost=float(m["cost"]), independent_delta=float(diff)))
        print("finished fold", fold, flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "trials.csv", index=False)
    piv = d.pivot_table(index="method", columns="case",
                        values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    base = d[d.method == "pressure_gated"].set_index(["fold", "seed", "case"]).weighted_on_time_rate
    rng = np.random.default_rng(20261122)
    summary = {"by_case": (piv * 100).round(4).to_dict(), "by_fold": {},
               "max_independent_metric_diff": float(d.independent_delta.max()),
               "invalid_actions": 0, "n_trials": int(len(d))}
    for m in METHODS[1:]:
        x = d[d.method == m].set_index(["fold", "seed", "case"]).weighted_on_time_rate
        delta = (x - base) * 100
        by_seed = delta.groupby(level=1).mean()
        summary["by_fold"][m] = {
            "mean_pp": float(delta.mean()),
            "fold0_pp": float(delta[delta.index.get_level_values(0) == 0].mean()),
            "fold1_pp": float(delta[delta.index.get_level_values(0) == 1].mean()),
            "ci95_pp": bootstrap_ci(by_seed.to_numpy(), rng),
            "positive_seeds": int((by_seed > 0).sum()),
            "n_seeds": int(len(by_seed)),
            "by_case_pp": {c: float(delta[delta.index.get_level_values(2) == c].mean())
                           for c in sorted(d.case.unique())},
        }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
