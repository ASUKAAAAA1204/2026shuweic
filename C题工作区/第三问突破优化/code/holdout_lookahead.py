"""Environment-group holdout check for the lookahead policy.

Two disjoint halves of the 16 training environments are used.  For each fold the
duration pools are fitted on the other half and every policy is replayed on the
held half with fresh seeds and the nominal mapping.  This is an independent,
environment-level stress test; it is not the official validation split.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "第三问改进正式方案" / "code"))
sys.path.insert(0, str(HERE))

from support import load, jobs_for
from engine import Engine
from deadline_policy import choose, pressure
from baseline import action
from lookahead_policy import select as lookahead_select
from robust_policy import select as robust_select

ROOT = HERE.parents[1]
OUT = HERE.parent / "results" / "lookahead_holdout"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    groups = json.loads((ROOT / "第三问突破优化" / "results" / "grouped_training" / "registration.json").read_text())["groups"]
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
                hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)}) for k, v in dur.items()}
                for method in ("pressure_gated", "robust20", "lookahead"):
                    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                               case["observation_limit"], 1.0, .5)
                    history = {}
                    while not e.terminal:
                        o = e.observe()
                        if method == "pressure_gated":
                            a = choose(o, pools, history) if pressure(o, pools, history) else action(o, med, "density", "minimum")
                        elif method == "robust20":
                            a = robust_select(o, pools, history, strength=20, feedback_strength=3)
                        else:
                            a = lookahead_select(o, pools, history, strength=40, feedback_strength=3, quantiles=5)
                        ev = e.step(a)
                        if a["type"] == "execute" and ev.get("completion") is not None:
                            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                            history.setdefault(f, []).append(ev["completion"] - ev["service_start"] - .5)
                    m = e.metrics()
                    assert not m["invalid_actions"]
                    rows.append(dict(fold=fold, case=case["case_id"], seed=seed, method=method,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"])))
        print("finished fold", fold, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "trials.csv", index=False)
    piv = d.pivot_table(index="method", columns="case", values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    base = d[d.method == "pressure_gated"].set_index(["fold", "seed", "case"]).weighted_on_time_rate
    summary = {"by_case": (piv * 100).round(4).to_dict(), "by_fold": {}}
    for m in ("robust20", "lookahead"):
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
