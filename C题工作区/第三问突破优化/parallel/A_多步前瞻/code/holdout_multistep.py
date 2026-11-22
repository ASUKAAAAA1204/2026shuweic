"""Environment-group holdout for role A (multi-step lookahead)."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FORMAL = ROOT / "第三问改进正式方案" / "code"
BREAK = HERE.parents[2]
sys.path.insert(0, str(FORMAL))
sys.path.insert(0, str(HERE))

from support import load, jobs_for
from engine import Engine
from deadline_policy import choose as base_choose, pressure as base_pressure
from baseline import action as base_action
import multi_step_policy as ms

METHODS = {
    "baseline": None,
    "v2": dict(depth=1, strength=20, nq_root=5, nq=2, beam=4),
    "d2b4q5": dict(depth=2, strength=20, nq_root=5, nq=5, beam=4),
}


def main():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    groups = json.loads((BREAK / "results" / "grouped_training" / "registration.json").read_text())["groups"]
    rows = []
    for fold, held in enumerate(groups):
        fit = seg[(seg.split == "train") & (seg.eligible == 1) & (~seg.environment_id.isin(held))]
        evp = seg[(seg.split == "train") & (seg.eligible == 1) & (seg.environment_id.isin(held))]
        pools = {f: g.duration_s.to_numpy(float) for f, g in fit.groupby("family_id")}
        med = {f: float(np.median(v)) for f, v in pools.items()}
        for case in cond["cases"]:
            jobs = jobs_for(tasks, case["case_id"])
            for seed in range(2026120100, 2026120112):
                rng = random.Random(seed)
                dur = {}
                for j in jobs:
                    c = evp[evp.family_id == j["family_id"]]
                    dur[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
                hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)}) for k, v in dur.items()}
                for name, cfg in METHODS.items():
                    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                               case["observation_limit"], 1.0, .5)
                    history = {}
                    while not e.terminal:
                        o = e.observe()
                        if cfg is None:
                            a = base_choose(o, pools, history) if base_pressure(o, pools, history) else base_action(o, med, "density", "minimum")
                        else:
                            a = ms.select(o, pools, history, feedback_strength=3, **cfg)
                        ev = e.step(a)
                        if a["type"] == "execute" and ev.get("completion") is not None:
                            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                            history.setdefault(f, []).append(ev["completion"] - ev["service_start"] - .5)
                    m = e.metrics()
                    assert not m["invalid_actions"]
                    rows.append(dict(fold=fold, case=case["case_id"], seed=seed, method=name,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"])))
        print("fold", fold, "done", flush=True)
    d = pd.DataFrame(rows)
    out = HERE / "results" / "holdout"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "trials.csv", index=False)
    piv = d.pivot_table(index="method", columns="case", values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    base = d[d.method == "baseline"].set_index(["fold", "seed", "case"]).weighted_on_time_rate
    summary = {"by_case": (piv * 100).round(4).to_dict(), "paired": {}}
    for m in METHODS:
        if m == "baseline":
            continue
        x = d[d.method == m].set_index(["fold", "seed", "case"]).weighted_on_time_rate
        delta = (x - base) * 100
        summary["paired"][m] = {
            "mean_pp": float(delta.mean()),
            "fold0_pp": float(delta[delta.index.get_level_values(0) == 0].mean()),
            "fold1_pp": float(delta[delta.index.get_level_values(0) == 1].mean()),
            "by_case_pp": {c: float(delta[delta.index.get_level_values(2) == c].mean()) for c in ("S1", "S2", "S3")},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print((piv * 100).round(2).to_string())
    print(json.dumps(summary["paired"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
