"""Run the official legal lookahead (strength=40, quantiles=5) on the full protocol.

Purpose: this is the *constructive legal lower bound* for the feasibility
question.  It is the exact method of 第三问突破优化/code/confirm_lookahead.py,
so reproducing its published nominal avg3 with our own harness is also a
protocol-identity check.  Output: results/legal_extra.csv.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parents[3] / "第三问改进正式方案" / "code"
BREAK = HERE.parents[3] / "第三问突破优化" / "code"
sys.path.insert(0, str(FORMAL))
sys.path.insert(0, str(BREAK))

from support import load, jobs_for
from engine import Engine, independent_metrics
import lookahead_policy as LP

ROOT = HERE.parents[3]
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
MAPPINGS = [(1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
S40 = dict(strength=40, feedback_strength=3, quantiles=5)


def main():
    cond, tasks, zones, train = load()
    pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]

    def hidden_for(jobs, seed):
        rng = random.Random(seed)
        out = {}
        for j in jobs:
            c = vpool[vpool.family_id == j["family_id"]]
            out[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
        return out

    rows, jrows, max_diff = [], [], 0.0
    for case in cond["cases"]:
        jobs = jobs_for(tasks, case["case_id"])
        for seed in cond["validation_seeds"]:
            t0 = time.time()
            dur = hidden_for(jobs, seed)
            hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)})
                      for k, v in dur.items()}
            for alpha, beta in MAPPINGS:
                e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                           case["observation_limit"], alpha, beta)
                history = {}
                while not e.terminal:
                    o = e.observe()
                    a = LP.select(o, pools, history, **S40)
                    ev = e.step(a)
                    if a["type"] == "execute" and ev.get("completion") is not None:
                        f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                        history.setdefault(f, []).append(
                            (ev["completion"] - ev["service_start"] - beta) / alpha)
                m = e.metrics()
                ind = independent_metrics(jobs, e.events, e.H)
                diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                max_diff = max(max_diff, diff)
                assert m["invalid_actions"] == 0 and diff < 1e-9
                rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                                 method="lookahead_s40",
                                 weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                                 weighted_completion_rate=float(m["weighted_completion_rate"]),
                                 cost=float(m["cost"]), observations=float(m["observations"])))
                for j in jobs:
                    st = e.status[j["job_id"]]
                    comp = st["completion"] if st["completed"] else None
                    jrows.append(dict(
                        case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                        method="lookahead_s40", job_id=j["job_id"], family=j["family_id"],
                        zone=j["zone_id"], weight=j["priority_weight"],
                        release=j["release_s"], due=j["due_s"],
                        attempted=bool(st["attempted"]), completed=bool(st["completed"]),
                        on_time=bool(st["completed"] and comp is not None
                                     and comp <= j["due_s"] + 1e-12),
                        completion=comp,
                        slack=(j["due_s"] - comp) if comp is not None else None,
                        latitude=j["due_s"] - j["release_s"]))
            print("case", case["case_id"], seed, "elapsed", round(time.time() - t0, 1),
                  flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "legal_extra.csv", index=False)
    pd.DataFrame(jrows).to_csv(OUT / "jobs_legal_extra.csv", index=False)
    n = d[(d.alpha == 1) & (d.beta == .5)]
    piv = n.pivot_table(index="method", columns="case", values="weighted_on_time_rate")
    piv["avg3"] = piv.mean(axis=1)
    print("nominal by case (pct):\n", (piv * 100).round(4))
    print("mapping avg (pct):", round(float(d.weighted_on_time_rate.mean() * 100), 4))
    (OUT / "legal_extra_diag.json").write_text(json.dumps(dict(
        max_independent_diff=max_diff, nominal_by_case={
            c: float(n[n.case == c].weighted_on_time_rate.mean() * 100)
            for c in ("S1", "S2", "S3")},
        nominal_avg3=float(n.weighted_on_time_rate.mean() * 100),
        mapping_avg5=float(d.weighted_on_time_rate.mean() * 100)), indent=2))


if __name__ == "__main__":
    main()
