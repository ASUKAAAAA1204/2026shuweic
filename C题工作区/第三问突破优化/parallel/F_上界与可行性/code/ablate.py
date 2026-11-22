"""Ablations for role F (nominal mapping, all 30 seeds x 3 cases).

1. Value of late-unlock: exact oracle with ancestors forbidden to run late
   (``allow_late=False``) vs the full oracle.  Isolates how much of the
   perfect-information optimum needs "run an ancestor late to unlock a child".
2. Relaxation tightness: cumulative necessary-condition knapsacks
   (eligibility only -> +deadline capacity -> +release windows -> +budget).
   Shows which component carries the slack between the relaxation and the oracle.
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

from support import load, jobs_for
import oracle as O

ROOT = HERE.parents[3]
OUT = HERE.parent / "results"
CASES = ["S1", "S2", "S3"]
ALPHA, BETA = 1.0, 0.5


def main():
    cond, tasks, zones, _ = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    rows = []
    for case in cond["cases"]:
        jobs = jobs_for(tasks, case["case_id"])
        H, B = case["horizon_s"], case["cost_budget"]
        for seed in cond["validation_seeds"]:
            rng = random.Random(seed)
            dur = {}
            for j in jobs:
                c = vpool[vpool.family_id == j["family_id"]]
                dur[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
            p, xy, z, D, D0 = O.build_arrays(jobs, dur, ALPHA, BETA, zones)
            full = O.solve_oracle_dp(jobs, p, zones, H, B, D, D0)["value"]
            strict = O.solve_oracle_dp(jobs, p, zones, H, B, D, D0,
                                       allow_late=False)["value"]
            ub_e, _ = O.relaxation_ub(jobs, p, zones, H, B, D0,
                                      use_deadline=False, use_release=False)
            ub_d, _ = O.relaxation_ub(jobs, p, zones, H, B, D0,
                                      use_deadline=True, use_release=False)
            ub_r, _ = O.relaxation_ub(jobs, p, zones, H, B, D0,
                                      use_deadline=True, use_release=True)
            rows.append(dict(case=case["case_id"], seed=seed, oracle_full=full,
                             oracle_no_late=strict, ub_elig=ub_e, ub_deadline=ub_d,
                             ub_release=ub_r))
        print("finished", case["case_id"], flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "ablation_trials.csv", index=False)
    agg = {}
    for c in CASES:
        s = d[d.case == c]
        agg[c] = {k: float(s[k].mean() * 100.0) for k in
                  ("oracle_full", "oracle_no_late", "ub_elig", "ub_deadline", "ub_release")}
        agg[c]["late_unlock_gain_pp"] = agg[c]["oracle_full"] - agg[c]["oracle_no_late"]
    overall = {k: float(d[k].mean() * 100.0) for k in
               ("oracle_full", "oracle_no_late", "ub_elig", "ub_deadline", "ub_release")}
    overall["late_unlock_gain_pp"] = overall["oracle_full"] - overall["oracle_no_late"]
    res = dict(by_case_pct=agg, avg3_pct=overall,
               notes="per-case mean over 30 seeds, then unweighted mean over 3 cases")
    (OUT / "ablation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                       encoding="utf8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
