"""Self-computed upper bounds (evaluation side only; not policies).

Uses role F's oracle.py but recomputes every number from the frozen protocol.
relaxation_ub is valid for every instance; its seed-average bounds the mean of
any legal online policy from above.  solve_oracle is the full-information
optimum (needs hidden durations, so it is *not* a legal policy).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import audit_runner as ar

ROLE = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--oracle-seeds", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args()
    import oracle

    cond, tasks, zones, train, seg = ar.load_all()
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    seeds = cond["validation_seeds"][:args.seeds]
    rows = []
    for case in cond["cases"]:
        jobs = ar.jobs_of(tasks, case["case_id"])
        for seed in seeds:
            dur = ar.hidden_for(jobs, seed, vpool)
            p, xy, z0, D, D0 = oracle.build_arrays(jobs, dur, 1.0, .5, zones)
            ub, _ = oracle.relaxation_ub(jobs, p, zones, case["horizon_s"], case["cost_budget"], D0=D0)
            rows.append(dict(case=case["case_id"], seed=seed, kind="relaxation_ub", value=float(ub)))
            if seed < seeds[0] + args.oracle_seeds:
                try:
                    sol = oracle.solve_oracle(jobs, p, zones, case["horizon_s"], case["cost_budget"],
                                              D=D, D0=D0, time_limit=args.time_limit)
                    rows.append(dict(case=case["case_id"], seed=seed, kind="oracle_optimum",
                                     value=float(sol["value"]), status=int(sol["status"])))
                except Exception as exc:  # noqa: BLE001
                    rows.append(dict(case=case["case_id"], seed=seed, kind="oracle_error",
                                     value=float("nan"), note=str(exc)[:200]))
        print("finished", case["case_id"], flush=True)
    d = pd.DataFrame(rows)
    out = ROLE / "results" / "upper_bound"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "trials.csv", index=False)
    summ = {k: {"mean3": float(g.value.mean()), "by_case": {c: float(v.value.mean()) for c, v in g.groupby("case")}}
            for k, g in d.groupby("kind")}
    (out / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summ, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
