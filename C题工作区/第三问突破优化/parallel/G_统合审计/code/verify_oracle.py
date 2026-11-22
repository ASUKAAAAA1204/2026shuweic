"""Independent re-computation of role F's exact perfect-information DP oracle.

Also checks the oracle dominates every audited online policy per (case, seed) at
the nominal mapping.  This replaces the disputed 93.77% figure with a verified
number produced here.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import audit_runner as ar

ROLE = HERE.parent


def main():
    import oracle
    cond, tasks, zones, train, seg = ar.load_all()
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    rows = []
    t0 = time.time()
    for case in cond["cases"]:
        jobs = ar.jobs_of(tasks, case["case_id"])
        for seed in cond["validation_seeds"]:
            dur = ar.hidden_for(jobs, seed, vpool)
            p, xy, z0, D, D0 = oracle.build_arrays(jobs, dur, 1., .5, zones)
            dp = oracle.solve_oracle_dp(jobs, p, zones, case["horizon_s"], case["cost_budget"], D, D0)
            ub, _ = oracle.relaxation_ub(jobs, p, zones, case["horizon_s"], case["cost_budget"], D0=D0)
            rows.append(dict(case=case["case_id"], seed=seed, oracle=float(dp["value"]),
                             relax_ub=float(ub), status=int(dp["status"])))
        print("done", case["case_id"], round(time.time() - t0, 1), "s", flush=True)
    d = pd.DataFrame(rows)
    out = ROLE / "results" / "oracle_dp"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "trials.csv", index=False)
    summary = {"oracle_mean3": float(d.oracle.mean()),
               "oracle_by_case": {k: float(v.oracle.mean()) for k, v in d.groupby("case")},
               "relax_ub_mean3": float(d.relax_ub.mean())}
    # dominance check against every audited policy set
    viol = {}
    for tag, method in [("E_selected", "E_env_s40_fb3"), ("A_full", "A_v2"),
                        ("C_mc_full", "C_mc_t4_s64_dt10"), ("B_full3", "B_K64"),
                        ("B_full3", "B_K128"), ("_validate", "audit_ref")]:
        p = pd.read_csv(ROLE / "results" / tag / "trials.csv")
        p = p[(p.alpha == 1) & (p.beta == .5) & (p.method == method)]
        j = p.merge(d, on=["case", "seed"], how="inner")
        viol[method] = dict(violations=int((j.weighted_on_time_rate > j.oracle + 1e-9).sum()),
                            n=len(j), max_excess_pp=float(((j.weighted_on_time_rate - j.oracle) * 100).max()))
    summary["dominance_vs_policies"] = viol
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
