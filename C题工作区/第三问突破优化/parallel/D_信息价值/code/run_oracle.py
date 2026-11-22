"""ANALYSIS-ONLY perfect-information bound (not a deployable policy).

The same v2_s20 decision rule is replayed with the job's true source duration
known (belief = point mass at alpha*d+beta).  This measures the absolute ceiling
of ANY duration-information channel under this decision rule, so that the value
of the prefix channel can be compared against a self-computed bound rather than
any unverified external number.  The bound is never used inside a deployable arm,
and every deployable arm still receives durations only through the engine.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import MAPPINGS, build_hidden, hidden_for, load_all  # noqa: E402
from runner import make_pools, run_episode  # noqa: E402
from support import jobs_for  # noqa: E402

OUT = HERE.parent / "results"


def main():
    cond, tasks, zones, train, seg = load_all()
    pools, med = make_pools(train)
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    rows = []
    t0 = time.time()
    for case in cond["cases"]:
        jobs = jobs_for(tasks, case["case_id"])
        for seed in cond["validation_seeds"]:
            sampled = hidden_for(jobs, seed, vpool)
            hidden = build_hidden(sampled, {})
            known = {k: v[0] for k, v in sampled.items()}
            for alpha, beta in MAPPINGS:
                for name, kw in (("pressure_gated", {}), ("v2_s20", {})):
                    m, ind, diff = run_episode(jobs, hidden, zones, case, alpha, beta,
                                               name, pools, med, None, **kw)
                    rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                                     method=name, ontime=float(m["weighted_on_time_rate"])))
                m, ind, diff = run_episode(jobs, hidden, zones, case, alpha, beta,
                                           "v2_s20", pools, med, None, known=known)
                assert not m["invalid_actions"] and diff < 1e-9
                rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                                 method="oracle_pi_analysis_only",
                                 ontime=float(m["weighted_on_time_rate"])))
        print(case["case_id"], round(time.time() - t0, 1), "s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "oracle_trials.csv", index=False)
    nom = d[d.alpha.eq(1) & d.beta.eq(.5)].groupby("method").ontime.mean() * 100
    by_case = d[d.alpha.eq(1) & d.beta.eq(.5)].pivot_table(index="method", columns="case",
                                                           values="ontime", aggfunc="mean") * 100
    overall = d.groupby("method").ontime.mean() * 100
    summary = {
        "nominal_mean_pct": {k: float(v) for k, v in nom.items()},
        "by_case_nominal_pct": {m: {c: float(by_case.loc[m, c]) for c in by_case.columns}
                                for m in by_case.index},
        "five_mapping_mean_pct": {k: float(v) for k, v in overall.items()},
        "oracle_gain_vs_v2_nominal_pp": float(nom["oracle_pi_analysis_only"] - nom["v2_s20"]),
        "oracle_gain_vs_baseline_nominal_pp": float(nom["oracle_pi_analysis_only"] - nom["pressure_gated"]),
        "note": "analysis-only ceiling for the fixed v2_s20 decision rule; not deployable; "
                "no deployable arm reads hidden durations",
    }
    (OUT / "oracle_bound.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
