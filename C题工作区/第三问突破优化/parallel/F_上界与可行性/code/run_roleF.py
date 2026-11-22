"""Role F runner: paired perfect-information oracle / relaxation UB / legal references.

Protocol (identical to confirm_lookahead.py):
  3 cases x 30 validation seeds x 5 mappings, identical hidden_for sampling,
  official engine.Engine, engine.independent_metrics audit.

Methods written to trials.csv:
  pressure_gated   legal online baseline (formal deadline_policy/baseline)
  v2_s20           legal online reference (vendored role-A multi-step, depth=1, strength=20)
  oracle           perfect-information exact optimum (subset DP), replayed through the engine
  ub_relax_knapsack provable relaxation upper bound (no simulation)

Usage: python run_roleF.py --case S1 [--limit N]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parents[3] / "第三问改进正式方案" / "code"
sys.path.insert(0, str(FORMAL))
sys.path.insert(0, str(HERE))

from support import jobs_for, load
from engine import Engine, independent_metrics
from deadline_policy import choose, pressure
from baseline import action
import oracle as O
import reference_policy

ROOT = HERE.parents[3]
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
MAPPINGS = [(1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
V2 = dict(depth=1, strength=20, feedback_strength=3, nq_root=5, nq=2, beam=3)


def policy_run(jobs, hidden, zones, case, alpha, beta, pools, med, kind):
    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
               case["observation_limit"], alpha, beta)
    history = {}
    while not e.terminal:
        o = e.observe()
        if kind == "pressure_gated":
            a = choose(o, pools, history) if pressure(o, pools, history) else action(o, med, "density", "minimum")
        elif kind == "v2_s20":
            a = reference_policy.select(o, pools, history, **V2)
        else:
            raise ValueError(kind)
        ev = e.step(a)
        if a["type"] == "execute" and ev.get("completion") is not None:
            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
            history.setdefault(f, []).append((ev["completion"] - ev["service_start"] - beta) / alpha)
    return e


def job_records(e, jobs, case_id, seed, alpha, beta, method):
    recs = []
    for j in jobs:
        st = e.status[j["job_id"]]
        comp = st["completion"] if st["completed"] else None
        on = bool(st["completed"] and comp is not None and comp <= j["due_s"] + 1e-12)
        recs.append(dict(case=case_id, seed=seed, alpha=alpha, beta=beta, method=method,
                         job_id=j["job_id"], family=j["family_id"], zone=j["zone_id"],
                         weight=j["priority_weight"], release=j["release_s"], due=j["due_s"],
                         attempted=bool(st["attempted"]), completed=bool(st["completed"]),
                         on_time=on, completion=comp,
                         slack=(j["due_s"] - comp) if comp is not None else None,
                         latitude=j["due_s"] - j["release_s"]))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["S1", "S2", "S3"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cond, tasks, zones, train = load()
    pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
    med = {f: float(np.median(v)) for f, v in pools.items()}
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]

    def hidden_for(jobs, seed):
        rng = random.Random(seed)
        out = {}
        for j in jobs:
            c = vpool[vpool.family_id == j["family_id"]]
            out[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
        return out

    case = next(c for c in cond["cases"] if c["case_id"] == args.case)
    jobs = jobs_for(tasks, case["case_id"])
    seeds = cond["validation_seeds"][: args.limit] if args.limit else cond["validation_seeds"]

    rows, jrows = [], []
    max_diff = 0.0
    for si, seed in enumerate(seeds):
        t_case = time.time()
        dur = hidden_for(jobs, seed)
        hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)}) for k, v in dur.items()}
        for alpha, beta in MAPPINGS:
            p, xy, z0, D, D0 = O.build_arrays(jobs, dur, alpha, beta, zones)

            # legal references
            for kind in ("pressure_gated", "v2_s20"):
                e = policy_run(jobs, hidden, zones, case, alpha, beta, pools, med, kind)
                m = e.metrics()
                ind = independent_metrics(jobs, e.events, e.H)
                diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                max_diff = max(max_diff, diff)
                assert m["invalid_actions"] == 0 and diff < 1e-9, (case["case_id"], seed, kind, diff)
                assert m["valid_protocol"], (case["case_id"], seed, kind)
                rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                                 method=kind,
                                 weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                                 weighted_completion_rate=float(m["weighted_completion_rate"]),
                                 cost=float(m["cost"]), observations=float(m["observations"])))
                jrows.extend(job_records(e, jobs, case["case_id"], seed, alpha, beta, kind))

            # exact perfect-information oracle (analysis only)
            dp = O.solve_oracle_dp(jobs, p, zones, case["horizon_s"], case["cost_budget"], D, D0)
            m, ind, diff, e = O.replay_plan(jobs, hidden, zones, case["horizon_s"],
                                            case["cost_budget"], case["observation_limit"],
                                            alpha, beta, dp["order"], return_engine=True)
            diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
            max_diff = max(max_diff, diff)
            assert m["invalid_actions"] == 0 and diff < 1e-9, (case["case_id"], seed, "oracle", diff)
            assert abs(m["weighted_on_time_rate"] - dp["value"]) < 1e-9, \
                (case["case_id"], seed, alpha, beta, m["weighted_on_time_rate"], dp["value"])
            assert m["cost"] <= case["cost_budget"] + 1e-9
            rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                             method="oracle",
                             weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                             weighted_completion_rate=float(m["weighted_completion_rate"]),
                             cost=float(m["cost"]), observations=float(m["observations"])))
            jrows.extend(job_records(e, jobs, case["case_id"], seed, alpha, beta, "oracle"))

            # relaxation upper bound
            ub, _ = O.relaxation_ub(jobs, p, zones, case["horizon_s"], case["cost_budget"], D0)
            assert ub >= dp["value"] - 1e-9, (case["case_id"], seed, ub, dp["value"])
            rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                             method="ub_relax_knapsack",
                             weighted_on_time_rate=float(ub),
                             weighted_completion_rate=np.nan, cost=np.nan, observations=np.nan))
        print("case", case["case_id"], "seed", si + 1, "/", len(seeds),
              "elapsed", round(time.time() - t_case, 1), flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / ("trials_%s.csv" % case["case_id"]), index=False)
    jd = pd.DataFrame(jrows)
    jd.to_csv(OUT / ("jobs_%s.csv" % case["case_id"]), index=False)
    print("WROTE", OUT / ("trials_%s.csv" % case["case_id"]), "max_independent_diff", max_diff)
    (OUT / ("diag_%s.json" % case["case_id"])).write_text(json.dumps(dict(
        case=case["case_id"], instances=len(rows), max_independent_diff=max_diff,
        n_seeds=len(seeds), mappings=[list(m) for m in MAPPINGS]), indent=2))


if __name__ == "__main__":
    main()
