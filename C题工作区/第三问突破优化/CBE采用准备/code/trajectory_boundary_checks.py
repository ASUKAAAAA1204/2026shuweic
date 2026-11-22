"""Check every decision in selected complete trajectories, not just time zero."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import runtime as rt
from engine import Engine


def main():
    out = Path(__file__).resolve().parents[1] / "results" / "trajectory_boundary"
    out.mkdir(parents=True, exist_ok=True)
    cond, tasks, zones, train, seg = rt.acc.ar.load_all()
    shared = rt.context(seg[(seg.split == "train") & (seg.eligible == 1)],
                        seg[(seg.split == "validation") & (seg.eligible == 1)])
    rows, timing_rows = [], []
    for case in cond["cases"]:
        jobs = rt.acc.ar.jobs_of(tasks, case["case_id"])
        for seed in cond["validation_seeds"][:3]:
            dur = rt.acc.ar.hidden_for(jobs, seed, shared["vpool"])
            for method in ("B", "C", "E"):
                select = rt.adapter(method)
                e = Engine(jobs, rt.acc.ar.wrap(dur), zones, case["horizon_s"], case["cost_budget"],
                           case["observation_limit"], 1., .5)
                history = {}
                ctx = {"med": shared["med"], "pools": shared["pools"], **shared["e"]}
                while not e.terminal:
                    o = e.observe()
                    before_o, before_h = copy.deepcopy(o), copy.deepcopy(history)
                    # Use an exact copy of the current RNG state for all shadows.
                    shadow_ctx = copy.deepcopy(ctx)
                    t0 = time.perf_counter()
                    actual = select(o, shared["pools"], history, ctx)
                    ms = (time.perf_counter() - t0) * 1000.
                    changed = copy.deepcopy(o)
                    changed["jobs"] = list(reversed(changed["jobs"]))
                    for j in changed["jobs"]:
                        j["features"] = [1e15, -1e15, {"poison": "unseen_future"}]
                        j["undisclosed_future_prefix"] = [99999.]
                    alternate = select(changed, shared["pools"], copy.deepcopy(history), shadow_ctx)
                    assert actual == alternate
                    assert o == before_o and history == before_h
                    event = e.step(actual)
                    if actual["type"] == "execute" and event.get("completion") is not None:
                        family = next(j["family_id"] for j in jobs if j["job_id"] == actual["job_id"])
                        history.setdefault(family, []).append(event["completion"] - event["service_start"] - .5)
                    rows.append({"case": case["case_id"], "seed": seed, "method": method,
                                 "decision": e.actions, "time": o["time"], "paired_action_equal": True,
                                 "public_input_unchanged": True, "history_size": sum(map(len, before_h.values())),
                                 "censored_feedback_added": event.get("completion") is None and history != before_h})
                    timing_rows.append({"method": method, "case": case["case_id"], "seed": seed,
                                        "decision": e.actions, "milliseconds": ms})
    d = pd.DataFrame(rows)
    assert not d.censored_feedback_added.any()
    d.to_csv(out / "checks.csv", index=False)
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(out / "decision_wall_times.csv", index=False)
    summary = {"decision_points": len(d), "trajectories": 27,
               "all_equal": bool(d.paired_action_equal.all()),
               "input_mutation_count": int((~d.public_input_unchanged).sum()),
               "censored_feedback_updates": int(d.censored_feedback_added.sum()),
               "timing": {m: {"n": len(g), "median_ms": float(g.milliseconds.median()),
                               "p95_ms": float(g.milliseconds.quantile(.95)),
                               "max_ms": float(g.milliseconds.max()),
                               "total_ms": float(g.milliseconds.sum())}
                          for m, g in timing.groupby("method")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
