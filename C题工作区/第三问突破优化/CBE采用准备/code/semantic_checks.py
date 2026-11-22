"""Adversarial checks for model semantics, information boundaries and resets."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import runtime as rt
from engine import Engine


def scalar_plan(plan, jobs, zones, samples, H, B):
    values = []
    for column in range(len(next(iter(samples.values())))):
        engine = Engine(jobs, rt.acc.ar.wrap({k: float(v[column]) for k, v in samples.items()}),
                        zones, H, B, 0, alpha=1., beta=0.)
        for jid in plan:
            if engine.terminal:
                break
            j = next(x for x in jobs if x["job_id"] == jid)
            obs = engine.observe()
            done = {x["job_id"] for x in obs["jobs"] if x["completed"]}
            if not set(j["predecessors"]) <= done:
                continue
            distance = math.dist(zones[engine.zone], zones[j["zone_id"]])
            if max(engine.time, j["release_s"]) + distance + .5 + .2 > H + 1e-12:
                continue
            if engine.cost + 2.02 + .2 * distance > B + 1e-12:
                continue
            if j["release_s"] > engine.time:
                if j["release_s"] > H:
                    continue
                engine.step({"type": "wait", "duration": j["release_s"] - engine.time})
            if engine.terminal:
                break
            distance = math.dist(zones[engine.zone], zones[j["zone_id"]])
            if engine.time + distance + .5 + .2 > H + 1e-12:
                continue
            if engine.cost + 2.02 + .2 * distance > B + 1e-12:
                continue
            engine.step({"type": "execute", "job_id": jid, "reserve": .2})
        weight = sum(j["priority_weight"] for j in jobs)
        values.append(engine.metrics()["weighted_on_time_rate"] * weight)
    return float(np.mean(values))


def mc_call(module, plan, jobs, zones, samples, H, B):
    keys = list(zones)
    zindex = {key: i for i, key in enumerate(keys)}
    zk = {j["job_id"]: zindex[j["zone_id"]] for j in jobs}
    D = np.asarray([[math.dist(zones[x], zones[y]) for y in keys] for x in keys])
    jj = {j["job_id"]: {"weight": j["priority_weight"], "release_s": j["release_s"],
                          "due_s": j["due_s"]} for j in jobs}
    preds = {j["job_id"]: set(j["predecessors"]) for j in jobs}
    K = len(next(iter(samples.values())))
    return module.mc_eval(plan, 0., zones["Z0"], 0., jj, {}, preds, H, B, samples,
                          K, zk, D, zindex["Z0"])


def main():
    out = Path(__file__).resolve().parents[1] / "results" / "semantic_checks"
    out.mkdir(parents=True, exist_ok=True)
    zones = {"Z0": (0., 0.), "Z1": (2., 0.)}
    counter_jobs = [dict(job_id="J0", family_id="A", release_s=0., due_s=10.,
                         priority_weight=1., zone_id="Z0", predecessors=[]),
                    dict(job_id="J1", family_id="A", release_s=0., due_s=10.,
                         priority_weight=5., zone_id="Z0", predecessors=[])]
    samples = {"J0": np.array([20.]), "J1": np.array([1.])}
    counter = {"original": mc_call(rt.c_original, ["J0", "J1"], counter_jobs, zones, samples, 10., 20.),
               "repaired": mc_call(rt.c_policy, ["J0", "J1"], counter_jobs, zones, samples, 10., 20.),
               "official_scalar": scalar_plan(["J0", "J1"], counter_jobs, zones, samples, 10., 20.)}
    assert counter["original"] == 5. and counter["repaired"] == counter["official_scalar"] == 0.
    rng = np.random.default_rng(20261123)
    errors, original_errors = [], []
    for trial in range(240):
        n = int(rng.integers(2, 7))
        jobs = []
        for i in range(n):
            jobs.append(dict(job_id=f"J{i}", family_id="A", release_s=float(rng.integers(0, 18)),
                             due_s=float(rng.integers(6, 24)), priority_weight=float(rng.integers(1, 6)),
                             zone_id="Z0" if rng.random() < .5 else "Z1",
                             predecessors=[f"J{i-1}"] if i and rng.random() < .4 else []))
        samples = {f"J{i}": rng.uniform(.02, 30., size=12) for i in range(n)}
        H, B = float(rng.integers(12, 30)), float(rng.integers(4, 18))
        plan = [j["job_id"] for j in jobs]
        actual = scalar_plan(plan, jobs, zones, samples, H, B)
        errors.append(abs(mc_call(rt.c_policy, plan, jobs, zones, samples, H, B) - actual))
        original_errors.append(abs(mc_call(rt.c_original, plan, jobs, zones, samples, H, B) - actual))
    assert max(errors) < 1e-9

    cond, tasks, zones, train, seg = rt.acc.ar.load_all()
    shared = rt.context(seg[(seg.split == "train") & (seg.eligible == 1)],
                        seg[(seg.split == "validation") & (seg.eligible == 1)])
    rows = []
    for case in cond["cases"]:
        jobs = rt.acc.ar.jobs_of(tasks, case["case_id"])
        e = Engine(jobs, rt.acc.ar.wrap({j["job_id"]: 1. for j in jobs}), zones,
                   case["horizon_s"], case["cost_budget"], case["observation_limit"], 1., .5)
        o = e.observe()
        altered = copy.deepcopy(o)
        for j in altered["jobs"]:
            j["features"] = [999999., -123456., {"fake_hidden_duration": 1e12}]
            j["undisclosed_future_prefix"] = [1e20]
        hidden_changed = Engine(jobs, rt.acc.ar.wrap({j["job_id"]: 1e8 for j in jobs}), zones,
                                e.H, e.B, e.K, 1., .5).observe()
        for method in ("B", "C", "E"):
            select = rt.adapter(method)
            ctx = lambda: {"med": shared["med"], "pools": shared["pools"], **shared["e"]}
            clean = select(o, shared["pools"], {}, ctx())
            feature = select(altered, shared["pools"], {}, ctx())
            hidden = select(hidden_changed, shared["pools"], {}, ctx())
            reversed_obs = dict(o, jobs=list(reversed(o["jobs"])))
            reordered = select(reversed_obs, shared["pools"], {}, ctx())
            repeated = select(o, shared["pools"], {}, ctx())
            assert clean == feature == hidden == reordered == repeated
            rows.append({"case": case["case_id"], "method": method, "kind": "initial_boundary",
                         "action": json.dumps(clean), "passed": True})
            # A new episode after an unrelated episode must reproduce all actions.
            first, first_events, _ = rt.run(case, jobs, zones, shared, method + "_probe", method,
                                            cond["validation_seeds"][0])
            rt.run(case, jobs, zones, shared, method + "_probe", method, cond["validation_seeds"][1])
            again, again_events, _ = rt.run(case, jobs, zones, shared, method + "_probe", method,
                                           cond["validation_seeds"][0])
            assert first_events == again_events
            rows.append({"case": case["case_id"], "method": method, "kind": "full_episode_reset",
                         "action": "all_events_equal", "passed": True})
    pd.DataFrame(rows).to_csv(out / "boundary_checks.csv", index=False)
    summary = {"c_censor_counterexample": counter, "random_fixed_plan_cases": 240,
               "scenarios_per_case": 12, "repaired_max_error": max(errors),
               "original_mismatched_cases": sum(x > 1e-9 for x in original_errors),
               "original_max_error": max(original_errors), "boundary_rows_passed": len(rows),
               "scope": "Initial observations and whole-episode fresh-state replay; no new environment generalization."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
