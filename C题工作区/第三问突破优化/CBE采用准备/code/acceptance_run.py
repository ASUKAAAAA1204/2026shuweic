"""Frozen, full-event acceptance replay for the retained B/C/E candidates.

This runner imports the candidate implementations as-is and writes only under
the acceptance directory. It pairs every policy against the formal baseline on
the same case, seed, and alpha/beta mapping.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BREAK = ROOT / "第三问突破优化"
G_CODE = BREAK / "parallel" / "G_统合审计" / "code"
if str(G_CODE) not in sys.path:
    sys.path.insert(0, str(G_CODE))
import audit_runner as ar
from engine import Engine
ar.METHODS["E_env_s40_fb3"] = dict(adapter=ar._make_e("env_s40_fb3"), kind="E")

METHODS = ("baseline", "B_K64", "C_mc_t4_s256_dt10", "E_env_s40_fb3")
MAPPINGS = ((1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def independent_rebuild(jobs, events, zones, horizon, budget, observation_limit):
    """Recompute legality, travel, elapsed time, costs, chains and reward."""
    by_id = {j["job_id"]: j for j in jobs}
    attempted, completed = set(), {}
    clock = cost = 0.0
    zone = tuple(zones["Z0"])
    reviews = invalid = censored = 0
    overrun = idle = 0.0
    violations = []
    for index, event in enumerate(events):
        action = event["action"]
        typ = action.get("type")
        t0 = clock
        if abs(event["decision_time"] - t0) > 1e-9:
            violations.append((index, "decision_time_mismatch"))
        if typ == "stop":
            continue
        if typ == "wait":
            duration = action.get("duration")
            if not isinstance(duration, (int, float)) or duration <= 0 or t0 + duration > horizon + 1e-9:
                violations.append((index, "illegal_wait")); invalid += 1
                continue
            clock = min(horizon, t0 + duration)
            continue
        jid = action.get("job_id")
        j = by_id.get(jid)
        if j is None or jid in attempted or j["release_s"] > t0 + 1e-9:
            violations.append((index, "unavailable_or_repeated_job")); invalid += 1
            continue
        if typ == "review":
            reviews += 1
            if reviews > observation_limit or t0 + 1 > horizon + 1e-9 or cost + 1 > budget + 1e-9:
                violations.append((index, "review_limit")); invalid += 1
                continue
            cost += 1.
            clock = t0 + 1.
        elif typ == "execute":
            if any(pred not in completed for pred in j["predecessors"]):
                violations.append((index, "predecessor_not_completed")); invalid += 1
                continue
            reserve = action.get("reserve")
            if not isinstance(reserve, (int, float)) or reserve < .2:
                violations.append((index, "invalid_reserve")); invalid += 1
                continue
            dest = tuple(zones[j["zone_id"]])
            distance = math.dist(zone, dest)
            start = t0 + distance + .5
            fee = 2. + .1 * reserve + .2 * distance
            if start + reserve > horizon + 1e-9 or cost + fee > budget + 1e-9:
                violations.append((index, "infeasible_execution")); invalid += 1
                continue
            if abs(start - event["service_start"]) > 1e-8 or abs(distance - event["distance"]) > 1e-8:
                violations.append((index, "physics_mismatch"))
            if abs(fee - event["fee"]) > 1e-8:
                violations.append((index, "fee_mismatch"))
            realized = event["realized_duration"]
            finish = start + realized
            attempted.add(jid)
            if finish <= horizon + 1e-9:
                completed[jid] = finish
                overrun += max(realized - reserve, 0.)
                idle += max(reserve - realized, 0.)
            else:
                censored += 1
                overrun += max(min(realized, horizon - start) - reserve, 0.)
            clock = min(horizon, start + max(realized, reserve))
            cost += fee
            zone = dest
        else:
            violations.append((index, "unknown_action")); invalid += 1
            continue
        if abs(clock - event["after"]["time"]) > 1e-8:
            violations.append((index, "clock_mismatch"))

    total_weight = sum(j["priority_weight"] for j in jobs)
    ontime = sum(j["priority_weight"] for j in jobs
                 if j["job_id"] in completed and completed[j["job_id"]] <= j["due_s"] + 1e-9)
    done_weight = sum(j["priority_weight"] for j in jobs if j["job_id"] in completed)
    tardy = sum(j["priority_weight"] * max(completed.get(j["job_id"], horizon) - j["due_s"], 0.)
                for j in jobs)
    return {
        "weighted_on_time_rate": ontime / total_weight,
        "weighted_completion_rate": done_weight / total_weight,
        "weighted_tardiness_s": tardy,
        "cost": cost,
        "observations": reviews,
        "elapsed_s": clock,
        "observed_overrun_s": overrun,
        "reserved_idle_s": idle,
        "censored_units": censored,
        "invalid_actions": invalid,
        "violations": violations,
    }


def make_context(method, shared):
    ctx = {"med": shared["med"], "pools": shared["pools"]}
    if method.startswith("E_"):
        ctx.update(shared["e"])
    return ctx


def run_episode(case, jobs, zones, pools, shared, method, seed, alpha, beta):
    vpool = shared["vpool"]
    hidden = ar.wrap(ar.hidden_for(jobs, seed, vpool))
    engine = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                    case["observation_limit"], alpha, beta)
    history, timings = {}, []
    ctx = make_context(method, shared)
    if method == "B_K64":
        import joint_policy
        ctx["b_pol"] = joint_policy.JointScenarioPolicy(
            K=64, strength=20, feedback_strength=3, seed=20261122,
            base="survival", stratified=True)
    adapter = ar.METHODS[method]["adapter"]
    while not engine.terminal:
        observation = engine.observe()
        start = time.perf_counter()
        action = adapter(observation, pools, history, ctx)
        timings.append(time.perf_counter() - start)
        event = engine.step(action)
        if action["type"] == "execute" and event.get("completion") is not None:
            family = next(j["family_id"] for j in jobs if j["job_id"] == action["job_id"])
            history.setdefault(family, []).append(
                (event["completion"] - event["service_start"] - beta) / alpha)
    metrics = engine.metrics()
    rebuilt = independent_rebuild(jobs, engine.events, zones, engine.H, engine.B, engine.K)
    numeric_diff = max(abs(float(metrics[k]) - float(rebuilt[k]))
                       for k in rebuilt if k != "violations")
    timing_ms = np.asarray(timings, float) * 1000.
    row = {
        "case": case["case_id"], "seed": seed, "alpha": alpha, "beta": beta,
        "method": method, "steps": len(engine.events),
        **{k: metrics[k] for k in (
            "weighted_on_time_rate", "weighted_completion_rate", "weighted_tardiness_s",
            "cost", "observations", "elapsed_s", "observed_overrun_s", "reserved_idle_s",
            "censored_units", "invalid_actions", "valid_protocol")},
        "independent_numeric_diff": numeric_diff,
        "independent_violation_count": len(rebuilt["violations"]),
        "decision_ms_median": float(np.median(timing_ms)) if len(timing_ms) else 0.,
        "decision_ms_p95": float(np.quantile(timing_ms, .95)) if len(timing_ms) else 0.,
        "decision_ms_max": float(timing_ms.max()) if len(timing_ms) else 0.,
    }
    return row, engine.events, rebuilt["violations"]


def main():
    out = HERE.parent / "results" / "full_replay"
    out.mkdir(parents=True, exist_ok=True)
    cond, tasks, zones, train, seg = ar.load_all()
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    train_pool = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
    med = {f: float(np.median(v)) for f, v in train_pool.items()}
    import robust_chain_policy as rcp
    from configs import CONFIGS
    cfg_name = "env_s40_fb3"
    fit = seg[(seg.split == "train") & (seg.eligible == 1)]
    records = [(env, str(env).split("_x_")[0], fam, g.duration_s.to_numpy(float))
               for (env, fam), g in fit.groupby(["environment_id", "family_id"])]
    prior = rcp.build_priors(records, strength=40., balance="env", winsor_q=None)
    shared = {"vpool": vpool, "pools": train_pool, "med": med,
              "e": {"e_prior": {cfg_name: prior}, "e_cfgs": {cfg_name: CONFIGS[cfg_name]}}}

    seed_list = cond["validation_seeds"][:30]
    cases = {c["case_id"]: c for c in cond["cases"]}
    jobs_by_case = {cid: ar.jobs_of(tasks, cid) for cid in cases}
    registry = {}
    for path in (BREAK / "parallel" / "B_联合场景前瞻" / "code" / "joint_policy.py",
                 BREAK / "parallel" / "C_尾部价值DP" / "code" / "dp_tail_policy.py",
                 BREAK / "parallel" / "E_稳健分布与选参" / "code" / "robust_chain_policy.py",
                 ROOT / "第三问改进正式方案" / "code" / "engine.py",
                 ROOT / "第三问改进正式方案" / "code" / "support.py"):
        registry[str(path.relative_to(ROOT))] = sha256(path)
    registry_path = out / "source_hashes.json"
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf8")

    rows, event_lines, violation_lines = [], [], []
    total = len(cases) * len(seed_list) * len(MAPPINGS) * len(METHODS)
    idx = 0
    for case_id, case in cases.items():
        jobs = jobs_by_case[case_id]
        for seed in seed_list:
            for alpha, beta in MAPPINGS:
                for method in METHODS:
                    idx += 1
                    row, events, violations = run_episode(
                        case, jobs, zones, train_pool, shared, method, seed, alpha, beta)
                    rows.append(row)
                    event_lines.append(json.dumps({"case": case_id, "seed": seed, "alpha": alpha,
                                                   "beta": beta, "method": method, "events": events},
                                                  separators=(",", ":"), ensure_ascii=False))
                    if violations:
                        violation_lines.append(json.dumps({"case": case_id, "seed": seed,
                                                           "alpha": alpha, "beta": beta,
                                                           "method": method, "violations": violations},
                                                          ensure_ascii=False))
                    if idx % 20 == 0:
                        print(f"{idx}/{total}", flush=True)
                    if idx % 100 == 0:
                        pd.DataFrame(rows).to_csv(out / "metrics.partial.csv", index=False)
                        with (out / "events.partial.jsonl").open("w", encoding="utf8") as f:
                            f.write("\n".join(event_lines) + "\n")

    metrics = pd.DataFrame(rows).sort_values(["method", "case", "seed", "alpha", "beta"])
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "events.jsonl").write_text("\n".join(event_lines) + "\n", encoding="utf8")
    (out / "independent_violations.jsonl").write_text("\n".join(violation_lines), encoding="utf8")
    summary = metrics.groupby("method").agg({
        "weighted_on_time_rate": "mean", "weighted_completion_rate": "mean",
        "weighted_tardiness_s": "mean", "cost": "mean", "observations": "mean",
        "reserved_idle_s": "mean", "observed_overrun_s": "mean", "censored_units": "mean",
        "invalid_actions": "sum", "independent_numeric_diff": "max",
        "independent_violation_count": "sum", "decision_ms_median": "median",
        "decision_ms_p95": "median", "decision_ms_max": "max"})
    summary.to_csv(out / "summary.csv")
    print(summary.to_string())
    print(f"episodes={len(metrics)} expected={total}; rows_with_independent_violations={len(violation_lines)}")


if __name__ == "__main__":
    main()
