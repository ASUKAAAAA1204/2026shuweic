"""Paired evaluation of joint-scenario lookahead on the official validation split.

Protocol (copied verbatim from ``confirm_lookahead.py``):
  3 cases x 30 validation seeds x 5 duration mappings, every (case, seed,
  mapping) cell pairs the baseline with each candidate.  Hidden durations are
  sampled with ``random.Random(seed)`` over ``jobs_for`` order from the
  validation&eligible pool.  Only the train split fits the duration belief.
  Test split is never read.

Usage:
  python run_joint.py --seeds 30 --mappings all --methods baseline,v2,joint_k32
  python run_joint.py --seeds 2 --mappings nominal --methods baseline,v2,joint_k8
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                  # C题工作区
FORMAL = ROOT / "第三问改进正式方案" / "code"
BREAK = HERE.parents[2]                 # 第三问突破优化
sys.path.insert(0, str(FORMAL))
sys.path.insert(0, str(HERE))

from support import load, jobs_for                       # noqa: E402
from engine import Engine, independent_metrics           # noqa: E402
from deadline_policy import choose, pressure              # noqa: E402
from baseline import action                              # noqa: E402
import v2_reference                                      # noqa: E402
from joint_policy import JointScenarioPolicy             # noqa: E402

MAPPINGS = [(1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.)]
# method name -> (rollout base policy, K, stratified sampling); baseline/v2 are not joint.
JOINT = {
    "joint_deadline_k32": ("deadline", 32, True),
    "joint_deadline_k64": ("deadline", 64, True),
    "joint_density_k8": ("density", 8, False), "joint_density_k16": ("density", 16, False),
    "joint_density_k32": ("density", 32, False), "joint_density_k64": ("density", 64, False),
    "joint_density_k64_strat": ("density", 64, True),
    "joint_survival_k8": ("survival", 8, True), "joint_survival_k16": ("survival", 16, True),
    "joint_survival_k32": ("survival", 32, True), "joint_survival_k64": ("survival", 64, True),
    "joint_survival_k64_iid": ("survival", 64, False),
    "joint_survival_k128": ("survival", 128, True),
}
DEFAULT_METHODS = ["baseline", "v2",
                   "joint_survival_k8", "joint_survival_k16",
                   "joint_survival_k32", "joint_survival_k64"]
STRENGTH = 20
FEEDBACK = 3
SCENARIO_SEED = 20261122

_DATA = {}


def init_worker():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
    med = {f: float(np.median(v)) for f, v in pools.items()}
    _DATA.update(cond=cond, tasks=tasks, zones=zones, train=train,
                 vpool=vpool, pools=pools, med=med)


def hidden_for(jobs, seed):
    """Verbatim copy of confirm_lookahead.hidden_for."""
    rng = random.Random(seed)
    vpool = _DATA["vpool"]
    out = {}
    for j in jobs:
        c = vpool[vpool.family_id == j["family_id"]]
        out[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
    return out


def run_cell(case_id, seed, mappings, methods):
    cond, tasks, zones = _DATA["cond"], _DATA["tasks"], _DATA["zones"]
    pools, med = _DATA["pools"], _DATA["med"]
    case = next(c for c in cond["cases"] if c["case_id"] == case_id)
    jobs = jobs_for(tasks, case_id)
    dur = hidden_for(jobs, seed)
    hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)})
              for k, v in dur.items()}
    rows = []
    for alpha, beta in mappings:
        for method in methods:
            policy = None
            if method in JOINT:
                base, K, strat = JOINT[method]
                policy = JointScenarioPolicy(K=K, strength=STRENGTH,
                                             feedback_strength=FEEDBACK,
                                             seed=SCENARIO_SEED, base=base,
                                             stratified=strat)
            e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                       case["observation_limit"], alpha, beta)
            history = {}
            began = time.perf_counter()
            mc_se = []
            decisions = 0
            while not e.terminal:
                o = e.observe()
                decisions += 1
                if method == "baseline":
                    a = choose(o, pools, history) if pressure(o, pools, history) \
                        else action(o, med, "density", "minimum")
                elif method == "v2":
                    a = v2_reference.select(o, pools, history, strength=STRENGTH,
                                            feedback_strength=FEEDBACK, depth=1,
                                            nq_root=5, nq=2, beam=3)
                else:
                    a = policy.select(o, pools, history)
                    mc_se.append(policy.last_se)
                ev = e.step(a)
                if a["type"] == "execute" and ev.get("completion") is not None:
                    f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                    history.setdefault(f, []).append(
                        (ev["completion"] - ev["service_start"] - beta) / alpha)
            m = e.metrics()
            ind = independent_metrics(jobs, e.events, e.H)
            diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
            assert not m["invalid_actions"] and diff < 1e-9, \
                (case_id, seed, method, alpha, beta, diff)
            rows.append(dict(case=case_id, seed=int(seed), alpha=float(alpha),
                             beta=float(beta), method=method,
                             weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                             weighted_completion_rate=float(m["weighted_completion_rate"]),
                             cost=float(m["cost"]), observations=float(m["observations"]),
                             seconds=float(time.perf_counter() - began),
                             independent_delta=float(diff), decisions=int(decisions),
                             mc_se=float(np.mean(mc_se)) if mc_se else 0.0))
    return rows


def bootstrap_ci(values, rng, draws=5000):
    values = np.asarray(values, float)
    i = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[i].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(d, methods, mappings, cfg):
    nominal = d[(np.isclose(d.alpha, 1.0)) & (np.isclose(d.beta, .5))]
    out = {"protocol": {
        "cases": sorted(d.case.unique().tolist()),
        "seeds": sorted(d.seed.unique().tolist()),
        "mappings": mappings,
        "methods": methods,
        "nominal_mapping": [1.0, 0.5],
        "scenario_seed": SCENARIO_SEED,
        "K_grid": sorted({k for _, k, _ in JOINT.values()}),
        "strength": STRENGTH,
        "feedback_strength": FEEDBACK,
        "joint_methods": {m: {"base": b, "K": k, "stratified": s}
                          for m, (b, k, s) in JOINT.items() if m in methods},
    }}
    # Nominal three-case means.
    piv = nominal.pivot_table(index="method", columns="case",
                              values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    out["nominal_avg3"] = {m: float(piv.loc[m, "avg3"]) for m in piv.index}
    out["nominal_by_case"] = {m: {c: float(piv.loc[m, c])
                                  for c in sorted(nominal.case.unique())}
                              for m in piv.index}
    # Five-mapping means.
    out["all_mapping_avg"] = {m: float(d[d.method == m].weighted_on_time_rate.mean())
                              for m in methods}
    out["all_mapping_by_mapping"] = {
        m: {f"{a},{b}": float(d[(d.method == m) & np.isclose(d.alpha, a)
                                & np.isclose(d.beta, b)].weighted_on_time_rate.mean())
            for a, b in mappings}
        for m in methods}
    # Paired deltas versus baseline (nominal) and versus v2.
    base = nominal[nominal.method == "baseline"].set_index(["case", "seed"]).weighted_on_time_rate
    rng = np.random.default_rng(20261122)
    out["paired_vs_baseline"] = {}
    out["paired_vs_v2"] = {}
    for m in methods:
        if m == "baseline":
            continue
        x = nominal[nominal.method == m].set_index(["case", "seed"]).weighted_on_time_rate
        dd = (x - base)
        by_seed = dd.groupby(level=1).mean()
        out["paired_vs_baseline"][m] = dict(
            mean_pp=float(dd.mean() * 100),
            by_case_pp={c: float(dd[dd.index.get_level_values(0) == c].mean() * 100)
                        for c in sorted(nominal.case.unique())},
            ci95_pp=bootstrap_ci(by_seed.to_numpy(), rng),
            positive_seeds=int((by_seed > 0).sum()), n_seeds=int(len(by_seed)),
            sd_seed_pp=float(by_seed.std(ddof=1) * 100),
        )
        # All five mappings, paired by seed (mean over cases and mappings).
        allb = d[d.method == "baseline"].set_index(["case", "seed", "alpha", "beta"]).weighted_on_time_rate
        allx = d[d.method == m].set_index(["case", "seed", "alpha", "beta"]).weighted_on_time_rate
        dall = (allx - allb)
        by_seed_all = dall.groupby(level=1).mean()
        out["paired_vs_baseline"][m]["all_mapping_mean_pp"] = float(dall.mean() * 100)
        out["paired_vs_baseline"][m]["all_mapping_ci95_pp"] = bootstrap_ci(by_seed_all.to_numpy(), rng)
        out["paired_vs_baseline"][m]["all_mapping_positive_seeds"] = int((by_seed_all > 0).sum())
        if "v2" in methods:
            v2 = nominal[nominal.method == "v2"].set_index(["case", "seed"]).weighted_on_time_rate
            dv = (x - v2)
            by_seed_v = dv.groupby(level=1).mean()
            out["paired_vs_v2"][m] = dict(
                mean_pp=float(dv.mean() * 100),
                by_case_pp={c: float(dv[dv.index.get_level_values(0) == c].mean() * 100)
                            for c in sorted(nominal.case.unique())},
                ci95_pp=bootstrap_ci(by_seed_v.to_numpy(), rng),
                positive_seeds=int((by_seed_v > 0).sum()), n_seeds=int(len(by_seed_v)),
            )
            allv2 = d[d.method == "v2"].set_index(["case", "seed", "alpha", "beta"]).weighted_on_time_rate
            dall_v2 = (allx - allv2)
            by_seed_all_v2 = dall_v2.groupby(level=1).mean()
            out["paired_vs_v2"][m]["all_mapping_mean_pp"] = float(dall_v2.mean() * 100)
            out["paired_vs_v2"][m]["all_mapping_ci95_pp"] = bootstrap_ci(
                by_seed_all_v2.to_numpy(), rng)
            out["paired_vs_v2"][m]["all_mapping_positive_seeds"] = int(
                (by_seed_all_v2 > 0).sum())
    # Cost / completion / observations / runtime on the nominal mapping.
    out["nominal_cost"] = {m: float(nominal[nominal.method == m].cost.mean()) for m in methods}
    out["nominal_completion"] = {m: float(nominal[nominal.method == m].weighted_completion_rate.mean())
                                 for m in methods}
    out["nominal_observations"] = {m: float(nominal[nominal.method == m].observations.mean())
                                   for m in methods}
    out["runtime_seconds_per_run"] = {m: float(d[d.method == m].seconds.mean()) for m in methods}
    out["runtime_total_seconds"] = float(d.seconds.sum())
    out["decision_mc_se"] = {m: float(d[d.method == m].mc_se.mean()) for m in methods if m in JOINT}
    out["max_independent_metric_diff"] = float(d.independent_delta.max())
    out["invalid_actions"] = 0
    out["n_trials"] = int(len(d))
    # Primary config: largest K in the required grid {8,16,32,64}, lowest MC SE
    # in that grid, fixed by rule rather than by the observed ranking.
    primary = "joint_survival_k64"
    baseline_nom = float(nominal[nominal.method == "baseline"].weighted_on_time_rate.mean())
    target = baseline_nom * 1.06
    out["primary"] = primary
    out["thresholds"] = {
        "baseline_nominal": baseline_nom,
        "relative_6pct_target": target,
        "nominal_result": out["nominal_avg3"].get(primary),
        "nominal_pass": bool(out["nominal_avg3"].get(primary, 0.0) >= target),
        "paired_ci_lower_positive": bool(
            out["paired_vs_baseline"].get(primary, {}).get("ci95_pp", [0, 0])[0] > 0),
        "all_five_mappings_ge_baseline": bool(all(
            out["all_mapping_by_mapping"][primary][f"{a},{b}"]
            >= out["all_mapping_by_mapping"]["baseline"][f"{a},{b}"]
            for a, b in mappings)),
        "independent_diff_lt_1e-9": bool(out["max_independent_metric_diff"] < 1e-9),
        "invalid_actions_zero": True,
    }
    out["selection"] = {
        "primary_K": 64,
        "primary_method": primary,
        "rule": "K=64 is the largest K in the required grid {8,16,32,64}; within that grid it has "
                "the smallest decision Monte Carlo SE.  K=128 is an out-of-grid diagnostic.",
        "all_K_reported": True,
        "selection_bias": "The K sweep was observed on the same validation split used for the headline. "
                          "K=64 is the max-K/max-precision rule, but K=16/K=32 also appear and are not "
                          "monotone, so the 90.96% point estimate is a selected estimate, not an unbiased "
                          "effect.  The independent environment holdout is reported separately.",
        "config": cfg,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--mappings", choices=["nominal", "all"], default="all")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cond, tasks, zones, train = load()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    mappings = MAPPINGS if args.mappings == "all" else [(1., .5)]
    seeds = cond["validation_seeds"][:args.seeds]
    cases = [c["case_id"] for c in cond["cases"]]
    cfg = dict(seeds=len(seeds), mappings=args.mappings, methods=methods)

    out = (HERE.parent / args.out / args.tag) if args.tag else (HERE.parent / args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks_list = [(c, s) for c in cases for s in seeds]
    rows = []
    began = time.time()
    with ProcessPoolExecutor(max_workers=args.procs, initializer=init_worker) as ex:
        futs = {ex.submit(run_cell, c, s, mappings, methods): (c, s)
                for c, s in tasks_list}
        for k, fut in enumerate(as_completed(futs), 1):
            c, s = futs[fut]
            rows.extend(fut.result())
            print(f"[{k}/{len(futs)}] {c} seed={s} rows={len(rows)} "
                  f"elapsed={time.time()-began:.0f}s", flush=True)

    d = pd.DataFrame(rows).sort_values(["case", "seed", "alpha", "beta", "method"])
    d.to_csv(out / "trials.csv", index=False)
    summary = summarize(d, methods, mappings, cfg)
    summary["wall_seconds"] = float(time.time() - began)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: summary[k] for k in
                      ("nominal_avg3", "all_mapping_avg", "paired_vs_baseline",
                       "paired_vs_v2", "max_independent_metric_diff")},
                     indent=2, ensure_ascii=False))
    print("written to", out)


if __name__ == "__main__":
    main()
