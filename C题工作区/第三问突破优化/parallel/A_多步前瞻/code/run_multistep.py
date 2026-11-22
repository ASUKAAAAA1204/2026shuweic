"""Paired runner for role A (correct multi-step lookahead).

Usage:
  python run_multistep.py --seeds 3  --mappings nominal --methods baseline,v2,d2b4,d3b4
  python run_multistep.py --seeds 30 --mappings all     --methods baseline,audit,v2,d2b4
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FORMAL = ROOT / "第三问改进正式方案" / "code"
BREAK = HERE.parents[2]
sys.path.insert(0, str(FORMAL))
sys.path.insert(0, str(BREAK / "code"))
sys.path.insert(0, str(HERE / "code"))

from support import load, jobs_for
from engine import Engine, independent_metrics
from deadline_policy import choose as base_choose, pressure as base_pressure
from baseline import action as base_action
import lookahead_policy_audit_variants as audit_variants
import multi_step_policy as ms

MAPPINGS = [(1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.)]
METHODS = {
    "baseline": dict(kind="baseline"),
    "audit": dict(kind="audit"),
    "v2": dict(kind="ms", depth=1, strength=20, nq_root=5, nq=2, beam=3),
    "d2b4": dict(kind="ms", depth=2, strength=20, nq_root=5, nq=2, beam=4),
    "d2b8": dict(kind="ms", depth=2, strength=20, nq_root=5, nq=2, beam=8),
    "d2b4q5": dict(kind="ms", depth=2, strength=20, nq_root=5, nq=5, beam=4),
    "d2b8q5": dict(kind="ms", depth=2, strength=20, nq_root=5, nq=5, beam=8),
    "d2b16": dict(kind="ms", depth=2, strength=20, nq_root=5, nq=2, beam=16),
    "d3b4": dict(kind="ms", depth=3, strength=20, nq_root=5, nq=2, beam=4),
    "d3b8": dict(kind="ms", depth=3, strength=20, nq_root=5, nq=2, beam=8),
    "d3b16": dict(kind="ms", depth=3, strength=20, nq_root=5, nq=2, beam=16),
    "d2b4s40": dict(kind="ms", depth=2, strength=40, nq_root=5, nq=2, beam=4),
    "adapt": dict(kind="adapt", strength=20, nq_root=5, nq=2, beam=4, depth_small=1, depth_large=2, size_gate=12),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mappings", choices=["nominal", "all"], default="nominal")
    ap.add_argument("--methods", default="baseline,v2,d2b4")
    ap.add_argument("--tag", default="bench")
    args = ap.parse_args()

    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
    med = {f: float(np.median(v)) for f, v in pools.items()}
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    maps = MAPPINGS if args.mappings == "all" else [(1., .5)]
    seeds = cond["validation_seeds"][:args.seeds]

    def hidden_for(jobs, seed):
        rng = random.Random(seed)
        out = {}
        for j in jobs:
            c = vpool[vpool.family_id == j["family_id"]]
            out[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
        return out

    rows = []
    max_diff = 0.0
    t0 = time.time()
    method_seconds = {m: 0.0 for m in methods}
    method_trials = {m: 0 for m in methods}
    for case in cond["cases"]:
        jobs = jobs_for(tasks, case["case_id"])
        for seed in seeds:
            dur = hidden_for(jobs, seed)
            hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)}) for k, v in dur.items()}
            for alpha, beta in maps:
                for name in methods:
                    cfg = METHODS[name]
                    t_start = time.perf_counter()
                    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                               case["observation_limit"], alpha, beta)
                    history = {}
                    while not e.terminal:
                        o = e.observe()
                        if cfg["kind"] == "baseline":
                            a = base_choose(o, pools, history) if base_pressure(o, pools, history) else base_action(o, med, "density", "minimum")
                        elif cfg["kind"] == "audit":
                            a = audit_variants.select(o, pools, history, strength=40, feedback_strength=3,
                                                      quantiles=5, fix_clock=True, fix_chain=True, fix_budget=True)
                        elif cfg["kind"] == "adapt":
                            d = cfg["depth_large"] if len(o["jobs"]) >= cfg["size_gate"] else cfg["depth_small"]
                            a = ms.select(o, pools, history, strength=cfg["strength"], feedback_strength=3,
                                          depth=d, nq_root=cfg["nq_root"], nq=cfg["nq"], beam=cfg["beam"])
                        else:
                            a = ms.select(o, pools, history, strength=cfg["strength"], feedback_strength=3,
                                          depth=cfg["depth"], nq_root=cfg["nq_root"], nq=cfg["nq"], beam=cfg["beam"])
                        ev = e.step(a)
                        if a["type"] == "execute" and ev.get("completion") is not None:
                            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                            history.setdefault(f, []).append((ev["completion"] - ev["service_start"] - beta) / alpha)
                    m = e.metrics()
                    ind = independent_metrics(jobs, e.events, e.H)
                    diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
                    max_diff = max(max_diff, diff)
                    elapsed = time.perf_counter() - t_start
                    method_seconds[name] += elapsed
                    method_trials[name] += 1
                    assert not m["invalid_actions"] and diff < 1e-9, (case["case_id"], seed, name, diff)
                    rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta, method=name,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                                     weighted_completion_rate=float(m["weighted_completion_rate"]),
                                     cost=float(m["cost"]), observations=float(m["observations"]),
                                     elapsed_s=elapsed))
        print("case", case["case_id"], "done", round(time.time() - t0, 1), "s", flush=True)

    d = pd.DataFrame(rows)
    out = HERE.parent / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "trials.csv", index=False)
    nominal = d[(d.alpha == 1.0) & (d.beta == .5)]
    piv = nominal.pivot_table(index="method", columns="case", values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    base = nominal[nominal.method == "baseline"].set_index(["case", "seed"]).weighted_on_time_rate
    summary = {"seeds": args.seeds, "mappings": args.mappings, "methods": methods,
               "max_independent_metric_diff": max_diff, "nominal": {}, "paired": {}, "all_mapping": {},
               "seconds_per_trial": {m: method_seconds[m] / max(method_trials[m], 1) for m in methods}}
    rng = np.random.default_rng(20261122)
    for m in methods:
        if m == "baseline":
            continue
        x = nominal[nominal.method == m].set_index(["case", "seed"]).weighted_on_time_rate
        dd = (x - base)
        by_seed = dd.groupby(level=1).mean()
        boot = [float(by_seed.sample(frac=1, replace=True, random_state=int(s)).mean()) for s in rng.integers(0, 10**6, 2000)]
        summary["paired"][m] = dict(
            mean_pp=float(dd.mean() * 100),
            by_case_pp={c: float(dd[dd.index.get_level_values(0) == c].mean() * 100) for c in ("S1", "S2", "S3")},
            ci95_pp=[float(np.percentile(boot, 2.5) * 100), float(np.percentile(boot, 97.5) * 100)],
            positive_seeds=int((by_seed > 0).sum()), n_seeds=int(len(by_seed)))
    for m in methods:
        summary["nominal"][m] = {c: float(piv.loc[m, c]) for c in ("S1", "S2", "S3", "avg3")}
        sub = d[d.method == m]
        summary["all_mapping"][m] = float(sub.weighted_on_time_rate.mean())
        summary.setdefault("per_mapping", {})[m] = {
            f"{a}/{b}": float(sub[(sub.alpha == a) & (sub.beta == b)].weighted_on_time_rate.mean())
            for a, b in MAPPINGS
            if not sub[(sub.alpha == a) & (sub.beta == b)].empty
        }
        summary.setdefault("all_mapping_by_case", {})[m] = {
            c: float(sub[sub.case == c].weighted_on_time_rate.mean()) for c in ("S1", "S2", "S3")
        }
        summary.setdefault("cost_mean", {})[m] = float(sub.cost.mean())
        summary.setdefault("completion_mean", {})[m] = float(sub.weighted_completion_rate.mean())
        summary.setdefault("observations_mean", {})[m] = float(sub.observations.mean())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print((piv * 100).round(2).sort_values("avg3", ascending=False).to_string())
    print(json.dumps(summary["paired"], indent=2, ensure_ascii=False))
    print("all-mapping:", {m: round(v * 100, 2) for m, v in summary["all_mapping"].items()})
    print("max diff", max_diff, "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
