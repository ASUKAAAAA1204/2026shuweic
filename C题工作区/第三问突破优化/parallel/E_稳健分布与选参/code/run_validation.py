"""Final one-shot validation confirmation for role E.

The configuration is fixed by environment-grouped CV *before* this script runs;
this script only confirms it.  Protocol is the mandatory paired protocol:
3 cases x 30 validation seeds x 5 mappings x {baseline, v2_s20, role_e}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import robust_chain_policy as rcp  # noqa: E402
from configs import CONFIGS  # noqa: E402

OUT = HERE.parent / "results" / "validation"
BOOT_SEED = 20261122


def bootstrap_ci(values, rng, n=4000):
    values = np.asarray(values, float)
    draws = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(n)]
    return [float(np.percentile(draws, 2.5) * 100), float(np.percentile(draws, 97.5) * 100)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pooled_s20_fb3")
    ap.add_argument("--methods", default="baseline,v2_s20,role_e")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--mappings", choices=["nominal", "all"], default="all")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    maps = common.MAPPINGS if args.mappings == "all" else [common.NOMINAL]

    common.add_v2_path()
    cond, tasks, zones, train, seg = common.load_all()
    out = OUT if not args.tag else OUT.parent / f"validation_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    pools = common.pools_from(common.train_rows(seg))
    records = common.records_from(common.train_rows(seg))
    med = {f: float(np.median(v)) for f, v in pools.items()}
    cfg = CONFIGS[args.config]
    prior = rcp.build_priors(records, strength=cfg["strength"], balance=cfg["balance"],
                             winsor_q=cfg.get("winsor_q"))
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    seeds = cond["validation_seeds"][:args.seeds]
    jobs_by_case = {c["case_id"]: common.jobs_for(tasks, c["case_id"]) for c in cond["cases"]}

    def ctx_for(method):
        return dict(pools=pools, med=med, prior=prior, params=cfg)

    rows, max_diff = common.run_grid(cond["cases"], jobs_by_case, seeds, maps, zones, vpool,
                                     methods, ctx_for, log_every=900)
    d = pd.DataFrame(rows)
    d.to_csv(out / "trials.csv", index=False)

    nominal = d[(d.alpha == 1.0) & (d.beta == 0.5)]
    pivot = nominal.pivot_table(index="method", columns="case",
                                values="weighted_on_time_rate", aggfunc="mean")
    pivot["avg3"] = pivot.mean(axis=1)
    base_key = nominal[nominal.method == "baseline"].set_index(["case", "seed"]).weighted_on_time_rate
    base_key_all = d[d.method == "baseline"].set_index(["case", "seed", "alpha", "beta"]) \
        .weighted_on_time_rate
    rng = np.random.default_rng(BOOT_SEED)
    summary = {
        "protocol": {"cases": [c["case_id"] for c in cond["cases"]], "seeds": len(seeds),
                     "mappings": maps, "methods": methods},
        "config_name": args.config, "config_params": cfg,
        "max_independent_metric_diff": max_diff,
        "nominal_avg": {m: float(pivot.loc[m, "avg3"]) for m in methods},
        "nominal_by_case": {m: {c: float(pivot.loc[m, c]) for c in ("S1", "S2", "S3")}
                            for m in methods},
        "all_mapping_avg": {m: float(d[d.method == m].weighted_on_time_rate.mean())
                            for m in methods},
        "per_mapping_avg": {}, "paired": {},
        "nominal_cost": {m: float(nominal[nominal.method == m].cost.mean()) for m in methods},
        "nominal_completion": {m: float(nominal[nominal.method == m].weighted_completion_rate.mean())
                               for m in methods},
        "all_mapping_completion": {m: float(d[d.method == m].weighted_completion_rate.mean())
                                   for m in methods},
        "nominal_observations": {m: float(nominal[nominal.method == m].observations.mean())
                                 for m in methods},
    }
    for alpha, beta in maps:
        sub = d[(d.alpha == alpha) & (d.beta == beta)]
        key = f"{alpha},{beta}"
        summary["per_mapping_avg"][key] = {m: float(sub[sub.method == m].weighted_on_time_rate.mean())
                                           for m in methods}
    for method in methods:
        if method == "baseline":
            continue
        x = nominal[nominal.method == method].set_index(["case", "seed"]).weighted_on_time_rate
        delta = (x - base_key)
        by_seed = delta.groupby(level=1).mean()
        xa = d[d.method == method].set_index(["case", "seed", "alpha", "beta"]).weighted_on_time_rate
        delta_all = xa - base_key_all
        by_seed_all = delta_all.groupby(level=1).mean()
        summary["paired"][method] = dict(
            nominal_delta_pp=float(delta.mean() * 100),
            nominal_delta_by_case_pp={c: float(delta[delta.index.get_level_values(0) == c].mean() * 100)
                                      for c in ("S1", "S2", "S3")},
            nominal_delta_ci95_pp=bootstrap_ci(by_seed, rng),
            nominal_positive_seeds=int((by_seed > 0).sum()), n_seeds=int(len(by_seed)),
            all_mapping_delta_pp=float(delta_all.mean() * 100),
            all_mapping_delta_ci95_pp=bootstrap_ci(by_seed_all, rng),
            all_mapping_positive_seeds=int((by_seed_all > 0).sum()),
            per_mapping_delta_pp={f"{a},{b}": float(delta_all.xs((a, b), level=("alpha", "beta")).mean() * 100)
                                  for a, b in maps},
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf8")
    print(pivot.mul(100).round(4).to_string())
    print(json.dumps({k: summary[k] for k in ("nominal_avg", "all_mapping_avg", "paired")},
                     indent=2, ensure_ascii=False))
    print("max diff", max_diff)


if __name__ == "__main__":
    main()
