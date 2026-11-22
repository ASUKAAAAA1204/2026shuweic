"""Environment-group holdout for role E, following `holdout_lookahead.py`.

Two disjoint halves of the 16 training environments; for each fold the duration
belief (pools, medians, priors) is fitted on the other half and every policy is
replayed on the held half with fresh seeds (2026120100..0119), nominal mapping.
This is an environment-level stress test, not the official validation split.
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

OUT = HERE.parent / "results" / "holdout"
GROUPS = json.loads((HERE.parents[2] / "results" / "grouped_training" /
                     "registration.json").read_text())["groups"]
SEED0 = 2026120100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="env_s40_fb3")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--methods", default="baseline,v2_s20,role_e")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    common.add_v2_path()
    cond, tasks, zones, train, seg = common.load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIGS[args.config]
    jobs_by_case = {c["case_id"]: common.jobs_for(tasks, c["case_id"]) for c in cond["cases"]}
    seeds = list(range(SEED0, SEED0 + args.seeds))
    rows = []
    max_diff = 0.0
    for fold, held in enumerate(GROUPS):
        fit = common.train_rows(seg, environments=set(seg.environment_id.unique()) - set(held))
        evp = common.train_rows(seg, environments=set(held))
        pools = common.pools_from(fit)
        med = {f: float(np.median(v)) for f, v in pools.items()}
        prior = rcp.build_priors(common.records_from(fit), strength=cfg["strength"],
                                 balance=cfg["balance"], winsor_q=cfg.get("winsor_q"))

        def ctx_for(method):
            return dict(pools=pools, med=med, prior=prior, params=cfg)

        fold_rows, diff = common.run_grid(cond["cases"], jobs_by_case, seeds,
                                          [common.NOMINAL], zones, evp, methods, ctx_for)
        for r in fold_rows:
            r["fold"] = fold
        rows.extend(fold_rows)
        max_diff = max(max_diff, diff)
        print("fold", fold, "done", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "trials.csv", index=False)
    base = d[d.method == "baseline"].set_index(["fold", "case", "seed"]).weighted_on_time_rate
    summary = {"config": args.config, "config_params": cfg,
               "seeds": [SEED0, SEED0 + args.seeds - 1],
               "mapping": list(common.NOMINAL),
               "max_independent_metric_diff": max_diff, "methods": {}}
    for method in methods:
        if method == "baseline":
            continue
        x = d[d.method == method].set_index(["fold", "case", "seed"]).weighted_on_time_rate
        delta = (x - base).dropna()
        by_fold = {int(k): float(v.mean() * 100) for k, v in delta.groupby(level=0)}
        by_cluster = delta.groupby(level=[0, 2]).mean().to_numpy()
        rng = np.random.default_rng(20261122)
        boot = [float(by_cluster[rng.integers(0, len(by_cluster), len(by_cluster))].mean() * 100)
                for _ in range(4000)]
        summary["methods"][method] = {
            "mean_pp": float(delta.mean() * 100),
            "fold0_pp": by_fold.get(0), "fold1_pp": by_fold.get(1),
            "fold_average_pp": float(np.mean(list(by_fold.values()))),
            "ci95_pp": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "by_case_pp": {c: float(delta[delta.index.get_level_values(1) == c].mean() * 100)
                           for c in ("S1", "S2", "S3")},
            "positive_clusters": int((delta.groupby(level=[0, 2]).mean() > 0).sum()),
            "n_clusters": int(delta.groupby(level=[0, 2]).mean().size),
            "otr_by_fold": {int(f): float(x[x.index.get_level_values(0) == f].mean())
                            for f in by_fold},
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf8")
    print(json.dumps(summary["methods"], indent=2, ensure_ascii=False))
    print("max diff", max_diff)


if __name__ == "__main__":
    main()
