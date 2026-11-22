"""Unified recomputation of every audited method from raw trials.

All numbers in report.md / summary.json come from this module.  The bootstrap is
seeded (20261122, 4000 replicates, seed-level cluster) so it is reproducible.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROLE = HERE.parent
MAPPINGS = [(1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.)]
NOM = (1., .5)
BOOT_SEED = 20261122
BOOT_REPS = 4000


def boot_ci(seed_delta, reps=BOOT_REPS, seed=BOOT_SEED):
    v = np.asarray(seed_delta, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(reps, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def nominal_stats(d, method, baseline_rows):
    n = d[(d.alpha == NOM[0]) & (d.beta == NOM[1])]
    mm = n[n.method == method]
    by_case = mm.groupby("case").weighted_on_time_rate.mean()
    bn = baseline_rows[(baseline_rows.alpha == NOM[0]) & (baseline_rows.beta == NOM[1])]
    pairs = mm.set_index(["case", "seed"]).weighted_on_time_rate.to_frame("m").join(
        bn.set_index(["case", "seed"]).weighted_on_time_rate.to_frame("b"), how="inner")
    delta = (pairs.m - pairs.b) * 100
    by_seed = delta.groupby(level="seed").mean()
    lo, hi = boot_ci(by_seed.values)
    return dict(
        mean3=float(by_case.mean()),
        cases={k: float(v) for k, v in by_case.items()},
        delta_mean_pp=float(delta.mean()),
        delta_cases={k: float(v) for k, v in ((pairs.m - pairs.b) * 100).groupby(level="case").mean().items()},
        ci95=[lo, hi],
        positive_seeds=int((by_seed > 0).sum()),
        n_seeds=int(len(by_seed)),
        cost=float(mm.cost.mean()),
        completion=float(mm.weighted_completion_rate.mean()),
        observations=float(mm.observations.mean()),
        max_independent_diff=float(mm.independent_diff.max()),
        invalid_actions=int(mm.invalid_actions.sum()),
        n_runs=int(len(mm)),
    )


def mapping_stats(d, method, baseline_rows):
    out = {}
    for a, b in MAPPINGS:
        mm = d[(d.method == method) & (d.alpha == a) & (d.beta == b)]
        bb = baseline_rows[(baseline_rows.alpha == a) & (baseline_rows.beta == b)]
        out[f"{a:.2f}/{b:.2f}"] = dict(mean=float(mm.weighted_on_time_rate.mean()),
                                       baseline=float(bb.weighted_on_time_rate.mean()),
                                       delta_pp=float((mm.weighted_on_time_rate.mean() - bb.weighted_on_time_rate.mean()) * 100))
    return out


def sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default=str(ROLE / "results" / "trials.csv"))
    ap.add_argument("--out", default=str(ROLE / "results" / "summary.json"))
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--code", default="")
    args = ap.parse_args()
    d = pd.read_csv(args.trials)
    d = d.drop_duplicates(["case", "seed", "alpha", "beta", "method"])
    base = d[d.method == args.baseline].copy()
    methods = [m for m in d.method.unique() if m != args.baseline]
    res = {"protocol": {"cases": sorted(d.case.unique()), "seeds": int(d.seed.nunique()),
                        "mappings": MAPPINGS, "bootstrap": {"seed": BOOT_SEED, "reps": BOOT_REPS, "cluster": "seed"},
                        "independent_metric_tol": 1e-9},
           "baseline": nominal_stats(d, args.baseline, base),
           "baseline_five_mapping_mean": float(d[d.method == args.baseline].weighted_on_time_rate.mean()),
           "methods": {}}
    for m in methods:
        st = nominal_stats(d, m, base)
        st["five_mapping_mean"] = float(d[d.method == m].weighted_on_time_rate.mean())
        st["per_mapping"] = mapping_stats(d, m, base)
        st["five_mapping_delta_pp"] = float((d[d.method == m].weighted_on_time_rate.mean()
                                             - d[d.method == args.baseline].weighted_on_time_rate.mean()) * 100)
        res["methods"][m] = st

    # acceptance flags (holdout is filled in later by holdout.py)
    gate = 90.198
    for m, st in res["methods"].items():
        checks = {
            "nominal_ge_90.198": st["mean3"] * 100 >= gate,
            "paired_ci_lower_gt_0": st["ci95"][0] > 0,
            "all_mappings_ge_baseline": all(v["delta_pp"] >= -1e-9 for v in st["per_mapping"].values()),
            "independent_diff_lt_1e-9": st["max_independent_diff"] < 1e-9,
            "no_invalid_actions": st["invalid_actions"] == 0,
        }
        st["gate_checks"] = checks
        st["gate_pass_without_holdout"] = all(checks.values())
    res["code_snapshot_sha256_16"] = {p: sha(p) for p in args.code.split(",") if p}
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({m: {"mean3": round(s["mean3"] * 100, 4), "five": round(s["five_mapping_mean"] * 100, 4),
                          "ci": [round(x, 3) for x in s["ci95"]], "pos": s["positive_seeds"],
                          "pass": s["gate_pass_without_holdout"]} for m, s in res["methods"].items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
