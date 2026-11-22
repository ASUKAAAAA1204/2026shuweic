"""Rebuild results/summary.json from results/trials.csv (no re-run).

Adds the fields required by the handoff contract: per-mapping paired CIs,
all-mapping paired deltas vs v2, seed-delta distribution, runtime/gain
trade-off, threshold flags and the environment-holdout block.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_joint as rj  # noqa: E402


def per_mapping_paired(d, method, ref, mappings, rng, draws=5000):
    out = {}
    for alpha, beta in mappings:
        key = (np.isclose(d.alpha, alpha)) & (np.isclose(d.beta, beta))
        x = d[key & (d.method == method)].set_index(["case", "seed"]).weighted_on_time_rate
        r = d[key & (d.method == ref)].set_index(["case", "seed"]).weighted_on_time_rate
        delta = (x - r) * 100
        by_seed = delta.groupby(level=1).mean()
        i = rng.integers(0, len(by_seed), size=(draws, len(by_seed)))
        means = by_seed.to_numpy()[i].mean(axis=1)
        out[f"{alpha},{beta}"] = {
            "mean_pp": float(delta.mean()),
            "ci95_pp": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "positive_seeds": int((by_seed > 0).sum()), "n_seeds": int(len(by_seed)),
        }
    return out


def main():
    out_dir = HERE.parent / "results"
    d = pd.read_csv(out_dir / "trials.csv")
    methods = ["baseline", "v2"] + [m for m in rj.JOINT if m in set(d.method)]
    methods = [m for m in methods if m in set(d.method)]
    mappings = rj.MAPPINGS
    summary = rj.summarize(d, methods, mappings, {"seeds": int(d.seed.nunique()),
                                                  "mappings": "all", "methods": methods})
    rng = np.random.default_rng(20261122)

    # Per-mapping paired deltas vs baseline.
    summary["per_mapping_paired_vs_baseline"] = {
        m: per_mapping_paired(d, m, "baseline", mappings, rng) for m in methods if m != "baseline"}
    summary["per_mapping_paired_vs_v2"] = {
        m: per_mapping_paired(d, m, "v2", mappings, rng) for m in methods if m != "v2"}
    # Seed-level delta distribution for the primary.
    primary = summary["primary"]
    nom = d[(np.isclose(d.alpha, 1.0)) & (np.isclose(d.beta, .5))]
    b = nom[nom.method == "baseline"].set_index(["case", "seed"]).weighted_on_time_rate
    x = nom[nom.method == primary].set_index(["case", "seed"]).weighted_on_time_rate
    sd = ((x - b).groupby(level=1).mean() * 100).sort_values()
    summary["primary_seed_delta_pp"] = {
        "min": float(sd.min()), "p10": float(sd.quantile(.1)),
        "median": float(sd.median()), "p90": float(sd.quantile(.9)),
        "max": float(sd.max()), "sd": float(sd.std(ddof=1)),
        "values": [float(v) for v in sd.to_numpy()],
    }
    # Worth-it trade-off (nominal, per run).
    rt = summary["runtime_seconds_per_run"]
    summary["worth_it"] = {
        "primary_runtime_s": rt.get(primary), "v2_runtime_s": rt.get("v2"),
        "baseline_runtime_s": rt.get("baseline"),
        "primary_speed_vs_v2": float(rt["v2"] / rt[primary]) if rt.get(primary) else None,
        "primary_speed_vs_baseline": float(rt["baseline"] / rt[primary]) if rt.get(primary) else None,
        "gain_vs_v2_nominal_pp": summary["paired_vs_v2"].get(primary, {}).get("mean_pp"),
        "gain_vs_baseline_nominal_pp": summary["paired_vs_baseline"].get(primary, {}).get("mean_pp"),
        "dominates_v2_nominal": bool(
            summary["paired_vs_v2"].get(primary, {}).get("mean_pp", -1) > 0
            and rt[primary] < rt["v2"]),
    }
    # Environment holdout block (independent of the validation split).
    hp = out_dir / "holdout_environment" / "summary.json"
    if hp.exists():
        hs = json.loads(hp.read_text(encoding="utf8"))
        summary["holdout"] = {
            "protocol": "2 disjoint halves of the 16 training environments, 20 seeds, nominal mapping",
            "primary_avg3": hs["by_case"]["avg3"].get(primary),
            "baseline_avg3": hs["by_case"]["avg3"].get("pressure_gated"),
            "primary_mean_pp": hs["by_fold"].get(primary, {}).get("mean_pp"),
            "primary_fold0_pp": hs["by_fold"].get(primary, {}).get("fold0_pp"),
            "primary_fold1_pp": hs["by_fold"].get(primary, {}).get("fold1_pp"),
            "primary_ci95_pp": hs["by_fold"].get(primary, {}).get("ci95_pp"),
            "average_nonnegative": bool(hs["by_fold"].get(primary, {}).get("mean_pp", -1) >= 0),
            "by_case": {c: hs["by_case"][c].get(primary) for c in ("S1", "S2", "S3")},
            "all_methods_mean_pp": {m: v["mean_pp"] for m, v in hs["by_fold"].items()},
            "max_independent_metric_diff": hs["max_independent_metric_diff"],
        }
        summary["thresholds"]["environment_holdout_nonnegative"] = summary["holdout"]["average_nonnegative"]

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: summary[k] for k in
                      ("nominal_avg3", "all_mapping_avg", "thresholds", "worth_it",
                       "primary_seed_delta_pp")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
