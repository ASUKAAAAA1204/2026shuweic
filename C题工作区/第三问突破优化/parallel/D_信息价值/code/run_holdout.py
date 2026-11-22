"""Two-fold environment-group holdout (same construction as holdout_lookahead.py).

For each fold the duration pools AND the prefix model are fitted on the other
half of the 16 training environments; replay uses the held half only, fresh
seeds 2026120100..2026120119 and the nominal mapping.  This is an independent
environment-level stress test, not the official validation split.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import (environment_groups, build_hidden, hidden_for,  # noqa: E402
                      load_all, pools_from, medians)
from prefix_model import PrefixModel, load_features  # noqa: E402
from runner import METHODS, gated_arm, run_episode  # noqa: E402
from support import jobs_for  # noqa: E402

OUT = HERE.parent / "results"
ARMS = ["pressure_gated", "v2_s20", "v2_free", "v2_paid", "v2_both", "v2_gated"]


def main():
    cond, tasks, zones, train, seg = load_all()
    groups = environment_groups()
    base_df = seg[(seg.split == "train") & (seg.eligible == 1)]
    rows = []
    t0 = time.time()
    for fold, held in enumerate(groups):
        fit = base_df[~base_df.environment_id.isin(held)]
        evp = base_df[base_df.environment_id.isin(held)]
        cache = load_features(list(fit.unit_id) + list(evp.unit_id))
        model = PrefixModel(fit, cache, kind="tree")
        METHODS["v2_gated"] = gated_arm(model, 3.0)
        pools = pools_from(fit)
        med = medians(pools)
        print("fold", fold, "fit", len(fit), "held", len(evp), "gate", METHODS["v2_gated"], flush=True)
        for case in cond["cases"]:
            jobs = jobs_for(tasks, case["case_id"])
            for seed in range(2026120100, 2026120120):
                hidden = build_hidden(hidden_for(jobs, seed, evp), cache)
                for name in ARMS:
                    m, ind, diff = run_episode(jobs, hidden, zones, case, 1.0, .5, name, pools, med, model)
                    assert not m["invalid_actions"] and diff < 1e-9
                    rows.append(dict(fold=fold, case=case["case_id"], seed=seed, method=name,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                                     weighted_completion_rate=float(m["weighted_completion_rate"]),
                                     cost=float(m["cost"]), observations=int(m["observations"])))
            print("fold", fold, case["case_id"], round(time.time() - t0, 1), "s", flush=True)
    d = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    d.to_csv(OUT / "holdout_trials.csv", index=False)
    base = d[d.method == "pressure_gated"].set_index(["fold", "case", "seed"]).weighted_on_time_rate
    summary = {"folds": groups, "per_arm": {}}
    for m in ARMS:
        if m == "pressure_gated":
            continue
        x = d[d.method == m].set_index(["fold", "case", "seed"]).weighted_on_time_rate
        delta = (x - base) * 100
        summary["per_arm"][m] = {
            "mean_delta_pp": float(delta.mean()),
            "fold0_delta_pp": float(delta[delta.index.get_level_values("fold") == 0].mean()),
            "fold1_delta_pp": float(delta[delta.index.get_level_values("fold") == 1].mean()),
            "passes_nonnegative": bool(delta.mean() >= 0),
            "nominal_mean": float(d[d.method == m].weighted_on_time_rate.mean()),
        }
    summary["_note"] = ("Pools and prefix model refitted per fold on the other environment half; "
                        "prefix features of held units are observed online only.")
    (OUT / "holdout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))
    print("elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
