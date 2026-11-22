"""Train-only prefix-information quality (environment-grouped OOF).

Both pre-registered learners are reported; the better OOF MAE is frozen as the
online model.  Improvement is measured against the family-median predictor, in
seconds, on held-out train environments only.

Distributional quality is also scored because the scheduler consumes a belief,
not a point: mean pinball loss over levels 0.1..0.9 for the conditional
distribution used online (pred * exp(OOF log-residual)), a location-shifted
family pool, and the family prior.  Validation prefixes are not read.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import load_all  # noqa: E402
from prefix_model import PrefixModel, load_features  # noqa: E402

OUT = HERE.parent / "results"
LEVELS = np.arange(1, 10) / 10


def pinball(samples, y, levels=LEVELS):
    q = np.quantile(samples, levels)
    return float(np.mean(2 * np.maximum(levels * (y - q), (levels - 1) * (y - q))))


def distribution_scores(model, prefix, pools):
    rows = model.rows
    oof = model.oof[prefix]
    preds, shift, prior = [], [], []
    for i, (_, row) in enumerate(rows.iterrows()):
        pool = np.asarray(pools[row.family_id], float)
        prior.append(pinball(pool, row.duration_s))
        if not np.isfinite(oof[i]):
            preds.append(np.nan)
            shift.append(np.nan)
            continue
        preds.append(pinball(model.duration_dist_from_features(
            model.cache[(row.unit_id, prefix)], prefix, row.family_id, pool), row.duration_s))
        ratio = float(np.clip(oof[i] / np.median(pool), .5, 2.))
        shift.append(pinball(pool * ratio, row.duration_s))
    return dict(
        pinball_replace=float(np.nanmean(preds)),
        pinball_shift=float(np.nanmean(shift)),
        pinball_family=float(np.nanmean(prior)),
        pinball_replace_improvement_pct=float(100 * (1 - np.nanmean(preds) / np.nanmean(prior))),
        pinball_shift_improvement_pct=float(100 * (1 - np.nanmean(shift) / np.nanmean(prior))),
    )


def main():
    cond, tasks, zones, train, seg = load_all()
    tr = seg[(seg.split == "train") & (seg.eligible == 1)]
    cache = load_features(list(tr.unit_id))
    pools = {f: g.duration_s.to_numpy(float) for f, g in tr.groupby("family_id")}
    coverage = {"train_rows": int(len(tr)),
                "feature_coverage": {str(p): float(sum(bool((cache[(u, p)] == cache[(u, p)]).any())
                                                      for u in tr.unit_id) / len(tr))
                                     for p in (.2, .6, 1.0)}}
    result = {"coverage": coverage,
              "train_duration_median_by_family": {f: float(np.median(v)) for f, v in pools.items()},
              "models": {}}
    for kind in ("ridge", "tree"):
        m = PrefixModel(tr, cache, kind=kind)
        for p in (.2, .6, 1.0):
            m.diagnostics[p].update(distribution_scores(m, p, pools))
        result["models"][kind] = m.summary()
        print(kind, json.dumps(m.summary(), indent=2), flush=True)
    best = min(result["models"], key=lambda k: result["models"][k]["prefixes"]["0.2"]["model_mae"])
    result["frozen_kind"] = best
    result["frozen_rule"] = "min train OOF MAE at the free 0.2 s prefix (train-only model choice)"
    bestp = result["models"][best]["prefixes"]["0.2"]
    result["passes_3pct_threshold"] = bool(bestp["improvement_pct"] >= 3.0)
    result["payoff_of_paid_prefix"] = {
        "0.6": result["models"][best]["prefixes"]["0.6"]["improvement_pct"],
        "1.0": result["models"][best]["prefixes"]["1.0"]["improvement_pct"],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "prefix_quality.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print("frozen", best, "0.2s improvement", bestp["improvement_pct"], "%",
          "verdict", "usable" if result["passes_3pct_threshold"] else "below 3% threshold")


if __name__ == "__main__":
    main()
