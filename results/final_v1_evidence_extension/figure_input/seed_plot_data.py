"""Extract the 80 per-seed points and verify the two archived stability summaries."""
from pathlib import Path

import numpy as np
import pandas as pd

EXT = Path(__file__).resolve().parents[1]
KEY = ["case_id", "seed", "alpha", "beta"]


def load_seed_points():
    sources = {"original": EXT.parent / "final_v1_stability/seed_trials.csv.gz",
               "replication": EXT / "internal_replication.csv.gz"}
    reference = pd.read_csv(EXT / "seed_batches.csv").set_index(["batch", "alpha"])
    hashes, streams, points = {}, {}, []
    for batch, path in sources.items():
        runs = pd.read_csv(path).query("alpha in [1., 1.5] and beta == .5")
        assert runs.valid_protocol.all() and runs.invalid_actions.eq(0).all()
        assert not runs.duplicated(KEY + ["internal_seed"]).any()
        by_instance = runs.groupby(KEY)
        assert by_instance.internal_seed.nunique().eq(20).all()
        assert by_instance.instance_hash.nunique().eq(1).all()
        hashes[batch] = by_instance.instance_hash.first().sort_index()
        streams[batch] = set(runs.internal_seed)
        assert len(streams[batch]) == 20
        for alpha, group in runs.groupby("alpha"):
            by_seed = group.groupby("internal_seed").weighted_on_time_rate
            assert len(by_seed) == 20 and by_seed.count().eq(90).all()
            means = (100 * by_seed.mean()).rename("rate_pct")
            counts = group.groupby(KEY + ["first_action"]).size()
            pair_pct = 100 * (counts * (counts - 1)).sum() / (90 * 20 * 19)
            ref = reference.loc[(batch, alpha)]
            np.testing.assert_allclose([means.mean(), pair_pct],
                                       [ref.mean_rate_pct, ref.first_match_pct], atol=1e-8, rtol=0)
            points.append(means.reset_index().assign(batch=batch, alpha=alpha))
    pd.testing.assert_series_equal(hashes["original"], hashes["replication"])
    assert streams["original"].isdisjoint(streams["replication"])
    result = pd.concat(points, ignore_index=True)
    assert len(result) == 80
    return result[["batch", "alpha", "internal_seed", "rate_pct"]]


if __name__ == "__main__":
    print(load_seed_points().groupby(["batch", "alpha"]).rate_pct.agg(["count", "mean"]))
