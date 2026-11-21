"""Check frozen open-loop duration limits on validation and test points."""
import numpy as np
import pandas as pd


def duration_checks(points, model, limits, split):
    keys = ["environment_id", "episode_uuid", "side", "window_id"]
    if points.duplicated(["window_id", "time"]).any() or points[keys].isna().any().any():
        raise ValueError("Each scoring point needs a unique window and time and complete grouping keys")
    values = points[["age", "error"]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Observation ages and distances must be finite and nonnegative")

    expected = pd.MultiIndex.from_product([model["scale"], [.02, .05, .1]])
    registered = limits.set_index(["dataset", "tolerance_fraction"])
    if not registered.index.is_unique or set(registered.index) != set(expected):
        raise ValueError("Each frozen source and tolerance must have one registered age limit")
    if set(points.dataset) - set(model["scale"]):
        raise ValueError("No frozen scale for one or more data sources")

    # Match the original replay's floating-point bucket boundaries exactly.
    points = points.assign(bucket=np.ceil(points.age * 30).clip(1, 30).astype(int))
    rows = []
    for (source, tolerance), limit in registered.iterrows():
        threshold = model["scale"][source] * tolerance
        radii = np.array([model["radii"]["damped"][f"{source}|{b}"] for b in range(1, 31)])
        if not np.isfinite(radii).all() or (radii < 0).any():
            raise ValueError(f"Invalid frozen radii for {source}")
        first_crossing = np.flatnonzero(radii > threshold)
        end = int(first_crossing[0]) if first_crossing.size else 30
        if end != limit.continuous_accepted_age_bins:
            raise ValueError(f"Frozen radii disagree with registered age limit for {source}, {tolerance}")
        group = points.loc[points.dataset == source]
        for scope, mask in (("prefix", group.bucket <= end), ("terminal", group.bucket == end)):
            eligible = group.loc[mask].assign(within=lambda g: g.error <= threshold)
            score = eligible.groupby(keys).within.mean()
            for levels in ([0, 1, 2], [0, 1], [0]):
                score = score.groupby(level=levels).mean()
            rows.append(dict(split=split, dataset=source, tolerance=tolerance, limit_bin=end,
                             scope=scope, points=len(eligible), windows=eligible.window_id.nunique(),
                             records=eligible.episode_uuid.nunique(), environments=len(score),
                             within=score.mean()))
    return pd.DataFrame(rows)
