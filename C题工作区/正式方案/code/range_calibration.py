"""Calibrate and audit paired Q1 old/new 90% ranges."""

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["calibrate"]
_COLUMNS = ["split", "dataset", "environment_id", "episode_uuid", "window_id",
            "age", "old_error", "new_error", "scale"]
_MODELS = (("original", "old_error"), ("B", "new_error"))
_HORIZONS = (0.2, 0.6, 1.0)


def _bin(age):
    return np.clip(np.ceil(np.asarray(age, dtype=float) * 30.0), 1, 30).astype(int)


def _fit(train):
    rows = []
    binned = train.assign(bin=_bin(train.age))
    for model, column in _MODELS:
        for (dataset, bin_id), group in binned.groupby(["dataset", "bin"], sort=True):
            radius = float(group[column].abs().quantile(0.9, interpolation="higher"))
            rows.append(dict(dataset=dataset, bin=int(bin_id), model=model,
                             radius=radius, points=len(group)))
    radii = pd.DataFrame(rows)
    assert not radii.empty and np.isfinite(radii.radius).all() and (radii.radius >= 0).all()
    return radii


def _equal_weight(frame, metrics):
    keys = ["dataset", "environment_id", "episode_uuid", "window_id"]
    frame = frame.groupby(keys, as_index=False)[metrics].mean()
    for level in (keys[:-1], keys[:-2], keys[:-3]):
        frame = frame.groupby(level, as_index=False)[metrics].mean()
    return {metric: float(frame[metric].mean()) for metric in metrics}


def _evaluate(points, radii):
    rows = []
    for split in ("validation", "test"):
        part = points[points.split == split]
        if part.empty:
            continue
        part = part.assign(bin=_bin(part.age))
        for model, column in _MODELS:
            lookup = radii[radii.model == model][["dataset", "bin", "radius"]]
            scored = part.merge(lookup, on=["dataset", "bin"], how="left", validate="many_to_one")
            assert scored.radius.notna().all(), f"missing calibrated radii for {split}/{model}"
            scored = scored.assign(
                _coverage=(scored[column].abs() <= scored.radius).astype(float),
                _radius=scored.radius / scored.scale,
            )
            for horizon in _HORIZONS:
                window = scored[scored.age <= horizon + 1e-12]
                if window.empty:
                    continue
                metrics = _equal_weight(window, ["_coverage", "_radius"])
                rows.append(dict(split=split, model=model, horizon=horizon,
                                 coverage=metrics["_coverage"],
                                 norm_radius=metrics["_radius"], points=len(window)))
    return rows


def calibrate(points, out):
    """Fit train-only ranges and write model/summary tables to ``out``."""
    assert isinstance(points, pd.DataFrame), "points must be a pandas DataFrame"
    assert set(_COLUMNS) <= set(points.columns), "missing required columns"
    data = points[_COLUMNS].copy()
    assert len(data), "points must not be empty"
    assert data.split.isin(["train", "validation", "test"]).all(), "invalid split value"
    for column in ("dataset", "environment_id", "episode_uuid", "window_id"):
        assert data[column].notna().all(), f"invalid {column}"
    assert np.isfinite(data[["age", "old_error", "new_error", "scale"]].to_numpy(float)).all()
    assert (data.age >= 0).all() and (data.scale > 0).all() and (data.split == "train").any()

    train = data[data.split == "train"]
    held = data[data.split.isin(["validation", "test"])]
    assert len(held) and not set(train.environment_id) & set(held.environment_id)
    radii = _fit(train)
    summary = pd.DataFrame(_evaluate(data, radii),
                           columns=["split", "model", "horizon", "coverage", "norm_radius", "points"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    radii.to_csv(out / "q1_range_model.csv", index=False)
    summary.to_csv(out / "q1_range_summary.csv", index=False)
    return summary
