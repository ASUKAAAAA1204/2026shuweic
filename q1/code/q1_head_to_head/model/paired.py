"""Paired, hierarchy-weighted model comparison with an environment bootstrap."""

import numpy as np
import pandas as pd

_MODELS = ("B", "K0", "K", "MLP")
_METRICS = ("nade", "ade", "fde")
_CONTRASTS = (("K0", "MLP"), ("K", "MLP"), ("MLP", "B"), ("K0", "B"), ("K", "B"))
_REQUIRED = ("window_id", "model", "horizon", "source", "env", "record", "side") + _METRICS
_SEED, _N_BOOT = 20261120, 5000


def summarize(metrics):
    """Return (environment metrics, model scores, paired gains)."""
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("metrics must be a pandas DataFrame")
    missing = set(_REQUIRED) - set(metrics.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    df = metrics.loc[:, list(_REQUIRED)].copy()
    if df.empty: raise ValueError("metrics must not be empty")
    if set(df["model"]) != set(_MODELS): raise ValueError("model names must be exactly B, K0, K, MLP")
    keys = ["window_id", "horizon", "source", "env", "record", "side"]
    if df[keys].isna().any().any(): raise ValueError("group keys must be non-null")
    try: values = df.loc[:, _METRICS].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc: raise ValueError("metrics must be numeric") from exc
    if not np.isfinite(values).all(): raise ValueError("metrics must be finite")
    if df.duplicated(["window_id", "model", "horizon"]).any(): raise ValueError("duplicate window/model/horizon rows")
    windows = df.groupby("window_id", sort=False)
    if (windows["model"].agg(lambda x: frozenset(x)) != frozenset(_MODELS)).any():
        raise ValueError("every window must contain all four matched models")
    if any((windows[c].nunique() > 1).any() for c in ("source", "env", "record", "side")):
        raise ValueError("window metadata must be constant across models")
    horizon_sets = df.groupby(["window_id", "model"])["horizon"].agg(frozenset).unstack("model")
    if horizon_sets.nunique(axis=1).ne(1).any(): raise ValueError("horizons must be matched across models")
    hierarchy = ["model", "horizon", "source", "env", "record", "side"]
    env = df.groupby(hierarchy + ["window_id"], as_index=False, sort=False)[list(_METRICS)].mean()
    env = env.groupby(hierarchy, as_index=False, sort=False)[list(_METRICS)].mean()
    env = env.groupby(hierarchy[:-1], as_index=False, sort=False)[list(_METRICS)].mean()
    env = env.groupby(hierarchy[:-2], as_index=False, sort=False)[list(_METRICS)].mean()
    env = env.sort_values(["model", "horizon", "source", "env"]).reset_index(drop=True)
    model_counts = env.groupby(["horizon", "source", "env"])["model"].nunique()
    if (model_counts != len(_MODELS)).any(): raise ValueError("environment groups must be matched across models")
    if len(env[["horizon", "source"]].drop_duplicates()) != env["horizon"].nunique() * env["source"].nunique():
        raise ValueError("every source must contain every horizon")

    cells = env.groupby(["model", "horizon", "source"], as_index=False)[list(_METRICS)].mean()
    # Equal metric weights define the scalar score at each source-horizon cell.
    cells["score"] = cells[list(_METRICS)].mean(axis=1)
    scores = cells.groupby("model", as_index=False)["score"].mean().set_index("model").reindex(_MODELS)
    scores = scores.rename_axis("model").reset_index()
    if not np.isfinite(scores["score"]).all(): raise ValueError("model scores must be finite")

    sources, horizons = sorted(env["source"].unique()), sorted(env["horizon"].unique())
    units = env[["source", "env"]].drop_duplicates().sort_values(["source", "env"]).reset_index(drop=True)
    unit_source = pd.Categorical(units["source"], categories=sources).codes
    unit_index = pd.MultiIndex.from_frame(units[["source", "env"]])
    row_units = unit_index.get_indexer(pd.MultiIndex.from_frame(env[["source", "env"]]))
    h_codes = pd.Categorical(env["horizon"], categories=horizons).codes
    n_u, n_h, n_s = len(units), len(horizons), len(sources)
    arrays, masks = [], []
    for model in _MODELS:
        take = (env["model"] == model).to_numpy()
        arr = np.full((n_u, n_h, len(_METRICS)), np.nan)
        arr[row_units[take], h_codes[take]] = env.loc[take, list(_METRICS)].to_numpy(float)
        mask = np.isfinite(arr).all(axis=2)
        arrays.append(np.where(mask[..., None], arr, 0.0))
        masks.append(mask)
    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]): raise ValueError("environment groups must be matched across models")

    # A draw is usable only when every source-horizon cell is represented.
    rng = np.random.default_rng(_SEED)
    boot = np.empty((_N_BOOT, len(_MODELS)))
    rejected, filled = 0, 0
    while filled < _N_BOOT:
        counts = np.bincount(rng.integers(0, n_u, n_u), minlength=n_u).astype(float)
        den = np.zeros((n_s, n_h))
        np.add.at(den, unit_source, counts[:, None] * masks[0])
        if not (den > 0).all():
            rejected += 1
            continue
        sums = np.zeros((len(_MODELS), n_s, n_h, len(_METRICS)))
        for k, arr in enumerate(arrays):
            np.add.at(sums[k], unit_source, counts[:, None, None] * arr)
        boot[filled] = (sums / den[None, :, :, None]).mean(axis=(1, 2, 3))
        filled += 1

    gains = np.column_stack([1.0 - boot[:, _MODELS.index(c)] / boot[:, _MODELS.index(r)]
                             for c, r in _CONTRASTS])
    point = scores.set_index("model")["score"]
    rows = []
    for j, (candidate, reference) in enumerate(_CONTRASTS):
        gain = 1.0 - point[candidate] / point[reference]
        if not np.isfinite(gain): raise ValueError("relative gains must be finite")
        bounds = np.quantile(gains[:, j], [0.025, 0.975, 0.0125, 0.9875])
        rows.append((candidate, reference, gain, *bounds, rejected))
    comparisons = pd.DataFrame(rows, columns=[
        "candidate", "reference", "gain", "lo95", "hi95", "lo975", "hi975", "rejected"])
    return env, scores, comparisons
