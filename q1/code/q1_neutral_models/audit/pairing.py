"""Paired NADE comparisons used by the neutral-model audit."""

from __future__ import annotations

import pandas as pd

_ENV_COLUMNS = ("model", "source", "env", "horizon", "nade")
_KEY_COLUMNS = ("model", "source", "env", "horizon")
_HORIZONS_PER_CELL = 3
_OVERALL_COLUMNS = ("candidate", "reference", "nade_difference", "gain_pct")
_ENVIRONMENT_COLUMNS = (
    "candidate",
    "reference",
    "nade_difference",
    "gain_pct",
    "source",
    "env",
)


def _validate(env: pd.DataFrame, summary: pd.Series) -> None:
    if not isinstance(env, pd.DataFrame):
        raise TypeError("env must be a pandas DataFrame")
    if not isinstance(summary, pd.Series):
        raise TypeError("summary must be a pandas Series")

    missing = [column for column in _ENV_COLUMNS if column not in env.columns]
    if missing:
        raise ValueError(f"env is missing required columns: {missing}")
    if env.empty:
        raise ValueError("env must not be empty")
    if summary.empty:
        raise ValueError("summary must not be empty")
    if env[list(_ENV_COLUMNS)].isna().any().any():
        raise ValueError("env contains missing values")
    if summary.isna().any():
        raise ValueError("summary contains missing values")
    if not pd.api.types.is_numeric_dtype(env["nade"]):
        raise ValueError("env['nade'] must be numeric")
    if not pd.api.types.is_numeric_dtype(summary):
        raise ValueError("summary values must be numeric")
    if not summary.index.is_unique:
        raise ValueError("summary index must be unique")
    if env.duplicated(subset=list(_KEY_COLUMNS)).any():
        raise ValueError("env contains duplicate model/source/env/horizon rows")


def _validate_coverage(env: pd.DataFrame) -> list[str]:
    """Return model order, failing unless every model has identical full coverage."""
    models = list(pd.unique(env["model"]))
    expected_keys = None

    for model, rows in env.groupby("model", sort=False):
        horizon_counts = rows.groupby(["source", "env"], sort=False)["horizon"].nunique()
        bad = horizon_counts[horizon_counts != _HORIZONS_PER_CELL]
        if not bad.empty:
            source, environment = bad.index[0]
            raise ValueError(
                f"model {model!r} covers {int(bad.iloc[0])} horizons for "
                f"source {source!r}, env {environment!r}; "
                f"expected {_HORIZONS_PER_CELL}"
            )

        keys = set(zip(rows["source"], rows["env"], rows["horizon"]))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(
                f"model {model!r} has inconsistent environment/horizon coverage"
            )

    return models


def _comparisons(models: list[str], available: set[str]) -> list[tuple[str, str]]:
    """Build requested (candidate, reference) pairs without duplicates."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(candidate: str, reference: str) -> None:
        pair = (candidate, reference)
        if candidate != reference and pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    if "B" not in models:
        raise ValueError("reference model 'B' is missing from env")
    for model in models:
        if model != "B":
            add(model, "B")

    if "K" in available:
        for reference in ("PackGate", "PackResidual", "M1", "K0"):
            if reference in available:
                add("K", reference)
    if "C" in available and "M1" in available:
        add("C", "M1")

    return pairs


def _difference(candidate: float, reference: float) -> tuple[float, float]:
    """Return candidate minus reference NADE and gain as a percentage of reference."""
    if reference == 0:
        raise ValueError("cannot compute gain_pct against a zero reference NADE")
    return candidate - reference, (reference - candidate) / reference * 100.0


def pairwise(
    env: pd.DataFrame, summary: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare model NADE against paired references, overall and per environment.

    Parameters
    ----------
    env:
        One row per model/source/environment/horizon with a ``nade`` value.
        Every model must cover the same three horizons for each source/environment
        pair; otherwise a ``ValueError`` is raised.
    summary:
        Overall NADE per model, indexed by model name.

    Returns
    -------
    (overall_pairs, environment_pairs):
        Overall pairs use the summary NADE. Environment pairs average the three
        horizon NADE values within each source/environment pair and report
        ``nade_difference = candidate - reference`` and
        ``gain_pct = (reference - candidate) / reference * 100``.
    """
    _validate(env, summary)
    models = _validate_coverage(env)

    overall_nade = summary.to_dict()
    missing = [model for model in models if model not in overall_nade]
    if missing:
        raise ValueError(f"summary is missing overall NADE for models: {missing}")
    if "B" not in overall_nade:
        raise ValueError("reference model 'B' is missing from summary")

    available = set(models) & set(summary.index)
    pairs = _comparisons(models, available)

    overall_rows = []
    for candidate, reference in pairs:
        difference, gain_pct = _difference(
            float(overall_nade[candidate]), float(overall_nade[reference])
        )
        overall_rows.append((candidate, reference, difference, gain_pct))
    overall_pairs = pd.DataFrame(overall_rows, columns=list(_OVERALL_COLUMNS))

    environment_means = (
        env.groupby(["source", "env", "model"], sort=False, as_index=False)["nade"]
        .mean()
    )
    mean_nade = {
        (row.model, row.source, row.env): float(row.nade)
        for row in environment_means.itertuples(index=False)
    }

    environment_rows = []
    cells = env[["source", "env"]].drop_duplicates()
    for source, environment in cells.itertuples(index=False):
        for candidate, reference in pairs:
            candidate_nade = mean_nade.get((candidate, source, environment))
            reference_nade = mean_nade.get((reference, source, environment))
            if candidate_nade is None or reference_nade is None:
                raise ValueError(
                    f"missing paired NADE for candidate {candidate!r} / "
                    f"reference {reference!r} in source {source!r}, "
                    f"env {environment!r}"
                )
            difference, gain_pct = _difference(candidate_nade, reference_nade)
            environment_rows.append(
                (
                    candidate,
                    reference,
                    difference,
                    gain_pct,
                    source,
                    environment,
                )
            )
    environment_pairs = pd.DataFrame(
        environment_rows, columns=list(_ENVIRONMENT_COLUMNS)
    )

    return overall_pairs, environment_pairs
