"""Validation quality summary for the Q1 persistence comparison.

The module intentionally performs no I/O other than writing its two requested
artifacts.  Predictions are supplied by the host through ``analyze_quality``.
Only the ``train`` split is used to estimate the persistence comparator; all
reported quality metrics come from the ``validation`` split.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import numpy as np
import pandas

_REQUIRED_COLUMNS = (
    "split",
    "dataset",
    "environment_id",
    "episode_uuid",
    "side",
    "index",
    "label",
    "probability",
    "constant_probability",
    "current_bad",
)

_SUBSETS = ("all", "currently_good", "currently_bad")

_LAPLACE_ALPHA = 1.0
_HORIZON_SECONDS = 0.2
_BOOTSTRAP_REPLICATES = 2000
_BOOTSTRAP_SEED = 20240517
_CONFIDENCE_LEVEL = 0.95

_SUMMARY_COLUMNS = [
    "dataset",
    "subset",
    "n_samples",
    "n_records",
    "n_environments",
    "positives",
    "mean_observed",
    "brier_logistic",
    "brier_constant",
    "brier_persistence",
    "mean_predicted_logistic",
    "mean_predicted_constant",
    "mean_predicted_persistence",
    "logistic_minus_persistence",
    "logistic_minus_persistence_ci95_low",
    "logistic_minus_persistence_ci95_high",
    "paired_n_environments",
    "paired_bootstrap_available",
]


def _normalise_split(series: pandas.Series) -> pandas.Series:
    """Return a lower-case split label, accepting a few common aliases."""

    values = series.astype("string").str.strip().str.lower()
    return values.replace(
        {
            "train": "train",
            "training": "train",
            "validation": "validation",
            "valid": "validation",
            "val": "validation",
        }
    )


def _coerce_boolean(series: pandas.Series) -> pandas.Series:
    """Coerce a boolean-like column to a plain bool Series.

    The expected input is already boolean, but accepting nullable booleans and
    the usual textual/numeric encodings makes the analysis robust to CSV
    round-trips by the host.
    """

    if pandas.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pandas.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0.0)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(("true", "t", "1", "1.0", "yes", "y")).fillna(False)


def _prepare_predictions(predictions: pandas.DataFrame) -> pandas.DataFrame:
    if not isinstance(predictions, pandas.DataFrame):
        raise TypeError("predictions must be a pandas.DataFrame")

    missing = [name for name in _REQUIRED_COLUMNS if name not in predictions.columns]
    if missing:
        raise ValueError(
            "predictions is missing required column(s): " + ", ".join(missing)
        )

    frame = predictions.loc[:, list(_REQUIRED_COLUMNS)].copy()
    frame["_split"] = _normalise_split(frame["split"])
    frame = frame.loc[frame["_split"].isin(("train", "validation"))].copy()

    frame["label"] = pandas.to_numeric(frame["label"], errors="coerce").astype(
        "float64"
    )
    frame["probability"] = pandas.to_numeric(
        frame["probability"], errors="coerce"
    ).astype("float64")
    frame["constant_probability"] = pandas.to_numeric(
        frame["constant_probability"], errors="coerce"
    ).astype("float64")
    frame["current_bad"] = _coerce_boolean(frame["current_bad"])
    return frame


def _fit_persistence(
    train: pandas.DataFrame,
) -> tuple[float, float, dict[str, Any]]:
    """Fit P(future bad | current state) on train with add-one smoothing."""

    labels = train["label"].to_numpy(dtype="float64", copy=False)
    current_bad = train["current_bad"].to_numpy(dtype=bool, copy=False)
    good_mask = ~current_bad

    n_good = int(good_mask.sum())
    n_bad = int(current_bad.sum())
    future_bad_given_good = int(np.nansum(labels[good_mask]))
    future_bad_given_bad = int(np.nansum(labels[current_bad]))

    # Add-one (Laplace) smoothing for a binary outcome.
    p_good = (future_bad_given_good + _LAPLACE_ALPHA) / (
        n_good + 2.0 * _LAPLACE_ALPHA
    )
    p_bad = (future_bad_given_bad + _LAPLACE_ALPHA) / (
        n_bad + 2.0 * _LAPLACE_ALPHA
    )

    details: dict[str, Any] = {
        "fit_split": "train",
        "horizon_seconds": _HORIZON_SECONDS,
        "laplace_alpha": _LAPLACE_ALPHA,
        "formula": "(future_bad + alpha) / (state_samples + 2 * alpha)",
        "p_future_bad_given_current_good": float(p_good),
        "p_future_bad_given_current_bad": float(p_bad),
        # Short aliases are convenient for downstream consumers.
        "p_bad_given_good": float(p_good),
        "p_bad_given_bad": float(p_bad),
        "current_good": {
            "n_samples": n_good,
            "n_future_bad": future_bad_given_good,
            "probability": float(p_good),
        },
        "current_bad": {
            "n_samples": n_bad,
            "n_future_bad": future_bad_given_bad,
            "probability": float(p_bad),
        },
        "train_counts": {
            "n_samples": int(len(train)),
            "n_records": _n_records(train),
            "n_environments": _n_environments(train),
            "n_current_good": n_good,
            "n_current_bad": n_bad,
            "n_future_bad": int(np.nansum(labels)),
        },
    }
    return float(p_good), float(p_bad), details


def _n_records(frame: pandas.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(
        frame.loc[:, ["environment_id", "episode_uuid"]].drop_duplicates().shape[0]
    )


def _n_environments(frame: pandas.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame["environment_id"].nunique(dropna=True))


def _counts(frame: pandas.DataFrame) -> dict[str, int]:
    return {
        "n_samples": int(len(frame)),
        "n_records": _n_records(frame),
        "n_environments": _n_environments(frame),
    }


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _environment_brier(
    frame: pandas.DataFrame, probability_column: str
) -> pandas.Series:
    """Per-environment Brier with equal weighting of records in an environment."""

    if frame.empty:
        return pandas.Series(dtype="float64")

    labels = frame["label"].to_numpy(dtype="float64", copy=False)
    probabilities = frame[probability_column].to_numpy(dtype="float64", copy=False)
    work = pandas.DataFrame(
        {
            "_environment": frame["environment_id"].to_numpy(copy=False),
            "_record": frame["episode_uuid"].to_numpy(copy=False),
            "_squared_error": (probabilities - labels) ** 2,
        }
    )

    # Mean within each record, then equal weight for records in an environment.
    record_brier = work.groupby(
        ["_environment", "_record"], sort=True, dropna=False
    )["_squared_error"].mean()
    return record_brier.groupby(level=0, sort=True, dropna=False).mean()


def _hierarchical_brier(frame: pandas.DataFrame, probability_column: str) -> float:
    """Mean within record -> equal records/environment -> equal environments."""

    per_environment = _environment_brier(frame, probability_column)
    if per_environment.empty:
        return float("nan")
    return float(per_environment.mean())


def _paired_difference(
    frame: pandas.DataFrame,
) -> tuple[float, float, float, int, bool]:
    """Return logistic-minus-persistence Brier difference and bootstrap CI.

    Environments, rather than samples or records, are the bootstrap unit.
    With fewer than two environments the point difference is still defined,
    but the environment bootstrap and therefore the interval are unavailable.
    """

    logistic = _environment_brier(frame, "probability")
    persistence = _environment_brier(frame, "_persistence_probability")
    paired = pandas.concat(
        [logistic.rename("logistic"), persistence.rename("persistence")],
        axis=1,
        join="inner",
    ).dropna()
    differences = paired["logistic"] - paired["persistence"]
    n_environments = int(differences.shape[0])

    if n_environments == 0:
        return float("nan"), float("nan"), float("nan"), 0, False

    point = float(differences.mean())
    if n_environments < 2:
        return point, float("nan"), float("nan"), n_environments, False

    values = differences.to_numpy(dtype="float64", copy=False)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    indices = rng.integers(
        0, n_environments, size=(_BOOTSTRAP_REPLICATES, n_environments)
    )
    bootstrap_means = values[indices].mean(axis=1)
    tail = (1.0 - _CONFIDENCE_LEVEL) / 2.0 * 100.0
    low, high = np.percentile(bootstrap_means, [tail, 100.0 - tail])
    return point, float(low), float(high), n_environments, True


def _subset_masks(frame: pandas.DataFrame) -> dict[str, pandas.Series]:
    return {
        "all": pandas.Series(True, index=frame.index, dtype=bool),
        "currently_good": ~frame["current_bad"].astype(bool),
        "currently_bad": frame["current_bad"].astype(bool),
    }


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars and non-finite floats for strict JSON."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(float(value))
    return value


def _sorted_values(values: pandas.Series) -> list[Any]:
    return sorted(
        (value for value in values.dropna().unique()),
        key=lambda value: (type(value).__name__, str(value)),
    )


def analyze_quality(predictions: pandas.DataFrame, out: pathlib.Path) -> None:
    """Compute the Q1 validation quality summary and persistence parameters.

    Parameters
    ----------
    predictions:
        Row-level predictions with the required columns.  Only rows whose
        ``split`` is train or validation are considered.
    out:
        Directory in which ``q1_quality_summary.csv`` and
        ``q1_quality_params.json`` are written.
    """

    frame = _prepare_predictions(predictions)
    train = frame.loc[frame["_split"] == "train"].copy()
    validation = frame.loc[frame["_split"] == "validation"].copy()

    p_good, p_bad, persistence_params = _fit_persistence(train)

    # Persistence is a two-state comparator fitted on train and evaluated only
    # on validation rows.
    validation["_persistence_probability"] = np.where(
        validation["current_bad"].to_numpy(dtype=bool, copy=False),
        p_bad,
        p_good,
    )

    summary_rows: list[dict[str, Any]] = []
    paired_details: dict[str, dict[str, Any]] = {}
    validation_counts_by_dataset: dict[str, dict[str, int]] = {}

    for dataset in _sorted_values(validation["dataset"]):
        dataset_validation = validation.loc[validation["dataset"] == dataset].copy()
        dataset_key = str(dataset)
        validation_counts_by_dataset[dataset_key] = _counts(dataset_validation)
        paired_details[dataset_key] = {}

        masks = _subset_masks(dataset_validation)
        for subset in _SUBSETS:
            subset_frame = dataset_validation.loc[masks[subset]].copy()
            labels = subset_frame["label"].to_numpy(dtype="float64", copy=False)

            counts = _counts(subset_frame)
            positive_count = (
                int(np.sum(labels > 0.5)) if labels.size else 0
            )
            mean_observed = _mean_or_nan(labels)

            logistic_probabilities = subset_frame["probability"].to_numpy(
                dtype="float64", copy=False
            )
            constant_probabilities = subset_frame[
                "constant_probability"
            ].to_numpy(dtype="float64", copy=False)
            persistence_probabilities = subset_frame[
                "_persistence_probability"
            ].to_numpy(dtype="float64", copy=False)

            brier_logistic = _hierarchical_brier(subset_frame, "probability")
            brier_constant = _hierarchical_brier(
                subset_frame, "constant_probability"
            )
            brier_persistence = _hierarchical_brier(
                subset_frame, "_persistence_probability"
            )

            (
                paired_difference,
                paired_ci_low,
                paired_ci_high,
                paired_n_environments,
                paired_available,
            ) = _paired_difference(subset_frame)

            summary_rows.append(
                {
                    "dataset": dataset,
                    "subset": subset,
                    **counts,
                    "positives": positive_count,
                    "mean_observed": mean_observed,
                    "brier_logistic": brier_logistic,
                    "brier_constant": brier_constant,
                    "brier_persistence": brier_persistence,
                    "mean_predicted_logistic": _mean_or_nan(logistic_probabilities),
                    "mean_predicted_constant": _mean_or_nan(constant_probabilities),
                    "mean_predicted_persistence": _mean_or_nan(
                        persistence_probabilities
                    ),
                    "logistic_minus_persistence": paired_difference,
                    "logistic_minus_persistence_ci95_low": paired_ci_low,
                    "logistic_minus_persistence_ci95_high": paired_ci_high,
                    "paired_n_environments": paired_n_environments,
                    "paired_bootstrap_available": paired_available,
                }
            )

            paired_details[dataset_key][subset] = {
                "logistic_minus_persistence": _finite_or_none(paired_difference),
                "ci95_low": _finite_or_none(paired_ci_low),
                "ci95_high": _finite_or_none(paired_ci_high),
                "n_environments": paired_n_environments,
                "bootstrap_available": paired_available,
            }

    summary = pandas.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS)

    params: dict[str, Any] = dict(persistence_params)
    params.update(
        {
            "record_key": ["environment_id", "episode_uuid"],
            "subset_definitions": {
                "all": "all validation rows",
                "currently_good": "validation rows with current_bad == False",
                "currently_bad": "validation rows with current_bad == True",
            },
            "brier_aggregation": (
                "mean squared error within record; equal weight per record "
                "within environment; equal weight per environment"
            ),
            "bootstrap": {
                "unit": "environment",
                "method": "percentile",
                "confidence_level": _CONFIDENCE_LEVEL,
                "replicates": _BOOTSTRAP_REPLICATES,
                "seed": _BOOTSTRAP_SEED,
                "single_environment": "unavailable",
            },
            "paired_metric": "brier_logistic - brier_persistence",
            "validation_counts_by_dataset": validation_counts_by_dataset,
            "paired_differences": paired_details,
        }
    )

    out_path = pathlib.Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        out_path / "q1_quality_summary.csv",
        index=False,
    )
    (out_path / "q1_quality_params.json").write_text(
        json.dumps(
            _json_safe(params),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
