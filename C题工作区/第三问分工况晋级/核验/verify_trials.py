#!/usr/bin/env python3
"""Standalone verifier for Q3 trials (official + extra).

Usage:
    python -B verify_trials.py --trials trials.csv --output verify.json

Only numpy and pandas are used.  The script reads records, validates the
schema and the exact seed/case/method grid, computes per-case means and
paired changes, runs a fixed-seed paired bootstrap, evaluates the gates and
writes a concise JSON report.  It never runs simulations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "case_id",
    "seed",
    "split",
    "method",
    "weighted_on_time_rate",
    "weighted_completion_rate",
    "weighted_tardiness_s",
    "cost",
    "observations",
    "invalid_actions",
]

NUMERIC_COLUMNS = [
    "weighted_on_time_rate",
    "weighted_completion_rate",
    "weighted_tardiness_s",
    "cost",
    "observations",
    "invalid_actions",
]

METHODS = ("old", "new")
SPLITS = ("official", "extra")

OFFICIAL_SEEDS = list(range(2026111700, 2026111730))
EXTRA_SEEDS = list(range(73000, 73120))

# Positional limits for the three cases (S1/S2/S3 or A/B/C or 1/2/3).
COST_LIMITS = (32.0, 34.0, 40.0)
OBSERVATION_LIMITS = (4, 6, 8)

# Official gates, applied to the new method of the corresponding case.
OFFICIAL_GATES = {
    0: {"name": "S1", "on_time": 0.94, "completion": 1.0},
    1: {"name": "S2", "on_time": 0.82, "completion": None},
    2: {"name": "S3", "on_time": 0.825, "completion": 0.90},
}

EPS = 1e-9
METRIC_OUT = {
    "weighted_on_time_rate": "on_time_rate",
    "weighted_completion_rate": "completion_rate",
    "weighted_tardiness_s": "tardiness_s",
    "cost": "cost",
    "observations": "observations",
    "invalid_actions": "invalid_actions",
}
CI_METRICS = [
    "weighted_on_time_rate",
    "weighted_completion_rate",
    "weighted_tardiness_s",
    "cost",
]


class SchemaError(Exception):
    """Structural problem with the input table."""


class DataError(Exception):
    """The input table is structurally valid but incomplete/inconsistent."""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def natural_key(value: str) -> Tuple[object, ...]:
    """Sort labels such as S1, S2, case10, A naturally."""
    parts = re.split(r"(\d+)", str(value))
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Q3 trial records and evaluate official/extra gates. "
            "Reads a CSV or CSV.GZ table; never runs simulations."
        )
    )
    parser.add_argument(
        "--trials",
        required=True,
        help="Input CSV or .csv.gz with the required trial columns.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSON report.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Paired bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=20261122,
        help="Fixed RNG seed for the paired bootstrap (default: 20261122).",
    )
    return parser.parse_args(argv)


def load_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise SchemaError(f"input file does not exist: {path}")
    try:
        df = pd.read_csv(path, compression="infer", encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001 - report any CSV/compression error
        raise SchemaError(f"could not read CSV: {exc}") from exc

    if df is None or len(df.columns) == 0:
        raise SchemaError("input table has no columns")
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError("missing required column(s): " + ", ".join(missing))
    if len(df) == 0:
        raise SchemaError("input table has no rows")
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate column types/values; return a normalized copy."""
    df = df.copy()

    for col in ("case_id", "split", "method"):
        series = df[col].astype("string").str.strip()
        if series.isna().any() or (series == "").any():
            raise SchemaError(f"{col} contains missing/empty values")
        df[col] = series.astype(str)

    df["split"] = df["split"].str.lower()
    bad_splits = sorted(set(df["split"]) - set(SPLITS))
    if bad_splits:
        raise SchemaError("invalid split value(s): " + ", ".join(bad_splits))

    df["method"] = df["method"].str.lower()
    bad_methods = sorted(set(df["method"]) - set(METHODS))
    if bad_methods:
        raise SchemaError("invalid method value(s): " + ", ".join(bad_methods))

    seed = pd.to_numeric(df["seed"], errors="coerce")
    if seed.isna().any():
        raise SchemaError("seed contains missing or non-numeric values")
    if not np.isfinite(seed.to_numpy(dtype=float)).all():
        raise SchemaError("seed contains non-finite values")
    seed_float = seed.to_numpy(dtype=float)
    if not np.equal(seed_float, np.floor(seed_float)).all():
        raise SchemaError("seed values must be integers")
    df["seed"] = seed_float.astype(np.int64)

    for col in NUMERIC_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            raise SchemaError(f"{col} contains missing or non-numeric values")
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise SchemaError(f"{col} contains non-finite values")
        df[col] = values

    for col in ("weighted_on_time_rate", "weighted_completion_rate"):
        values = df[col].to_numpy(dtype=float)
        if ((values < -EPS) | (values > 1.0 + EPS)).any():
            raise SchemaError(f"{col} must lie in [0, 1]")

    for col in ("weighted_tardiness_s", "cost", "observations", "invalid_actions"):
        values = df[col].to_numpy(dtype=float)
        if (values < -EPS).any():
            raise SchemaError(f"{col} must be non-negative")

    for col in ("observations", "invalid_actions"):
        values = df[col].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise SchemaError(f"{col} values must be integers")
        df[col] = values.astype(np.int64)

    return df


def validate_grid(df: pd.DataFrame) -> List[str]:
    """Check uniqueness and the exact case/split/seed/method grid."""
    key_cols = ["case_id", "split", "seed", "method"]
    duplicated = df.duplicated(subset=key_cols, keep=False)
    if duplicated.any():
        dup = df.loc[duplicated, key_cols].head(8).to_dict("records")
        raise DataError(
            f"duplicate case/split/seed/method rows ({int(duplicated.sum())}); "
            f"examples: {dup}"
        )

    cases = sorted(df["case_id"].unique().tolist(), key=natural_key)
    if cases != ["S1", "S2", "S3"]:
        raise DataError(f"expected S1, S2, S3; found {cases}")

    splits = set(df["split"].unique().tolist())
    if splits != set(SPLITS):
        raise DataError(
            "expected both splits official and extra; found: "
            + ", ".join(sorted(splits))
        )

    expected_seed_sets = {
        "official": set(OFFICIAL_SEEDS),
        "extra": set(EXTRA_SEEDS),
    }
    for split in SPLITS:
        actual = set(df.loc[df["split"] == split, "seed"].unique().tolist())
        expected = expected_seed_sets[split]
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise DataError(
                f"{split} seed grid mismatch: missing={len(missing)}"
                f"{' ' + str(missing[:5]) if missing else ''}, "
                f"unexpected={len(unexpected)}"
                f"{' ' + str(unexpected[:5]) if unexpected else ''}"
            )

    actual_keys = set(
        map(tuple, df[key_cols].itertuples(index=False, name=None))
    )
    expected_keys = set()
    for split in SPLITS:
        for case in cases:
            for seed in expected_seed_sets[split]:
                for method in METHODS:
                    expected_keys.add((case, split, seed, method))

    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys
    if missing_keys or unexpected_keys:
        raise DataError(
            "case/split/seed/method grid mismatch: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )

    expected_rows = sum(len(expected_seed_sets[s]) for s in SPLITS) * len(cases) * len(METHODS)
    if len(df) != expected_rows:
        raise DataError(f"expected {expected_rows} rows, found {len(df)}")

    return cases


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def method_means(df: pd.DataFrame, case: str, split: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        block = df[
            (df["case_id"] == case)
            & (df["split"] == split)
            & (df["method"] == method)
        ]
        out[method] = {
            METRIC_OUT[col]: round_float(block[col].mean())
            for col in [
                "weighted_on_time_rate",
                "weighted_completion_rate",
                "weighted_tardiness_s",
                "cost",
                "observations",
                "invalid_actions",
            ]
        }
    return out


def paired_diff_vectors(
    df: pd.DataFrame,
    case: str,
    split: str,
    metrics: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return the common ordered seeds and new-minus-old vectors."""
    old = df[
        (df["case_id"] == case)
        & (df["split"] == split)
        & (df["method"] == "old")
    ].set_index("seed").sort_index()
    new = df[
        (df["case_id"] == case)
        & (df["split"] == split)
        & (df["method"] == "new")
    ].set_index("seed").sort_index()

    seeds = np.array(sorted(set(old.index) & set(new.index)), dtype=np.int64)
    if len(seeds) == 0:
        raise DataError(f"no paired seeds for case={case!r}, split={split!r}")
    if len(seeds) != len(old.index) or len(seeds) != len(new.index):
        raise DataError(f"unpaired rows for case={case!r}, split={split!r}")

    diffs = {
        metric: (
            new.loc[seeds, metric].to_numpy(dtype=float)
            - old.loc[seeds, metric].to_numpy(dtype=float)
        )
        for metric in metrics
    }
    return seeds, diffs


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> List[float]:
    """Percentile paired bootstrap CI for the mean."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [None, None]  # type: ignore[list-item]
    if n_boot <= 0 or values.size < 2:
        mean = float(values.mean())
        return [round_float(mean), round_float(mean)]
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round_float(lo), round_float(hi)]


def paired_block(
    df: pd.DataFrame,
    case: str,
    split: str,
    rng: np.random.Generator,
    n_boot: int,
) -> Dict[str, object]:
    seeds, diffs = paired_diff_vectors(df, case, split, CI_METRICS)
    ci = {
        METRIC_OUT[metric]: bootstrap_mean_ci(diffs[metric], rng, n_boot)
        for metric in CI_METRICS
    }
    return {
        "paired_seeds": int(len(seeds)),
        "mean_delta": {
            METRIC_OUT[metric]: round_float(diffs[metric].mean())
            for metric in CI_METRICS
        },
        "bootstrap_ci95": ci,
    }


def extra_overall_ontime_delta(
    df: pd.DataFrame,
    cases: Sequence[str],
    rng: np.random.Generator,
    n_boot: int,
) -> Dict[str, object]:
    per_seed_rows: List[np.ndarray] = []
    seeds_ref: np.ndarray | None = None
    for case in cases:
        seeds, diffs = paired_diff_vectors(
            df, case, "extra", ["weighted_on_time_rate"]
        )
        seeds_ref = seeds if seeds_ref is None else seeds_ref
        per_seed_rows.append(diffs["weighted_on_time_rate"])

    if seeds_ref is None:
        raise DataError("no extra rows found")
    matrix = np.vstack(per_seed_rows).T  # seeds x cases
    row_means = matrix.mean(axis=1)
    overall = float(matrix.mean())

    if n_boot > 0 and row_means.size >= 2:
        idx = rng.integers(0, row_means.size, size=(n_boot, row_means.size))
        boot = row_means[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ci = [round_float(lo), round_float(hi)]
    else:
        ci = [round_float(overall), round_float(overall)]

    return {
        "mean_paired_on_time_delta": round_float(overall),
        "bootstrap_ci95": ci,
        "seeds": int(row_means.size),
        "cases": list(cases),
    }


def raw_mean(
    df: pd.DataFrame, case: str, split: str, method: str, column: str
) -> float:
    block = df[
        (df["case_id"] == case)
        & (df["split"] == split)
        & (df["method"] == method)
    ]
    return float(block[column].mean())


def build_report(
    df: pd.DataFrame,
    cases: Sequence[str],
    trials_path: str,
    n_boot: int,
    rng_seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(rng_seed)
    case_limits = {
        case: {
            "max_cost": COST_LIMITS[i],
            "max_observations": OBSERVATION_LIMITS[i],
        }
        for i, case in enumerate(cases)
    }

    cases_out: Dict[str, object] = {}
    for case in cases:
        case_block: Dict[str, object] = {}
        for split in SPLITS:
            means = method_means(df, case, split)
            change = {
                metric: round_float(means["new"][metric] - means["old"][metric])
                for metric in means["new"]
            }
            sub = df[(df["case_id"] == case) & (df["split"] == split)]
            case_block[split] = {
                "means": means,
                "change_new_minus_old": change,
                "paired": paired_block(df, case, split, rng, n_boot),
                "max_cost": round_float(sub["cost"].max()),
                "max_observations": int(sub["observations"].max()),
                "illegal_actions": int(sub["invalid_actions"].sum()),
                "rows": int(len(sub)),
            }
        cases_out[case] = case_block

    # ---------------- gates ----------------
    official_gates: Dict[str, object] = {}
    for i, case in enumerate(cases):
        spec = OFFICIAL_GATES[i]
        on_time = raw_mean(
            df, case, "official", "new", "weighted_on_time_rate"
        )
        completion = raw_mean(
            df, case, "official", "new", "weighted_completion_rate"
        )
        old_completion = raw_mean(
            df, case, "official", "old", "weighted_completion_rate"
        )
        paired_completion_delta = completion - old_completion

        failures: List[str] = []
        if on_time < spec["on_time"] - EPS:
            failures.append(
                f"on_time_rate {on_time:.6f} < {spec['on_time']}"
            )
        if spec["completion"] is not None:
            if completion < float(spec["completion"]) - EPS:
                failures.append(
                    f"completion_rate {completion:.6f} < {spec['completion']}"
                )
            completion_rule = f">= {spec['completion']}"
        else:
            if paired_completion_delta < -EPS:
                failures.append(
                    "completion_rate "
                    f"{completion:.6f} < paired old {old_completion:.6f}"
                )
            completion_rule = ">= paired old"

        official_gates[spec["name"]] = {
            "case": case,
            "pass": not failures,
            "on_time_rate": round_float(on_time),
            "completion_rate": round_float(completion),
            "old_completion_rate": round_float(old_completion),
            "paired_completion_delta": round_float(paired_completion_delta),
            "thresholds": {
                "on_time_rate": spec["on_time"],
                "completion_rate": spec["completion"]
                if spec["completion"] is not None
                else completion_rule,
            },
            "bootstrap_ci95_completion_delta": (
                cases_out[case]["official"]["paired"]["bootstrap_ci95"][
                    "completion_rate"
                ]
            ),
            "failures": failures,
        }

    extra = extra_overall_ontime_delta(df, cases, rng, n_boot)
    per_case_extra = {
        case: raw_mean(df, case, "extra", "new", "weighted_on_time_rate")
              - raw_mean(df, case, "extra", "old", "weighted_on_time_rate")
        for case in cases
    }
    extra_gate = {
        "pass": all(value >= -EPS for value in per_case_extra.values()),
        "per_case_delta": per_case_extra,
        "mean_paired_on_time_delta": extra["mean_paired_on_time_delta"],
        "bootstrap_ci95": extra["bootstrap_ci95"],
        "seeds": extra["seeds"],
    }

    record_gates: Dict[str, object] = {}
    per_case_records: Dict[str, object] = {}
    illegal_total = int(df["invalid_actions"].sum())
    for i, case in enumerate(cases):
        block = df[df["case_id"] == case]
        max_cost = float(block["cost"].max())
        max_obs = int(block["observations"].max())
        case_illegal = int(block["invalid_actions"].sum())
        per_case_records[case] = {
            "max_cost": round_float(max_cost),
            "cost_limit": COST_LIMITS[i],
            "cost_pass": max_cost <= COST_LIMITS[i] + EPS,
            "max_observations": max_obs,
            "observation_limit": OBSERVATION_LIMITS[i],
            "observation_pass": max_obs <= OBSERVATION_LIMITS[i],
            "illegal_actions": case_illegal,
        }
    record_gates["no_illegal_actions"] = {
        "pass": illegal_total == 0,
        "total_illegal_actions": illegal_total,
        "per_case": {
            case: int(df.loc[df["case_id"] == case, "invalid_actions"].sum())
            for case in cases
        },
    }
    record_gates["cost_limits"] = {
        "pass": all(v["cost_pass"] for v in per_case_records.values()),
        "per_case": {
            case: {
                "max_cost": v["max_cost"],
                "limit": v["cost_limit"],
                "pass": v["cost_pass"],
            }
            for case, v in per_case_records.items()
        },
    }
    record_gates["observation_limits"] = {
        "pass": all(v["observation_pass"] for v in per_case_records.values()),
        "per_case": {
            case: {
                "max_observations": v["max_observations"],
                "limit": v["observation_limit"],
                "pass": v["observation_pass"],
            }
            for case, v in per_case_records.items()
        },
    }

    all_gate_values = [g["pass"] for g in official_gates.values()]
    all_gate_values.append(bool(extra_gate["pass"]))
    all_gate_values.extend(bool(g["pass"]) for g in record_gates.values())
    overall = all(all_gate_values)

    return {
        "status": "pass" if overall else "fail",
        "overall_pass": overall,
        "trials": trials_path,
        "bootstrap": {
            "method": "paired percentile bootstrap over seeds",
            "samples": n_boot,
            "rng_seed": rng_seed,
            "ci": 0.95,
        },
        "counts": {
            "rows": int(len(df)),
            "official": int((df["split"] == "official").sum()),
            "extra": int((df["split"] == "extra").sum()),
            "official_seeds": len(OFFICIAL_SEEDS),
            "extra_seeds": len(EXTRA_SEEDS),
            "cases": list(cases),
            "methods": list(METHODS),
        },
        "case_order": list(cases),
        "case_limits": case_limits,
        "cases": cases_out,
        "gates": {
            "official": official_gates,
            "extra": {"mean_paired_on_time_delta": extra_gate},
            "records": record_gates,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_samples < 0:
        eprint("ERROR: --bootstrap-samples must be non-negative")
        return 2

    try:
        df = load_table(args.trials)
        df = validate_schema(df)
        cases = validate_grid(df)
    except SchemaError as exc:
        eprint(f"ERROR: invalid schema: {exc}")
        return 2
    except DataError as exc:
        eprint(f"ERROR: incomplete data: {exc}")
        return 1

    report = build_report(
        df, cases, args.trials, args.bootstrap_samples, args.rng_seed
    )

    try:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except Exception as exc:  # noqa: BLE001 - surface any write error
        eprint(f"ERROR: could not write output JSON: {exc}")
        return 2

    if not report["overall_pass"]:
        failed = []
        for name, gate in report["gates"]["official"].items():
            if not gate["pass"]:
                failed.append(name)
        if not report["gates"]["extra"]["mean_paired_on_time_delta"]["pass"]:
            failed.append("extra_mean_paired_on_time_delta")
        for name, gate in report["gates"]["records"].items():
            if not gate["pass"]:
                failed.append(name)
        eprint("GATES FAILED: " + ", ".join(failed))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())



