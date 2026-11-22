#!/usr/bin/env python3
"""Independent gate for one nominal S3 paired-trial CSV."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = 2048
METHODS = ("tier3", "candidate")
SEED_BASE = 7_820_000
ROUND_STEP = 3_000
RESAMPLES = 10_000
CHUNK = 500
RNG_SEED = 557_221
COLUMNS = (
    "seed", "method", "weighted_on_time_rate", "weighted_completion_rate",
    "weighted_tardiness_s", "cost", "invalid_actions", "observations",
    "valid_protocol",
)
NUMERIC = COLUMNS[2:8]
TRUE_VALUES = {"true", "1", "1.0", "yes"}


class InvalidData(ValueError):
    pass


def fail(message):
    raise InvalidData(message)


def require_finite(frame, name):
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        fail(f"{name} contains a missing or non-finite value")
    return values


def load(path, round_number):
    try:
        frame = pd.read_csv(
            path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        fail(f"cannot read CSV: {exc}")

    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        fail("missing columns: " + ", ".join(missing))
    expected_rows = N * len(METHODS)
    if len(frame) != expected_rows:
        fail(f"expected {expected_rows} rows, found {len(frame)}")

    frame["method"] = frame["method"].str.strip()
    if set(frame["method"]) != set(METHODS):
        fail("methods must be exactly tier3 and candidate")

    seed_values = pd.to_numeric(frame["seed"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(seed_values).all() or not np.equal(
        seed_values, np.floor(seed_values)
    ).all():
        fail("seed contains a missing or non-integer value")
    frame["seed"] = seed_values.astype(np.int64)

    start = SEED_BASE + ROUND_STEP * (round_number - 1)
    expected_seeds = np.arange(start, start + N, dtype=np.int64)
    if (
        frame["seed"].nunique() != N
        or set(frame["seed"].tolist()) != set(expected_seeds.tolist())
    ):
        fail(f"seeds must be distinct consecutive values {start}..{start + N - 1}")
    if frame.duplicated(["seed", "method"]).any():
        fail("duplicate seed/method rows")
    if any((frame["method"] == method).sum() != N for method in METHODS):
        fail("full method grid is required")

    protocol = frame["valid_protocol"].str.strip().str.lower()
    if protocol.eq("").any() or not set(protocol).issubset(TRUE_VALUES):
        fail("valid_protocol must be true on every record")

    values = {name: require_finite(frame, name) for name in NUMERIC}
    for name in ("weighted_on_time_rate","weighted_completion_rate"):
        if ((values[name] < 0) | (values[name] > 1)).any():
            fail(f"{name} must be in [0, 1]")
    if (values["weighted_on_time_rate"] > values["weighted_completion_rate"] + 1e-12).any():
        fail("on-time rate exceeds completion rate")
    if (values["invalid_actions"] != 0).any():
        fail("invalid_actions must be zero")
    if (values["cost"] < 0).any() or (values["cost"] > 40 + 1e-9).any():
        fail("cost must be between 0 and 40")
    if (values["observations"] < 0).any() or (values["observations"] > 8).any():
        fail("observations must be between 0 and 8")
    if (values["weighted_tardiness_s"] < 0).any():
        fail("weighted_tardiness_s cannot be negative")

    table = pd.DataFrame({"seed": frame["seed"], "method": frame["method"]})
    for name, column in values.items():
        table[name] = column
    tiers = {}
    for method in METHODS:
        part = table[table["method"] == method].set_index("seed").sort_index()
        if part.index.duplicated().any() or not np.array_equal(
            part.index.to_numpy(), expected_seeds
        ):
            fail(f"full method grid is required for {method}")
        tiers[method] = part
    return tiers, start


def bootstrap(candidate_rate, delta_rate):
    rng = np.random.default_rng(RNG_SEED)
    absolute_means = []
    delta_means = []
    done = 0
    while done < RESAMPLES:
        size = min(CHUNK, RESAMPLES - done)
        indices = rng.integers(0, N, size=(size, N))
        absolute_means.append(candidate_rate[indices].mean(axis=1))
        delta_means.append(delta_rate[indices].mean(axis=1))
        done += size
    absolute_means = np.concatenate(absolute_means)
    delta_means = np.concatenate(delta_means)
    absolute_bounds = np.percentile(absolute_means, [2.5, 97.5])
    delta_bounds = np.percentile(delta_means, [2.5, 97.5])
    return (
        float(absolute_bounds[0]),
        float(absolute_bounds[1]),
        float(delta_bounds[0]),
        float(delta_bounds[1]),
    )


def audit(csv_path, round_number):
    tiers, start = load(csv_path, round_number)
    tier3 = tiers["tier3"]
    candidate = tiers["candidate"]

    tier3_ontime = tier3["weighted_on_time_rate"].to_numpy(dtype=float)
    candidate_ontime = candidate["weighted_on_time_rate"].to_numpy(dtype=float)
    tier3_completion = tier3["weighted_completion_rate"].to_numpy(dtype=float)
    candidate_completion = candidate["weighted_completion_rate"].to_numpy(dtype=float)
    ontime_delta = candidate_ontime - tier3_ontime
    completion_delta = candidate_completion - tier3_completion

    absolute_low, absolute_high, gain_low, gain_high = bootstrap(
        candidate_ontime, ontime_delta
    )
    tier3_mean = float(tier3_ontime.mean())
    candidate_mean = float(candidate_ontime.mean())
    gain = float(ontime_delta.mean())
    tier3_completion_mean = float(tier3_completion.mean())
    candidate_completion_mean = float(candidate_completion.mean())
    completion_gain = float(completion_delta.mean())
    passed = bool(
        absolute_low >= 0.85 and gain_low > 0 and completion_gain >= 0
    )

    return {
        "round": round_number, "start": start, "count": N, "passed": passed,
        "on_time": {"tier3": tier3_mean, "candidate": candidate_mean,
                    "candidate_ci95": [absolute_low, absolute_high], "gain": gain,
                    "gain_ci95": [gain_low, gain_high]},
        "completion": {"tier3": tier3_completion_mean, "candidate": candidate_completion_mean,
                       "gain": completion_gain},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one nominal S3 paired-trial CSV."
    )
    parser.add_argument("csv", type=Path, help="paired trial CSV")
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    parser.add_argument("--output", type=Path, required=True, help="JSON audit path")
    args = parser.parse_args(argv)
    if args.round_number < 1:
        parser.error("--round must be a positive integer")
    if args.output.resolve() == args.csv.resolve():
        parser.error("--output must differ from the input CSV")

    try:
        report = audit(args.csv, args.round_number)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (InvalidData, OSError, ValueError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"passed={report['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
