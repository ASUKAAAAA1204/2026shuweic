#!/usr/bin/env python3
"""Read-only verifier for nominal tier3/tier4 release-replay trials.

Usage:
    python -B -X utf8 verify_nominal_replay.py TRIALS.csv --conditions conditions.json

TRIALS.csv must hold exactly one row for every
split x case x seed x release-version key, with
  split    in {official, promotion}
  case_id  in {S1, S2, S3}
  version  in {tier3, tier4}
  promotion seeds exactly 7_832_000..7_834_047
  official seeds exactly conditions["validation_seeds"]

Every numeric field must be finite, valid_protocol must be a true Boolean,
invalid_actions must be zero, and the frozen cases S1/S2 must have identical
score, cost and review counts between tier3 and tier4 for every seed.

The verifier never writes files and never runs simulations.  It prints one
compact JSON object to stdout and exits 0 on success, 1 on failed checks, and
2 when the inputs cannot be read or interpreted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

PROMOTION_START = 7_832_000
PROMOTION_STOP = 7_834_048  # exclusive
PROMOTION_COUNT = PROMOTION_STOP - PROMOTION_START

SPLITS = ("official", "promotion")
CASES = ("S1", "S2", "S3")
VERSIONS = ("tier3", "tier4")
FROZEN_CASES = ("S1", "S2")

REQUIRED_COLUMNS = (
    "split",
    "case_id",
    "seed",
    "version",
    "weighted_on_time_rate",
    "weighted_completion_rate",
    "weighted_tardiness_s",
    "cost",
    "invalid_actions",
    "observations",
    "valid_protocol",
)

NUMERIC_COLUMNS = (
    "weighted_on_time_rate",
    "weighted_completion_rate",
    "weighted_tardiness_s",
    "cost",
    "invalid_actions",
    "observations",
)

# Frozen releases must reproduce scores, accumulated cost and paid reviews.
FROZEN_FIELDS = (
    "weighted_on_time_rate",
    "weighted_completion_rate",
    "weighted_tardiness_s",
    "cost",
    "observations",
)

TRUE_TEXT = frozenset({"true", "1", "1.0", "yes"})
FALSE_TEXT = frozenset({"false", "0", "0.0", "no"})


class InputError(Exception):
    """Input data cannot be read or interpreted."""


def _text(raw, name, line):
    value = raw[name].strip()
    if not value:
        raise InputError(f"line {line}: {name} is empty")
    return value


def _number(raw, name, line):
    text = _text(raw, name, line)
    try:
        value = float(text)
    except ValueError as exc:
        raise InputError(f"line {line}: {name} is not numeric: {text!r}") from exc
    if not math.isfinite(value):
        raise InputError(f"line {line}: {name} is not finite")
    return value


def _boolean(raw, name, line):
    text = _text(raw, name, line).lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise InputError(f"line {line}: {name} must be a Boolean")


def _seed(value, source):
    if isinstance(value, bool):
        raise InputError(f"{source}: a Boolean is not a seed")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise InputError(f"{source}: seed must be a finite integer")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError as exc:
            raise InputError(f"{source}: seed is not numeric: {text!r}") from exc
        if not math.isfinite(number) or not number.is_integer():
            raise InputError(f"{source}: seed must be a finite integer")
        return int(number)
    raise InputError(f"{source}: unsupported seed value {value!r}")


def load_official_seeds(path):
    if not path.is_file():
        raise InputError(f"conditions file not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read conditions JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputError("conditions JSON must contain an object")
    if "validation_seeds" not in payload:
        raise InputError('conditions JSON lacks the "validation_seeds" key')
    raw = payload["validation_seeds"]
    if not isinstance(raw, list) or not raw:
        raise InputError("validation_seeds must be a non-empty list")
    seeds = tuple(_seed(value, "validation_seeds") for value in raw)
    if len(set(seeds)) != len(seeds):
        raise InputError("validation_seeds contains duplicates")
    return seeds


def _parse_record(raw, line):
    split = _text(raw, "split", line).lower()
    if split not in SPLITS:
        raise InputError(f"line {line}: split must be official or promotion")
    case_id = _text(raw, "case_id", line).upper()
    if case_id not in CASES:
        raise InputError(f"line {line}: case_id must be S1, S2 or S3")
    version = _text(raw, "version", line).lower()
    if version not in VERSIONS:
        raise InputError(f"line {line}: version must be tier3 or tier4")
    seed = _number(raw, "seed", line)
    if not seed.is_integer():
        raise InputError(f"line {line}: seed must be an integer")
    record = {
        "split": split,
        "case_id": case_id,
        "seed": int(seed),
        "version": version,
    }
    for name in NUMERIC_COLUMNS:
        record[name] = _number(raw, name, line)
    record["valid_protocol"] = _boolean(raw, "valid_protocol", line)
    return record


def _open_text(path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def load_trials(path):
    if not path.is_file():
        raise InputError(f"trials CSV not found: {path}")
    records = {}
    try:
        with _open_text(path) as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise InputError("trials CSV is empty") from exc
            header = [field.strip() for field in header]
            if len(set(header)) != len(header):
                raise InputError("trials CSV has duplicate column names")
            missing = [name for name in REQUIRED_COLUMNS if name not in header]
            if missing:
                raise InputError("missing required column(s): " + ", ".join(missing))
            positions = {name: header.index(name) for name in REQUIRED_COLUMNS}
            for line, fields in enumerate(reader, start=2):
                if not fields:
                    continue
                if len(fields) != len(header):
                    raise InputError(
                        f"line {line}: expected {len(header)} columns, found {len(fields)}"
                    )
                raw = {name: fields[positions[name]] for name in REQUIRED_COLUMNS}
                record = _parse_record(raw, line)
                key = (
                    record["split"],
                    record["case_id"],
                    record["seed"],
                    record["version"],
                )
                if key in records:
                    raise InputError(f"line {line}: duplicate key {_format_key(key)}")
                records[key] = record
    except (OSError, EOFError, csv.Error, UnicodeError) as exc:
        raise InputError(f"cannot read trials CSV: {exc}") from exc
    if not records:
        raise InputError("trials CSV has no data rows")
    return records


def _format_key(key):
    split, case_id, seed, version = key
    return f"{split}/{case_id}/{seed}/{version}"


def _preview(keys, limit=5):
    ordered = sorted(keys)
    shown = ", ".join(_format_key(key) for key in ordered[:limit])
    if len(ordered) > limit:
        shown += ", ..."
    return f"{len(ordered)} ({shown})"


def verify(records, official_seeds):
    expected_seeds = {
        "official": tuple(official_seeds),
        "promotion": range(PROMOTION_START, PROMOTION_STOP),
    }
    expected_keys = {
        (split, case_id, seed, version)
        for split in SPLITS
        for seed in expected_seeds[split]
        for case_id in CASES
        for version in VERSIONS
    }
    actual_keys = set(records)

    errors = []
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    if missing:
        errors.append("missing key(s): " + _preview(missing))
    if unexpected:
        errors.append("unexpected key(s): " + _preview(unexpected))

    protocol_failures = sum(
        1 for record in records.values() if not record["valid_protocol"]
    )
    if protocol_failures:
        errors.append(f"valid_protocol is false on {protocol_failures} row(s)")

    invalid_rows = [
        (key, record["invalid_actions"])
        for key, record in records.items()
        if record["invalid_actions"] != 0
    ]
    if invalid_rows:
        examples = ", ".join(
            f"{_format_key(key)}={value:g}"
            for key, value in sorted(invalid_rows)[:5]
        )
        errors.append(
            "invalid_actions must be zero on every row: "
            f"{len(invalid_rows)} row(s), e.g. {examples}"
        )

    mismatches = []
    for split in SPLITS:
        for seed in expected_seeds[split]:
            for case_id in FROZEN_CASES:
                left = records.get((split, case_id, seed, "tier3"))
                right = records.get((split, case_id, seed, "tier4"))
                if left is None or right is None:
                    continue
                for field in FROZEN_FIELDS:
                    if left[field] != right[field]:
                        mismatches.append(
                            (split, case_id, seed, field, left[field], right[field])
                        )
    if mismatches:
        split_seed_count = len({(item[0], item[2]) for item in mismatches})
        fields = sorted({item[3] for item in mismatches})
        first = mismatches[0]
        errors.append(
            f"S1/S2 releases differ on {len(mismatches)} value(s) across "
            f"{split_seed_count} split-seed pair(s); fields={fields}; "
            f"first={first[0]}/{first[1]}/{first[2]}/{first[3]} "
            f"({first[4]!r} vs {first[5]!r})"
        )

    stats = {
        "rows": len(records),
        "expected_rows": len(expected_keys),
        "official_seeds": len(official_seeds),
        "promotion_seeds": PROMOTION_COUNT,
        "invalid_actions": sum(
            record["invalid_actions"] for record in records.values()
        ),
        "protocol_failures": protocol_failures,
        "frozen_mismatches": len(mismatches),
    }
    return errors, stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify a nominal tier3/tier4 release replay CSV (read-only).",
        epilog="Prints one JSON object to stdout and writes no files.",
    )
    parser.add_argument("trials", type=Path, help="trials CSV (optionally .csv.gz)")
    parser.add_argument(
        "--conditions",
        type=Path,
        required=True,
        help='JSON file whose top-level "validation_seeds" lists official seeds',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        official_seeds = load_official_seeds(args.conditions)
        records = load_trials(args.trials)
    except InputError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 2

    errors, stats = verify(records, official_seeds)
    if errors:
        print(
            json.dumps(
                {"passed": False, **stats, "errors": errors},
                ensure_ascii=False,
            )
        )
        return 1

    report = {
        "passed": True,
        "rows": stats["rows"],
        "official_seeds": stats["official_seeds"],
        "promotion_seeds": stats["promotion_seeds"],
        "cases": list(CASES),
        "versions": list(VERSIONS),
        "invalid_actions": 0,
        "frozen_cases": list(FROZEN_CASES),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
