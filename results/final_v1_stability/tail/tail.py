#!/usr/bin/env python3
"""Equal-window tail audit for frozen v1.0.0-final B / B_K64 results.

Descriptive statistics only. Inputs are read-only gzip CSV archives; no model is
loaded, trained, modified, or run. All outputs are written to this directory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE
ROOT = HERE.parents[2]
if not (ROOT / "results" / "final_v1_sensitivity").is_dir():
    raise SystemExit(f"workspace root not found from {HERE}")

SOURCES = {
    "q1_dev": "results/final_v1_sensitivity/q1_history_windows.csv.gz",
    "q2_dev": "results/final_v1_sensitivity/q2/per_window.csv.gz",
    "q1_arch": "tmp/final_v1_20261122/C题工作区/正式方案/results/q1_windows.csv.gz",
    "q2_arch": "tmp/final_v1_20261122/C题工作区/正式方案/results/q2_windows.csv.gz",
    "q3_arch": "tmp/final_v1_20261122/C题工作区/正式方案/results/q3/b_k64/trials.csv.gz",
}
OUTPUT_FILES = (
    "q1_summary.csv",
    "q2_summary.csv",
    "q3_summary.csv",
    "q3_worst3.csv",
    "paired_tail.csv",
    "validation.json",
)
Q1_HISTORIES = ("full", "0.75s", "0.30s")
HORIZONS = ("0.2", "0.6", "1.0")
Q2_DELAYS = ("0.0", "0.1", "0.2")
FORBIDDEN_PARTS = {".git", ".aws", ".env", "secrets"}
TOL = 1e-12


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile(values, p):
    """Linear empirical quantile (Hyndman-Fan type 7)."""
    xs = sorted(values)
    if not xs:
        raise ValueError("empty sample")
    if len(xs) == 1:
        return xs[0]
    h = (len(xs) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    return xs[lo] if lo == hi else xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def mean(values):
    return sum(values) / len(values)


def nade_block(values):
    """Equal-window NADE summary; worst decile = ceil(0.1*n) largest values."""
    n = len(values)
    k = math.ceil(0.1 * n)
    return {
        "n_windows": n,
        "nade_median": quantile(values, 0.5),
        "nade_mean": mean(values),
        "nade_p90": quantile(values, 0.9),
        "nade_p95": quantile(values, 0.95),
        "worst_decile_n": k,
        "worst_decile_mean_nade": mean(sorted(values, reverse=True)[:k]),
    }


def key_map(rows, keys, value):
    """Map a tuple key to a float value; return (map, duplicate-row count)."""
    out = {}
    duplicates = 0
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in out:
            duplicates += 1
        out[key] = float(row[value])
    return out, duplicates


def trial_id(row):
    return "|".join(
        (row["case_id"], row["seed"], row["alpha"], row["beta"], row["method"])
    )


def fmt(value):
    return repr(value) if isinstance(value, float) else value


def write_csv(name, fields, rows):
    with (OUTPUT_DIR / name).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in fields})


def record_check(checks, failures, name, passed, **detail):
    checks[name] = {"passed": bool(passed), **detail}
    if not passed:
        failures.append(name)


def pair_tail(comparison, nominal_label, stress_label, nominal_rows, stress_rows):
    """Paired tail on the nominal worst decile, ties broken by window_id."""
    nominal, nominal_dup = key_map(nominal_rows, ("window_id",), "nade")
    stress, stress_dup = key_map(stress_rows, ("window_id",), "nade")
    if nominal_dup or stress_dup:
        raise ValueError(f"duplicate window IDs in paired comparison {comparison}")
    if set(nominal) != set(stress):
        raise ValueError(f"ID mismatch in paired comparison {comparison}")

    ids = sorted(nominal)
    k = math.ceil(0.1 * len(ids))
    baseline_tail = sorted(ids, key=lambda i: (-nominal[i], i))[:k]
    stress_tail = {
        i for i in sorted(ids, key=lambda i: (-stress[i], i))[:k]
    }
    overlap_count = sum(1 for i in baseline_tail if i in stress_tail)
    nominal_tail_mean = mean([nominal[i] for i in baseline_tail])
    stress_tail_mean = mean([stress[i] for i in baseline_tail])
    return {
        "comparison": comparison,
        "nominal": nominal_label,
        "stress": stress_label,
        "n_paired": len(ids),
        "tail_n": k,
        "nominal_tail_cutoff": min(nominal[i] for i in baseline_tail),
        "nominal_tail_mean": nominal_tail_mean,
        "stress_tail_mean": stress_tail_mean,
        "tail_change_mean": mean(
            [stress[i] - nominal[i] for i in baseline_tail]
        ),
        "overlap_count": overlap_count,
        "overlap_rate": overlap_count / k,
        "stress_tail_n": len(stress_tail),
    }


# ---------------------------------------------------------------- load inputs
source_paths = {key: ROOT / rel for key, rel in SOURCES.items()}
data = {key: read_gz(path) for key, path in source_paths.items()}
input_meta = [
    {
        "key": key,
        "path": rel,
        "bytes": source_paths[key].stat().st_size,
        "sha256": digest(source_paths[key]),
        "rows": len(data[key]),
    }
    for key, rel in SOURCES.items()
]

checks = {}
failures = []

forbidden_hits = [
    rel
    for rel in SOURCES.values()
    if any(part.lower() in FORBIDDEN_PARTS for part in Path(rel).parts)
]
record_check(
    checks,
    failures,
    "no_forbidden_or_secret_paths",
    not forbidden_hits,
    hits=forbidden_hits,
)
record_check(
    checks,
    failures,
    "outputs_cwd_only",
    all((OUTPUT_DIR / name).resolve().parent == OUTPUT_DIR for name in OUTPUT_FILES),
    directory=str(OUTPUT_DIR),
    files=list(OUTPUT_FILES),
)

# ------------------------------------------------------- Q1 equal-window NADE
q1_dev, q1_arch = data["q1_dev"], data["q1_arch"]
q1_dev_groups = {
    (hist, hz): [r for r in q1_dev if r["history"] == hist and r["horizon"] == hz]
    for hist in Q1_HISTORIES
    for hz in HORIZONS
}
q1_dev_detail = {
    f"{hist}|{hz}": {
        "rows": len(grp),
        "unique_window_ids": len({r["window_id"] for r in grp}),
    }
    for (hist, hz), grp in q1_dev_groups.items()
}
record_check(
    checks,
    failures,
    "q1_development_cardinality",
    len(q1_dev) == 1908
    and {(r["history"], r["horizon"]) for r in q1_dev}
    == {(h, hz) for h in Q1_HISTORIES for hz in HORIZONS}
    and all(
        d["rows"] == 212 and d["unique_window_ids"] == 212
        for d in q1_dev_detail.values()
    ),
    expected_rows=1908,
    expected_per_group=212,
    groups=q1_dev_detail,
)

q1_nominal_rows = [r for r in q1_dev if r["history"] == "full"]
q1_ref_rows = [
    r for r in q1_arch if r["split"] == "validation" and r["model"] == "B"
]
q1_nominal, q1_nominal_dup = key_map(
    q1_nominal_rows, ("window_id", "horizon"), "nade"
)
q1_ref, q1_ref_dup = key_map(q1_ref_rows, ("window_id", "horizon"), "nade")
q1_common = sorted(set(q1_nominal) & set(q1_ref))
q1_maxdiff = max(
    (abs(q1_nominal[k] - q1_ref[k]) for k in q1_common), default=None
)
record_check(
    checks,
    failures,
    "q1_nominal_development_vs_archived_B_validation",
    q1_nominal_dup == 0
    and q1_ref_dup == 0
    and set(q1_nominal) == set(q1_ref)
    and q1_maxdiff is not None
    and q1_maxdiff <= TOL,
    matched_by=["window_id", "horizon"],
    nominal_rows=len(q1_nominal_rows),
    archived_B_rows=len(q1_ref_rows),
    common_keys=len(q1_common),
    missing_from_archive=len(set(q1_nominal) - set(q1_ref)),
    extra_in_archive=len(set(q1_ref) - set(q1_nominal)),
    max_abs_nade_diff=q1_maxdiff,
    tolerance=TOL,
)

q1_test_b = [
    r for r in q1_arch if r["split"] == "test" and r["model"] == "B"
]
q1_test_by_horizon = {
    hz: [r for r in q1_test_b if r["horizon"] == hz] for hz in HORIZONS
}
q1_test_ids = {
    hz: {r["window_id"] for r in rows}
    for hz, rows in q1_test_by_horizon.items()
}
q1_test_dup = len(q1_test_b) - len(
    {(r["window_id"], r["horizon"]) for r in q1_test_b}
)
q1_test_detail = {
    hz: {"rows": len(rows), "unique_ids": len(q1_test_ids[hz])}
    for hz, rows in q1_test_by_horizon.items()
}
record_check(
    checks,
    failures,
    "q1_archived_final_test_B",
    len(q1_test_b) == 1170
    and all(
        d["rows"] == 390 and d["unique_ids"] == 390
        for d in q1_test_detail.values()
    )
    and q1_test_ids["0.2"] == q1_test_ids["0.6"] == q1_test_ids["1.0"]
    and q1_test_dup == 0
    and {r["model"] for r in q1_test_b} == {"B"}
    and {r["split"] for r in q1_test_b} == {"test"},
    expected_rows=1170,
    expected_ids_per_horizon=390,
    horizons=q1_test_detail,
    duplicate_window_horizon_rows=q1_test_dup,
)

# ------------------------------------------------------- Q2 equal-window NADE
q2_dev, q2_arch = data["q2_dev"], data["q2_arch"]
q2_dev_groups = {
    delay: [r for r in q2_dev if r["delay"] == delay] for delay in Q2_DELAYS
}
q2_envs = sorted({r["environment_id"] for r in q2_dev})
q2_env_groups = {
    (delay, env): [
        r
        for r in q2_dev
        if r["delay"] == delay and r["environment_id"] == env
    ]
    for delay in Q2_DELAYS
    for env in q2_envs
}
q2_dev_detail = {
    delay: {
        "rows": len(rows),
        "unique_window_ids": len({r["window_id"] for r in rows}),
    }
    for delay, rows in q2_dev_groups.items()
}
q2_env_counts = {
    env: {delay: len(q2_env_groups[(delay, env)]) for delay in Q2_DELAYS}
    for env in q2_envs
}
record_check(
    checks,
    failures,
    "q2_development_cardinality",
    len(q2_dev) == 636
    and {r["split"] for r in q2_dev} == {"validation"}
    and {r["delay"] for r in q2_dev} == set(Q2_DELAYS)
    and all(
        d["rows"] == 212 and d["unique_window_ids"] == 212
        for d in q2_dev_detail.values()
    )
    and all(
        all(count > 0 for count in counts.values())
        and len(set(counts.values())) == 1
        for counts in q2_env_counts.values()
    )
    and all(
        sum(q2_env_counts[env][delay] for env in q2_envs) == 212
        for delay in Q2_DELAYS
    ),
    expected_rows=636,
    expected_per_delay=212,
    delays=q2_dev_detail,
    environments=q2_env_counts,
)

q2_nominal_rows = q2_dev_groups["0.1"]
q2_ref_rows = [
    r for r in q2_arch if r["split"] == "validation" and r["model"] == "B"
]
q2_nominal, q2_nominal_dup = key_map(q2_nominal_rows, ("window_id",), "nade")
q2_ref, q2_ref_dup = key_map(q2_ref_rows, ("window_id",), "nade")
q2_common = sorted(set(q2_nominal) & set(q2_ref))
q2_maxdiff = max(
    (abs(q2_nominal[k] - q2_ref[k]) for k in q2_common), default=None
)
record_check(
    checks,
    failures,
    "q2_nominal_development_vs_archived_B_validation",
    q2_nominal_dup == 0
    and q2_ref_dup == 0
    and set(q2_nominal) == set(q2_ref)
    and q2_maxdiff is not None
    and q2_maxdiff <= TOL,
    matched_by=["window_id"],
    nominal_rows=len(q2_nominal_rows),
    archived_B_rows=len(q2_ref_rows),
    common_keys=len(q2_common),
    missing_from_archive=len(set(q2_nominal) - set(q2_ref)),
    extra_in_archive=len(set(q2_ref) - set(q2_nominal)),
    max_abs_nade_diff=q2_maxdiff,
    tolerance=TOL,
)

q1_dev_ids = {r["window_id"] for r in q1_nominal_rows}
q2_dev_ids = {r["window_id"] for r in q2_nominal_rows}
record_check(
    checks,
    failures,
    "q1_q2_development_window_ids_equal",
    q1_dev_ids == q2_dev_ids,
    q1_development_ids=len(q1_dev_ids),
    q2_development_ids=len(q2_dev_ids),
    common_ids=len(q1_dev_ids & q2_dev_ids),
)

q2_test_b = [
    r for r in q2_arch if r["split"] == "test" and r["model"] == "B"
]
q2_test_ids = {r["window_id"] for r in q2_test_b}
q2_test_dup = len(q2_test_b) - len(q2_test_ids)
record_check(
    checks,
    failures,
    "q2_archived_final_test_B",
    len(q2_test_b) == 390
    and len(q2_test_ids) == 390
    and q2_test_dup == 0
    and {r["model"] for r in q2_test_b} == {"B"}
    and {r["split"] for r in q2_test_b} == {"test"},
    expected_rows=390,
    unique_ids=len(q2_test_ids),
    duplicate_ids=q2_test_dup,
)
record_check(
    checks,
    failures,
    "q1_q2_test_window_ids_equal",
    q1_test_ids["1.0"] == q2_test_ids,
    q1_test_ids=len(q1_test_ids["1.0"]),
    q2_test_ids=len(q2_test_ids),
    common_ids=len(q1_test_ids["1.0"] & q2_test_ids),
)

# ---------------------------------------------------------------- Q3 B_K64
q3 = data["q3_arch"]
q3_cases = sorted({r["case_id"] for r in q3})
q3_mappings = sorted(
    {(r["alpha"], r["beta"]) for r in q3},
    key=lambda ab: (float(ab[0]), float(ab[1])),
)
q3_seeds = sorted({r["seed"] for r in q3}, key=int)
q3_group_counts = Counter(
    (r["case_id"], r["alpha"], r["beta"]) for r in q3
)
q3_trial_keys = {
    (r["case_id"], r["seed"], r["alpha"], r["beta"], r["method"]) for r in q3
}
q3_group_ok = set(q3_group_counts) == {
    (case, alpha, beta)
    for case in q3_cases
    for alpha, beta in q3_mappings
} and all(n == 30 for n in q3_group_counts.values())
record_check(
    checks,
    failures,
    "q3_B_K64_official_seed_cardinality",
    len(q3) == 450
    and {r["method"] for r in q3} == {"B_K64"}
    and len(q3_cases) == 3
    and len(q3_mappings) == 5
    and len(q3_seeds) == 30
    and q3_group_ok
    and len(q3_trial_keys) == 450
    and all(r["valid_protocol"].strip().lower() == "true" for r in q3)
    and all(float(r["invalid_actions"]) == 0 for r in q3),
    cases=q3_cases,
    mappings=[list(ab) for ab in q3_mappings],
    seeds=q3_seeds,
    rows=len(q3),
    unique_trial_ids=len(q3_trial_keys),
    trials_per_case_mapping=30,
    valid_protocol_all=True,
)

# ---------------------------------------------------------------- Q1 outputs
q1_rows = []
for hist in Q1_HISTORIES:
    for hz in HORIZONS:
        values = [float(r["nade"]) for r in q1_dev_groups[(hist, hz)]]
        q1_rows.append(
            {
                "scope": "development",
                "history": hist,
                "horizon": hz,
                "source": SOURCES["q1_dev"],
                **nade_block(values),
            }
        )
for hz in HORIZONS:
    values = [float(r["nade"]) for r in q1_test_by_horizon[hz]]
    q1_rows.append(
        {
            "scope": "archived_final_test_B",
            "history": "nominal",
            "horizon": hz,
            "source": SOURCES["q1_arch"],
            **nade_block(values),
        }
    )

# ---------------------------------------------------------------- Q2 outputs
q2_rows = []
for delay in Q2_DELAYS:
    values = [float(r["nade"]) for r in q2_dev_groups[delay]]
    q2_rows.append(
        {
            "scope": "development",
            "level": "overall",
            "delay": delay,
            "environment_id": "ALL",
            "source": SOURCES["q2_dev"],
            **nade_block(values),
        }
    )
for env in q2_envs:
    for delay in Q2_DELAYS:
        values = [float(r["nade"]) for r in q2_env_groups[(delay, env)]]
        q2_rows.append(
            {
                "scope": "development",
                "level": "environment",
                "delay": delay,
                "environment_id": env,
                "source": SOURCES["q2_dev"],
                **nade_block(values),
            }
        )
q2_test_values = [float(r["nade"]) for r in q2_test_b]
q2_rows.append(
    {
        "scope": "archived_final_test_B",
        "level": "overall",
        "delay": "nominal",
        "environment_id": "ALL",
        "source": SOURCES["q2_arch"],
        **nade_block(q2_test_values),
    }
)

# ---------------------------------------------------------------- Q3 outputs
q3_rows = []
q3_worst_rows = []
for case in q3_cases:
    for alpha, beta in q3_mappings:
        grp = [
            r
            for r in q3
            if r["case_id"] == case
            and r["alpha"] == alpha
            and r["beta"] == beta
        ]
        on_time = [float(r["weighted_on_time_rate"]) for r in grp]
        completion = [float(r["weighted_completion_rate"]) for r in grp]
        tardiness = [float(r["weighted_tardiness_s"]) for r in grp]
        worst_n = math.ceil(0.1 * len(grp))
        ordered = sorted(
            grp,
            key=lambda r: (float(r["weighted_on_time_rate"]), trial_id(r)),
        )
        worst = ordered[:worst_n]
        selected_ids = {trial_id(r) for r in worst}
        cutoff = float(worst[-1]["weighted_on_time_rate"])
        cutoff_ties = sum(1 for value in on_time if value == cutoff)
        excluded_ties = sum(
            1
            for r in grp
            if float(r["weighted_on_time_rate"]) == cutoff
            and trial_id(r) not in selected_ids
        )
        q3_rows.append(
            {
                "case_id": case,
                "alpha": alpha,
                "beta": beta,
                "method": "B_K64",
                "n_trials": len(grp),
                "n_seeds": len({r["seed"] for r in grp}),
                "mean_on_time": mean(on_time),
                "on_time_p10": quantile(on_time, 0.1),
                "worst3_mean_on_time": mean(
                    [float(r["weighted_on_time_rate"]) for r in worst]
                ),
                "worst3_cutoff_on_time": cutoff,
                "worst3_cutoff_tie_count": cutoff_ties,
                "worst3_cutoff_excluded": excluded_ties,
                "mean_completion": mean(completion),
                "mean_tardiness": mean(tardiness),
                "worst3_trial_ids": ";".join(trial_id(r) for r in worst),
                "source": SOURCES["q3_arch"],
            }
        )
        for rank, row in enumerate(worst, 1):
            q3_worst_rows.append(
                {
                    "case_id": case,
                    "alpha": alpha,
                    "beta": beta,
                    "method": "B_K64",
                    "seed": row["seed"],
                    "trial_id": trial_id(row),
                    "worst_rank": rank,
                    "weighted_on_time_rate": float(
                        row["weighted_on_time_rate"]
                    ),
                    "weighted_completion_rate": float(
                        row["weighted_completion_rate"]
                    ),
                    "weighted_tardiness_s": float(
                        row["weighted_tardiness_s"]
                    ),
                    "source": SOURCES["q3_arch"],
                }
            )

# ---------------------------------------------------------------- paired tail
paired_rows = [
    pair_tail(
        "q1_horizon_1.0",
        "history=full",
        "history=0.30s",
        [r for r in q1_dev if r["history"] == "full" and r["horizon"] == "1.0"],
        [
            r
            for r in q1_dev
            if r["history"] == "0.30s" and r["horizon"] == "1.0"
        ],
    ),
    pair_tail(
        "q2_delay",
        "delay=0.1",
        "delay=0.2",
        q2_dev_groups["0.1"],
        q2_dev_groups["0.2"],
    ),
]
record_check(
    checks,
    failures,
    "paired_tail_sample_sizes",
    all(row["n_paired"] == 212 and row["tail_n"] == 22 for row in paired_rows),
    comparisons={
        row["comparison"]: {
            "n_paired": row["n_paired"],
            "tail_n": row["tail_n"],
            "overlap_count": row["overlap_count"],
        }
        for row in paired_rows
    },
)

# ---------------------------------------------------------------- write files
write_csv(
    "q1_summary.csv",
    [
        "scope",
        "history",
        "horizon",
        "source",
        "n_windows",
        "nade_median",
        "nade_mean",
        "nade_p90",
        "nade_p95",
        "worst_decile_n",
        "worst_decile_mean_nade",
    ],
    q1_rows,
)
write_csv(
    "q2_summary.csv",
    [
        "scope",
        "level",
        "delay",
        "environment_id",
        "source",
        "n_windows",
        "nade_median",
        "nade_mean",
        "nade_p90",
        "nade_p95",
        "worst_decile_n",
        "worst_decile_mean_nade",
    ],
    q2_rows,
)
write_csv(
    "q3_summary.csv",
    [
        "case_id",
        "alpha",
        "beta",
        "method",
        "n_trials",
        "n_seeds",
        "mean_on_time",
        "on_time_p10",
        "worst3_mean_on_time",
        "worst3_cutoff_on_time",
        "worst3_cutoff_tie_count",
        "worst3_cutoff_excluded",
        "mean_completion",
        "mean_tardiness",
        "worst3_trial_ids",
        "source",
    ],
    q3_rows,
)
write_csv(
    "q3_worst3.csv",
    [
        "case_id",
        "alpha",
        "beta",
        "method",
        "seed",
        "trial_id",
        "worst_rank",
        "weighted_on_time_rate",
        "weighted_completion_rate",
        "weighted_tardiness_s",
        "source",
    ],
    q3_worst_rows,
)
write_csv(
    "paired_tail.csv",
    [
        "comparison",
        "nominal",
        "stress",
        "n_paired",
        "tail_n",
        "nominal_tail_cutoff",
        "nominal_tail_mean",
        "stress_tail_mean",
        "tail_change_mean",
        "overlap_count",
        "overlap_rate",
        "stress_tail_n",
    ],
    paired_rows,
)

protocol = {
    "tag": "v1.0.0-final",
    "models": ["B", "B_K64"],
    "model_actions": "none; read-only archived and frozen-model CSVs",
    "aggregation": "empirical equal-window; one window is one observation; no hierarchical pooling",
    "quantiles": "linear empirical quantiles (Hyndman-Fan type 7)",
    "worst_decile": "ceil(0.1*n) largest NADE values; arithmetic mean reported",
    "paired_tail": "baseline worst decile selected under nominal; same window IDs scored under stress; change = stress - nominal",
    "overlap_rate": "|baseline worst-decile IDs intersect stress worst-decile IDs| / tail_n; overlap_count is the intersection size",
    "tie_break": "Q1/Q2 value descending then window_id ascending; Q3 worst3 value ascending then trial_id ascending",
    "q3_ties": "cutoff tie count includes the selected worst 3; excluded counts equal-valued trials outside the selected worst 3",
    "q3": {
        "method": "B_K64",
        "seeds": 30,
        "cases": 3,
        "mappings": 5,
        "trials_per_case_mapping": 30,
        "baseline_metric": "weighted_on_time_rate",
        "low_p10": "linear 10th percentile of weighted_on_time_rate",
        "worst3": "mean of 3 lowest weighted_on_time_rate trials",
        "scenario_pooling": "none; each case x mapping reported separately",
    },
    "descriptive_only": True,
    "secrets_read": False,
}
validation = {
    "protocol": protocol,
    "inputs": input_meta,
    "checks": checks,
    "outputs": list(OUTPUT_FILES),
    "failures": failures,
    "passed": not failures,
}
with (OUTPUT_DIR / "validation.json").open(
    "w", encoding="utf-8", newline="\n"
) as fh:
    json.dump(validation, fh, indent=2, ensure_ascii=False)
    fh.write("\n")

if failures:
    raise SystemExit("validation failed: " + ", ".join(failures))
print(f"wrote {len(OUTPUT_FILES)} files to {OUTPUT_DIR}; validation passed")
