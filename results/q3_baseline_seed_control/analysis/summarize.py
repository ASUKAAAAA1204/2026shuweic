"""Paired seed stability audit for the archived baseline reruns.

The rerun file must reproduce the exact instance registry used by the two
archived B_K64 stability batches. Only instance hashes are compared: the
baseline policy is expected to produce different outcomes on the same inputs.
"""

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent
REFERENCE = {
    "original": OUT.parent / "final_v1_stability/seed_trials.csv.gz",
    "replication": OUT.parent / "final_v1_evidence_extension/internal_replication.csv.gz",
}
EXPECTED_ROWS = {"original": 9000, "replication": 3600}
KEY = ["case_id", "seed", "alpha", "beta", "internal_seed"]
INSTANCE = ["case_id", "seed", "alpha", "beta"]
RATE = "weighted_on_time_rate"



def canonical_action(value):
    """Ignore JSON key order and whitespace when comparing first actions."""
    try:
        return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value).strip()


def pair_agreement(values):
    """Share of unordered pairs sharing a value, in percent."""
    series = pd.Series(list(values))
    n = len(series)
    if n < 2:
        return 0.0
    counts = series.value_counts(dropna=False)
    return 100.0 * float((counts * (counts - 1)).sum()) / (n * (n - 1))


def read_trials():
    trials = pd.read_csv(OUT / "trials.csv.gz")
    trials["method"] = "baseline"
    return trials


def match_reference(trials):
    """Require one-to-one keys and identical instance hashes for both batches."""
    assert set(trials.batch) == set(EXPECTED_ROWS), sorted(set(trials.batch))
    compared = [trials]
    for batch, path in REFERENCE.items():
        part = trials[trials.batch == batch]
        reference = pd.read_csv(path)
        expected = EXPECTED_ROWS[batch]
        assert len(part) == expected, (batch, len(part), expected)
        assert len(reference) == expected, (batch, len(reference), expected)
        assert not part.duplicated(KEY).any(), f"duplicate keys in {batch} trials"
        assert not reference.duplicated(KEY).any(), f"duplicate keys in {batch} reference"

        merged = part.merge(
            reference[KEY + ["instance_hash"]],
            on=KEY,
            how="outer",
            validate="one_to_one",
            suffixes=("", "_reference"),
            indicator=True,
        )
        assert merged["_merge"].eq("both").all(), f"{batch} keys do not match reference"
        assert merged["instance_hash"].eq(merged["instance_hash_reference"]).all(), \
            f"{batch} instance hashes differ from reference"

        for name, frame in (("trials", part), ("reference", reference)):
            grouped = frame.groupby(INSTANCE)
            assert grouped.internal_seed.nunique().eq(20).all(), \
                f"{batch} {name}: every instance must have 20 seeds"
            assert grouped.instance_hash.nunique().eq(1).all(), \
                f"{batch} {name}: instance hash must be constant across seeds"
        compared.append(reference.assign(method="B_K64", batch=batch))
    return pd.concat(compared, ignore_index=True)


def summarize(trials):
    trials = trials.copy()
    trials["first_action_canonical"] = trials.groupby("method", sort=False)[
        "first_action"
    ].transform(lambda values: values.map(canonical_action))

    rows = []
    for (batch, method, alpha, beta), group in trials.groupby(
            ["batch", "method", "alpha", "beta"], sort=True):
        instance_sd = []
        first_pairs = []
        path_pairs = []
        varying = 0
        instances = 0
        for _, instance in group.groupby(["case_id", "seed"], sort=False):
            assert len(instance) == 20
            assert instance.internal_seed.nunique() == 20
            rates = instance[RATE].astype(float)
            instance_sd.append(float((rates - rates.iloc[0]).std(ddof=1)) * 100.0)
            first_pairs.append(pair_agreement(instance["first_action_canonical"]))
            path_pairs.append(pair_agreement(instance["path_hash"]))
            varying += int(rates.nunique(dropna=False) > 1)
            instances += 1
        rows.append({
            "batch": batch,
            "method": method,
            "alpha": alpha,
            "beta": beta,
            "mean_rate_pct": float(group[RATE].mean()) * 100.0,
            "mean_within_instance_sd_pp": float(pd.Series(instance_sd).mean()),
            "first_action_pair_agreement_pct": float(pd.Series(first_pairs).mean()),
            "path_pair_agreement_pct": float(pd.Series(path_pairs).mean()),
            "n_instances_rate_varies": varying,
            "n_instances": instances,
        })
    return pd.DataFrame(rows)


def main():
    trials = match_reference(read_trials())
    summary = summarize(trials)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
