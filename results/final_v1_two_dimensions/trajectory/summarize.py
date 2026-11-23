"""Build ``trajectory_cells.csv`` from the frozen Q1/Q2 window tables.

Q1 rows sweep history x horizon (five histories x three horizons); Q2 rows
sweep history x delay (``full``/``short`` x five delays).  For every
split x condition the table reports

* ``hier_mean`` - window mean within episode, episode within environment,
  environment within dataset, then an equal-weight mean across datasets;
* ``tail05``/``tail10``/``tail20`` - equal-window mean of the worst
  ceil(5/10/20% x n) NADE values (largest NADE is worst);
* ``n`` and ``n05``/``n10``/``n20`` - window count and tail sizes.

The script verifies window-set identity across conditions, completeness,
finiteness, tail ordering and tail counts before writing a UTF-8-BOM CSV
next to itself.  It performs no network access and writes only this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = SCRIPT_DIR.parent
Q1_PATH = EVIDENCE_DIR / "q1_windows.csv.gz"
Q2_PATH = EVIDENCE_DIR / "q2_windows.csv.gz"
OUT_PATH = SCRIPT_DIR / "trajectory_cells.csv"

SPLITS = ("validation", "test")
Q1_HISTORIES = ("full", "0.75s", "0.60s", "0.45s", "0.30s")
Q1_HORIZONS = (0.2, 0.6, 1.0)
Q2_HISTORIES = ("full", "short")
Q2_DELAYS = (0.1, 0.125, 0.15, 0.175, 0.2)
TAIL_PERCENTS = (5, 10, 20)
META_COLUMNS = ("dataset", "environment_id", "episode_uuid")
OUT_COLUMNS = (
    "question", "split", "history", "horizon", "delay", "n",
    "hier_mean", "tail05", "tail10", "tail20", "n05", "n10", "n20",
)


def ceil_percent(percent: int, n: int) -> int:
    """Return ``ceil(percent / 100 * n)`` using exact integer arithmetic."""
    return (percent * n + 99) // 100


def hierarchical_mean(frame: pd.DataFrame) -> float:
    """Average window->episode->environment->dataset, then datasets equally."""
    episode = frame.groupby(list(META_COLUMNS), as_index=False)["nade"].mean()
    environment = episode.groupby(
        ["dataset", "environment_id"], as_index=False)["nade"].mean()
    dataset = environment.groupby("dataset", as_index=False)["nade"].mean()
    return float(dataset["nade"].mean())


def equal_window_tail(nade: np.ndarray, percent: int) -> tuple[float, int]:
    """Mean of the ``ceil(percent% * n)`` worst (largest) NADE values."""
    ordered = np.sort(np.asarray(nade, dtype=float))[::-1]
    count = ceil_percent(percent, int(ordered.size))
    return float(ordered[:count].mean()), count


def load_windows(path: Path, key: tuple[str, ...], label: str) -> pd.DataFrame:
    """Load a window table and validate its keys, metadata and NADE values."""
    frame = pd.read_csv(path)
    required = set(key) | set(META_COLUMNS) | {"nade"}
    missing = required.difference(frame.columns)
    assert not missing, f"{label}: missing columns {sorted(missing)}"
    for column in ("split", "window_id", *META_COLUMNS, "history"):
        assert frame[column].notna().all(), f"{label}: null {column}"
    assert np.isfinite(frame["nade"].to_numpy(float)).all(), \
        f"{label}: non-finite NADE"
    assert frame.duplicated(list(key)).sum() == 0, \
        f"{label}: duplicate keys {key}"
    meta_nunique = frame.groupby("window_id")[list(META_COLUMNS)].nunique()
    assert (meta_nunique == 1).to_numpy().all(), \
        f"{label}: window metadata not unique"
    return frame


def condition_mask(frame: pd.DataFrame, split: str, history: str,
                   value: float, column: str) -> np.ndarray:
    """Boolean mask selecting one split/history/value condition."""
    values = frame[column].to_numpy(float)
    return (
        (frame["split"].to_numpy() == split)
        & (frame["history"].to_numpy() == history)
        & np.isclose(values, value, rtol=0.0, atol=1e-9)
    )


def assert_identical_windows(conditions, label: str) -> frozenset:
    """Assert every condition covers the same windows with the same metadata."""
    reference = None
    metadata = None
    for name, frame in conditions:
        assert len(frame) > 0, f"{label}: empty condition {name}"
        windows = frozenset(frame["window_id"])
        current = (frame.drop_duplicates("window_id")
                   .set_index("window_id")[list(META_COLUMNS)]
                   .sort_index())
        if reference is None:
            reference, metadata = windows, current
        else:
            assert windows == reference, f"{label}: window set differs in {name}"
            try:
                pd.testing.assert_frame_equal(current, metadata)
            except AssertionError as exc:
                raise AssertionError(
                    f"{label}: window metadata differs in {name}") from exc
    return reference


def build_cell(question: str, split: str, history: str, horizon: float,
               delay: float, frame: pd.DataFrame) -> dict:
    """Summarise one split x condition into a single output row."""
    assert len(frame) > 0, f"{question}/{split}/{history}: empty condition"
    n = int(frame["window_id"].nunique())
    assert len(frame) == n, \
        f"{question}/{split}/{history}: expected one row per window"
    nade = frame["nade"].to_numpy(float)
    assert np.isfinite(nade).all(), \
        f"{question}/{split}/{history}: non-finite NADE"
    record = {
        "question": question,
        "split": split,
        "history": history,
        "horizon": horizon,
        "delay": delay,
        "n": n,
        "hier_mean": hierarchical_mean(frame),
    }
    for percent in TAIL_PERCENTS:
        tail, count = equal_window_tail(nade, percent)
        record[f"tail{percent:02d}"] = tail
        record[f"n{percent:02d}"] = count
    return record


def verify_output(out: pd.DataFrame, window_counts: dict) -> None:
    """Verify completeness, finiteness, tail ordering and tail counts."""
    assert tuple(out.columns) == OUT_COLUMNS
    assert out["question"].isin(["q1", "q2"]).all()
    assert set(out["split"]) == set(SPLITS)

    non_null = ["question", "split", "history", "n", "hier_mean",
                "tail05", "tail10", "tail20", "n05", "n10", "n20"]
    assert out[non_null].notna().to_numpy().all(), "null output cells"

    numeric = ["n", "hier_mean", "tail05", "tail10", "tail20",
               "n05", "n10", "n20"]
    assert np.isfinite(out[numeric].to_numpy(float)).all(), \
        "non-finite output cells"

    q1_rows = out["question"] == "q1"
    q2_rows = out["question"] == "q2"
    assert out.loc[q1_rows, "horizon"].notna().all()
    assert out.loc[q1_rows, "delay"].isna().all()
    assert out.loc[q2_rows, "delay"].notna().all()
    assert out.loc[q2_rows, "horizon"].isna().all()

    # Worst-5% mean >= worst-10% mean >= worst-20% mean.
    assert (out["tail05"] >= out["tail10"] - 1e-12).all(), "tail05 < tail10"
    assert (out["tail10"] >= out["tail20"] - 1e-12).all(), "tail10 < tail20"

    for percent, column in zip(TAIL_PERCENTS, ("n05", "n10", "n20")):
        expected = out["n"].map(lambda n: ceil_percent(percent, int(n)))
        assert (out[column] == expected).all(), f"{column} count mismatch"
        assert (out[column] <= out["n"]).all(), f"{column} exceeds n"
    assert (out["n05"] <= out["n10"]).all()
    assert (out["n10"] <= out["n20"]).all()

    for split, count in window_counts.items():
        assert (out.loc[out["split"] == split, "n"] == count).all(), \
            f"{split}: unexpected condition window count"

    per_split_question = out.groupby(["split", "question"]).size().to_dict()
    for split in SPLITS:
        assert per_split_question[(split, "q1")] == \
            len(Q1_HISTORIES) * len(Q1_HORIZONS)
        assert per_split_question[(split, "q2")] == \
            len(Q2_HISTORIES) * len(Q2_DELAYS)


def main() -> None:
    q1 = load_windows(
        Q1_PATH, ("split", "window_id", "history", "horizon"), "Q1")
    q2 = load_windows(
        Q2_PATH, ("split", "window_id", "history", "delay"), "Q2")

    q1["horizon"] = q1["horizon"].astype(float)
    q2["delay"] = q2["delay"].astype(float)

    assert set(q1["split"]) == set(SPLITS), "Q1 unexpected splits"
    assert set(q2["split"]) == set(SPLITS), "Q2 unexpected splits"
    assert set(q1["history"]) == set(Q1_HISTORIES), "Q1 unexpected histories"
    assert set(q2["history"]) == set(Q2_HISTORIES), "Q2 unexpected histories"
    assert np.allclose(np.sort(q1["horizon"].unique()), Q1_HORIZONS,
                       rtol=0.0, atol=1e-9), "Q1 unexpected horizons"
    assert np.allclose(np.sort(q2["delay"].unique()), Q2_DELAYS,
                       rtol=0.0, atol=1e-9), "Q2 unexpected delays"

    rows: list[dict] = []
    window_counts: dict[str, int] = {}
    consumed = {"q1": 0, "q2": 0}

    for split in SPLITS:
        q1_conditions = []
        for history in Q1_HISTORIES:
            for horizon in Q1_HORIZONS:
                frame = q1[condition_mask(q1, split, history, horizon,
                                          "horizon")]
                q1_conditions.append((f"{history}@{horizon}", frame))
                rows.append(build_cell("q1", split, history, horizon,
                                       np.nan, frame))
                consumed["q1"] += len(frame)

        q2_conditions = []
        for history in Q2_HISTORIES:
            for delay in Q2_DELAYS:
                frame = q2[condition_mask(q2, split, history, delay,
                                          "delay")]
                q2_conditions.append((f"{history}@{delay}", frame))
                rows.append(build_cell("q2", split, history, np.nan,
                                       delay, frame))
                consumed["q2"] += len(frame)

        q1_windows = assert_identical_windows(q1_conditions, f"Q1/{split}")
        q2_windows = assert_identical_windows(q2_conditions, f"Q2/{split}")
        assert q1_windows == q2_windows, \
            f"{split}: Q1 and Q2 window sets differ"
        window_counts[split] = len(q1_windows)

    assert consumed["q1"] == len(q1), "Q1 rows unaccounted for"
    assert consumed["q2"] == len(q2), "Q2 rows unaccounted for"

    out = pd.DataFrame(rows, columns=list(OUT_COLUMNS))
    expected_rows = len(SPLITS) * (
        len(Q1_HISTORIES) * len(Q1_HORIZONS)
        + len(Q2_HISTORIES) * len(Q2_DELAYS))
    assert len(out) == expected_rows, "unexpected output row count"
    assert window_counts == {"validation": 212, "test": 390}
    verify_output(out, window_counts)

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig", na_rep="")

    raw = OUT_PATH.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "output is missing the UTF-8 BOM"
    back = pd.read_csv(OUT_PATH, encoding="utf-8-sig")
    assert len(back) == len(out), "round-trip row mismatch"
    for column in ("n", "hier_mean", "tail05", "tail10", "tail20",
                   "n05", "n10", "n20"):
        np.testing.assert_allclose(
            back[column].to_numpy(float),
            out[column].to_numpy(float),
            rtol=0.0, atol=1e-12)

    print(f"wrote {OUT_PATH} ({len(out)} rows, splits={SPLITS})")


if __name__ == "__main__":
    main()

