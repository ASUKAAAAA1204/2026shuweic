#!/usr/bin/env python
"""Pre-fixed, no-selection replay of the official Q1/Q2 validation and test windows.

Q1 scores the frozen B forecast at a one-second horizon with full versus
0.30 s visible history.  Q2 replays the full official windows with the fixed
0.3/0.6 s requests, budget 2, at 0.1 s (nominal) and 0.2 s (stress) delay.
Every official window is evaluated; no selection, tuning or source edits.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[2]
SPACE = WORKSPACE / "tmp/final_v1_20261122/C题工作区"
FORMAL = SPACE / "正式方案"
FROZEN = FORMAL / "results"
SENSITIVITY = WORKSPACE / "results/final_v1_sensitivity"
RECORDS = WORKSPACE / "C_Problem_Data/v1/v1/records"
PACKAGE = SPACE / "独立方案B_旋转模型"
DELIVERY = PACKAGE / "第二问/delivery"
MODEL_PATH = PACKAGE / "第一问/model/model_parameters.npz"
DELIVERY_MODEL_PATH = DELIVERY / "model/rotation_full.npz"
RANGE_MODEL_PATH = FROZEN / "q1_range_model.csv"
COMMIT = "9f8bc05fae9d8ce32da7fa7c5e28a82960085819"
MODEL_SHA256 = "8f34fa90c6484f53a53deed01374d1e373aacee8c1328a9e7ad7c2bdbd4d6237"
SPLITS = ("validation", "test")
Q1_HORIZON = 1.0
BUDGET = 2
COLUMNS = ["question", "split", "window_id", "side", "episode_uuid",
           "environment_id", "dataset", "condition", "nade", "coverage",
           "norm_diameter", "points"]
CONDITIONS = {
    "q1": (("nominal", np.inf), ("stress", 0.30)),
    "q2": (("nominal", 0.1), ("stress", 0.2)),
}

sys.path[:0] = [str(SENSITIVITY), str(SENSITIVITY / "q2"),
                str(FORMAL / "code"), str(DELIVERY / "code")]

import run_q1_q3  # noqa: E402
import run_q2  # noqa: E402
from adopted import AdoptedModel  # noqa: E402
from replay_engine import FixedPolicy, replay  # noqa: E402
from rotation_inference import predict_model  # noqa: E402


def snapshot():
    """Frozen repository HEAD and the model/range files used by the replay."""
    return {
        "head": subprocess.check_output(
            ["git", "-C", str(SPACE), "rev-parse", "HEAD"], text=True).strip(),
        "model_path": str(MODEL_PATH),
        "model_sha256": run_q2.digest(MODEL_PATH),
        "delivery_model_path": str(DELIVERY_MODEL_PATH),
        "delivery_model_sha256": run_q2.digest(DELIVERY_MODEL_PATH),
        "range_model_path": str(RANGE_MODEL_PATH),
        "range_model_sha256": run_q2.digest(RANGE_MODEL_PATH),
    }


def verify_snapshot(state):
    """The frozen source and model must match the registered commit and bytes."""
    assert state["head"] == COMMIT, state["head"]
    assert state["model_sha256"] == MODEL_SHA256, state["model_sha256"]
    assert state["delivery_model_sha256"] == MODEL_SHA256, state["delivery_model_sha256"]
    assert MODEL_PATH.read_bytes() == DELIVERY_MODEL_PATH.read_bytes()
    return state


def official_windows():
    """The official validation and test windows, in frozen manifest order."""
    windows = pd.read_csv(run_q2.PROTOCOL / "data/windows.csv")
    windows = windows[windows.split.isin(SPLITS)].copy()
    counts = windows.groupby("split").size().to_dict()
    assert counts == {"validation": 212, "test": 390}, counts
    assert windows.window_id.is_unique
    return windows


def replay_q1(model, windows):
    """One-second Q1 scores with full and 0.30 s visible history."""
    rows = []
    for episode, group in windows.groupby("episode_uuid", sort=False):
        with np.load(RECORDS / f"{episode}.npz", allow_pickle=False) as data:
            raw = dict(data)
        for w in group.itertuples():
            clock = raw[w.side + "__t_s"]
            start, stop = int(w.start_index), int(w.end_index_exclusive)
            anchor = clock[start - 1]
            ages = clock[start:stop] - anchor
            truth = raw[w.side + "__position"][start:stop].mean(axis=1)
            supported = ages <= Q1_HORIZON + 1e-12
            ages_supported, truth_supported = ages[supported], truth[supported]
            assert ages_supported.size, w.window_id
            scale = model.paid["scale"][w.dataset]
            for condition, span in CONDITIONS["q1"]:
                past = run_q1_q3.history_since(raw, anchor, span)
                observed_clock = past[w.side + "__t_s"]
                assert observed_clock[-1] == anchor
                position, radius = model.predict_position(
                    past, w.side, len(observed_clock) - 1, ages_supported, w.dataset)
                error = np.linalg.norm(position - truth_supported, axis=1)
                rows.append({
                    "question": "q1", "split": w.split, "window_id": w.window_id,
                    "side": w.side, "episode_uuid": episode,
                    "environment_id": w.environment_id, "dataset": w.dataset,
                    "condition": condition,
                    "nade": float(error.mean() / scale),
                    "coverage": float((error <= radius).mean()),
                    "norm_diameter": float(2 * (radius / scale).mean()),
                    "points": int(supported.sum()),
                })
    return pd.DataFrame(rows)


def replay_q2(model, windows):
    """Full-window Q2 replay with the frozen state/radius helper and fixed policy."""
    cfg, prior, weights = model.paid, model.prior, model.parameters
    rows = []
    for episode, group in windows.groupby("episode_uuid", sort=False):
        with np.load(RECORDS / f"{episode}.npz", allow_pickle=False) as data:
            raw = dict(data)
        for w in group.itertuples():
            side = w.side
            t = raw[side + "__t_s"]
            x = raw[side + "__position"].mean(axis=1)
            start, stop = int(w.start_index), int(w.end_index_exclusive)
            anchor = float(t[start - 1])
            times, reference = t[start:stop], x[start:stop]
            assert len(times) >= 2 and np.all(np.diff(times) > 0)
            initial = predict_model(weights, run_q2.past_only(raw, anchor), side,
                                    anchor, times - anchor, w.dataset) + x[start - 1]
            scale = cfg["scale"][w.dataset]
            for condition, delay in CONDITIONS["q2"]:
                State, radii = run_q2.range_state(times, initial, w.dataset, cfg, prior)
                run = replay(times, reference, t[start - 2:start], x[start - 2:start],
                             FixedPolicy(), delay=delay, budget=BUDGET, state_factory=State)
                prediction = np.asarray([point["prediction"] for point in run["points"]])
                truth = np.asarray([point["reference"] for point in run["points"]])
                errors = np.linalg.norm(prediction - truth, axis=1) / scale
                radii = np.asarray(radii, float) / scale
                assert len(radii) == len(errors) == len(times)
                assert np.isfinite(errors).all() and np.isfinite(radii).all()
                rows.append({
                    "question": "q2", "split": w.split, "window_id": w.window_id,
                    "side": side, "episode_uuid": episode,
                    "environment_id": w.environment_id, "dataset": w.dataset,
                    "condition": condition,
                    "nade": float(errors.mean()),
                    "coverage": float((errors <= radii).mean()),
                    "norm_diameter": float(2 * radii.mean()),
                    "points": len(times),
                })
    return pd.DataFrame(rows)


def validate_rows(rows, windows):
    """One complete, finite row per question, split, window and condition."""
    counts = windows.groupby("split").size().to_dict()
    assert len(rows) == 2 * sum(counts.values()) * 2, len(rows)
    assert set(rows.question) == {"q1", "q2"}
    assert set(rows.condition) == {"nominal", "stress"}
    assert not rows.duplicated(["question", "split", "window_id", "condition"]).any()
    for question in ("q1", "q2"):
        for split in SPLITS:
            for condition in ("nominal", "stress"):
                part = rows[(rows.question == question) & (rows.split == split) &
                            (rows.condition == condition)]
                assert len(part) == counts[split], (question, split, condition, len(part))
    assert np.isfinite(rows[["nade", "coverage", "norm_diameter"]].to_numpy(float)).all()
    assert rows.points.gt(0).all()


def hierarchy(frame, metric):
    """Equal-weight roll-up used by the frozen Q1 range summary."""
    keys = ["dataset", "environment_id", "episode_uuid", "window_id"]
    table = frame.groupby(keys, as_index=False)[metric].mean()
    for level in (keys[:-1], keys[:-2], keys[:-3]):
        table = table.groupby(level, as_index=False)[metric].mean()
    return float(table[metric].mean())


def max_difference(observed, expected):
    return float(np.max(np.abs(np.asarray(observed, float) -
                               np.asarray(expected, float))))


def regressions(rows):
    """Frozen-result regressions requested for the nominal and stress rows."""
    nominal = rows[rows.condition == "nominal"]
    stress = rows[rows.condition == "stress"]
    q1_nominal = nominal[nominal.question == "q1"]
    q2_nominal = nominal[nominal.question == "q2"]

    # Q1 and Q2 nominal NADE on every validation and test window.
    q1_ref = pd.read_csv(FROZEN / "q1_windows.csv.gz")
    q1_ref = q1_ref[(q1_ref.model == "B") &
                    np.isclose(q1_ref.horizon, Q1_HORIZON)]
    q1_pair = q1_nominal.merge(q1_ref, on=["split", "window_id"],
                               suffixes=("_ours", "_frozen"), validate="one_to_one")
    assert len(q1_pair) == 602
    q1_nade_difference = max_difference(q1_pair.nade_ours, q1_pair.nade_frozen)
    assert q1_nade_difference < 1e-10

    q2_ref = pd.read_csv(FROZEN / "q2_windows.csv.gz")
    q2_ref = q2_ref[q2_ref.model == "B"]
    q2_pair = q2_nominal.merge(q2_ref, on=["split", "window_id"],
                               suffixes=("_ours", "_frozen"), validate="one_to_one")
    assert len(q2_pair) == 602
    q2_nade_difference = max_difference(q2_pair.nade_ours, q2_pair.nade_frozen)
    q2_coverage_difference = max_difference(q2_pair.coverage_ours,
                                            q2_pair.coverage_frozen)
    q2_width_difference = max_difference(q2_pair.norm_diameter, q2_pair.width)
    assert max(q2_nade_difference, q2_coverage_difference,
               q2_width_difference) < 1e-10

    # Q1 nominal coverage and 2 * radius, rolled up with the official hierarchy.
    q1_range_ref = pd.read_csv(FROZEN / "q1_range_summary.csv")
    q1_range_ref = q1_range_ref[(q1_range_ref.model == "B") &
                                np.isclose(q1_range_ref.horizon, Q1_HORIZON)
                                ].set_index("split")
    q1_range = {}
    for split in SPLITS:
        part = q1_nominal[q1_nominal.split == split]
        observed_coverage = hierarchy(part, "coverage")
        observed_diameter = hierarchy(part, "norm_diameter")
        expected = q1_range_ref.loc[split]
        coverage_difference = observed_coverage - float(expected.coverage)
        diameter_difference = observed_diameter - 2 * float(expected.norm_radius)
        points = int(part.points.sum())
        assert abs(coverage_difference) < 1e-12
        assert abs(diameter_difference) < 1e-12
        assert points == int(expected.points)
        q1_range[split] = {
            "windows": len(part),
            "observed_coverage": observed_coverage,
            "expected_coverage": float(expected.coverage),
            "coverage_difference": float(coverage_difference),
            "observed_norm_diameter": observed_diameter,
            "expected_norm_diameter": 2 * float(expected.norm_radius),
            "norm_diameter_difference": float(diameter_difference),
            "points": points,
            "expected_points": int(expected.points),
        }

    # Validation stress rows against the previously archived sensitivity runs.
    q1_stress_ref = pd.read_csv(SENSITIVITY / "q1_history_windows.csv.gz")
    q1_stress_ref = q1_stress_ref[(q1_stress_ref.history == "0.30s") &
                                  np.isclose(q1_stress_ref.horizon, Q1_HORIZON)]
    q1_stress = stress[(stress.question == "q1") & (stress.split == "validation")]
    q1_stress_pair = q1_stress.merge(q1_stress_ref, on="window_id",
                                     suffixes=("_ours", "_frozen"),
                                     validate="one_to_one")
    assert len(q1_stress_pair) == 212
    q1_stress_differences = {
        "nade": max_difference(q1_stress_pair.nade_ours, q1_stress_pair.nade_frozen),
        "coverage": max_difference(q1_stress_pair.coverage_ours,
                                   q1_stress_pair.coverage_frozen),
    }
    assert q1_stress_pair.points_ours.eq(q1_stress_pair.points_frozen).all()
    assert max(q1_stress_differences.values()) < 1e-10

    q2_stress_ref = pd.read_csv(SENSITIVITY / "q2/per_window.csv.gz")
    q2_stress_ref = q2_stress_ref[np.isclose(q2_stress_ref.delay, 0.2)]
    q2_stress = stress[(stress.question == "q2") & (stress.split == "validation")]
    q2_stress_pair = q2_stress.merge(q2_stress_ref, on="window_id",
                                     suffixes=("_ours", "_frozen"),
                                     validate="one_to_one")
    assert len(q2_stress_pair) == 212
    q2_stress_differences = {
        "nade": max_difference(q2_stress_pair.nade_ours, q2_stress_pair.nade_frozen),
        "coverage": max_difference(q2_stress_pair.coverage_ours,
                                   q2_stress_pair.coverage_frozen),
        "norm_diameter": max_difference(q2_stress_pair.norm_diameter_ours,
                                        q2_stress_pair.norm_diameter_frozen),
    }
    assert q2_stress_pair.points_ours.eq(q2_stress_pair.points_frozen).all()
    assert max(q2_stress_differences.values()) < 1e-10

    return {
        "q1_nominal_per_window_nade": {
            "source": "正式方案/results/q1_windows.csv.gz model=B horizon=1.0",
            "windows": len(q1_pair),
            "by_split": {split: int((q1_pair.split == split).sum()) for split in SPLITS},
            "max_abs_difference": q1_nade_difference,
        },
        "q2_nominal_per_window": {
            "source": "正式方案/results/q2_windows.csv.gz model=B",
            "windows": len(q2_pair),
            "by_split": {split: int((q2_pair.split == split).sum()) for split in SPLITS},
            "max_abs_nade_difference": q2_nade_difference,
            "max_abs_coverage_difference": q2_coverage_difference,
            "max_abs_width_difference": q2_width_difference,
        },
        "q1_nominal_range_hierarchy": {
            "source": "正式方案/results/q1_range_summary.csv model=B horizon=1.0",
            "splits": q1_range,
        },
        "validation_stress_q1": {
            "source": "final_v1_sensitivity/q1_history_windows.csv.gz "
                      "history=0.30s horizon=1.0",
            "windows": len(q1_stress_pair),
            "max_abs_differences": q1_stress_differences,
            "points_equal": True,
        },
        "validation_stress_q2": {
            "source": "final_v1_sensitivity/q2/per_window.csv.gz delay=0.2",
            "windows": len(q2_stress_pair),
            "max_abs_differences": q2_stress_differences,
            "points_equal": True,
        },
    }


def main():
    before = verify_snapshot(snapshot())
    windows = official_windows()
    model = AdoptedModel()
    rows = pd.concat([replay_q1(model, windows), replay_q2(model, windows)],
                     ignore_index=True)[COLUMNS]
    validate_rows(rows, windows)
    checks = regressions(rows)
    rows.to_csv(OUT / "windows.csv.gz", index=False, compression="gzip")
    after = verify_snapshot(snapshot())
    assert after == before

    counts = windows.groupby("split").size().to_dict()
    audit = {
        "protocol": {
            "kind": "pre-fixed protocol replay without selection",
            "windows": counts,
            "q1": {
                "horizon_s": Q1_HORIZON,
                "conditions": {"nominal": "full visible history",
                               "stress": "0.30 s visible history"},
                "history": "frozen run_q1_q3.history_since crops every clocked channel",
                "predictor": "frozen AdoptedModel.predict_position (model B)",
                "position_channels": 3,
            },
            "q2": {
                "window": "full official one-second window",
                "conditions": {"nominal": "delay 0.1 s", "stress": "delay 0.2 s"},
                "fixed_requests_s": [0.3, 0.6],
                "budget": BUDGET,
                "state_radius": "frozen run_q2.range_state; prior/return radii unchanged",
            },
        },
        "inputs": {
            "source_space": str(SPACE),
            "protocol_windows": str(run_q2.PROTOCOL / "data/windows.csv"),
            "records": str(RECORDS),
            "episodes": int(windows.episode_uuid.nunique()),
            "frozen_results": str(FROZEN),
            "sensitivity_references": str(SENSITIVITY),
        },
        "integrity": {
            "expected_commit": COMMIT,
            "expected_model_sha256": MODEL_SHA256,
            "before": before,
            "after": after,
            "head_verified_before": before["head"] == COMMIT,
            "head_verified_after": after["head"] == COMMIT,
            "model_verified_before": before["model_sha256"] == MODEL_SHA256,
            "model_verified_after": after["model_sha256"] == MODEL_SHA256,
            "snapshot_unchanged": after == before,
        },
        "output": {
            "file": str(OUT / "windows.csv.gz"),
            "rows": len(rows),
            "columns": COLUMNS,
        },
        "checks": checks,
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"wrote {OUT / 'windows.csv.gz'} ({len(rows)} rows)")
    print(f"q1 nominal NADE max diff: {checks['q1_nominal_per_window_nade']['max_abs_difference']:.3g}")
    q2_check = checks["q2_nominal_per_window"]
    print("q2 nominal max diffs: "
          f"NADE {q2_check['max_abs_nade_difference']:.3g}, "
          f"coverage {q2_check['max_abs_coverage_difference']:.3g}, "
          f"width {q2_check['max_abs_width_difference']:.3g}")
    print(f"wrote {OUT / 'audit.json'}")


if __name__ == "__main__":
    main()
