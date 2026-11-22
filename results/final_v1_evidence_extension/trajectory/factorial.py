#!/usr/bin/env python
"""Q2 2x2 factorial sensitivity on the frozen official validation/test windows.

Visible history (full versus 0.30 s) x single-return delay (0.10 versus 0.20 s)
with the fixed 0.3/0.6 s request schedule, budget 2 and frozen model B.  The
history factor enters only the initial forecast through the frozen
``run_q1_q3.history_since`` helper, which crops every clocked input channel on
its own clock.  The official replay engine still receives the two observed
positions before the window start, so no future observation is visible to the
forecast or to the policy.

Outputs in this directory: per_window.csv.gz, summary.csv, audit.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
PRIOR_TRAJECTORY = WORKSPACE / "results/final_v1_remaining/trajectory"
SENSITIVITY = WORKSPACE / "results/final_v1_sensitivity"
PRIOR_WINDOWS = PRIOR_TRAJECTORY / "windows.csv.gz"
PRIOR_Q2 = SENSITIVITY / "q2/per_window.csv.gz"
PRIOR_ENVIRONMENTS = (WORKSPACE / "results/final_v1_remaining/"
                                   "environment_results.csv")
OUT_WINDOWS = HERE / "per_window.csv.gz"
OUT_SUMMARY = HERE / "summary.csv"
OUT_AUDIT = HERE / "audit.json"

sys.path.insert(0, str(PRIOR_TRAJECTORY))
import replay as base  # noqa: E402

run_q1_q3 = base.run_q1_q3
run_q2 = base.run_q2
history_since = run_q1_q3.history_since
past_only = run_q2.past_only
range_state = run_q2.range_state
aggregate = run_q2.aggregate
digest = run_q2.digest
AdoptedModel = base.AdoptedModel
FixedPolicy = base.FixedPolicy
engine_replay = base.replay
predict_model = base.predict_model
snapshot = base.snapshot
verify_snapshot = base.verify_snapshot
SPLITS = base.SPLITS
BUDGET = base.BUDGET
RECORDS = base.RECORDS
COMMIT = base.COMMIT
MODEL_SHA256 = base.MODEL_SHA256
SPACE = base.SPACE
FORMAL = base.FORMAL
DELIVERY = base.DELIVERY

HISTORY_SPAN = 0.30
HISTORIES = (("full", np.inf), ("short", HISTORY_SPAN))
DELAYS = (0.1, 0.2)
METRICS = ("nade", "coverage", "norm_diameter")
SCENARIOS = {
    ("full", 0.1): "full_d01",
    ("full", 0.2): "full_d02",
    ("short", 0.1): "short_d01",
    ("short", 0.2): "short_d02",
}
EFFECTS = ("history_main", "delay_main", "did", "full_delay",
           "short_delay", "history_d01", "history_d02")
COLUMNS = [
    "split", "dataset", "environment_id", "episode_uuid", "side", "window_id",
    "anchor_s", "anchor_index", "start_index", "end_index_exclusive",
    "interval", "interval_s", "interval_start_s", "interval_end_s",
    "history_start", "window_s", "history", "history_span_s", "delay",
    "scenario", "points", "nade", "nfde", "coverage", "norm_diameter",
    "returns", "successful_returns", "requests", "waiting_s",
    "nonoverlap_selected",
]
SUMMARY_COLUMNS = [
    "selection", "scope", "split", "dataset", "environment_id", "sources",
    "metric", "metric_role", "n_windows", "n_hands", "n_episodes",
    "n_environments", "n_datasets", "n_points",
    "full_d01", "full_d02", "short_d01", "short_d02",
    "history_main", "delay_main", "did", "full_delay", "short_delay",
    "history_d01", "history_d02", "sign_did", "sign_short_delay",
]


def sign(value, tolerance=1e-15):
    value = float(value)
    return int(value > tolerance) - int(value < -tolerance)


def equal_arrays(left, right):
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)


def selection_map(windows):
    """Deterministic, outcome-free greedy non-overlap selection.

    Within one episode and one hand, sort intervals by start, end and window
    id, then keep a window only when its [start, end) interval begins at or
    after the end of the last selected interval.
    """
    selected = {}
    for (episode, side), group in windows.groupby(["episode_uuid", "side"],
                                                  sort=False):
        ordered = group.sort_values(
            ["start_index", "end_index_exclusive", "window_id"],
            kind="mergesort")
        last_end = -1
        for row in ordered.itertuples():
            start = int(row.start_index)
            stop = int(row.end_index_exclusive)
            assert stop > start
            keep = start >= last_end
            selected[row.window_id] = bool(keep)
            if keep:
                last_end = stop
    assert set(selected) == set(windows.window_id)
    return selected


def check_disjoint(windows, selected):
    """Every kept [start, end) interval is disjoint from the next kept one."""
    checked = 0
    for (episode, side), group in windows.groupby(["episode_uuid", "side"]):
        chosen = group[group.window_id.map(selected)].sort_values(
            ["start_index", "end_index_exclusive", "window_id"],
            kind="mergesort")
        intervals = list(zip(chosen.start_index.astype(int),
                             chosen.end_index_exclusive.astype(int)))
        for (start_a, stop_a), (start_b, stop_b) in zip(intervals, intervals[1:]):
            assert start_b >= stop_a, (episode, side, (start_a, stop_a),
                                       (start_b, stop_b))
            checked += 1
    return checked


def crop_assertions(past, raw, anchor, span):
    """Every channel is cropped on its own clock to [anchor - span, anchor]."""
    for key, values in past.items():
        values = np.asarray(values)
        clock = raw.get(key.split("__")[0] + "__t_s")
        if key.endswith("__t_s"):
            clock_values = np.asarray(values, float)
            assert clock_values.size
            assert np.nanmax(clock_values) <= anchor + 1e-12, key
            if np.isfinite(span):
                assert np.nanmin(clock_values) >= anchor - span - 1e-12, key
        if clock is not None and values.ndim and len(values) == len(clock):
            lower = anchor - span if np.isfinite(span) else -np.inf
            expected = int(np.count_nonzero(
                (clock >= lower) & (clock <= anchor)))
            assert len(values) == expected, (key, len(values), expected)


def perturb_future(raw, anchor):
    """Change only rows whose clocks are after the anchor."""
    changed = dict(raw)
    for key, values in raw.items():
        clock = raw.get(key.split("__")[0] + "__t_s")
        if clock is None or not values.ndim or len(values) != len(clock):
            continue
        future = clock > anchor
        if not future.any():
            continue
        copy = values.copy()
        if copy.dtype.kind == "f":
            copy[future] = np.nan
        elif copy.dtype.kind == "b":
            copy[future] = ~copy[future]
        elif copy.dtype.kind in "iu":
            copy[future] = 10 ** 9
        else:
            continue
        changed[key] = copy
    return changed


def replay_windows(model, windows, selected):
    """Every window under the four history x delay scenarios."""
    cfg, prior, weights = model.paid, model.prior, model.parameters
    rows = []
    stats = {
        "full_history_equals_past_only_windows": 0,
        "full_history_array_mismatches": 0,
        "max_initial_forecast_difference_full_vs_official": 0.0,
        "crop_assertions": 0,
        "replay_history_observations_checked": 0,
    }
    for episode, group in windows.groupby("episode_uuid", sort=False):
        with np.load(RECORDS / f"{episode}.npz", allow_pickle=False) as data:
            raw = {key: data[key].copy() for key in data.files}
        for w in group.itertuples():
            side = w.side
            t = raw[side + "__t_s"]
            positions = raw[side + "__position"].mean(axis=1)
            start = int(w.start_index)
            stop = int(w.end_index_exclusive)
            assert start >= 2 and stop > start
            anchor = float(t[start - 1])
            times = np.asarray(t[start:stop], float)
            reference = np.asarray(positions[start:stop], float)
            assert len(times) >= 2 and np.all(np.diff(times) > 0)
            scale = float(cfg["scale"][w.dataset])

            full = history_since(raw, anchor, np.inf)
            short = history_since(raw, anchor, HISTORY_SPAN)
            official = past_only(raw, anchor)
            assert list(full) == list(official)
            mismatches = [key for key in full
                          if not equal_arrays(full[key], official[key])]
            assert not mismatches, (w.window_id, mismatches)
            stats["full_history_equals_past_only_windows"] += 1
            stats["full_history_array_mismatches"] += len(mismatches)

            for label, span, past in (("full", np.inf, full),
                                      ("short", HISTORY_SPAN, short)):
                crop_assertions(past, raw, anchor, span)
                stats["crop_assertions"] += 1

            initial_official = predict_model(
                weights, official, side, anchor, times - anchor,
                w.dataset) + positions[start - 1]
            initial_full = predict_model(
                weights, full, side, anchor, times - anchor,
                w.dataset) + positions[start - 1]
            difference = float(np.max(np.abs(initial_official -
                                             initial_full)))
            stats["max_initial_forecast_difference_full_vs_official"] = max(
                stats["max_initial_forecast_difference_full_vs_official"],
                difference)
            assert difference == 0.0, w.window_id

            initials = {"full": initial_full}
            initials["short"] = predict_model(
                weights, short, side, anchor, times - anchor,
                w.dataset) + positions[start - 1]
            for label, initial in initials.items():
                assert np.isfinite(initial).all(), (w.window_id, label)

            history_two = np.asarray(t[start - 2:start], float)
            position_two = np.asarray(positions[start - 2:start], float)
            assert np.all(history_two <= anchor + 1e-12)
            assert np.all(np.diff(history_two) > 0)
            stats["replay_history_observations_checked"] += 1

            for label, span in HISTORIES:
                initial = initials[label]
                for delay in DELAYS:
                    State, radii = range_state(times, initial, w.dataset,
                                               cfg, prior)
                    run = engine_replay(
                        times, reference, t[start - 2:start],
                        positions[start - 2:start], FixedPolicy(),
                        delay=delay, budget=BUDGET, state_factory=State)
                    prediction = np.asarray([point["prediction"]
                                             for point in run["points"]])
                    truth = np.asarray([point["reference"]
                                        for point in run["points"]])
                    errors = np.linalg.norm(prediction - truth, axis=1) / scale
                    scaled_radii = np.asarray(radii, float) / scale
                    assert len(scaled_radii) == len(errors) == len(times)
                    assert np.isfinite(errors).all()
                    assert np.isfinite(scaled_radii).all()
                    rows.append({
                        "split": w.split,
                        "dataset": w.dataset,
                        "environment_id": w.environment_id,
                        "episode_uuid": episode,
                        "side": side,
                        "window_id": w.window_id,
                        "anchor_s": anchor,
                        "anchor_index": start - 1,
                        "start_index": start,
                        "end_index_exclusive": stop,
                        "interval": f"[{start},{stop})",
                        "interval_s": (f"[{float(w.start_s):.6f},"
                                       f"{float(w.end_s):.6f})"),
                        "interval_start_s": float(times[0]),
                        "interval_end_s": float(times[-1]),
                        "history_start": int(w.history_start),
                        "window_s": float(w.end_s - w.start_s),
                        "history": label,
                        "history_span_s": float(span),
                        "delay": float(delay),
                        "scenario": SCENARIOS[(label, delay)],
                        "points": int(len(times)),
                        "nade": float(errors.mean()),
                        "nfde": float(errors[-1]),
                        "coverage": float((errors <= scaled_radii).mean()),
                        "norm_diameter": float(2 * scaled_radii.mean()),
                        "returns": int(sum(action["event"] == "return"
                                           for action in run["actions"])),
                        "successful_returns": int(
                            run["metrics"]["successful_returns"]),
                        "requests": int(run["metrics"]["requests"]),
                        "waiting_s": float(run["metrics"]["waiting_s"]),
                        "nonoverlap_selected": bool(selected[w.window_id]),
                    })
    return pd.DataFrame(rows)[COLUMNS], stats


def leak_perturbation(model, windows):
    """Alter every post-anchor row and show the initial forecast is unchanged."""
    cfg, prior, weights = model.paid, model.prior, model.parameters
    chosen = (windows.sort_values(["split", "environment_id", "window_id"],
                                  kind="mergesort")
              .groupby("environment_id", sort=False).head(1))
    assert len(chosen) == windows.environment_id.nunique() == 8
    maximum = {"full": 0.0, "short": 0.0}
    checked = 0
    for w in chosen.itertuples():
        with np.load(RECORDS / f"{w.episode_uuid}.npz",
                     allow_pickle=False) as data:
            raw = {key: data[key].copy() for key in data.files}
        side = w.side
        t = raw[side + "__t_s"]
        positions = raw[side + "__position"].mean(axis=1)
        start = int(w.start_index)
        stop = int(w.end_index_exclusive)
        anchor = float(t[start - 1])
        times = np.asarray(t[start:stop], float)
        perturbed = perturb_future(raw, anchor)
        touched = 0
        for key, values in raw.items():
            clock = raw.get(key.split("__")[0] + "__t_s")
            if (clock is not None and values.ndim and
                    len(values) == len(clock) and (clock > anchor).any()):
                touched += 1
        assert touched > 0
        for label, span in HISTORIES:
            past = history_since(raw, anchor, span)
            past_perturbed = history_since(perturbed, anchor, span)
            base_initial = predict_model(
                weights, past, side, anchor, times - anchor,
                w.dataset) + positions[start - 1]
            changed_initial = predict_model(
                weights, past_perturbed, side, anchor, times - anchor,
                w.dataset) + positions[start - 1]
            assert equal_arrays(base_initial, changed_initial), \
                (w.window_id, label)
            difference = float(np.max(np.abs(base_initial - changed_initial)))
            maximum[label] = max(maximum[label], difference)
            checked += 1
    return {
        "windows": len(chosen),
        "environments": int(chosen.environment_id.nunique()),
        "history_scenarios": checked,
        "channels_touched_per_window": "all clocked channels after anchor",
        "max_initial_difference": maximum,
        "unchanged": all(value == 0.0 for value in maximum.values()),
    }


def scope_counts(meta):
    return {
        "n_windows": int(len(meta)),
        "n_hands": int(meta[["episode_uuid", "side"]]
                       .drop_duplicates().shape[0]),
        "n_episodes": int(meta.episode_uuid.nunique()),
        "n_environments": int(meta.environment_id.nunique()),
        "n_datasets": int(meta.dataset.nunique()),
        "n_points": int(meta.points.sum()),
    }


def contrasts(wide):
    """Per-window cell means, main effects and additive interaction."""
    for metric in METRICS:
        f1 = wide[f"{metric}__full_d01"]
        f2 = wide[f"{metric}__full_d02"]
        s1 = wide[f"{metric}__short_d01"]
        s2 = wide[f"{metric}__short_d02"]
        wide[f"{metric}__history_main"] = ((s1 - f1) + (s2 - f2)) / 2
        wide[f"{metric}__delay_main"] = ((f2 - f1) + (s2 - s1)) / 2
        wide[f"{metric}__did"] = (s2 - s1) - (f2 - f1)
        wide[f"{metric}__full_delay"] = f2 - f1
        wide[f"{metric}__short_delay"] = s2 - s1
        wide[f"{metric}__history_d01"] = s1 - f1
        wide[f"{metric}__history_d02"] = s2 - f2
    return wide


def build_summary(per_window):
    meta_columns = ["split", "dataset", "environment_id", "episode_uuid",
                    "side", "window_id", "points", "nonoverlap_selected"]
    meta = (per_window.drop_duplicates("window_id")[meta_columns]
            .set_index("window_id"))
    assert len(meta) == per_window.window_id.nunique() == 602
    wide = meta.copy()
    for (label, delay), name in SCENARIOS.items():
        part = per_window[(per_window.history == label) &
                          (per_window.delay == delay)].set_index("window_id")
        assert len(part) == len(wide) and part.index.is_unique
        for metric in METRICS:
            wide[f"{metric}__{name}"] = part[metric].reindex(wide.index)
    assert not wide.isna().any().any()
    wide = contrasts(wide)

    metric_columns = [f"{metric}__{name}" for metric in METRICS
                      for name in SCENARIOS.values()] + \
                     [f"{metric}__{effect}" for metric in METRICS
                      for effect in EFFECTS]

    descriptors = [dict(scope="overall", split="all", dataset="",
                        environment_id="")]
    descriptors += [dict(scope="split", split=split, dataset="",
                         environment_id="") for split in SPLITS]
    environment_keys = (meta.reset_index()[["split", "environment_id"]]
                        .drop_duplicates()
                        .sort_values(["split", "environment_id"],
                                     kind="mergesort"))
    assert len(environment_keys) == 8
    assert environment_keys.groupby("environment_id").split.nunique().eq(1).all()
    assert environment_keys.groupby("split").size().to_dict() == {
        "validation": 4, "test": 4}
    descriptors += [dict(scope="environment", split=row.split,
                         dataset="", environment_id=row.environment_id)
                    for row in environment_keys.itertuples()]

    records = []
    for selection in ("all", "nonoverlap"):
        frame = wide if selection == "all" else \
            wide[wide.nonoverlap_selected]
        selected_meta = meta if selection == "all" else \
            meta[meta.nonoverlap_selected]
        for descriptor in descriptors:
            if descriptor["scope"] == "overall":
                part = frame
                subset = selected_meta
                keys = ["selection"]
            elif descriptor["scope"] == "split":
                part = frame[frame.split == descriptor["split"]]
                subset = selected_meta[
                    selected_meta.split == descriptor["split"]]
                keys = ["split", "selection"]
            else:
                # Environment identity is the environment_id; two of the
                # eight environments carry two dataset branches.  The
                # official hierarchy averages those dataset branches with
                # equal dataset weight after the hand/episode levels.
                mask = ((frame.split == descriptor["split"]) &
                        (frame.environment_id ==
                         descriptor["environment_id"]))
                part = frame[mask]
                subset = selected_meta[
                    (selected_meta.split == descriptor["split"]) &
                    (selected_meta.environment_id ==
                     descriptor["environment_id"])]
                keys = ["split", "environment_id", "selection"]
            assert len(part) == len(subset) and len(part) > 0
            grouped = part.assign(selection=selection)
            result = aggregate(grouped, keys, metric_columns)
            assert len(result) == 1
            row = result.iloc[0]
            counts = scope_counts(subset)
            sources = "|".join(sorted(subset.dataset.unique()))
            for metric in METRICS:
                cell = {name: float(row[f"{metric}__{name}"])
                        for name in SCENARIOS.values()}
                effects = {name: float(row[f"{metric}__{name}"])
                           for name in EFFECTS}
                # Official hierarchy is linear, so the aggregated interaction
                # equals the interaction of the four aggregated cells.
                assert abs(effects["did"] - (
                    (cell["short_d02"] - cell["short_d01"]) -
                    (cell["full_d02"] - cell["full_d01"]))) < 1e-15
                assert abs(effects["history_main"] - (
                    ((cell["short_d01"] - cell["full_d01"]) +
                     (cell["short_d02"] - cell["full_d02"])) / 2)) < 1e-15
                assert abs(effects["delay_main"] - (
                    ((cell["full_d02"] - cell["full_d01"]) +
                     (cell["short_d02"] - cell["short_d01"])) / 2)) < 1e-15
                records.append({
                    "selection": selection,
                    "scope": descriptor["scope"],
                    "split": descriptor["split"],
                    "dataset": descriptor["dataset"],
                    "environment_id": descriptor["environment_id"],
                    "sources": sources,
                    "metric": metric,
                    "metric_role": ("pre-specified" if metric == "nade"
                                    else "descriptive"),
                    **counts,
                    **cell,
                    **effects,
                    "sign_did": sign(effects["did"]),
                    "sign_short_delay": sign(effects["short_delay"]),
                })
    summary = pd.DataFrame(records)[SUMMARY_COLUMNS]
    return wide, summary


def direction_checks(summary):
    """Direction of the NADE delay effect in each of the eight environments."""
    environments = summary[(summary.scope == "environment") &
                           (summary.metric == "nade")]
    output = {}
    sources = (environments.drop_duplicates(["split", "environment_id"])
               .set_index(["split", "environment_id"]).sources)
    for effect in ("did", "short_delay"):
        table = environments.pivot(index=["split", "environment_id"],
                                   columns="selection", values=effect)
        assert len(table) == 8
        assert set(table.columns) == {"all", "nonoverlap"}
        details = []
        pairwise = []
        for index, row in table.iterrows():
            split, environment = index
            signs = {"all": sign(row["all"]),
                     "nonoverlap": sign(row["nonoverlap"])}
            details.append({
                "split": split,
                "environment_id": environment,
                "sources": sources.loc[(split, environment)],
                "all": float(row["all"]),
                "nonoverlap": float(row["nonoverlap"]),
                "sign_all": signs["all"],
                "sign_nonoverlap": signs["nonoverlap"],
                "same_sign": bool(signs["all"] == signs["nonoverlap"] != 0),
            })
            pairwise.append(bool(signs["all"] == signs["nonoverlap"] != 0))
        consistent = {}
        sign_counts = {}
        for selection in ("all", "nonoverlap"):
            values = [sign(row[selection]) for _, row in table.iterrows()]
            consistent[selection] = bool(len(set(values)) == 1 and
                                         values[0] != 0)
            sign_counts[selection] = {
                "positive": int(sum(value == 1 for value in values)),
                "negative": int(sum(value == -1 for value in values)),
                "zero": int(sum(value == 0 for value in values)),
            }
        overall = summary[(summary.scope == "overall") &
                          (summary.metric == "nade")]
        output[effect] = {
            "by_environment": details,
            "sign_counts": sign_counts,
            "pairwise_sign_agreement_all8": bool(all(pairwise)),
            "same_sign_across_all8": consistent,
            "replicated": bool(all(pairwise) and consistent["all"] and
                               consistent["nonoverlap"]),
            "overall": {row.selection: {
                "value": float(getattr(row, effect)),
                "sign": sign(getattr(row, effect)),
            } for row in overall.itertuples()},
        }
    return output


def compare_prior(our, reference, columns, split_check=True):
    pair = our.merge(reference, on="window_id", suffixes=("_ours", "_ref"),
                     validate="one_to_one")
    assert len(pair) == len(our) == len(reference)
    if split_check:
        assert pair.split_ours.eq(pair.split_ref).all()
    differences = {
        column: float(np.max(np.abs(
            pair[column + "_ours"].to_numpy(float) -
            pair[column + "_ref"].to_numpy(float))))
        for column in columns
    }
    return {
        "windows": int(len(pair)),
        "by_split": {split: int((pair.split_ours == split).sum())
                     for split in SPLITS},
        "max_abs_differences": differences,
        "max_abs_difference": float(max(differences.values())),
        "points_equal": bool(pair.points_ours.eq(pair.points_ref).all()),
    }


def main():
    before = verify_snapshot(snapshot())
    tag = subprocess.check_output(
        ["git", "-C", str(SPACE), "rev-parse", "v1.0.0-final^{}"],
        text=True).strip()
    assert tag == COMMIT, tag
    source_paths = {
        "results/final_v1_remaining/trajectory/replay.py":
            PRIOR_TRAJECTORY / "replay.py",
        "results/final_v1_sensitivity/q2/run_q2.py":
            SENSITIVITY / "q2/run_q2.py",
        "results/final_v1_sensitivity/run_q1_q3.py":
            SENSITIVITY / "run_q1_q3.py",
        "正式方案/code/adopted.py": FORMAL / "code/adopted.py",
        "第二问/delivery/code/replay_engine.py":
            DELIVERY / "code/replay_engine.py",
        "第二问/delivery/code/rotation_inference.py":
            DELIVERY / "code/rotation_inference.py",
        "第二问/delivery/code/run_complete.py":
            DELIVERY / "code/run_complete.py",
    }
    source_hashes = {name: digest(path)
                     for name, path in source_paths.items()}
    prior_hash = digest(PRIOR_WINDOWS)
    prior_q2_hash = digest(PRIOR_Q2)
    prior_environments_hash = digest(PRIOR_ENVIRONMENTS)

    windows = base.official_windows()
    assert len(windows) == 602 and windows.window_id.is_unique
    assert windows.groupby("split").size().to_dict() == {
        "validation": 212, "test": 390}
    assert windows.environment_id.nunique() == 8
    assert sorted(windows.groupby("split").environment_id.nunique()
                  .to_dict().items()) == [("test", 4), ("validation", 4)]
    window_span = (windows.end_s - windows.start_s).to_numpy(float)
    assert np.max(np.abs(window_span - 1.0)) < 1e-12

    selected = selection_map(windows)
    disjoint_pairs = check_disjoint(windows, selected)
    assert selection_map(windows) == selected
    selected_count = int(sum(selected.values()))

    model = AdoptedModel()
    per_window, stats = replay_windows(model, windows, selected)
    assert len(per_window) == 602 * 4
    assert not per_window.duplicated(["window_id", "history", "delay"]).any()
    for label, delay in SCENARIOS:
        part = per_window[(per_window.history == label) &
                          (per_window.delay == delay)]
        assert len(part) == 602
        assert part.groupby("split").size().to_dict() == {
            "validation": 212, "test": 390}
    assert np.isfinite(per_window[list(METRICS) + ["nfde"]].to_numpy(float)).all()
    assert per_window.points.gt(0).all()
    assert per_window.requests.eq(BUDGET).all()
    assert per_window.successful_returns.eq(BUDGET).all()
    assert per_window.returns.eq(BUDGET).all()
    assert np.allclose(per_window.waiting_s,
                       per_window.delay * BUDGET, rtol=0, atol=1e-12)
    alignment = per_window.groupby("window_id").nonoverlap_selected.nunique()
    assert alignment.eq(1).all()

    wide, summary = build_summary(per_window)
    effects = direction_checks(summary)
    leak = leak_perturbation(model, windows)

    prior = pd.read_csv(PRIOR_WINDOWS)
    prior = prior[prior.question == "q2"].copy()
    prior_checks = {}
    for condition, delay in (("nominal", 0.1), ("stress", 0.2)):
        ours = per_window[(per_window.history == "full") &
                          (per_window.delay == delay)]
        reference = prior[prior.condition == condition]
        check = compare_prior(ours, reference,
                              ["nade", "coverage", "norm_diameter"])
        assert check["windows"] == 602
        assert check["points_equal"]
        assert check["max_abs_difference"] < 1e-10
        prior_checks[f"full_d{delay:.1f}_vs_{condition}"] = {
            "source": "results/final_v1_remaining/trajectory/"
                      "windows.csv.gz question=q2 model=B",
            **check,
            "nade_difference_below_1e-10":
                check["max_abs_differences"]["nade"] < 1e-10,
        }

    sensitivity = pd.read_csv(PRIOR_Q2)
    sensitivity = sensitivity[sensitivity.split == "validation"]
    sensitivity_checks = {}
    for delay in DELAYS:
        ours = per_window[(per_window.history == "full") &
                          (per_window.delay == delay) &
                          (per_window.split == "validation")]
        reference = sensitivity[sensitivity.delay == delay]
        check = compare_prior(ours, reference,
                              ["nade", "coverage", "norm_diameter"])
        assert check["windows"] == 212
        assert check["points_equal"]
        assert check["max_abs_difference"] < 1e-10
        sensitivity_checks[f"full_d{delay:.1f}"] = {
            "source": "results/final_v1_sensitivity/q2/per_window.csv.gz "
                      "split=validation",
            **check,
        }

    # The archived environment table uses the same official hierarchy:
    # window -> hand -> episode -> (dataset, environment) -> dataset.
    environment_reference = pd.read_csv(PRIOR_ENVIRONMENTS)
    environment_reference = environment_reference[
        (environment_reference.question == "q2") &
        (environment_reference.scope == "environment")].copy()
    assert len(environment_reference) == 8
    environment_rows = []
    for (split, environment), group in summary[
            (summary.scope == "environment") &
            (summary.selection == "all")].groupby(["split",
                                                   "environment_id"]):
        record = {
            "split": split,
            "environment": environment,
            "windows": int(group.n_windows.iloc[0]),
            "episodes": int(group.n_episodes.iloc[0]),
            "sources": group.sources.iloc[0],
        }
        for metric in METRICS:
            prior_name = {"nade": "nade", "coverage": "coverage",
                          "norm_diameter": "diameter"}[metric]
            part = group[group.metric == metric].iloc[0]
            record[f"{prior_name}_nominal"] = float(part.full_d01)
            record[f"{prior_name}_stress"] = float(part.full_d02)
        environment_rows.append(record)
    environment_pair = pd.DataFrame(environment_rows).merge(
        environment_reference, on=["split", "environment"],
        suffixes=("_ours", "_ref"), validate="one_to_one")
    assert len(environment_pair) == 8
    environment_differences = {
        f"{name}_{condition}": float(np.max(np.abs(
            environment_pair[f"{name}_{condition}_ours"] -
            environment_pair[f"{name}_{condition}_ref"])))
        for name, condition in ((n, c) for n in
                                ("nade", "coverage", "diameter")
                                for c in ("nominal", "stress"))
    }
    assert max(environment_differences.values()) < 1e-10
    assert environment_pair.windows_ours.eq(
        environment_pair.windows_ref).all()
    assert environment_pair.episodes_ours.eq(
        environment_pair.episodes_ref).all()
    assert environment_pair.sources_ours.eq(
        environment_pair.sources_ref).all()
    environment_check = {
        "source": "results/final_v1_remaining/environment_results.csv "
                  "question=q2 scope=environment",
        "environments": int(len(environment_pair)),
        "max_abs_differences": environment_differences,
        "max_abs_difference": float(max(environment_differences.values())),
        "windows_equal": True,
        "episodes_equal": True,
        "sources_equal": True,
    }

    per_window.to_csv(OUT_WINDOWS, index=False, compression="gzip")
    summary.to_csv(OUT_SUMMARY, index=False)

    after = verify_snapshot(snapshot())
    assert after == before
    after_tag = subprocess.check_output(
        ["git", "-C", str(SPACE), "rev-parse", "v1.0.0-final^{}"],
        text=True).strip()
    assert after_tag == tag
    after_source_hashes = {name: digest(path)
                           for name, path in source_paths.items()}
    assert after_source_hashes == source_hashes
    assert digest(PRIOR_WINDOWS) == prior_hash
    assert digest(PRIOR_Q2) == prior_q2_hash

    overall = summary[(summary.scope == "overall") &
                      (summary.metric == "nade")].set_index("selection")

    audit = {
        "protocol": {
            "kind": "pre-specified Q2 2x2 factorial replay; no selection",
            "question": 2,
            "model": "B",
            "frozen_commit": COMMIT,
            "tag": {"name": "v1.0.0-final", "resolved": tag,
                    "locked": tag == COMMIT},
            "windows": {"validation": 212, "test": 390, "total": 602,
                        "environments": 8,
                        "environments_by_split": {"validation": 4,
                                                  "test": 4},
                        "dataset_environment_branches": 10,
                        "environment_identity":
                            "environment_id; two of the eight environments "
                            "carry two dataset branches, which the official "
                            "hierarchy averages with equal dataset weight"},
            "mask": {
                "kind": "fixed official one-second [start, end) window",
                "window_span_s": {
                    "min": float(window_span.min()),
                    "max": float(window_span.max()),
                },
                "scored_frames": {str(int(k)): int(v) for k, v in
                                  (windows.end_index_exclusive -
                                   windows.start_index).value_counts()
                                  .sort_index().items()},
                "note": "No age re-masking; this is exactly the frozen Q2 "
                        "replay window.  The last frame can sit on the "
                        "one-second boundary.",
            },
            "history_factor": {
                "full": "history_since(raw, anchor, inf)",
                "short": "history_since(raw, anchor, 0.30)",
                "channels": "all clocked input channels, each cropped on its "
                            "own __t_s clock",
                "entry_point": "initial forecast only",
                "initial_forecast": "predict_model(weights, cropped history, "
                                    "side, anchor, ages, dataset) + observed "
                                    "anchor position",
                "replay_history": "two observed pre-anchor positions "
                                  "t[start-2:start], x[start-2:start]",
            },
            "delay_factor": {"nominal_s": 0.1, "stress_s": 0.2},
            "requests_s": [0.3, 0.6],
            "budget": BUDGET,
            "state_radius": "frozen run_q2.range_state; official prior/"
                            "return radii unchanged",
            "hierarchy": ["window", "hand", "episode", "environment",
                          "dataset"],
            "effects": {
                "cell": "full/short history at delay 0.1/0.2",
                "history_main": "average over delays of short minus full",
                "delay_main": "average over histories of 0.2 minus 0.1",
                "did": "additive interaction = (short 0.2 - short 0.1) - "
                       "(full 0.2 - full 0.1)",
                "short_delay": "short 0.2 - short 0.1",
                "full_delay": "full 0.2 - full 0.1",
            },
        },
        "inputs": {
            "workspace": str(WORKSPACE),
            "source_space": str(SPACE),
            "protocol_windows": str(run_q2.PROTOCOL / "data/windows.csv"),
            "records": str(RECORDS),
            "episodes": int(windows.episode_uuid.nunique()),
            "model_path": str(base.MODEL_PATH),
            "delivery_model_path": str(base.DELIVERY_MODEL_PATH),
            "range_model_path": str(base.RANGE_MODEL_PATH),
            "prior_windows": {"path": str(PRIOR_WINDOWS),
                              "sha256": prior_hash},
            "prior_q2_per_window": {"path": str(PRIOR_Q2),
                                    "sha256": prior_q2_hash},
            "prior_environment_results": {
                "path": str(PRIOR_ENVIRONMENTS),
                "sha256": prior_environments_hash,
            },
            "source_sha256": source_hashes,
            "credentials_used": False,
            "model_or_source_edits": False,
        },
        "integrity": {
            "expected_commit": COMMIT,
            "expected_model_sha256": MODEL_SHA256,
            "before": before,
            "after": after,
            "snapshot_unchanged": after == before,
            "head_verified": before["head"] == COMMIT == after["head"],
            "tag_verified": tag == COMMIT == after_tag,
            "model_verified": (before["model_sha256"] == MODEL_SHA256 ==
                               after["model_sha256"]),
            "source_hashes_unchanged": after_source_hashes == source_hashes,
            "prior_files_unchanged": (
                digest(PRIOR_WINDOWS) == prior_hash and
                digest(PRIOR_Q2) == prior_q2_hash and
                digest(PRIOR_ENVIRONMENTS) == prior_environments_hash),
        },
        "checks": {
            "window_counts": {"validation": 212, "test": 390,
                              "total": 602, "unique": True},
            "rows": {
                "per_window": len(per_window),
                "expected": 602 * 4,
                "scenarios": {name: int((per_window.scenario == name).sum())
                              for name in SCENARIOS.values()},
                "unique_window_history_delay": True,
                "finite_metrics": True,
                "points_positive": True,
                "requests_equal_budget": True,
                "returns_equal_budget": True,
                "waiting_equals_delay_times_budget": True,
                "scenario_alignment_across_selections": True,
            },
            "full_history_equivalence": {
                "full_history_since_equals_frozen_past_only": {
                    "windows": stats[
                        "full_history_equals_past_only_windows"],
                    "array_mismatches": stats[
                        "full_history_array_mismatches"],
                    "comparison": "all keys, np.array_equal(equal_nan=True)",
                },
                "max_initial_forecast_difference_vs_official": stats[
                    "max_initial_forecast_difference_full_vs_official"],
                "crop_assertions": stats["crop_assertions"],
            },
            "prior_regressions": {
                "full_history_vs_final_v1_remaining": prior_checks,
                "full_history_validation_vs_sensitivity_q2":
                    sensitivity_checks,
                "full_history_environment_level_vs_final_v1_remaining":
                    environment_check,
            },
            "no_future_leak": {
                "replay_history_observations_checked":
                    stats["replay_history_observations_checked"],
                "replay_history_rule": "only t[start-2:start] and "
                                       "x[start-2:start], both at or before "
                                       "the anchor, enter the engine",
                "future_perturbation": leak,
                "outcome_blind_selection": True,
            },
            "nonoverlap_sensitivity": {
                "rule": "within (episode_uuid, side), sort by (start_index, "
                        "end_index_exclusive, window_id); keep a window when "
                        "start_index >= end of the last kept window",
                "uses_outcomes": False,
                "selected_windows": selected_count,
                "excluded_windows": len(windows) - selected_count,
                "disjoint_adjacent_pairs_checked": disjoint_pairs,
                "selected_by_split": {
                    split: int(sum(selected[w] for w in
                                   windows[windows.split == split].window_id))
                    for split in SPLITS},
                "recomputed_selection_identical": True,
                "summary_repeated": True,
                "summary_equals_all_windows": bool(np.allclose(
                    summary[summary.selection == "nonoverlap"][
                        list(EFFECTS) + list(SCENARIOS.values())
                    ].to_numpy(float),
                    summary[summary.selection == "all"][
                        list(EFFECTS) + list(SCENARIOS.values())
                    ].to_numpy(float),
                    rtol=0, atol=1e-15)),
            },
            "direction": effects,
            "direction_conclusion": {
                effect: ("replicated: same non-zero sign in all eight "
                         "environments under both selections"
                         if result["replicated"] else
                         "not replicated: environment signs are not uniform "
                         "under both selections")
                for effect, result in effects.items()
            },
        },
        "output": {
            "directory": str(HERE),
            "files": [str(OUT_WINDOWS), str(OUT_SUMMARY), str(OUT_AUDIT)],
            "outside_directory_writes": False,
            "per_window": {
                "file": str(OUT_WINDOWS),
                "rows": len(per_window),
                "columns": COLUMNS,
                "compression": "gzip",
            },
            "summary": {
                "file": str(OUT_SUMMARY),
                "rows": len(summary),
                "columns": SUMMARY_COLUMNS,
                "selections": ["all", "nonoverlap"],
                "scopes": ["overall", "split", "environment"],
            },
        },
    }
    OUT_AUDIT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf8")

    print("overall NADE cells and effects")
    print(overall[list(SCENARIOS.values()) + list(EFFECTS)]
          .to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nby split NADE")
    print(summary[(summary.scope == "split") & (summary.metric == "nade")]
          [["selection", "split", "n_windows", "did", "short_delay"]]
          .to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nby environment NADE interaction")
    print(summary[(summary.scope == "environment") &
                  (summary.metric == "nade")]
          [["selection", "split", "environment_id", "n_windows", "did",
            "short_delay"]]
          .to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nprior max abs differences")
    print("  full 0.1:", prior_checks["full_d0.1_vs_nominal"]
          ["max_abs_difference"])
    print("  full 0.2:", prior_checks["full_d0.2_vs_stress"]
          ["max_abs_difference"])
    print("\nnon-overlap selected:", selected_count, "of", len(windows))
    print("future perturbation unchanged:", leak["unchanged"])
    for effect, result in effects.items():
        print(f"{effect} direction replicated: {result['replicated']} "
              f"(pairwise {result['pairwise_sign_agreement_all8']}, "
              f"all8 {result['same_sign_across_all8']})")
    print(f"wrote {OUT_WINDOWS} ({len(per_window)} rows)")
    print(f"wrote {OUT_SUMMARY} ({len(summary)} rows)")
    print(f"wrote {OUT_AUDIT}")


if __name__ == "__main__":
    main()
