#!/usr/bin/env python
"""Q2 deployed-model delay sensitivity on the frozen 212 validation windows.

Protocol: full one-second windows, fixed 0.3/0.6 s requests, budget 2,
delays 0/0.1/0.2 s.  Only the deployed B state/forecast is evaluated.  The
official per-point radii (prior before the first return, return model after)
are left unchanged at every delay.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SENSITIVITY = HERE.parent
RESULTS = SENSITIVITY.parent
SPACE = RESULTS.parent
SNAPSHOT = SPACE / "tmp/final_v1_20261122/C题工作区"
FORMAL = SNAPSHOT / "正式方案"
PACKAGE = SNAPSHOT / "独立方案B_旋转模型"
DELIVERY = PACKAGE / "第二问/delivery"
PROTOCOL = SNAPSHOT / "阶段交付/stage1_review/corrected_results"

sys.path.insert(0, str(FORMAL / "code"))
sys.path.insert(0, str(DELIVERY / "code"))

from adopted import AdoptedModel  # noqa: E402
from replay_engine import FixedPolicy, MotionState, replay  # noqa: E402
from rotation_inference import predict_model  # noqa: E402
from rotation_state import factory  # noqa: E402
from run_complete import aggregate, radius  # noqa: E402

DELAYS = (0.0, 0.1, 0.2)
NOMINAL = 0.1
BUDGET = 2


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_records(uids):
    """Frozen raw-record directory, resolved relative to this file."""
    candidates = (
        SNAPSHOT / "处理结果/v1/records",
        SPACE / "C_Problem_Data/v1/v1/records",
        RESULTS / "scheme_b_replay_4faa8cf/C题工作区/处理结果/v1/records",
    )
    for candidate in candidates:
        if all((candidate / f"{uid}.npz").is_file() for uid in uids):
            return candidate
    raise FileNotFoundError("no frozen raw-record directory contains all validation episodes")


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def past_only(raw, anchor):
    past = {}
    for key, values in raw.items():
        clock = raw.get(key.split("__")[0] + "__t_s")
        if clock is not None and values.ndim and len(values) == len(clock):
            past[key] = values[clock <= anchor].copy()
        else:
            past[key] = values.copy()
    return past


def range_state(times, initial, source, paid, prior):
    """The replay_adopted state wrapper: radius is recorded on every predict."""
    radii = []
    return_model = paid["return_models"][source]
    prior_model = prior[source]
    scale = paid["scale"][source]

    class WithRange(factory(times, initial, MotionState)):
        def predict(self, now):
            prediction = super().predict(now)
            params = return_model if self.returns else prior_model
            radii.append(radius(params, now - self.last_time,
                                np.linalg.norm(self.velocity), scale))
            return prediction

    return WithRange, radii


def main():
    model = AdoptedModel()
    cfg, prior, weights = model.paid, model.prior, model.parameters
    assert PACKAGE.joinpath("第一问/model/model_parameters.npz").read_bytes() == \
        DELIVERY.joinpath("model/rotation_full.npz").read_bytes()

    windows = pd.read_csv(DELIVERY / "data/windows.csv")
    windows = windows[windows.split == "validation"].copy()
    assert len(windows) == 212 and windows.window_id.is_unique
    official = pd.read_csv(PROTOCOL / "data/windows.csv")
    official = official[official.split == "validation"]
    columns = ["window_id", "episode_uuid", "dataset", "environment_id", "side",
               "start_index", "end_index_exclusive"]
    assert windows[columns].reset_index(drop=True).equals(
        official[columns].reset_index(drop=True))

    records = find_records(windows.episode_uuid.unique())
    expected = json.loads((DELIVERY / "results/input_hashes.json").read_text("utf8"))
    expected = {key.replace("\\", "/").rsplit("/", 1)[-1]: value for key, value in expected.items()}
    validated = {}
    for uid in windows.episode_uuid.unique():
        path = records / f"{uid}.npz"
        actual = digest(path)
        assert expected.get(f"{uid}.npz") == actual, f"frozen input mismatch: {path}"
        validated[f"{uid}.npz"] = actual

    rows = []
    for uid, group in windows.groupby("episode_uuid", sort=False):
        raw = load_arrays(records / f"{uid}.npz")
        for row in group.itertuples():
            side = row.side
            t = raw[side + "__t_s"]
            x = raw[side + "__position"].mean(1)
            i, j = int(row.start_index), int(row.end_index_exclusive)
            anchor = float(t[i - 1])
            times = t[i:j]
            reference = x[i:j]
            initial = predict_model(weights, past_only(raw, anchor), side, anchor,
                                    times - anchor, row.dataset) + x[i - 1]
            scale = cfg["scale"][row.dataset]
            assert len(times) >= 2 and np.all(np.diff(times) > 0)
            for delay in DELAYS:
                State, radii = range_state(times, initial, row.dataset, cfg, prior)
                run = replay(times, reference, t[i - 2:i], x[i - 2:i], FixedPolicy(),
                             delay=delay, budget=BUDGET, state_factory=State)
                prediction = np.asarray([point["prediction"] for point in run["points"]])
                truth = np.asarray([point["reference"] for point in run["points"]])
                errors = np.linalg.norm(prediction - truth, axis=1) / scale
                radii = np.asarray(radii, float) / scale
                assert len(radii) == len(errors) and len(run["points"]) == len(times)
                assert np.isfinite(errors).all() and np.isfinite(radii).all()
                returns = sum(action["event"] == "return" for action in run["actions"])
                rows.append(dict(
                    window_id=row.window_id, episode_uuid=uid, dataset=row.dataset,
                    environment_id=row.environment_id, split=row.split, side=side,
                    delay=delay, points=len(times), nade=float(errors.mean()),
                    nfde=float(errors[-1]), coverage=float((errors <= radii).mean()),
                    norm_diameter=float(2 * radii.mean()),
                    returns=returns,
                    successful_returns=int(run["metrics"]["successful_returns"]),
                    requests=int(run["metrics"]["requests"]),
                    waiting_s=float(run["metrics"]["waiting_s"])))

    per_window = pd.DataFrame(rows)
    assert len(per_window) == 212 * len(DELAYS)
    assert per_window.groupby("delay").size().eq(212).all()
    assert per_window[["returns", "successful_returns", "requests"]].eq(BUDGET).all().all()
    assert np.isfinite(per_window[["nade", "coverage", "norm_diameter"]].to_numpy()).all()
    per_window.to_csv(HERE / "per_window.csv.gz", index=False)

    metrics = ["nade", "nfde", "coverage", "norm_diameter", "returns",
               "successful_returns", "requests", "waiting_s"]
    summary = aggregate(per_window, ["delay"], metrics)
    summary.insert(1, "windows", per_window.groupby("delay").size().reindex(summary.delay).to_numpy())
    summary.insert(7, "total_returns", per_window.groupby("delay").returns.sum()
                   .reindex(summary.delay).to_numpy(dtype=int))
    summary.to_csv(HERE / "summary.csv", index=False)

    print(summary.to_string(index=False))

    # Archived references: nominal B row of the official Q2 summary and the
    # rotation_switch delay rows of the archived position-sensitivity study.
    q2 = pd.read_csv(FORMAL / "results/q2_summary.csv")
    q2 = q2[(q2.split == "validation") & (q2.model == "B")].iloc[0]
    nominal = summary[np.isclose(summary.delay, NOMINAL)].iloc[0]
    q2_diff = {key: float(abs(nominal[ours] - q2[key]))
               for ours, key in (("nade", "nade"), ("nfde", "nfde"),
                                 ("coverage", "coverage"), ("norm_diameter", "width"),
                                 ("requests", "requests"), ("waiting_s", "waiting_s"))}
    print("nominal q2 max difference", max(q2_diff.values()), q2_diff)

    sensitivity = pd.read_csv(DELIVERY / "evidence/position_sensitivity/summary.csv")
    sensitivity = sensitivity[(sensitivity.split == "validation") &
                              (sensitivity.method == "rotation_switch") &
                              np.isclose(sensitivity.horizon, 1.0)]
    compare = []
    for delay in DELAYS:
        archived = sensitivity[np.isclose(sensitivity.delay, delay)].iloc[0]
        ours = summary[np.isclose(summary.delay, delay)].iloc[0]
        compare.append(dict(delay=delay, archived=float(archived.nade),
                            observed=float(ours.nade),
                            difference=float(ours.nade - archived.nade),
                            archived_nfde=float(archived.nfde),
                            observed_nfde=float(ours.nfde),
                            nfde_difference=float(ours.nfde - archived.nfde),
                            archived_requests=float(archived.requests),
                            observed_requests=float(ours.requests)))
    print(pd.DataFrame(compare).to_string(index=False))

    reference = pd.read_csv(DELIVERY / "reference/validation_windows.csv")
    nominal_rows = per_window[np.isclose(per_window.delay, NOMINAL)]
    reference = nominal_rows.merge(reference, on="window_id", validate="one_to_one",
                                   suffixes=("_new", "_archived"))
    assert len(reference) == 212
    window_diff = {key: float((reference[key + "_new"] - reference[key + "_archived"]).abs().max())
                   for key in ("nade", "nfde", "requests", "waiting_s")}
    print("nominal per-window reference max difference", max(window_diff.values()), window_diff)
    assert max(q2_diff.values()) < 1e-10 and max(window_diff.values()) < 1e-10
    assert max(abs(item["difference"]) for item in compare) < 1e-10

    audit = dict(
        protocol=dict(windows=len(windows), delays=list(DELAYS), requests=[0.3, 0.6],
                      budget=BUDGET, mask="fixed full one-second window",
                      model="deployed B", radii="official per-point prior/return params unchanged"),
        inputs=dict(records=str(records), validation_records=len(validated),
                    hashes_verified_against="delivery/results/input_hashes.json",
                    model_sha256=digest(DELIVERY / "model/rotation_full.npz"),
                    test_reads=0),
        event_ordering=dict(
            zero_delay=("on the first scoring frame at/after .3/.6 the request is made, "
                        "its equal-time return is processed before prediction at that same "
                        "frame, and the after-return radius applies; observed NADE reproduces "
                        "the archived delay-0 value (difference <1e-16), so no scoring/event-"
                        "order discrepancy exists"),
            positive_delay=("a return is processed on the first scoring frame at/after its "
                            "arrival, before that frame's prediction; nominal and 0.2 s NADE "
                            "both reproduce the archive"),
            zero_delay_archived_nade_difference=float(compare[0]["difference"])),
        checks=dict(per_window_rows=len(per_window),
                    all_windows_all_delays=bool(len(per_window) == 636),
                    all_requests_equal_budget=bool(per_window.requests.eq(BUDGET).all()),
                    all_returns_positive=bool(per_window.returns.gt(0).all()),
                    total_return_events={str(k): int(v) for k, v in
                                         per_window.groupby("delay").returns.sum().items()},
                    nominal_per_window_reference=dict(
                        source="delivery/reference/validation_windows.csv rotation_switch",
                        max_abs_differences=window_diff,
                        max_abs_difference=float(max(window_diff.values()))),
                    nominal_q2_reference=dict(
                        source="正式方案/results/q2_summary.csv validation/B",
                        nade=float(q2.nade), nfde=float(q2.nfde),
                        coverage=float(q2.coverage), width=float(q2.width),
                        differences=q2_diff,
                        max_abs_difference=float(max(q2_diff.values()))),
                    archived_position_sensitivity=compare),
    )
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf8")


if __name__ == "__main__":
    main()
