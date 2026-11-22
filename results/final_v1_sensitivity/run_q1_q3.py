"""Reproduce the final-model history stress test and paired scheduling summaries."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TAG = ROOT / "tmp/final_v1_20261122"
SPACE = TAG / "C题工作区"
FINAL = SPACE / "正式方案"
FROZEN = FINAL / "results"
COMMIT = "9f8bc05fae9d8ce32da7fa7c5e28a82960085819"
sys.path.insert(0, str(FINAL / "code"))
from adopted import AdoptedModel
from trajectory import PROTOCOL


def history_since(raw, anchor, span):
    """Crop each channel on its own clock. Preserve the predictor's availability flags."""
    masks = {key.split("__")[0]: (values >= anchor - span) & (values <= anchor)
             for key, values in raw.items() if key.endswith("__t_s")}
    result = {}
    for key, values in raw.items():
        mask = masks.get(key.split("__")[0])
        result[key] = values[mask] if mask is not None and values.ndim and len(values) == len(mask) else values
    return result


def q1():
    model = AdoptedModel()
    windows = pd.read_csv(PROTOCOL / "data/windows.csv").query("split == 'validation'")
    assert len(windows) == windows.window_id.nunique() == 212
    rows = []
    for episode, group in windows.groupby("episode_uuid"):
        with np.load(ROOT / "C_Problem_Data/v1/v1/records" / f"{episode}.npz", allow_pickle=False) as data:
            raw = dict(data)
        for w in group.itertuples():
            clock = raw[w.side + "__t_s"]
            start, stop = w.start_index, w.end_index_exclusive
            anchor = clock[start - 1]
            ages = clock[start:stop] - anchor
            true_position = raw[w.side + "__position"][start:stop].mean(axis=1)
            # The registered window may end just after 1 s. Q1 scores by age from the anchor.
            supported = ages <= 1.0 + 1e-12
            ages, true_position = ages[supported], true_position[supported]
            scale = model.paid["scale"][w.dataset]
            for label, span in [("full", np.inf), ("0.75s", 0.75), ("0.30s", 0.30)]:
                past = history_since(raw, anchor, span)
                observed_clock = past[w.side + "__t_s"]
                assert observed_clock[-1] == anchor
                position, radius = model.predict_position(past, w.side, len(observed_clock)-1, ages, w.dataset)
                error = np.linalg.norm(position - true_position, axis=1)
                for horizon in (0.2, 0.6, 1.0):
                    keep = ages <= horizon + 1e-12
                    rows.append(dict(history=label, horizon=horizon, dataset=w.dataset,
                                     environment_id=w.environment_id, episode_uuid=episode, window_id=w.window_id,
                                     nade=error[keep].mean()/scale, nfde=error[keep][-1]/scale,
                                     coverage=(error[keep] <= radius[keep]).mean(), points=int(keep.sum())))
    scores = pd.DataFrame(rows)
    assert len(scores) == 212 * 3 * 3 and not scores.duplicated(["window_id", "history", "horizon"]).any()
    assert np.isfinite(scores[["nade", "nfde", "coverage"]]).all().all()
    # Each row is one window. Roll up episodes, environments and sources with equal weight.
    keys = ["history", "horizon", "dataset", "environment_id", "episode_uuid"]
    means = scores
    for n in (5, 4, 3, 2):
        means = means.groupby(keys[:n], as_index=False)[["nade", "nfde", "coverage"]].mean()
    nominal = means.query("history == 'full'").set_index("horizon").sort_index()
    ref = pd.read_csv(FROZEN / "q1_windows.csv.gz").query("split == 'validation' and model == 'B'")
    pairs = scores.query("history == 'full'").merge(ref, on=["window_id", "horizon"], validate="one_to_one")
    assert len(pairs) == 636
    for metric in ("nade", "nfde"):
        np.testing.assert_allclose(pairs[metric+"_x"], pairs[metric+"_y"], rtol=0, atol=1e-10)
    ref = pd.read_csv(FROZEN/"q1_range_summary.csv").query("split == 'validation' and model == 'B'")
    np.testing.assert_allclose(nominal.coverage, ref.set_index("horizon").sort_index().coverage, atol=1e-12, rtol=0)
    means.to_csv(OUT/"q1_history.csv", index=False)
    scores.to_csv(OUT/"q1_history_windows.csv.gz", index=False)


def q3():
    trials = pd.read_csv(FROZEN / "q3/b_k64/trials.csv.gz")
    keys = ["alpha", "beta", "case_id", "seed"]
    assert set(trials.method) == {"B_K64"} and len(trials) == 450 and not trials.duplicated(keys).any()
    assert trials.valid_protocol.all() and trials.invalid_actions.eq(0).all()
    matrix = trials.pivot(index="seed", columns=["alpha", "beta", "case_id"], values="weighted_on_time_rate").sort_index(axis=1)
    assert matrix.shape == (30, 15) and matrix.notna().all().all()
    rng = np.random.default_rng(20261122)
    draws = rng.integers(0, 30, size=(10000, 30))
    nominal = matrix[1.0, 0.5].mean(axis=1)
    rows = []
    for (alpha, beta), part in trials.groupby(["alpha", "beta"]):
        cases = matrix[alpha, beta]
        assert set(cases) == {"S1", "S2", "S3"}
        mean = cases.mean(axis=1)
        change = 100 * (mean - nominal).to_numpy()
        ci = np.quantile(change[draws].mean(axis=1), [.025, .975])
        rows.append(dict(alpha=alpha, beta=beta, **cases.mean().to_dict(),
                         on_time=mean.mean(), delta_pp=change.mean(), low_pp=ci[0], high_pp=ci[1],
                         completion=part.weighted_completion_rate.mean(), tardiness_s=part.weighted_tardiness_s.mean()))
    result = pd.DataFrame(rows)
    np.testing.assert_allclose([nominal.mean(), result.on_time.mean()], [.90960648,.88960648], atol=1e-7, rtol=0)
    result.to_csv(OUT/"q3_mapping.csv", index=False)


if __name__ == "__main__":
    for ref in ("HEAD", "v1.0.0-final^{}"):
        assert subprocess.check_output(["git", "-C", str(TAG), "rev-parse", ref], text=True).strip() == COMMIT
    q1()
    q3()
