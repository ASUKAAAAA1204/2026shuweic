"""Replay the five prespecified trajectory levels and match archived endpoints."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "final_v1_evidence_extension"
sys.path.insert(0, str(PRIOR / "trajectory"))
import factorial as q2

KEY = ["split", "dataset", "environment_id", "episode_uuid", "window_id"]


def q1_grid(model, windows):
    rows = []
    for episode, group in windows.groupby("episode_uuid"):
        with np.load(q2.RECORDS / f"{episode}.npz", allow_pickle=False) as archive:
            raw = dict(archive)
        for w in group.itertuples():
            i, j = w.start_index, w.end_index_exclusive
            clock = raw[w.side + "__t_s"]
            anchor = clock[i-1]
            ages = clock[i:j] - anchor
            supported = ages <= 1 + 1e-12
            truth = raw[w.side + "__position"][i:j].mean(1)[supported]
            ages = ages[supported]
            meta = {k: getattr(w, k) for k in KEY}
            for span in (.30, .45, .60, .75, np.inf):
                past = q2.history_since(raw, anchor, span)
                ix = len(past[w.side + "__t_s"]) - 1
                pos, _ = model.predict_position(past, w.side, ix, ages, w.dataset)
                error = np.linalg.norm(pos-truth, axis=1) / model.paid["scale"][w.dataset]
                for h in (.2, .6, 1.):
                    rows.append(dict(**meta, history="full" if np.isinf(span) else f"{span:.2f}s",
                                     horizon=h, nade=error[ages <= h+1e-12].mean()))
    return pd.DataFrame(rows)


def match(d, name, ref, keys):
    paired = ref.merge(d, on=keys, validate="one_to_one")
    assert len(paired) == len(ref)
    np.testing.assert_allclose(paired.nade_x, paired.nade_y, rtol=0, atol=1e-10)
    assert np.isfinite(d.nade).all() and d.groupby(keys).size().eq(1).all()
    d.to_csv(OUT / f"{name}_windows.csv.gz", index=False)
    print(name, len(d), "archive rows matched", len(paired), flush=True)


if __name__ == "__main__":
    initial = q2.verify_snapshot(q2.snapshot())
    windows = q2.base.official_windows()
    model = q2.AdoptedModel()
    q1 = q1_grid(model, windows)
    assert len(q1) == 602*15
    match(q1, "q1", pd.read_csv(PRIOR / "history_horizon_windows.csv.gz"), KEY + ["history","horizon"])
    # Supply the additional delay levels to the existing replay routine in this process.
    q2.DELAYS = (.10,.125,.15,.175,.20)
    q2.SCENARIOS = {(label,d): f"{label}_{d}" for label,_ in q2.HISTORIES for d in q2.DELAYS}
    d, _ = q2.replay_windows(model, windows, q2.selection_map(windows))
    keep = KEY + ["history","delay","nade"]
    assert len(d) == 602*10
    match(d[keep], "q2", pd.read_csv(PRIOR / "trajectory/per_window.csv.gz")[keep], KEY + ["history","delay"])
    assert initial == q2.snapshot()

