"""Independent check of the frozen Q2 main table from timestamps, payloads and states."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run import KEYS, OUT, Q1, REPO, SOL, blob, csv, hierarchical

OPEN = {"hold", "cv", "damped"}
METHODS = OPEN | {"fixed", "budget", "threshold", "delay_aware"}
TOL = 1e-10


def primary(frame):
    paid = ~frame.method.isin(OPEN)
    return frame.loc[(frame.length_s == 1) & (frame.delay_s == np.where(paid, .1, 0))]


def trajectories(index):
    cache = {}
    for uid in index.episode_uuid.unique():
        with np.load(io.BytesIO(blob(f"q1/C_Problem_Data/v1/v1/records/{uid}.npz", Q1))) as z:
            for side in ("left_hand", "right_hand"):
                cache[uid, side] = (z[side+"__t_s"], z[side+"__position"].mean(axis=1),
                                    z[side+"__reference_candidate"])
    return cache


def check_split(split, catalog, model):
    base = SOL + f"results/S5_{split}/"
    index = catalog.loc[catalog.split == split].set_index("window_id")
    cache = trajectories(index)
    traces = primary(csv(base+"points.csv.gz", compression="gzip"))
    logs = primary(csv(base+"actions.csv.gz", compression="gzip"))
    requests = logs.loc[logs.event == "request"].set_index(["window_id","method","request_id"])
    returns = logs.loc[logs.event == "return"].set_index(requests.index.names)
    assert requests.index.is_unique and returns.index.is_unique
    assert set(requests.index) == set(returns.index)
    assert set(traces.window_id) == set(index.index)
    grouped = traces.groupby(["window_id","method"], sort=False)
    assert set(traces.method) == METHODS and len(grouped) == len(index) * len(METHODS)
    by_action = {k:g for k,g in requests.reset_index().groupby(["window_id","method"])}
    cert = dict(split=split, windows=len(index), methods=len(METHODS), points=len(traces),
                method_windows=len(grouped), requests=len(requests), returns=len(returns),
                max_coordinate_difference=0.)
    scores = []
    for (wid, method), group in grouped:
        w = index.loc[wid]
        assert group[KEYS].eq(w[KEYS]).all().all()
        all_t, all_x, valid = cache[w.episode_uuid,w.side]
        i, j = int(w.start_index), int(w.end_index_exclusive)
        t, x = all_t[i:j], all_x[i:j]
        assert valid[i-2:j].all()
        np.testing.assert_array_equal(t, group.time)
        np.testing.assert_array_equal(x, group[["ref_x","ref_y","ref_z"]])
        latest = np.full(len(t), i-1, dtype=int)
        previous = np.full(len(t), i-2, dtype=int)
        actions = by_action.get((wid,method))
        count, wait = 0, 0.
        if actions is not None:
            prior_index, prior_due = i-1, -np.inf
            for req in actions.sort_values("request_id").itertuples():
                count += 1
                ret = returns.loc[wid, method, req.request_id]
                sample = np.searchsorted(t, req.time)
                due = req.time + .1
                delivery = np.searchsorted(t+1e-12, due)
                assert req.request_id == count and count <= 2
                assert sample < len(t) and delivery < len(t)
                assert req.time == req.sample_time == t[sample] and req.time >= prior_due-1e-12
                assert abs(req.arrival-due) < TOL and abs(ret.time-due) < TOL
                assert ret.processed_at == t[delivery] and ret.sample_time == req.time and ret.valid
                previous[delivery:], latest[delivery:] = prior_index, i+sample
                prior_index, prior_due = i+sample, due
                wait += req.arrival - req.time
        if method in OPEN:
            assert count == 0
        delta = t - all_t[latest]
        velocity = (all_x[latest]-all_x[previous]) / (all_t[latest]-all_t[previous])[:,None]
        motion = method if method in OPEN else "damped"
        factor = {"cv":delta, "hold":np.zeros_like(delta), "damped":-.2*np.expm1(-delta/.2)}[motion]
        prediction = all_x[latest] + factor[:,None]*velocity
        diff = np.max(np.abs(prediction-group[["pred_x","pred_y","pred_z"]].to_numpy()))
        cert["max_coordinate_difference"] = max(cert["max_coordinate_difference"],float(diff))
        assert diff < TOL
        np.testing.assert_allclose(delta, group.age, atol=TOL, rtol=0)
        error = np.linalg.norm(prediction-x,axis=1)
        np.testing.assert_allclose(error, group.error, atol=TOL, rtol=0)
        bins = np.clip(np.ceil(delta*30).astype(int),1,30)
        radius = np.array([model["radii"][motion][f"{w.dataset}|{b}"] for b in bins])
        np.testing.assert_array_equal(radius,group.radius)
        kset = [f"{w.dataset}|{b}" for b in bins]
        post = np.array([model["post_radius_exploratory"].get(key,r) if updated else r
                         for key,r,updated in zip(kset,radius,latest>=i,strict=True)])
        np.testing.assert_array_equal(post, group.radius_separate_exploratory)
        scale = model["scale"][w.dataset]
        scores.append(dict(**w[KEYS].to_dict(),window_id=wid,method=method,
            nade=error.mean()/scale,nfde=error[-1]/scale,coverage=(error<=radius).mean(),
            path_covered=float((error<=radius).all()),width=2*radius.mean()/scale,
            coverage_separate_exploratory=(error<=post).mean(),width_separate_exploratory=2*post.mean()/scale,
            within=(error<=.05*scale).mean(),requests=count,cost=count,waiting_s=wait))
    rows = pd.DataFrame(scores)
    metrics = [k for k in rows if k not in KEYS+["window_id","method"]]
    calculated = hierarchical(rows,["method"],metrics).set_index("method").sort_index()
    official = primary(csv(base+"summary.csv")).set_index("method").sort_index()
    official["within"] = 1-official["exceed_0.05"]
    np.testing.assert_allclose(calculated, official[metrics], atol=TOL,rtol=0)
    cert["max_main_metric_difference"] = float((calculated-official[metrics]).abs().max().max())
    return cert,calculated.reset_index().assign(split=split)


def main():
    catalog = csv("C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv")
    official_path = REPO / "赛题分析报告/record_split.csv"
    actual = pd.read_csv(official_path).sort_values("episode_uuid").reset_index(drop=True)
    roster = csv("q1/C_Problem_Data/record_split.csv",Q1).sort_values("episode_uuid").reset_index(drop=True)
    pd.testing.assert_frame_equal(roster,actual)
    model = json.loads(blob(SOL+"frozen_model.json"))
    checks, tables = [], []
    for split in ("validation","test"):
        check, table = check_split(split,catalog,model)
        checks.append(check)
        tables.append(table)
    paired = pd.read_csv(OUT/"replay_windows.csv.gz").query("plan=='both'")
    # Per-source and pooled-window estimates are sensitivity checks on the prespecified hierarchy.
    source = paired.groupby(["split","motion",*KEYS]).nade.mean()
    for depth in (5,4,3):
        source = source.groupby(level=list(range(depth))).mean()
    source = source.unstack("motion").reset_index()
    tested = paired.loc[paired.split=="test"]
    pool = tested.groupby("motion").nade.mean().rename("equal_windows")
    formal = hierarchical(tested, ["motion"], ["nade"]).set_index("motion").nade.rename("equal_sources")
    weights = pd.concat([formal,pool],axis=1)
    pd.concat(tables,ignore_index=True).to_csv(OUT/"verified_main.csv",index=False)
    source.to_csv(OUT/"motion_by_source.csv",index=False)
    weights.to_csv(OUT/"test_weight_sensitivity.csv")
    proof = {"roster_rows_matched":len(roster),"frozen_replay":checks,
             "coordinate_tolerance":TOL,"main_metric_tolerance":TOL}
    (OUT/"final_checks.json").write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(proof,indent=2))


if __name__ == "__main__":
    main()
