"""Reproduce Q2 evidence using the pinned T.1 engine, inputs and evaluation clocks."""
import hashlib
import importlib.util
import io
import json
from itertools import combinations
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from age.duration import duration_checks
from recovery import recovery_events

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
OUT = PACKAGE / "results"
REV = "b0a8526a0db5dbad9a3ec376238830153a2f9855"
Q1 = "bf380c4a1ec61deeb1ccb4206ad6098619f37157"
SOL = "C题工作区/第二问求解/"
EXP = "C题工作区/第二问探索/results/"
KEYS = ["dataset", "environment_id", "episode_uuid", "side"]
SPLITS = ["train", "validation", "test"]
SCHEDULES = {"none": (), "first": (.3,), "second": (.6,), "both": (.3, .6)}


def blob(path, rev=REV):
    return subprocess.check_output(["git", "-C", str(REPO), "show", rev + ":" + path])


def csv(path, rev=REV, **opts):
    return pd.read_csv(io.BytesIO(blob(path, rev)), float_precision="round_trip", **opts)


def load_engine(directory, model):
    data = blob(SOL + "code/engine.py")
    # Git normalized the original engine's single final CRLF to LF.
    data = data.removesuffix(b"\n") + b"\r\n"
    assert hashlib.sha256(data).hexdigest() == model["code_sha256"]["engine.py"]
    path = Path(directory) / "engine.py"
    path.write_bytes(data)
    spec = importlib.util.spec_from_file_location("q2_frozen_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def hierarchical(frame, by, metrics):
    """From one score per window/event to a source-balanced overall score."""
    for depth in (4, 3, 2, 1, 0):
        frame = frame.groupby(by + KEYS[:depth], as_index=False)[metrics].mean()
    return frame


def replay_window(engine, t, x, w, scale):
    i, j = w.start_index, w.end_index_exclusive
    assert np.all(t[i:j] - t[i] <= 1.0 + 1e-9)
    results = []
    common_actions = None
    variants = [("damped", plan) for plan in SCHEDULES] + [("hold", "both"), ("cv", "both")]
    for motion, plan in variants:
        trial = engine.replay(t[i:j], x[i:j], t[i-2:i], x[i-2:i],
                              engine.FixedPolicy(SCHEDULES[plan]), delay=.1,
                              motion=motion, window_end=float(t[i]+1))
        met = trial["metrics"]
        assert met["requests"] == met["successful_returns"] == len(SCHEDULES[plan])
        if plan == "both":
            if common_actions is None:
                common_actions = trial["actions"]
            assert trial["actions"] == common_actions
        results.append((motion, plan, met["ade"]/scale, met["requests"], met["waiting_s"]))
    return results


def main():
    model_data = blob(SOL + "frozen_model.json")
    model = json.loads(model_data)
    # The archived evaluation identifies the Windows worktree representation.
    model_hash = hashlib.sha256(model_data.replace(b"\n", b"\r\n")).hexdigest()
    assert model["selected"] == "fixed" and model["calibrator_formal"] == "open_radius"
    assert model["fixed_schedule"] == list(SCHEDULES["both"])
    scales = json.loads(blob(SOL + "results/S0/fold_scales.json"))
    assert model["scale"] == scales["development"]
    catalog = csv("C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv")
    roster = csv("q1/C_Problem_Data/record_split.csv", Q1)
    assert roster.episode_uuid.is_unique and catalog.window_id.is_unique
    assert not roster[["scene_id","environment_id","operator_id"]].isna().any().any()
    verified = catalog.merge(roster, on=["episode_uuid","dataset","environment_id","split"], validate="many_to_one")
    assert len(verified) == len(catalog) and set(roster.episode_uuid) == set(catalog.episode_uuid)
    segs = csv("q1/C_Problem_Data/v1/v1/segments_private.csv", Q1).groupby("episode_uuid")
    hashes = pd.concat([csv(SOL+"results/S0/input_hashes.csv")[["episode_uuid","sha256"]],
                        csv(SOL+"results/S5_test/input_hashes.csv")], ignore_index=True)
    assert hashes.episode_uuid.is_unique and set(hashes.episode_uuid) == set(roster.episode_uuid)
    hashes = hashes.set_index("episode_uuid").sha256
    overlaps = []
    for a,b in combinations(SPLITS,2):
        for field in ["environment_id","scene_id","operator_id"]:
            n = len(set(roster.loc[roster.split==a,field]) & set(roster.loc[roster.split==b,field]))
            overlaps.append((a,b,field,n))

    inputs, labels, runs, streams, eligibility = [], [], [], [], []
    by_hand = catalog.groupby(["episode_uuid","side"])
    with tempfile.TemporaryDirectory() as tmp:
        engine = load_engine(tmp,model)
        for record in roster.itertuples():
            uid = record.episode_uuid
            raw = blob("q1/C_Problem_Data/v1/v1/records/"+uid+".npz", Q1)
            assert hashlib.sha256(raw).hexdigest() == hashes[uid]
            scale = scales[record.environment_id if record.split=="train" else "development"][record.dataset]
            annotations = segs.get_group(uid)
            with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
                for side in ("left_hand","right_hand"):
                    t = arrays[side+"__t_s"]
                    x = arrays[side+"__position"].mean(axis=1)
                    valid = arrays[side+"__reference_candidate"].astype(bool)
                    bad = arrays[side+"__bad"].astype(bool)
                    used = np.zeros(t.size,dtype=bool)
                    meta = (record.split,record.dataset,record.environment_id,uid,side)
                    e,r = recovery_events(t,x,valid,bad,scale)
                    # Bad-segment coordinates are never a scoring or recovery input.
                    hidden = x.copy()
                    hidden[bad] = np.nan
                    assert (e,r) == recovery_events(t,hidden,valid,bad,scale)
                    eligibility.extend([(*meta,*row) for row in e])
                    streams.extend([(*meta,*row) for row in r])
                    windows = by_hand.get_group((uid,side)) if (uid,side) in by_hand.groups else catalog.iloc[:0]
                    for w in windows.itertuples():
                        i,j = w.start_index,w.end_index_exclusive
                        assert i-w.history_start>=2 and valid[w.history_start:j].all() and not used[i:j].any()
                        used[i:j] = True
                        for row in replay_window(engine,t,x,w,scale):
                            runs.append((*meta,w.window_id,*row))
                        group = annotations[(annotations.start_s<=t[i]) & (t[i]<annotations.end_s)]
                        assert len(group)==1
                        labels.append((w.window_id,"crosses_end" if t[j-1]>=group.end_s.iloc[0] else "internal"))
                    inputs.append((*meta,len(t),int(valid.sum()),int(used.sum()),len(windows)))
    meta = ["split",*KEYS]
    scores = pd.DataFrame(runs,columns=meta+["window_id","motion","plan","nade","requests","wait_s"])
    assert not scores.duplicated(["window_id","motion","plan"]).any()
    fixed = scores.query("motion=='damped' and plan=='both'")
    assert fixed.groupby("split").size().to_dict()=={"train":1527,"validation":212,"test":390}
    tables = {"split_overlaps":pd.DataFrame(overlaps,columns=["first","second","field","shared"])}
    hand = pd.DataFrame(inputs,columns=meta+["samples","qualified","scored","windows"])
    retention = hand.groupby(["split","dataset"],as_index=False).agg(
        records=("episode_uuid","nunique"),environments=("environment_id","nunique"),
        **{k:(k,"sum") for k in ["samples","qualified","scored","windows"]})
    retention["unqualified"] = retention.samples-retention.qualified
    retention["qualified_outside"] = retention.qualified-retention.scored
    assert (retention.qualified_outside>=0).all()
    tables["source_retention"] = retention
    tables["replay_windows"] = scores
    tables["replay_summary"] = hierarchical(scores,["split","motion","plan"],["nade","requests","wait_s"])
    paired = csv(EXP+"train_paired_windows.csv").rename(columns={"method":"plan"})
    trial_train = scores.query("split=='train' and motion=='damped'")
    merged = paired.merge(trial_train,on=["window_id","plan"],validate="one_to_one")
    assert len(merged)==len(paired)==len(trial_train)
    errors = {"train_schedules":float((merged.nade_x-merged.nade_y).abs().max())}

    limits = csv(SOL+"empirical_prediction_duration.csv")
    age_results = []
    for split in ("validation","test"):
        folder = SOL+"results/S5_"+split+"/"
        assert json.loads(blob(folder+"audit.json"))["frozen_model_sha256"] == model_hash
        archived = csv(folder+"windows.csv").query("method=='fixed' and length_s==1 and delay_s==.1")
        ours = fixed[fixed.split==split].set_index("window_id").nade
        archived = archived.set_index("window_id").nade
        assert ours.index.is_unique and set(ours.index)==set(archived.index)
        errors[split+"_fixed"] = float((ours-archived).abs().max())
        points = csv(folder+"points.csv.gz",compression="gzip",
                     usecols=KEYS+["window_id","method","length_s","delay_s","time","age","error"])
        points = points.query("method=='damped' and length_s==1 and delay_s==0")
        assert set(points.window_id)==set(ours.index)
        age_results.append(duration_checks(points,model,limits,split))
    assert max(errors.values())<1e-10, errors
    tables["duration_checks"] = pd.concat(age_results,ignore_index=True)

    tagged = fixed.merge(pd.DataFrame(labels,columns=["window_id","boundary"]),on="window_id",validate="one_to_one")
    tables["stage_windows"] = tagged[meta+["window_id","boundary","nade"]]
    env = tagged.groupby(["split","boundary",*KEYS],as_index=False).nade.mean()
    for depth in (3,2):
        env = env.groupby(["split","boundary",*KEYS[:depth]],as_index=False).nade.mean()
    cells = env.pivot(index=["split","dataset","environment_id"],columns="boundary",values="nade").dropna().reset_index()
    cells["difference"] = cells.crosses_end-cells.internal
    tables["stage_pairs"] = cells
    stage = cells.groupby(["split","dataset"])[["crosses_end","internal","difference"]].mean().groupby("split").mean()
    stage["matched_source_envs"] = cells.groupby("split").size()
    stage["environments"] = cells.groupby("split").environment_id.nunique()
    stage["sources"] = cells.groupby("split").dataset.nunique()
    stage["crossing_windows_all"] = tagged.query("boundary=='crosses_end'").groupby("split").size()
    tables["stage_summary"] = stage.reset_index()
    np.testing.assert_allclose(stage.loc["train",["crosses_end","internal","difference"]].to_numpy(dtype=float),
                               csv(EXP+"stage_diagnostic/matched_boundary_summary.csv").iloc[:,1],atol=1e-10)
    elig = pd.DataFrame(eligibility,columns=meta+["event_start","reason"])
    events = pd.DataFrame(streams,columns=meta+["event_start","method","nade","within","points"])
    rec = hierarchical(events,["split","method"],["nade","within"])
    support = events.groupby(["split","method"]).agg(events=("event_start","size"),records=("episode_uuid","nunique"),
                                                      environments=("environment_id","nunique"),sources=("dataset","nunique"))
    tables["recovery_summary"] = rec.merge(support.reset_index(),on=["split","method"])
    tables["recovery_events"] = events
    tables["recovery_eligibility"] = elig.groupby(["split","dataset","reason"]).size().reset_index(name="events")
    original = csv(EXP+"natural_recovery/summary.csv").query("length_s==.2").set_index("method")
    actual = rec.query("split=='train'").set_index("method").reindex(original.index)
    np.testing.assert_allclose(actual.nade,original.nade_pre,atol=1e-10,rtol=0)
    np.testing.assert_allclose(actual.within,1-original.exceed_pre_05,atol=1e-10,rtol=0)
    assert elig.query("reason=='eligible'").shape[0]*3==len(events)
    audit = {"q2_commit":REV,"q1_inputs_commit":Q1,"model_sha256":model_hash,
             "records_hash_matched":len(hashes),"windows":len(fixed),"replay_variants":len(scores),
             "max_absolute_nade_diff":errors,"duration_method":"frozen no-request damped, 1 s, ceil(age*30)",
             "recovery_bad_coordinates_nan_check":True,
             "model_refitted":False,"evaluation":"descriptive reanalysis of previously scored splits"}
    OUT.mkdir(parents=True,exist_ok=True)
    for name, table in tables.items():
        suffix = ".csv.gz" if name=="replay_windows" else ".csv"
        if suffix.endswith(".gz"):
            table.to_csv(OUT/(name+suffix), index=False, lineterminator="\n",
                         compression={"method": "gzip", "mtime": 0})
        else:
            table.to_csv(OUT/(name+suffix), index=False)
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()
