"""Replay the frozen model from the original hand arrays; assemble fold-local scores."""
from pathlib import Path
import hashlib
import os
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q1_revised"
DATA = Path(os.environ.get("Q1_FULL_V1", ROOT / "C_Problem_Data/v1/v1")).resolve()
FROZEN = ROOT / "code/q1.py"
CAT = ROOT.parent / "C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv"
HORIZONS = (.2, .6, 1.)
KEY = ["window_id","horizon"]

def save(frame, name):
    frame.to_csv(OUT/name, index=False)

def hierarchy(frame, by, metrics):
    keys = by + ["source","env","record"]
    result = frame.groupby(keys,dropna=False)[metrics].mean()
    for _ in range(3):
        keys = keys[:-1]
        result = result.groupby(level=keys).mean()
    return result.reset_index()

def frozen_windows():
    spec = importlib.util.spec_from_file_location("frozen_q1",FROZEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {w.meta["window_id"]:w for w in module.inputs()[0]}

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    idx=pd.read_csv(DATA/"record_index.csv")
    official=pd.read_csv(next((ROOT/"C_Problem_Data").rglob("record_split.csv"))).set_index("episode_uuid")
    assert idx.episode_uuid.is_unique and idx.groupby("environment_id").split.nunique().max()==1
    for k in ["split","environment_id","dataset"]:
        np.testing.assert_array_equal(official.loc[idx.episode_uuid,k],idx[k])
    cat=pd.read_csv(CAT)
    w0=frozen_windows()
    hist=pd.read_csv(ROOT/"results/q1/motion_windows.csv").query("model=='damped'")
    assert set(w0)==set(hist.window_id)
    head_d={}; usage=[]; meta=[]; metrics=[]; points=[]; hashes=[]; maxdiff=0.
    paths=[CAT,FROZEN,DATA/"input_sha256.csv",DATA/"record_index.csv",DATA/"natural_unusable_intervals.csv",DATA/"segments_private.csv",
           next((ROOT/"C_Problem_Data").rglob("record_split.csv")),ROOT/"reports/第一问_统一预处理与缺帧恢复_实验设计.md",
           ROOT/"results/q1/quality_predictions.csv",ROOT/"code/q1_curves/models.py"]
    for row in idx.itertuples():
        uid=row.episode_uuid; group=cat[cat.episode_uuid==uid]
        path=DATA/"records"/f"{uid}.npz"; paths.append(path)
        with np.load(path,allow_pickle=False) as z:
            for side in ("left_hand","right_hand"):
                def get(k): return z[side+"__"+k]
                t,pos,quat,known,bad=map(get,("t_s","position","quaternion","quality_known","bad"))
                dt=np.diff(t); gap=np.r_[False,dt>3*np.median(dt[dt>0])]
                good=(np.isfinite(pos).all((1,2)) & np.isfinite(quat).all((1,2)) &
                      (abs(np.linalg.norm(quat,axis=2)-1)<=.001).all(1) & known & ~bad & np.r_[True,dt>0])
                np.testing.assert_array_equal(good,get("reference_candidate"))
                np.testing.assert_array_equal(gap,get("gap_before"))
                block=group[group.side==side]
                usage.append(dict(record=uid,source=row.dataset,env=row.environment_id,split=row.split,side=side,
                                  n=len(t),reference=good.sum(),bad=bad.sum(),unknown=(~known).sum(),gaps=gap.sum(),
                                  formal=len(block),use="structure_only" if row.split=="test" else "fit" if row.split=="train" else "development"))
                if row.split=="test":continue
                x=pos.mean(1);np.testing.assert_array_equal(t,z["head__t_s"])
                head_ok=z["head__finite"] & z["head__time_increasing"] & z["head__quaternion_unit_1e3"]
                if row.split=="train":
                    head_d[(uid,side)]=np.linalg.norm(x[good & head_ok]-z["head__position"][good & head_ok,0],axis=1)
                for w in block.itertuples():
                    i,j=w.start_index,w.end_index_exclusive
                    assert good[w.history_start:j].all() and not gap[w.history_start+1:j].any()
                    age=t[i:j]-t[i-1]; rel=t[i:j]-t[i]; v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2])
                    delta=x[i:j]-x[i-1]; pred=.2*(-np.expm1(-age/.2))[:,None]*v
                    old=w0[w.window_id]
                    np.testing.assert_array_equal(old.age,age);np.testing.assert_array_equal(old.delta,delta)
                    maxdiff=max(maxdiff,np.abs(pred-old.features[:,0]).max())
                    err=np.linalg.norm(delta-pred,axis=1)
                    base=dict(window_id=w.window_id,source=row.dataset,env=row.environment_id,record=uid,
                              side=side,split=row.split,family=old.meta["family"],phase=old.meta["phase"])
                    meta.append(dict(**base,first_hidden=i,last_visible=i-1,end_exclusive=j,history_start=w.history_start,
                                     endpoint=np.linalg.norm(delta[-1])))
                    points.extend(dict(**base,frame_index=i+k,age=h,error=e) for k,(h,e) in enumerate(zip(age,err)))
                    for h in HORIZONS:
                        for clock,time in [("legacy",rel),("age",age)]:
                            take=time<=h+1e-12
                            metrics.append(dict(**base,horizon=h,clock=clock,n=take.sum(),last_age=age[take][-1],
                                                ade=err[take].mean(),fde=err[take][-1]))
    for path in paths:
        label = ("C_Problem_Data/v1/v1/" + path.relative_to(DATA).as_posix()
                 if path.is_relative_to(DATA) else Path(os.path.relpath(path, ROOT)).as_posix())
        hashes.append(dict(path=label, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(pd.DataFrame(hashes),"inputs.csv");save(pd.DataFrame(usage),"record_usage.csv")
    windows=pd.DataFrame(meta);save(windows,"window_manifest.csv")
    df=pd.DataFrame(metrics)
    cmp=df.query("clock=='legacy'").merge(hist.rename(columns={"horizon_s":"horizon"}),on=KEY,validate="one_to_one")
    assert len(cmp)==len(hist)==len(windows)*3
    deltas={k:float(abs(cmp[k+"_x"]-cmp[k+"_y"]).max()) for k in ["ade","fde"]}
    assert max(maxdiff,*deltas.values())<=1e-10
    scales=[]; members=[]
    for held in [*sorted(windows.query("split=='train'").env.unique()),"development"]:
        train=windows.query("split=='train' and env!=@held")
        for source,g in train.groupby("source"):
            ids=idx[(idx.split=="train") & (idx.dataset==source) & (idx.environment_id!=held)].episode_uuid
            distances=np.concatenate([head_d[(uid,s)] for uid in ids for s in ("left_hand","right_hand")])
            scales.append(dict(held=held,source=source,head=np.median(distances),endpoint=g.endpoint.median(),n_head=len(distances),n_windows=len(g)))
            members.extend(dict(held=held,record=uid,source=source) for uid in ids)
    save(pd.DataFrame(members),"fit_records.csv");sc=pd.DataFrame(scales);save(sc,"scales.csv")
    df["held"]=np.where(df.split=="train",df.env,"development")
    df=df.merge(sc,on=["held","source"],validate="many_to_one")
    for col in ["head","endpoint"]:df[col+"_ade"]=df.ade/df[col]
    save(df,"clock_windows.csv.gz")
    summary=hierarchy(df,["split","clock","horizon"],["head_ade","endpoint_ade","ade","fde"])
    save(summary,"clock_scores.csv");save(pd.DataFrame(points),"continuous_points.csv.gz")
    audit=dict(windows=len(windows),keys_checked=len(cmp),max_position_delta=maxdiff,old_score_delta=deltas,
               mask_diff=0, gap_diff=0,structural_counts=cat.split.value_counts().to_dict())
    (OUT/"audit_A.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    print(audit)

if __name__=="__main__":main()
