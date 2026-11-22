"""Offline context for Q1 and natural gaps; semantic annotations never enter prediction."""
import numpy as np
import pandas as pd


def run(processed,out):
    q1=pd.read_csv(out/"q1_windows.csv.gz")
    from trajectory import PROTOCOL
    windows=pd.read_csv(PROTOCOL/"data/windows.csv")
    segments=pd.read_csv(processed/"segments_private.csv")
    context=[]
    for uid,g in windows.groupby("episode_uuid"):
        spans=segments[segments.episode_uuid==uid]
        for w in g.itertuples():
            s=spans[(spans.start_s<=w.start_s)&(spans.end_s>w.start_s)]
            family=stage="ambiguous" if len(s) else "unannotated"
            if len(s)==1:
                r=s.iloc[0]
                family=r.family_id
                stage=("early","middle","late")[min(2,int(3*(w.start_s-r.start_s)/r.duration_s))]
            context.append(dict(window_id=w.window_id,stage=stage,family=family))
    data=q1.merge(pd.DataFrame(context),on="window_id",validate="many_to_one")
    for tag in ("stage","family"):
        keys=["split","model",tag,"horizon","dataset","environment_id","episode_uuid"]
        g=data
        for n in range(len(keys),2,-1):
            g=g.groupby(keys[:n],as_index=False)[["nade","nfde"]].mean()
        g.to_csv(out/f"q1_{tag}.csv",index=False)
    # Use the first subsequent eligible frame only for a finite, fully observed
    # gap duration. Retain both censoring flags and count all excluded endpoints.
    gaps=pd.read_csv(processed/"natural_unusable_intervals.csv")
    gaps=gaps[gaps.channel.isin(["left_hand","right_hand"])].copy()
    ix=pd.read_csv(processed/"record_index.csv")[["episode_uuid","dataset"]]
    gaps=gaps.merge(ix,on="episode_uuid",validate="many_to_one")
    durations=[]
    for uid,group in gaps.groupby("episode_uuid"):
        with np.load(processed/"records"/f"{uid}.npz") as raw:
            for e in group.itertuples():
                t=raw[e.channel+"__t_s"]
                left=e.start_index==0
                right=e.end_index_exclusive==len(t)
                d=np.nan if left or right else t[e.end_index_exclusive]-t[e.start_index]
                durations.append(dict(split=e.split,dataset=e.dataset,side=e.channel,left=left,right=right,duration=d))
    d=pd.DataFrame(durations)
    d.groupby(["split","dataset"]).agg(events=("duration","size"),complete=("duration","count"),
        left_censored=("left","sum"),right_censored=("right","sum"),median_s=("duration","median"),
        p90_s=("duration",lambda x:x.quantile(.9))).to_csv(out/"natural_duration.csv")
    e=pd.read_csv(out/"q1_recovery_events.csv.gz").dropna(subset=["error","age"]).copy()
    e["bin"]=np.maximum(1,np.ceil(e.age*30)).astype(int)
    keys=["dataset","phase","bin"]
    fit=e.query("split=='train'").groupby(keys).error.quantile(.9,interpolation="higher").rename("radius")
    fit.to_csv(out/"q1_natural_radius_model.csv")
    held=e.query("split!='train'").join(fit,on=keys)
    held["covered"]=(held.error<=held.radius).astype(float).where(held.radius.notna())
    counts=held.groupby(["split","phase"]).covered.agg(scorable="size",with_radius="count")
    levels=["split","phase","dataset","environment_id","episode_uuid"]
    for n in (5,4,3,2):
        held=held.groupby(levels[:n],as_index=False).covered.mean()
    held.join(counts,on=levels[:2]).to_csv(out/"q1_natural_range.csv",index=False)
