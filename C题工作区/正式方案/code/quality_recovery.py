"""Source-quality risk and recovery when a continuous clean prefix is unavailable."""
import numpy as np
import pandas as pd


def quality_state(history_bad):
    if history_bad[-1]:
        return "flagged"
    return "recent_recovery" if history_bad.any() else "clean_history"


def cohort_outputs(record,raw,intervals,scale):
    samples,events=[],[]
    for side in ("left_hand","right_hand"):
        t,x,bad,known,good,gap=[raw[side+"__"+k] for k in
            ("t_s","position","bad","quality_known","reference_candidate","gap_before")]
        x=x.mean(axis=1)
        for i in range(6,len(t)-7,6):
            lo=t.searchsorted(t[i]-.2)
            hi=t.searchsorted(t[i]+.2,side="right")
            if hi<=i+1 or not known[lo:hi].all() or not np.isfinite(x[lo:i+1]).all():
                continue
            samples.append(dict(record,side=side,index=i,state=quality_state(bad[lo:i+1]),
                                label=int(bad[i+1:hi].any())))
        # Velocity reinitializes on two consecutive valid observations. A single
        # returned observation resets position and decays the last valid speed.
        ids=np.flatnonzero(good)
        v=np.zeros_like(x)
        for a,b in zip(ids[:-1],ids[1:]):
            dt=t[b]-t[a]
            v[b]=(x[b]-x[a])/dt if b==a+1 and not gap[b] else v[a]*np.exp(-dt/.2)
        for event in intervals[intervals.channel==side].itertuples():
            for phase,k in [("pre_first",event.end_index_exclusive),("pre_second",event.end_index_exclusive+1)]:
                prev=np.searchsorted(ids,k)-1
                ok=k<len(t) and good[k] and (phase=="pre_first" or not gap[k]) and prev>=0
                error=age=np.nan
                if ok:
                    last=ids[prev]
                    age=t[k]-t[last]
                    prediction=x[last]-.2*np.expm1(-age/.2)*v[last]
                    error=np.linalg.norm(prediction-x[k])
                events.append(dict(record,side=side,start=event.start_index,phase=phase,age=age,error=error,
                                   norm_error=error/scale))
    return samples,events


def finish_quality(rows,events,out,protocol):
    data=pd.DataFrame(rows)
    ref=pd.read_csv(protocol/"q12/reliability_predictions.csv")
    key=["episode_uuid","side","index"]
    check=data.query("split!='test'").merge(ref[key+["label"]],on=key,validate="one_to_one")
    assert len(check)==len(ref)==17724 and (check.label_x==check.label_y).all()
    train=data.query("split=='train'")
    rates,predictions=[],[]
    for held in [*sorted(train.environment_id.unique()),"full"]:
        fit=train if held=="full" else train[train.environment_id!=held]
        score=data[data.split!="train"] if held=="full" else train[train.environment_id==held]
        table=fit.groupby("state").label.agg(["sum","size"])
        table["probability"]=(table["sum"]+1)/(table["size"]+2)
        rates.append(table.reset_index().assign(held=held))
        scored=score.copy()
        scored["brier"]=(scored.state.map(table.probability)-scored.label)**2
        assert scored.brier.notna().all()
        predictions.append(scored)
    pd.concat(rates).to_csv(out/"q1_quality_model.csv",index=False)
    sample=pd.concat(predictions)
    counts=sample.groupby("split").label.agg(anchors="size",positives="sum")
    levels=["split","dataset","environment_id","episode_uuid"]
    for n in (4,3,2,1):
        sample=sample.groupby(levels[:n],as_index=False).brier.mean()
    sample.join(counts,on="split").to_csv(out/"q1_quality.csv",index=False)
    events=pd.DataFrame(events)
    events.to_csv(out/"q1_recovery_events.csv.gz",index=False)
    keys=["split","phase","dataset","environment_id","episode_uuid"]
    counts=events.groupby(keys[:2]).norm_error.agg(events="size",scored="count")
    grouped=events
    for n in (5,4,3,2):
        grouped=grouped.groupby(keys[:n],as_index=False).norm_error.mean()
    grouped.join(counts,on=keys[:2]).to_csv(out/"q1_recovery.csv",index=False)
