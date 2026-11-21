"""Refit the Q1 quality models from causal features, never from archived scores."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_position import ROOT, DATA, OUT, save
from paired_stats import contrasts
sys.path.insert(0,str(ROOT/"code/q1_reliability_candidates"))
import compare as old
from run import rebuild,recovery_state,probabilities,three_state_rates

def main():
    raw,_=rebuild()
    raw["state"]=recovery_state(raw)
    segments=pd.read_csv(DATA/"segments_private.csv")
    for (uid,side),g in raw.groupby(["episode_uuid","side"]):
        with np.load(DATA/"records"/f"{uid}.npz") as f:t=f[side+"__t_s"][g["index"]]
        ss=segments[segments.episode_uuid==uid]
        fam=np.full(len(t),"unannotated",object);phase=fam.copy();count=np.zeros(len(t),int)
        for s in ss.itertuples():
            mask=(t>=s.start_s)&(t<s.end_s)
            fam[mask]=s.family_id;count+=mask
            phase[mask]=np.array(["early","middle","late"])[np.minimum(2,(3*(t[mask]-s.start_s)/s.duration_s).astype(int))]
        fam[count>1]=phase[count>1]="ambiguous"
        raw.loc[g.index,"family"]=fam;raw.loc[g.index,"phase"]=phase
    train=raw[raw.split=="train"];rates=[];pred=[]
    for held in [*sorted(train.environment_id.unique()),"development"]:
        fit,target=(train,raw[raw.split=="validation"]) if held=="development" else (train[train.environment_id!=held],train[train.environment_id==held])
        assert not set(fit.environment_id)&set(target.environment_id)
        x=target.copy()
        q3=three_state_rates(fit);rates.append(q3.reset_index().assign(held=held))
        x["State"]=x.current_bad.map(probabilities(fit));x["State3"]=x.state.map(q3.p).fillna(x.State)
        x["Motion"]=old.clean_tree(fit,x,old.KIN,x.State3)
        pred.append(x)
    df=pd.concat(pred)
    cols=["State","State3","Motion"]
    ref=pd.read_csv(ROOT/"results/q1_reliability_candidates/predictions.csv.gz",float_precision="round_trip")
    pair=df.merge(ref,on=old.KEY,validate="one_to_one",suffixes=("","_arch"))
    assert len(pair)==len(df)==len(ref)==17724
    for k in ["state","label","environment_id","split"]:np.testing.assert_array_equal(pair[k],pair[k+"_arch"])
    delta={k:float(abs(pair[k]-pair[k+"_arch"]).max()) for k in cols}
    assert max(delta.values())<1e-10
    save(pd.concat(rates),"quality_rates.csv");save(df,"quality_predictions.csv.gz")
    grouped=[];ranking=[];alarm=[];envs=[]
    for split,g in df.groupby("split"):
        for subset,gr in [("all",g),*g.groupby("state")]:
            errs=gr[cols].sub(gr.label,axis=0).pow(2).join(gr[old.GROUP])
            byenv=errs.groupby(old.GROUP).mean().groupby(level=old.GROUP[:2]).mean().reset_index()
            byenv=byenv.rename(columns={"dataset":"source","environment_id":"env"})
            envs.append(byenv.assign(split=split,subset=subset))
            score=byenv.groupby("source")[cols].mean().mean()
            grouped.append(dict(split=split,subset=subset,n=len(gr),positives=gr.label.sum(),**score))
        clean=g[g.state=="clean_history"]
        for (src,env),b in clean.groupby(["dataset","environment_id"]):
            for m in cols:ranking.append(dict(split=split,source=src,env=env,model=m,**old.ranking_metrics(b.label,b[m])))
    envs=pd.concat(envs);save(envs,"quality_environments.csv");save(pd.DataFrame(grouped),"quality_scores.csv")
    stats=[contrasts(g,[("State3","State"),("Motion","State3")]).assign(split=split)
           for split,g in envs.query("subset=='all'").groupby("split")]
    save(pd.concat(stats),"quality_paired.csv")
    for (src),g in df.query("state=='clean_history'").groupby("dataset"):
        fit=g[g.split=="train"];val=g[g.split=="validation"]
        for m in cols:
            cut=fit[m].nlargest(int(np.ceil(.1*len(fit)))).iloc[-1]
            fraction=(.1*len(fit)-(fit[m]>cut).sum())/(fit[m]==cut).sum()
            fire=(val[m]>cut).astype(float)+(val[m]==cut)*fraction
            alarm.append(dict(source=src,model=m,n=len(val),positive=val.label.sum(),alerts=fire.sum(),
                              hits=fire@val.label,false_alerts=fire.sum()-fire@val.label,threshold=cut,tie=fraction,
                              unique_positive_times=len(val[val.label.eq(1)].drop_duplicates(["episode_uuid","index"]))))
    save(pd.DataFrame(alarm),"quality_alarm.csv");r=pd.DataFrame(ranking);save(r,"quality_ranking.csv")
    save(r.dropna(subset=["ap"]).groupby(["split","source","model"])[["ap","recall10","prevalence"]].mean().groupby(level=["split","model"]).mean().reset_index(),"quality_ap.csv")
    levels=["split","dataset","environment_id","family","phase","state"]
    counts=df.groupby(levels).agg(anchors=("label","size"),positive=("label","sum"),records=("episode_uuid","nunique"))
    cells=df[cols].sub(df.label,axis=0).pow(2).join(df[levels+["episode_uuid"]])
    grouped=cells.groupby(levels+["episode_uuid"])[cols].mean().groupby(level=levels).mean()
    save(counts.join(grouped).reset_index(),"quality_stage.csv")
    (OUT/"audit_quality.json").write_text(json.dumps(delta,indent=2))
    print(delta)

if __name__=="__main__":
    with threadpool_limits(limits=1):main()
