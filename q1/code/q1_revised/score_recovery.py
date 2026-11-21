"""Compute recovery replay coverage and errors on both available and paired samples."""
import numpy as np
import pandas as pd
from audit_position import OUT,save
from paired_stats import contrasts

def score(d,keys,units,metrics,filename):
    """Available-error and common-observation analyses, with explicit denominators."""
    assert not d.duplicated(keys+units+["model"]).any()
    observed=d[metrics[0]].notna()
    d=d.copy()
    d["available"]=observed
    d["common"]=d.groupby(keys+units).available.transform("sum")==d.model.nunique()
    rows=[]
    for mode,group in [("available",d),("common",d[d.common])]:
        levels=keys+["model","source","env","record"]
        # The coverage denominator is defined before restricting to common support.
        for cell,g in group.groupby(keys+["model"]):
            rec=g.groupby(["source","env","record"])[metrics+["available"]].mean()
            env=rec.groupby(level=["source","env"]).mean()
            val=env.groupby(level="source").mean().mean()
            rows.append(dict(zip(keys+["model"],cell))|dict(scope=mode,n=len(g),predicted=int(g.available.sum()),
                       common=int(g.common.sum()),records=g.record.nunique(),envs=g.env.nunique(),**val))
        if mode=="common" and len(group):
            e=group.groupby(levels)[metrics].mean().groupby(level=levels[:-1]).mean().reset_index()
            save(e,filename+"_env.csv")
    save(pd.DataFrame(rows),filename+"_summary.csv")

def main():
    dest=OUT/"recovery"
    scale=pd.read_csv(OUT/"scales.csv")[["held","source","head"]]
    def withscale(d):
        return d.assign(held=np.where(d.split=="train",d.env,"development")).merge(scale,on=["held","source"],validate="many_to_one")
    a=pd.read_csv(dest/"synthetic_metrics.csv.gz")
    m=pd.read_csv(OUT/"window_manifest.csv")[["window_id","source","record","env","split","side","family","phase"]]
    a=a.merge(m[["window_id"]+[n for n in m.columns if n!="window_id" and n not in a]],on="window_id",validate="many_to_one")
    a=withscale(a)
    a["nade"]=a.ade/a["head"];a["nfde"]=a.fde/a["head"]
    score(a,["split","scenario","horizon"],["window_id"],["ade","fde","nade","nfde"],"recovery_artificial")
    b=withscale(pd.read_csv(dest/"natural_points.csv.gz"))
    b["norm_error"]=b.error/b["head"]
    score(b,["split","phase"],["record","side","event"],["error","norm_error"],"recovery_natural")
    # Mechanism comparisons use identical scorable events/windows for all policies.
    paired=[]
    for stem,keys,metric in [("artificial",["split","scenario","horizon"],"nade"),("natural",["split","phase"],"norm_error")]:
        envs=pd.read_csv(OUT/f"recovery_{stem}_env.csv")
        for group,rows in envs.groupby(keys):
            wide=rows.pivot(index=["source","env"],columns="model",values=metric).reset_index()
            result=contrasts(wide,[("decay","hold_two"),("decay","zero_velocity"),("decay","recent6_hold")])
            paired.append(result.assign(**dict(zip(keys,group)),interface=stem))
    save(pd.concat(paired),"recovery_contrasts.csv")
    ref=pd.read_csv(OUT/"natural_reference.csv").rename(columns={"start":"event","phase":"event_phase","stage":"task_stage"})
    by=b.merge(ref[["record","side","event","event_phase","family","task_stage"]],left_on=["record","side","event","phase"],
               right_on=["record","side","event","event_phase"],validate="many_to_one")
    save(by.groupby(["split","source","env","family","task_stage","phase","model"]).agg(
        events=("event","size"),scored=("error","count"),records=("record","nunique"),mean_error=("error","mean")).reset_index(),"recovery_stage.csv")
if __name__=="__main__":main()
