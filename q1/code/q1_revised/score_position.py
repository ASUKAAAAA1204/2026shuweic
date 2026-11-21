"""Aggregate paired formal-window predictions, retaining source, record and environment weights."""
import pandas as pd
import numpy as np
from audit_position import OUT,save
from paired_stats import contrasts

def main():
    path=OUT/"curve/window_metrics.csv.gz"
    data=pd.read_csv(path)
    model_seed=sorted(data.loc[data.model=="C","seed"].unique())[0]
    allseeds=data[data.model=="C"].copy()
    data=data[(data.model!="C")|(data.seed==model_seed)].drop_duplicates(["window_id","horizon","model"])
    baseline=pd.read_csv(OUT/"clock_windows.csv.gz").query("clock=='age'")
    ck=data[data.model=="B"].merge(baseline,on=["window_id","horizon"],validate="one_to_one")
    assert len(ck)==5217
    np.testing.assert_allclose(ck.ade_x,ck.ade_y,atol=1e-10,rtol=0)
    np.testing.assert_allclose(ck.nade,ck.head_ade,atol=1e-10,rtol=0)
    counts=data.groupby(["window_id","horizon"]).model.nunique()
    assert counts.eq(4).all() and len(counts)==5217
    info=pd.read_csv(OUT/"window_manifest.csv")[["window_id","family","phase"]]
    data=data.drop(columns=["family","phase"],errors="ignore").merge(info,on="window_id",validate="many_to_one")
    mets=["ade","fde","nade","nfde"]
    groups=["split","model","horizon","source","env","record"]
    rec=data.groupby(groups)[mets].mean();env=rec.groupby(level=groups[:-1]).mean()
    src=env.groupby(level=groups[:-2]).mean();hor=src.groupby(level=groups[:3]).mean()
    save(env.reset_index(),"position_environments.csv");save(src.reset_index(),"position_sources.csv")
    save(hor.reset_index(),"position_horizons.csv");save(hor.groupby(level=groups[:2]).mean().reset_index(),"position_overall.csv")
    comp=[]
    for split,b in env.reset_index().groupby("split"):
        av=b.groupby(["env","source","model"]).nade.mean().unstack("model").reset_index()
        comp.append(contrasts(av,[("B","CV"),("B","Hold"),("C","B")]).assign(split=split))
    save(pd.concat(comp),"position_paired.csv")
    sg=["split","family","phase","source","env","model","horizon"]
    g=data.groupby(sg)
    stage=g.agg(windows=("window_id","size"),records=("record","nunique")).join(
        data.groupby(sg+["record"])[mets].mean().groupby(level=sg).mean())
    save(stage.reset_index(),"position_stage.csv")
    keys=["split","seed","horizon","source","env","record"]
    by=allseeds.groupby(keys).nade.mean()
    for _ in range(4):
        keys=keys[:-1];by=by.groupby(level=keys).mean()
    save(by.reset_index(),"mlp_seeds.csv")
    print(pd.concat(comp).to_string(index=False))

if __name__=="__main__":main()
