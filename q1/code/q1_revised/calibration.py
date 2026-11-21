"""Cross-fit source/age residual radii within each observation interface."""
import argparse
import json
import numpy as np
import pandas as pd
from audit_position import OUT, ROOT, HORIZONS, save

def boot_coverage(env):
    pivot=env.pivot(index="env",columns="source",values="coverage").to_numpy()
    if len(pivot)<2:return np.nan,np.nan
    rng=np.random.default_rng(20261121);b=[];remaining=5000
    while remaining:
        ii=rng.integers(len(pivot),size=(remaining,len(pivot)))
        ok=np.isfinite(pivot[ii]).any(axis=1).all(axis=1)
        if ok.any():b.extend(np.nanmean(pivot[ii[ok]],axis=1).mean(axis=1))
        remaining-=int(ok.sum())
    return np.quantile(b,[.025,.975])

def run(points,interface,continuous):
    pts=points.dropna(subset=["error","age"]).copy()
    pts["bin"]=np.maximum(1,np.ceil(pts.age*30).astype(int))
    if continuous:pts["bin"]=pts.bin.clip(upper=30)
    train=pts[pts.split=="train"]
    ticks=[];cal=[];stats=[]
    for held in [*sorted(pts.loc[pts.split=="train","env"].unique()),"development"]:
        fit,target=(train,pts[pts.split=="validation"]) if held=="development" else (train[train.env!=held],train[train.env==held])
        if target.empty:continue
        assert set(fit.env).isdisjoint(target.env)
        grid=fit.groupby(["source","bin"]).agg(radius=("error",lambda x:np.quantile(x,.9,method="higher")),
                 fit_n=("error","size"),fit_envs=("env","nunique"),fit_records=("record","nunique")).reset_index()
        cal.append(grid.assign(held=held,interface=interface))
        pred=target.merge(grid,on=["source","bin"],how="left",validate="many_to_one")
        if continuous:pred=pred[pred.age<=1+1e-12].copy()
        pred["held"]=held;ticks.append(pred)
    if not ticks:return
    ticks=pd.concat(ticks,ignore_index=True)
    scale=pd.read_csv(OUT/"scales.csv")[["held","source","head"]]
    ticks=ticks.merge(scale,on=["held","source"],validate="many_to_one")
    ticks["coverage"]=(ticks.error<=ticks.radius).astype(float).where(ticks.radius.notna())
    ticks["norm_radius"]=ticks.radius/ticks["head"]
    ticks["sparse"]=ticks.fit_envs.fillna(0)<2
    save(pd.concat(cal),f"cal_{interface}_radii.csv")
    save(ticks,f"cal_{interface}_points.csv.gz")
    metrics=["coverage","radius","norm_radius"]
    for split,g in ticks.groupby("split"):
        for h in (HORIZONS if continuous else (0.,)):
            selected=g[g.age<=h+1e-12] if continuous else g
            use=selected.dropna(subset=["radius"])
            if not len(use):continue
            keys=["source","env","record"]
            if continuous:keys.append("window_id")
            rec=use.groupby(keys)[metrics].mean()
            if continuous:rec=rec.groupby(level=keys[:3]).mean()
            env=rec.groupby(level=["source","env"]).mean().reset_index()
            env.to_csv(OUT/f"cal_{interface}_{split}_{h}_env.csv",index=False)
            for src,es in [("ALL",env),*env.groupby("source")]:
                scope=selected if src=="ALL" else selected[selected.source==src]
                val=es.groupby("source")[metrics].mean().mean()
                lo,hi=boot_coverage(es)
                stats.append(dict(split=split,source=src,horizon=h,interface=interface,n=len(scope),eligible=scope.radius.notna().sum(),
                        records=scope.record.nunique(),envs=scope.env.nunique(),fit_lt2_env=scope["sparse"].sum(),lo=lo,hi=hi,**val))
    save(pd.DataFrame(stats),f"cal_{interface}_summary.csv")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["continuous","natural"]);args=parser.parse_args()
    if args.mode=="continuous":
        p=pd.read_csv(OUT/"continuous_points.csv.gz",float_precision="round_trip")
        train=p[p.split=="train"].copy();train["bin"]=np.clip(np.ceil(train.age*30).astype(int),1,30)
        old=json.loads((ROOT/"results/q1/motion_model.json").read_text())["radius90"]
        err=[]
        for s,g in train.groupby("source"):
            for b in range(1,31):
                v=g.loc[g.bin==b,"error"]
                rad=np.quantile(v if len(v) else g.error,.9,method="higher")
                err.append(abs(rad-old[s]["damped"][b-1]))
        assert max(err)<1e-10
        (OUT/"cal_old_check.json").write_text(json.dumps(dict(max_radius_delta=max(err),bins=len(err))))
        run(p,"continuous",True)
    else:
        p=pd.read_csv(OUT/"recovery/natural_points.csv.gz")
        p=p.loc[p.model=="decay"]
        assert len(p)>0
        for phase,g in p.groupby("phase"):run(g,phase,False)
if __name__=="__main__":main()
