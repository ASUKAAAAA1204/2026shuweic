"""Apply the fixed contrasts, tail and replication criteria written in design.md."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PRIOR=OUT.parent/"final_v1_stability"
RATE="weighted_on_time_rate"
KEY=["case_id","seed","alpha","beta"]


def paired(name,delta):
    v=np.asarray(delta,float)*100
    rng=np.random.default_rng(20261122)
    idx=rng.integers(0,len(v),size=(10000,len(v)))
    q=np.quantile(v[idx].mean(axis=1),[.025,.975])
    return dict(contrast=name,n=len(v),delta_pp=v.mean(),lower_pp=q[0],upper_pp=q[1],halfwidth_pp=(q[1]-q[0])/2)


def summarize_grid(grid):
    mat=grid.pivot(index="seed",columns=["alpha","beta","case_id"],values=RATE).sort_index(axis=1)
    assert mat.shape==(300,27) and mat.notna().all().all()
    av={(a,b):mat[a,b].mean(axis=1) for a in [.75,1.,1.5] for b in [0.,.5,1.]}
    contrasts=[]
    for n in [30,300]:
        contrasts.append(paired("long-minus-nominal",(av[1.5,.5]-av[1.,.5]).iloc[:n]))
        contrasts.append(paired("joint-minus-additive",(av[1.5,1.]-av[1.5,.5]-av[1.,1.]+av[1.,.5]).iloc[:n]))
    pd.DataFrame(contrasts).to_csv(OUT/"paired_grid.csv",index=False)
    table=grid.groupby(["alpha","beta","case_id"])[RATE].mean().unstack()
    table["overall"]=table.mean(axis=1)
    table.to_csv(OUT/"grid_means.csv")
    tail=[]
    for (case,a,b),g in grid.query("alpha in [1.,1.5] and beta==0.5").groupby(["case_id","alpha","beta"]):
        g=g.sort_values("seed")
        for block,part in [("all",g),("first150",g.iloc[:150]),("last150",g.iloc[150:])]:
            values=part[RATE]
            k=len(part)//10
            lowest=values.nsmallest(k)
            tail.append(dict(case_id=case,alpha=a,beta=b,block=block,n=len(part),worst_n=k,p10=values.quantile(.1),
                median=values.median(),mean=values.mean(),worst_mean=lowest.mean(),cutoff=lowest.max(),
                tied_at_cutoff=int((values==lowest.max()).sum())))
    tail=pd.DataFrame(tail)
    tail.to_csv(OUT/"tail_blocks.csv",index=False)
    checks=tail.pivot(index=["case_id","alpha"],columns="block",values="worst_mean")
    checks["gap_pp"]=100*(checks.first150-checks.last150).abs()
    checks["within_5pp"]=checks.gap_pp<=5+1e-10
    checks.to_csv(OUT/"tail_replication.csv")


def summarize_streams():
    old=pd.read_csv(PRIOR/"seed_trials.csv.gz").query("alpha in [1.,1.5] and beta==0.5")
    new=pd.read_csv(OUT/"internal_replication.csv.gz")
    assert set(old.internal_seed).isdisjoint(new.internal_seed)
    a=old.groupby(KEY).instance_hash.first().sort_index()
    b=new.groupby(KEY).instance_hash.first().sort_index()
    pd.testing.assert_series_equal(a,b)
    metrics=[]
    for name,data in [("original",old),("replication",new)]:
        assert data.groupby(KEY).internal_seed.nunique().eq(20).all()
        profiles=[]
        for (case,seed,alpha,beta),g in data.groupby(KEY):
            counts=g.first_action.value_counts()
            profiles.append(dict(case_id=case,seed=seed,alpha=alpha,mean_rate_pct=100*g[RATE].mean(),
                                 sample_sd_pp=100*g[RATE].std(ddof=1),
                                 first_match_pct=100*(counts*(counts-1)).sum()/(20*19)))
        summary=pd.DataFrame(profiles).groupby("alpha")[["mean_rate_pct","sample_sd_pp","first_match_pct"]].mean()
        for alpha,row in summary.iterrows():metrics.append(dict(batch=name,alpha=alpha,**row.to_dict()))
    metrics=pd.DataFrame(metrics)
    metrics.to_csv(OUT/"seed_batches.csv",index=False)
    diff=metrics.set_index(["batch","alpha"]).loc["replication"]-metrics.set_index(["batch","alpha"]).loc["original"]
    diff["mean_pass"]=diff.mean_rate_pct.abs()<=1
    diff["sd_pass"]=diff.sample_sd_pp.abs()<=.5
    diff["first_pass"]=diff.first_match_pct.abs()<=10
    diff.to_csv(OUT/"seed_replication_checks.csv")


if __name__=="__main__":
    summarize_grid(pd.read_csv(OUT/"duration_grid.csv.gz"))
    summarize_streams()
