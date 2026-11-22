"""Paired environment diagnostics; each environment is kept intact under resampling."""
from itertools import product
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
METRICS=["nade","coverage","norm_diameter"]


def aggregate(frame,question):
    # Match released Q1/Q2 hierarchies, including Q2's within-record hand weighting.
    keys=["condition","dataset","environment_id","episode_uuid"]
    if question=="q2": keys.append("side")
    means=frame
    for n in range(len(keys),0,-1):
        means=means.groupby(keys[:n],as_index=False)[METRICS].mean()
    return means.set_index("condition").loc[["nominal","stress"]]


def main():
    d=pd.read_csv(OUT/"trajectory/windows.csv.gz")
    assert len(d)==(212+390)*2*2 and not d.duplicated(["question","split","window_id","condition"]).any()
    assert d.coverage.between(0,1).all() and d.norm_diameter.ge(0).all()
    records=[]; uncertainty=[]
    for (question,split),g in d.groupby(["question","split"]):
        assert set(g.condition)=={"nominal","stress"}
        assert g.groupby("window_id").condition.nunique().eq(2).all()
        envs=sorted(g.environment_id.unique())
        groups=[("all","all",g)]
        groups.extend(("environment",e,g[g.environment_id==e]) for e in envs)
        groups.extend(("omit_environment",e,g[g.environment_id!=e]) for e in envs)
        env_values=[]
        for scope,label,sample in groups:
            values=aggregate(sample,question)
            n,s=values.loc["nominal"],values.loc["stress"]
            records.append(dict(question=question,split=split,scope=scope,environment=label,
                windows=sample.window_id.nunique(),episodes=sample.episode_uuid.nunique(),sources="|".join(sorted(sample.dataset.unique())),
                nade_nominal=n.nade,nade_stress=s.nade,nade_diff=s.nade-n.nade,nade_change_pct=100*(s.nade/n.nade-1),
                coverage_nominal=n.coverage,coverage_stress=s.coverage,coverage_change_pp=100*(s.coverage-n.coverage),
                nominal_gap_pp=100*(n.coverage-.9),stress_gap_pp=100*(s.coverage-.9),
                diameter_nominal=n.norm_diameter,diameter_stress=s.norm_diameter,
                diameter_change_pct=100*(s.norm_diameter/n.norm_diameter-1)))
            if scope=="environment":env_values.append(values.to_numpy())
        env_values=np.array(env_values)
        assert len(envs)==4
        draws=np.array(list(product(range(4),repeat=4)))
        # A separate environment-equal estimand; resample both conditions and all records together.
        pooled=env_values.mean(axis=0)
        boot=env_values[draws].mean(axis=1)
        for i,metric in enumerate(METRICS):
            def contrast(a,b):
                return 100*(b-a) if metric=="coverage" else 100*(b/a-1)
            changes=contrast(env_values[:,0,i],env_values[:,1,i])
            boot_delta=contrast(boot[:,0,i],boot[:,1,i])
            plus=int((changes>1e-10).sum()); minus=int((changes< -1e-10).sum()); n=plus+minus
            exact=min(1,2*sum(comb(n,k) for k in range(min(plus,minus)+1))/(2**n)) if n else 1.
            lo,hi=np.quantile(boot_delta,[.025,.975])
            uncertainty.append(dict(question=question,split=split,metric=metric,unit="pp" if metric=="coverage" else "relative_pct",
                envs=4,n_positive=plus,n_negative=minus,n_tied=4-n,exact_twosided_sign_p=exact,
                env_equal_delta=contrast(pooled[0,i],pooled[1,i]),resample_p025=lo,resample_p975=hi,
                min_environment=changes.min(),max_environment=changes.max(),resamples=len(draws)))
    environments=pd.DataFrame(records)
    environments.to_csv(OUT/"environment_results.csv",index=False)
    pd.DataFrame(uncertainty).to_csv(OUT/"paired_uncertainty.csv",index=False)
    # Recover the frozen official hierarchy at the nominal point in each split.
    src=OUT.parents[1]/"tmp/final_v1_20261122/C题工作区/正式方案/results"
    q1=pd.read_csv(src/"q1_horizons.csv").query("model=='B' and horizon==1").set_index("split")
    q1range=pd.read_csv(src/"q1_range_summary.csv").query("model=='B' and horizon==1").set_index("split")
    q2=pd.read_csv(src/"q2_summary.csv").query("model=='B'").set_index("split")
    for row in environments.query("scope=='all'").itertuples():
        if row.question=="q1":
            expected=[q1.loc[row.split,"nade"],q1range.loc[row.split,"coverage"],2*q1range.loc[row.split,"norm_radius"]]
        else: expected=q2.loc[row.split,["nade","coverage","width"]].to_numpy(float)
        np.testing.assert_allclose([row.nade_nominal,row.coverage_nominal,row.diameter_nominal],expected,atol=1e-10,rtol=0)


if __name__=="__main__":main()
