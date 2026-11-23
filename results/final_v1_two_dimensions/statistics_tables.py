"""Produce plot-ready rates, paired contrasts and replication errors from trial scores."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "final_v1_evidence_extension"
RATE = "weighted_on_time_rate"
GROUPS = ["S1", "S2", "S3", "overall"]
METRICS = ["mean", "tail05", "tail10", "tail20"]
REFERENCE = (1., .5)


def metrics(array):
    # Last axes are seed and case. Rank independently in every case, then average cases.
    ordered = np.sort(array, axis=-2)
    n = ordered.shape[-2]
    cols = [ordered.mean(axis=-2)] + [ordered[..., :(n*p+99)//100, :].mean(axis=-2) for p in (5, 10, 20)]
    result = np.stack(cols, axis=-1)
    return np.concatenate((result, result.mean(axis=-2, keepdims=True)), axis=-2)


def schedules():
    data = pd.concat([pd.read_csv(path) for path in
                      (PRIOR/"duration_grid.csv.gz", OUT/"fresh_trials.csv.gz", OUT/"local_trials.csv.gz", OUT/"grid_completion_trials.csv.gz")])
    assert len(data) == 27000 and not data.duplicated(["case_id","seed","alpha","beta"]).any()
    assert data.valid_protocol.all() and data.invalid_actions.eq(0).all() and data.independent_diff.max()<1e-9
    assert data.internal_seed.eq(20261122).all() and data[RATE].between(0,1).all()
    data["batch"] = np.where(data.seed<918400300,"original","fresh")
    rows=[]
    for batch, g in data.groupby("batch"):
        start = 918400000 if batch=="original" else 918400900
        assert set(g.seed)==set(range(start,start+300))
        draws = np.random.default_rng(26301).integers(300,size=(5000,300))
        def samples(cell):
            arr=cell.pivot(index="seed",columns="case_id",values=RATE).reindex(index=range(start,start+300),columns=GROUPS[:3]).to_numpy()*100
            assert np.isfinite(arr).all()
            return metrics(arr), np.concatenate([metrics(arr[idx]) for idx in np.array_split(draws,10)])
        cells=g.groupby(["alpha","beta"])
        assert set(cells.groups) == set(itertools.product((.75,.95,1.,1.05,1.5),(0.,.5,1.)))
        base, base_boot = samples(cells.get_group(REFERENCE))
        for (a,b), cell in cells:
            values, boot=samples(cell)
            diffs=values-base
            lo,hi=np.quantile(boot-base_boot,[.025,.975],axis=0)
            for i,case in enumerate(GROUPS):
                for j,metric in enumerate(METRICS):
                    rows.append(dict(batch=batch,case_id=case,alpha=a,beta=b,metric=metric,n_seeds=300,
                                     rate_pct=values[i,j],delta_pp=diffs[i,j],paired_lo_pp=lo[i,j],paired_hi_pp=hi[i,j]))
    tab=pd.DataFrame(rows)
    tab.to_csv(OUT/"schedule_cells.csv",index=False,encoding="utf-8-sig")
    keys=["case_id","alpha","beta","metric"]
    comparison=tab.pivot(index=keys,columns="batch",values="rate_pct").reset_index()
    comparison["fresh_minus_original_pp"]=comparison.fresh-comparison.original
    comparison.to_csv(OUT/"schedule_replication.csv",index=False,encoding="utf-8-sig")
    # Verify archived summaries with independent estimators from original batch.
    old=pd.read_csv(PRIOR/"grid_means.csv").melt(["alpha","beta"],var_name="case_id",value_name="ref")
    joined=tab.query("batch=='original' and metric=='mean'").merge(old,on=keys[:-1],validate="1:1")
    assert len(joined)==36
    np.testing.assert_allclose(joined.rate_pct,100*joined.ref,atol=1e-10)
    ref=pd.read_csv(PRIOR/"tail_replication.csv")
    joined=tab.query("batch=='original' and metric=='tail10' and beta==0.5").merge(ref,on=["case_id","alpha"],validate="1:1")
    assert len(joined)==6
    np.testing.assert_allclose(joined.rate_pct,100*joined["all"],atol=1e-10)


def trajectories():
    rows=[]
    for q, axis, nominal in [("q1","history","full"),("q2","delay",.1)]:
        d=pd.read_csv(OUT/f"{q}_windows.csv.gz")
        d=d[d.horizon==1] if q=="q1" else d[d.history=="full"]
        base=d[d[axis]==nominal]
        threshold=base.loc[base.split=="validation","nade"].quantile(.9)
        for (split,value),group in d.groupby(["split",axis]):
            g=group.merge(base[["window_id","nade"]],on="window_id",suffixes=("","_base"),validate="1:1")
            above=g.nade>threshold
            before=g.nade_base>threshold
            rows.append(dict(question=q,split=split,level=value,n=len(g),threshold=threshold,
                             exceeded=int(above.sum()),exceed_pct=100*above.mean(),
                             entered=int((above&~before).sum()),recovered=int((before&~above).sum())))
    tab=pd.DataFrame(rows)
    assert len(tab)==20
    tab.to_csv(OUT/"trajectory_thresholds.csv",index=False,encoding="utf-8-sig")


if __name__=="__main__":
    schedules()
    trajectories()

