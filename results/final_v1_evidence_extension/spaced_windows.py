"""Time-separated subset chosen before considering any prediction errors."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SPACE=ROOT/"tmp/final_v1_20261122/C题工作区"


def select():
    windows=pd.read_csv(SPACE/"阶段交付/stage1_review/corrected_results/data/windows.csv")
    selected=[]
    for uid,g in windows.query("split in ['test','validation']").groupby("episode_uuid"):
        with np.load(ROOT/"C_Problem_Data/v1/v1/records"/f"{uid}.npz",allow_pickle=False) as raw:
            for side,part in g.groupby("side"):
                t=raw[side+"__t_s"]
                last=-np.inf
                for w in part.sort_values(["start_index","window_id"]).itertuples():
                    start,end=t[w.start_index],t[w.end_index_exclusive-1]
                    if start-last>=2.0-1e-9:
                        selected.append(dict(window_id=w.window_id,side=side,episode_uuid=uid,
                                             split=w.split,environment_id=w.environment_id,first_s=start,last_s=end))
                        last=end
    d=pd.DataFrame(selected)
    assert d.window_id.is_unique and d.environment_id.nunique()==8
    d.to_csv(OUT/"spaced_selection.csv",index=False)
    return d


def summarize(metrics,selection):
    d=metrics[metrics.window_id.isin(selection.window_id)]
    rows=[]
    for (split,environment),g in d.groupby(["split","environment_id"]):
        means=g
        keys=["delay","dataset","environment_id","episode_uuid","side"]
        for level in (5,4,3,2,1):
            means=means.groupby(keys[:level],as_index=False).nade.mean()
        x=means.set_index("delay").nade
        assert set(x.index)=={.1,.2} and g.window_id.nunique()*2==len(g)
        rows.append(dict(split=split,environment_id=environment,n=g.window_id.nunique(),
                         nade_01=x[.1],nade_02=x[.2],change_pct=100*(x[.2]/x[.1]-1)))
    res=pd.DataFrame(rows)
    assert len(res)==8 and res.n.sum()==len(selection)
    res.to_csv(OUT/"spaced_environments.csv",index=False)


if __name__=="__main__":
    selection=select()
    metrics=pd.read_csv(OUT/"trajectory/per_window.csv.gz")
    summarize(metrics.loc[metrics.history.eq("short")],selection)
