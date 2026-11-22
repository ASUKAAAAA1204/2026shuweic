"""Replicate the history × scoring-horizon interaction on frozen trajectory windows."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/"final_v1_remaining/trajectory"))
import replay as prior


def main():
    snap=prior.verify_snapshot(prior.snapshot())
    windows=prior.official_windows()
    model=prior.AdoptedModel()
    rows=[]
    for episode,group in windows.groupby("episode_uuid"):
        with np.load(prior.RECORDS/f"{episode}.npz",allow_pickle=False) as data:
            raw=dict(data)
        for w in group.itertuples():
            t=raw[w.side+"__t_s"]
            i,j=w.start_index,w.end_index_exclusive
            anchor=t[i-1]
            ages=t[i:j]-anchor
            keep=ages<=1+1e-12
            truth=raw[w.side+"__position"][i:j].mean(1)[keep]
            ages=ages[keep]
            for history,span in [("full",np.inf),("0.30s",.3)]:
                past=prior.run_q1_q3.history_since(raw,anchor,span)
                ix=len(past[w.side+"__t_s"])-1
                pos,_=model.predict_position(past,w.side,ix,ages,w.dataset)
                error=np.linalg.norm(pos-truth,axis=1)/model.paid["scale"][w.dataset]
                for h in (.2,.6,1.):
                    rows.append(dict(split=w.split,window_id=w.window_id,dataset=w.dataset,
                        environment_id=w.environment_id,episode_uuid=episode,history=history,
                        horizon=h,nade=error[ages<=h+1e-12].mean()))
    d=pd.DataFrame(rows)
    assert len(d)==602*6
    old=pd.read_csv(prior.SENSITIVITY/"q1_history_windows.csv.gz").query("history in ['full','0.30s']")
    idx=["window_id","history","horizon"]
    matched=d.query("split=='validation'").merge(old,on=idx,validate="one_to_one")
    assert len(matched)==212*6
    np.testing.assert_allclose(matched.nade_x,matched.nade_y,atol=1e-10,rtol=0)
    ref=pd.read_csv(prior.FROZEN/"q1_windows.csv.gz").query("model=='B' and split=='test'")
    matched=d.query("split=='test' and history=='full'").merge(ref,on=["window_id","horizon"],validate="one_to_one")
    assert len(matched)==390*3
    np.testing.assert_allclose(matched.nade_x,matched.nade_y,atol=1e-10,rtol=0)
    prev=pd.read_csv(OUT.parent/"final_v1_remaining/trajectory/windows.csv.gz").query("question=='q1' and condition=='stress'")
    matched=d.query("history=='0.30s' and horizon==1").merge(prev,on="window_id",validate="one_to_one")
    assert len(matched)==602
    np.testing.assert_allclose(matched.nade_x,matched.nade_y,atol=1e-10,rtol=0)
    result=[]
    for split,frame in d.groupby("split"):
        for label,g in [("all",frame),*frame.groupby("environment_id")]:
            group=["history","horizon","dataset","environment_id","episode_uuid"]
            for n in (5,4,3,2): g=g.groupby(group[:n],as_index=False).nade.mean()
            paired=g.pivot(index="horizon",columns="history",values="nade")
            delta=paired["0.30s"]-paired["full"]
            result.append(dict(split=split,environment=label,diff_02=delta.loc[.2],
                               diff_06=delta.loc[.6],diff_10=delta.loc[1.],interaction=delta.loc[1.]-delta.loc[.2]))
    pd.DataFrame(result).to_csv(OUT/"history_interaction.csv",index=False)
    d.to_csv(OUT/"history_horizon_windows.csv.gz",index=False)
    assert snap==prior.snapshot()


if __name__=="__main__":main()
