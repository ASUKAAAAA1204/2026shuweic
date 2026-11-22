"""Carry validation nominal P90 thresholds to original test windows."""
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent
frame=pd.read_csv(OUT.parent/"final_v1_remaining/trajectory/windows.csv.gz")
rows=[]
for q,all_rows in frame.groupby("question"):
    val=all_rows.query("split=='validation' and condition=='nominal'")
    assert val.window_id.nunique()==len(val)==212
    threshold=val.nade.quantile(.9)
    test=all_rows.query("split=='test'")
    for label,g in [("all",test),*test.groupby("environment_id")]:
        scores=g.pivot(index="window_id",columns="condition",values="nade")
        assert scores.notna().all().all()
        bad=scores.gt(threshold)
        rows.append(dict(question=q,environment=label,n=len(bad),threshold=threshold,
                         exceed_nominal=bad.nominal.mean(),exceed_stress=bad.stress.mean(),
                         both=int((bad.nominal&bad.stress).sum()),new=int((~bad.nominal&bad.stress).sum()),
                         recovered=int((bad.nominal&~bad.stress).sum())))
pd.DataFrame(rows).to_csv(OUT/"tail_threshold_transfer.csv",index=False)
