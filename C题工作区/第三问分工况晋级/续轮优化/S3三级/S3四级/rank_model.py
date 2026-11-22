"""Estimate within-environment duration ranks using eligible training motion prefixes."""
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from cue_model import ROOT,cues,features,CFG


def fit(data):
    rows=pd.read_csv(CFG/"source_segment_index.csv").query("split=='train' and eligible==1").reset_index(drop=True)
    x=[]
    for uid in rows.unit_id:
        with np.load(data/"private"/(uid+"_0.2.npz"),allow_pickle=False) as z:x.append(cues(features(z)))
    x=np.stack(x)
    y=np.log(rows.duration_s.to_numpy())
    # Within-group centering removes differences in mean duration and pose between training sites.
    xx=np.zeros_like(x);yy=np.zeros_like(y)
    for ids in rows.groupby(["environment_id","dataset","family_id"]).indices.values():
        xx[ids]=x[ids]-np.median(x[ids],axis=0)
        yy[ids]=np.clip(y[ids]-np.median(y[ids]),-1.6,1.6)
    scale=np.maximum(xx.std(axis=0),[.06,.05,.05,.008,.12,.08,.05,.05,.008,.10])
    xx=np.clip(xx/scale,-3.,3.)
    global_coef=np.linalg.solve(xx.T@xx+40*np.eye(xx.shape[1]),xx.T@yy)
    model={}
    for f,ids in rows.groupby("family_id").indices.items():
        X=xx[ids]
        coef=np.linalg.solve(X.T@X+30*np.eye(X.shape[1]),X.T@yy[ids]+30*global_coef)
        model[f]=(coef/scale).tolist()
    (ROOT/"rank_model.json").write_bytes(json.dumps(model,indent=2).encode())


@lru_cache(maxsize=1)
def model():return json.loads((ROOT/"rank_model.json").read_bytes())

@lru_cache(maxsize=256)
def predict(f,public):
    return np.dot(cues(public),model()[f])

if __name__=="__main__":fit(Path(sys.argv[1]))
