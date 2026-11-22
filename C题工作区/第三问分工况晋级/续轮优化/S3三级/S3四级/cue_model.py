"""Fit duration contrasts only on broad kinematic groups in eligible training clips."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from kinematics import features
from adaptive import ROOT as PRIOR_ROOT
from search import CFG

def cues(x):
    # Both hands' visible prefix. Rounded coarse summaries; no absolute coordinates or clocks.
    x=np.asarray(x)
    return np.array([x[12],(x[2]+x[8])/2,(x[3]+x[9])/2,(x[5]+x[11])/2,
                     max(x[1],x[7]),min(x[1],x[7]),x[2]-x[8],abs(x[3]-x[9]),
                     x[4]+x[10],min(x[0],x[6])])

def train(data):
    df=pd.read_csv(CFG/"source_segment_index.csv").query("eligible==1 and split=='train'")
    items=[]
    for r in df.itertuples():
        with np.load(data/"private"/(r.unit_id+"_0.2.npz"),allow_pickle=False) as z: x=cues(features(z))
        items.append((r.family_id,r.duration_s,x))
    out={}
    for f in sorted(df.family_id.unique()):
        X=np.stack([x for family,y,x in items if f==family])
        y=np.array([y for family,y,x in items if f==family])
        out[f]=[]
        for i in range(X.shape[1]):
            threshold=float(np.median(X[:,i]))
            group=X[:,i]>threshold
            # Winsorized log duration prevents one long clip from determining a contrast.
            log=np.clip(np.log(y),*np.quantile(np.log(y),[.05,.95]))
            offsets=[]
            for k in (False,True):
                v=log[group==k]-log.mean()
                offsets.append(float(v.sum()/(len(v)+4)))
            out[f].append(dict(t=threshold,offset=offsets))
    (ROOT/"cue_model.json").write_bytes(json.dumps(out,indent=2).encode())

if __name__=="__main__":
    train(Path(sys.argv[1]))
