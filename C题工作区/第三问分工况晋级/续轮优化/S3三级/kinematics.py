"""Aggregate the free motion prefix and fit shallow duration trees on train rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
FAMILIES=("A","P","L")


def features(z):
    left,right=(z[s+"__position"] for s in ("left_hand","right_hand"))
    head=z["head__position"][:,0]
    def median(values,default=0.):
        v=np.asarray(values)
        v=v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else default
    out=[]
    hands=[]
    for side,x in (("left_hand",left),("right_hand",right)):
        t=z[side+"__t_s"]
        good=z[side+"__quality_known"] & ~z[side+"__bad"] & np.isfinite(x).all(axis=(1,2))
        hand=x.mean(axis=1)
        hands.append(hand)
        dt=np.diff(t)
        pair=good[:-1]&good[1:]&(dt>0)
        speed=np.linalg.norm(np.diff(hand,axis=0)[pair],axis=1)/dt[pair]
        htime=z["head__t_s"]
        if len(htime) and good.any():
            near=np.abs(htime[:,None]-t[good]).argmin(axis=0)
            offset=hand[good]-head[near]
        else:
            offset=np.empty((0,3))
        spread=np.linalg.norm(x[good]-x[good,:1],axis=2)
        aperture=np.linalg.norm(x[good,4]-x[good,8],axis=1)
        out.extend([round(float(np.mean(good)),1) if len(t) else 0.,
                    round(median(speed),1),round(median(np.linalg.norm(offset,axis=1)),1),
                    round(median(offset[:,2]),1),round(median(spread),2),round(median(aperture),2)])
    if len(hands[0]) and len(hands[1]):
        out.append(round(median(np.linalg.norm(hands[0].mean(0)-hands[1].mean(0))),1))
    else:out.append(0.)
    return out


def train_tree(x,y,ids,rng,depth):
    v=y[ids]
    # Coarse observable groups, with at least six train observations per leaf.
    if depth==0 or len(v)<14:
        return float(np.median(v))
    best=None
    for col in rng.choice(x.shape[1],size=5,replace=False):
        vals=x[ids,col]
        for q in (.25,.5,.75):
            t=float(np.quantile(vals,q)); mask=vals<=t
            if min(mask.sum(),(~mask).sum())<6:continue
            loss=np.square(v[mask]-v[mask].mean()).sum()+np.square(v[~mask]-v[~mask].mean()).sum()
            if best is None or loss<best[0]:best=loss,int(col),t,mask
    if best is None:return float(np.median(v))
    _,col,t,mask=best
    return [col,t,train_tree(x,y,ids[mask],rng,depth-1),train_tree(x,y,ids[~mask],rng,depth-1)]


def estimate(trees,x):
    values=[]
    for node in trees:
        while isinstance(node,list):
            col,t,left,right=node
            node=left if x[col]<=t else right
        values.append(node)
    return float(np.mean(values))


def prepare(data):
    index=pd.read_csv(ROOT.parents[3]/"赛题分析报告/source_segment_index.csv")
    rows=[]; replay={}
    for r in index.query("eligible==1 and split in ['train','validation']").itertuples():
        with np.load(data/"private"/(r.unit_id+"_0.2.npz"),allow_pickle=False) as z:
            f=features(z)
        if r.split=="train":rows.append((r.family_id,r.duration_s,f))
        else:replay[r.unit_id]=f
    x=np.array([r[2] for r in rows]);fam=np.array([r[0] for r in rows]);y=np.log([r[1] for r in rows])
    rng=np.random.default_rng(92811)
    base={f:float(np.median(y[fam==f])) for f in FAMILIES}
    residual=y-np.array([base[f] for f in fam])
    trees=[train_tree(x,residual,rng.choice(len(y),size=len(y),replace=True),rng,4) for _ in range(64)]
    for name,value in [("kinematics.json",dict(trees=trees)),("prefix_for_replay.json",replay)]:
        (ROOT/name).write_bytes(json.dumps(value,allow_nan=False,separators=(",",":")).encode())


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("data",type=Path)
    prepare(p.parse_args().data)
