"""One frozen nominee per disjoint seed block. Stop on the first passing release."""
import argparse
import gzip
import json
import sys
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from adaptive import ROOT, Adaptive, frozen_dispatcher
from bellman import Bellman
from experiment import Bench, MET, save

OUT=ROOT/"results"
DEV=list(range(440000,440480))


def bench():
    b=Bench("S3")
    b.prefix=json.loads((ROOT/"prefix_for_replay.json").read_bytes())
    return b


def measure(b,model,seeds,log=None):
    make=(lambda: frozen_dispatcher("S3",2)) if model is None else (lambda: Bellman(model) if model.get("kind")=="dp" else Adaptive(model))
    return b.run(None,seeds,log,make)


def paired(before,after):
    if not before.index.equals(after.index):
        raise ValueError("Seed mismatch")
    rng=np.random.default_rng(66517)
    indices=rng.integers(len(before),size=(10000,len(before)))
    out={}
    for metric in MET:
        v=(after[metric]-before[metric]).to_numpy()
        low,hi=np.quantile(v[indices].mean(axis=1),[.025,.975])
        out[metric]=dict(old=before[metric].mean(),new=after[metric].mean(),gain=v.mean(),
                         low=float(low),high=float(hi))
    return out


def passed(s):
    return bool(s[MET[0]]["new"]>=.85 and s[MET[0]]["low"]>0 and s[MET[1]]["gain"]>=0)


def grid(r):
    if r==1:
        return [dict(motion=w,width=b,hist=h,cw=c,bonus=lw) for w,b,h,c,lw in
                [(1,64,3,.8,.5),(.5,64,4,.8,.5),(1.75,64,4,.8,.5),(2,84,7,.5,.7),
                 (.85,96,8,1.,.5),(.65,48,2,.6,1.5)]]
    return json.loads((ROOT/f"grid_{r}.json").read_bytes())


def screen(r):
    folder=OUT/f"design_{r}";folder.mkdir(exist_ok=True,parents=True)
    b=bench()
    rows=[]
    models=grid(r)
    for i,m in enumerate(models):
        d=measure(b,m,DEV[:80])
        rows.append(dict(id=i,model=json.dumps(m),**d[MET].mean().to_dict()))
    small=pd.DataFrame(rows)
    rank=small[MET[0]]+.4*small[MET[1]]
    top=rank.nlargest(min(3,len(rank))).index
    small.to_csv(folder/"screen80.csv",index=False)
    for i in top:
        d=measure(b,models[i],DEV)
        d.to_csv(folder/f"dev_{i}.csv")
        print(r,i,json.dumps(models[i]),d[MET].mean().round(5).to_dict(),flush=True)
    # Previous confirmation seeds are now development only.
    choices=[]
    for i in top:
        m=pd.read_csv(folder/f"dev_{i}.csv")[MET].mean()
        score=m[MET[0]]-max(0.,.91543-m[MET[1]])*3
        choices.append((score,int(i)))
    best=max(choices)[1]
    save(folder/"selected.json",models[best])
    return models[best]


def gate(r,model,design):
    if any(OUT.glob("round_*/passed.lock")):
        raise RuntimeError("Already promoted")
    folder=OUT/f"round_{r}";folder.mkdir(exist_ok=True,parents=True)
    snapshot=folder/"nominee.json"
    if snapshot.exists() and json.loads(snapshot.read_bytes())!=model:raise ValueError("Nominee changed")
    for prev in OUT.glob("round_*/nominee.json"):
        if prev!=snapshot and model==json.loads(prev.read_bytes()):raise ValueError("Duplicate nominee")
    save(snapshot,model)
    b=bench();start=6810000+1000*(r-1);ids=range(start,start+640)
    old=measure(b,None,ids)
    with gzip.open(folder/"actions.jsonl.gz","wt",encoding="utf8") as log:new=measure(b,model,ids,log)
    assert len(old)==len(new)==640
    pd.concat([old.assign(method="tier2"),new.assign(method="candidate")]).to_csv(folder/"paired_trials.csv")
    s=paired(old,new)
    state=dict(round=r,design=design,start=start,count=len(ids),passed=passed(s),metrics=s,
               nominee_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    save(folder/"verdict.json",state)
    if state["passed"]:save(folder/"passed.lock",state)
    print(json.dumps(state),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("design",type=int);a.add_argument("--submit",type=int);opt=a.parse_args()
    if any(OUT.glob("round_*/passed.lock")):raise SystemExit("Already promoted")
    if opt.submit:gate(opt.submit,json.loads((OUT/f"design_{opt.design}/selected.json").read_bytes()),opt.design)
    else:screen(opt.design)
