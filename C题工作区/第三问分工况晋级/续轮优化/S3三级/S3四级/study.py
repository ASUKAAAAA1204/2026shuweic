"""Develop on previously used seeds, then evaluate a frozen nominee on a disjoint block."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fourth import HERE,Policy,previous
from run import bench
from experiment import MET,save

OUT=HERE/"results"
DEV=list(range(440000,440480))
# Stagger the small screen across the old block instead of using its first rows.
SMALL=DEV[::4]

def digest(p):
    return hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()

def measure(b,p,seeds,log=None):
    return b.run(None,seeds,log,lambda:previous("S3") if p is None else Policy(p))

def stats(old,new):
    if not old.index.equals(new.index):raise ValueError("Unpaired trials")
    rng=np.random.default_rng(557221)
    n=len(old)
    # Avoid a large temporary (10000 x n) for each metric.
    abs_boot=[];diff_boot=[]
    after=new[MET[0]].to_numpy();delta=after-old[MET[0]].to_numpy()
    for _ in range(100):
        ix=rng.integers(n,size=(100,n))
        abs_boot.extend(after[ix].mean(axis=1))
        diff_boot.extend(delta[ix].mean(axis=1))
    s={m:dict(old=old[m].mean(),new=new[m].mean(),gain=(new[m]-old[m]).mean()) for m in MET}
    s[MET[0]].update(absolute_ci=np.quantile(abs_boot,[.025,.975]).tolist(),
                     gain_ci=np.quantile(diff_boot,[.025,.975]).tolist())
    m=s[MET[0]]
    return s,bool(m["absolute_ci"][0]>=.85 and m["gain_ci"][0]>0 and s[MET[1]]["gain"]>=0)

def screen(design):
    f=OUT/f"design_{design}";f.mkdir(parents=True,exist_ok=True)
    grid=json.loads((HERE/f"grid_{design}.json").read_bytes())
    b=bench(); rows=[]
    for i,p in enumerate(grid):
        path=f/f"small_{i}.csv"
        if not path.exists():measure(b,p,SMALL).to_csv(path)
        rows.append(dict(id=i,**pd.read_csv(path)[MET].mean()))
    x=pd.DataFrame(rows)
    basefile=OUT/"dev_tier3.csv"
    if not basefile.exists():measure(b,None,DEV).to_csv(basefile)
    base=pd.read_csv(basefile).set_index("seed")
    x["rank"]=x[MET[0]]-1.8*(base.loc[SMALL,MET[1]].mean()-x[MET[1]]).clip(lower=0)
    x.to_csv(f/"screen.csv",index=False)
    finalists=[]
    for i in x.nlargest(min(4,len(x)),"rank").id.to_list():
        path=f/f"dev_{i}.csv"
        if not path.exists():measure(b,grid[i],DEV).to_csv(path)
        d=pd.read_csv(path)[MET].mean()
        finalists.append(dict(id=i,**d,score=d[MET[0]]-2.5*max(0,base[MET[1]].mean()-d[MET[1]])))
    rank=pd.DataFrame(finalists).sort_values("score",ascending=False)
    rank.to_csv(f/"finalists.csv",index=False)
    save(f/"nominee.json",grid[int(rank.iloc[0].id)])
    print(rank.round(5).to_string(index=False),flush=True)

def gate(design,r):
    if any(OUT.glob("round_*/passed.lock")):raise RuntimeError("A release is already frozen")
    model=json.loads((OUT/f"design_{design}/nominee.json").read_bytes())
    f=OUT/f"round_{r}";f.mkdir(parents=True,exist_ok=True)
    if (f/"verdict.json").exists():raise RuntimeError("Round already evaluated")
    for existing in OUT.glob("round_*/nominee.json"):
        if existing.parent!=f and json.loads(existing.read_bytes())==model:raise ValueError("Cannot retry identical nominee")
    snap=f/"nominee.json";save(snap,model)
    files=["fourth.py","cue_model.py","cue_model.json","../dynamic.cpp","../bellman.py","../published.json","study.py"]
    if model.get("guard"):files+=["guard_api.py","guard.cpp"]
    if model.get("env_weight"):files+=["posterior.py","../../../../../赛题分析报告/source_segment_index.csv"]
    if model.get("rank_weight"):files+=["rank_model.py","rank_model.json"]
    lock={name:digest(HERE/name) for name in files}
    (f/"policy_snapshot.txt").write_bytes((HERE/"fourth.py").read_bytes())
    save(f/"before.json",dict(design=design,round=r,candidate_hash=digest(snap),files=lock))
    seeds=range(7820000+3000*(r-1),7820000+3000*(r-1)+2048)
    b=bench();old=measure(b,None,seeds)
    with gzip.open(f/"events.jsonl.gz","wt",encoding="utf8") as log:new=measure(b,model,seeds,log)
    pd.concat([old.assign(method="tier3"),new.assign(method="candidate")]).to_csv(f/"paired.csv")
    metrics,passed=stats(old,new)
    result=dict(round=r,design=design,seeds=[seeds[0],seeds[-1]],n=len(seeds),passed=passed,metrics=metrics)
    save(f/"verdict.json",result)
    if passed:save(f/"passed.lock",result)
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("design",type=int);p.add_argument("--submit",type=int)
    a=p.parse_args()
    if any(OUT.glob("round_*/passed.lock")):raise SystemExit("Already promoted")
    if a.submit:gate(a.design,a.submit)
    else:screen(a.design)
