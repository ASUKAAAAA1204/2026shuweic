"""Train a priority dispatch policy per published case. Uses training units only."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
CFG=ROOT.parents[1]/"赛题分析报告"
OLD=ROOT.parent/"第三问改进正式方案"
OUT=ROOT/"results"
GATES={"S1":(.94,1.),"S2":(.82,.9222222222222222),"S3":(.825,.9)}


def settings(case):
    config=json.loads((CFG/"common_conditions.json").read_bytes())
    c=next(c for c in config["cases"] if c["case_id"]==case)
    tasks=pd.read_csv(CFG/"tasks.csv").fillna("").query("case_id==@case").sort_values("job_id")
    zones=pd.read_csv(CFG/"service_zones.csv").set_index("zone_id")[["x","y"]].T.to_dict("list")
    durations=pd.DataFrame(json.loads((OLD/"model/training_family_samples.json").read_bytes()))
    pools={f:g.duration_s.to_numpy() for f,g in durations.groupby("family_id")}
    return c,tasks,zones,pools


class Job(ct.Structure):
    _fields_=[(k,ct.c_double) for k in ("release","due","weight","x","y","typical")]+[("parents",ct.c_int)]


class TrainingReplay:
    def __init__(self,case):
        self.case,self.tasks,self.zones,self.pools=settings(case)
        self.n=len(self.tasks)
        ids=list(self.tasks.job_id)
        self.jobs=(Job*self.n)(*[Job(r.release_s,r.due_s,r.priority_weight,*self.zones[r.zone_id],
            float(np.median(self.pools[r.family_id]))+.5,
            sum(1<<ids.index(k) for k in r.predecessors.split("|") if k)) for r in self.tasks.itertuples()])
        source=ROOT/"search_core.cpp"
        binary=Path(tempfile.gettempdir())/("q3_dispatch_"+hashlib.sha256(source.read_bytes()).hexdigest()[:12]+".dll")
        if not binary.exists():
            tmp=binary.with_name(binary.stem+str(os.getpid())+".dll")
            subprocess.run(["g++","-std=c++17","-O3","-shared","-static","-o",str(tmp),str(source)],check=True)
            os.replace(tmp,binary)
        self.library=ct.CDLL(str(binary))
        self.evaluate=self.library.evaluate
        arr=np.ctypeslib.ndpointer(np.float64,flags="C_CONTIGUOUS")
        self.evaluate.argtypes=[ct.POINTER(Job),ct.c_int,ct.c_double,ct.c_double,arr,ct.c_int,arr,ct.c_int,arr]
        self.evaluate.restype=None

    def sample(self,seed,count):
        rng=np.random.default_rng(seed)
        return np.ascontiguousarray(np.stack([rng.choice(self.pools[f],count)+.5 for f in self.tasks.family_id],axis=1))

    def score(self,p,samples):
        p=np.ascontiguousarray(p,dtype=float)
        out=np.empty((len(p),3))
        self.evaluate(self.jobs,self.n,self.case["horizon_s"],self.case["cost_budget"],samples,len(samples),p,len(p),out)
        return out


def train(case,round_id):
    run=OUT/f"round_{round_id}"
    run.mkdir(parents=True,exist_ok=True)
    target=run/f"{case}.json"
    if target.exists() or (OUT/f"frozen_{case}.json").exists():return
    env=TrainingReplay(case)
    n=env.n
    rng=np.random.default_rng(5412+round_id*100+n)
    priorities=np.array(env.tasks.priority_weight,dtype=float)
    weights=[.25,.65,1.15]
    finalists=[]
    history=[]
    for island,cw in enumerate(weights):
        population=rng.normal(0,1.6,(88,n+5))
        population[:,:n]+=priorities-.035*env.tasks.due_s.to_numpy()
        population[:,n:]=rng.uniform([0,0,0,0,.45],[3,5,2,4,1.8],size=(88,5))
        for gen in range(24):
            scores=env.score(population,env.sample(260000+round_id*299+gen//3,96))
            objective=scores[:,0]+cw*scores[:,1]-.00008*scores[:,2]
            order=np.argsort(objective)[::-1]
            elite=population[order[:12]].copy()
            if gen%4==3:finalists.extend(elite[:3].copy())
            history.append(dict(generation=gen,island=island,train_on=scores[order[0],0],train_completed=scores[order[0],1]))
            left=elite[rng.integers(len(elite),size=76)]
            right=elite[rng.integers(len(elite),size=76)]
            child=np.where(rng.random(left.shape)<.75,left,right)
            child+=rng.normal(0,.8 if gen<12 else .4,child.shape)*(rng.random(child.shape)<.42)
            child[:,n:]=np.clip(child[:,n:],[0,0,0,0,.3],[7,12,5,12,3])
            population=np.vstack([elite,child])
    params=np.array(finalists)
    final=env.score(params,env.sample(930000+round_id,1500))
    rank=final[:,0]+(.7 if case=="S1" else .55)*final[:,1]-.00007*final[:,2]
    best=int(rank.argmax())
    pd.DataFrame(history).to_csv(run/f"{case}_training.csv",index=False)
    pd.DataFrame(final,columns=["on_time","completion","tardiness"]).assign(rank_score=rank).to_csv(run/f"{case}_shortlist.csv",index=False)
    target.write_text(json.dumps(dict(case=case,round=round_id,kind="priority",params=params[best].tolist(),
        jobs=list(env.tasks.job_id),typical={f:float(np.median(v))+.5 for f,v in env.pools.items()},
        training_selection=final[best].tolist()),indent=2),encoding="utf8")
    print("training selection",case,final[best],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("round",type=int)
    ap.add_argument("cases",nargs="*",default=list(GATES))
    args=ap.parse_args()
    for case in args.cases:train(case,args.round)
