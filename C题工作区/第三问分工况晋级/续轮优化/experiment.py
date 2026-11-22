"""Screen on existing development draws, then test one nominee per case and attempt."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from policy import HERE, Route
from evaluate import CFG, Engine, independent_metrics, jobs_for, sample_hidden, settings

OUT = HERE / "results"
MET = ["weighted_on_time_rate","weighted_completion_rate","weighted_tardiness_s","cost"]
OFFICIAL = json.loads((CFG/"common_conditions.json").read_bytes())["validation_seeds"]


def save(path, data):
    path.write_bytes(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf8"))


class Bench:
    def __init__(self, case):
        self.case = case
        self.c,tasks,self.zones,_ = settings(case)
        self.jobs = jobs_for(tasks,case)
        df = pd.read_csv(CFG/"source_segment_index.csv").query("split=='validation' and eligible==1")
        self.pool = {f:list(g.itertuples()) for f,g in df.groupby("family_id")}
        self.hidden = {}
        self.prefix = json.loads((HERE/"replay_public_prefix.json").read_bytes())

    def run(self, model, seeds, log=None, policy_factory=None):
        rows = []
        for seed in seeds:
            if seed not in self.hidden:
                self.hidden[seed] = sample_hidden((self.jobs,self.pool),seed)
                for item in self.hidden[seed].values():
                    item["features"]["0.2"] = self.prefix[item["unit_id"]]
            c = self.c
            sim = Engine(self.jobs,self.hidden[seed],self.zones,c["horizon_s"],c["cost_budget"],c["observation_limit"],1.,.5)
            policy = policy_factory() if policy_factory is not None else Route(self.case,model)
            while not sim.terminal:
                a = policy.choose(sim.observe())
                e = sim.step(a)
                if e.get("completion") is not None:
                    policy.record_completion(sim.jobs[a["job_id"]]["family_id"],e["service_start"],e["completion"])
            metrics = sim.metrics()
            check = independent_metrics(self.jobs,sim.events,sim.H)
            if (metrics["invalid_actions"] or metrics["cost"]>sim.B+1e-9 or metrics["observations"]>sim.K or
                    any(abs(metrics[k]-check[k])>1e-9 for k in metrics)):
                raise ValueError((self.case,seed,metrics))
            if log:
                log.write(json.dumps(dict(case=self.case,seed=seed,events=sim.events))+"\n")
            rows.append(dict(seed=seed,**metrics))
        return pd.DataFrame(rows).set_index("seed")


def grid(attempt,case):
    if attempt == 1:
        return [dict(kind="robust",cap=cap,scale=scale,mix=mix,prior_count=prior,completion_weight=.25,
                     stratified=True,scenarios=48,steps=60)
                for cap,scale,mix,prior in [(.80,1,.5,4),(.85,1,1,10),(.75,.85,.15,6),
                    (.90,.85,.6,10),(.75,1,.85,2),(.9,1.1,.15,3)]]
    if attempt == 4:
        return [dict(kind="beam",cap=.9,scale=1.,mix=mix,quantile=q,hist=h,width=36,cw=cw,margin=m)
                for q,mix,h,cw,m in [(.5,.3,4,.35,3),(.6,.4,6,.35,4),(.55,.75,6,.4,4),
                    (.60,.9,10,.20,6),(.65,.35,8,.40,5),(.45,.7,8,.22,7)]]
    if attempt == 2:
        return [dict(kind="motion_roll",cap=cap,scale=scale,mix=.10,prior_count=prior,completion_weight=cw,
                     stratified=True,scenarios=64,steps=75,bandwidth=bw,motion_weight=w)
                for cap,scale,prior,cw,bw,w in [(.96,1,5,.5,.35,.85),(.99,1,12,.65,.60,1),
                    (.92,1,8,.45,.4,1),(.97,1,4,.65,.65,.75),(.94,1.05,12,.85,.45,.8),
                    (.96,.95,20,.5,.35,1)]]
    if attempt == 3:
        return [dict(kind="stochastic",cap=cap,scale=1.,mix=mix,hist=h,width=width,cw=cw,margin=m,
                     scenarios=16,optimism=opt,bandwidth=.45,motion_weight=w)
                for cap,mix,h,width,cw,m,opt,w in [(.93,.20,4,24,.65,2,.45,.85),(.97,.1,6,32,.50,3,.6,.9),
                (.99,.05,4,36,.65,1,.75,.75),(.92,.3,8,28,.6,2,.70,1),(.98,.05,3,36,.8,2,.5,.0)]]
    if attempt == 5:
        configs = ([(.97,6,48,.8,1.5,1.,.4),(.98,5,48,1.4,3,1.,0),(.96,4,48,.8,0,.65,.6),
                    (.99,3,60,1.5,0,.78,0),(.97,4,36,.6,2,.57,.5),(.97,8,48,2,3,.90,.85)]
                   if case=="S2" else
                   [(.99,3,64,.8,.5,1,0),(.98,5,56,1.2,0,1,0),(.98,2,48,.65,1.5,1,.2),
                    (.99,3,72,.7,1,1,.3),(.95,5,64,.6,2,1,.45),(.99,6,56,1.4,1,1,0)])
        return [dict(kind="stochastic",cap=cap,scale=1.,mix=.05,hist=h,width=width,cw=cw,margin=2,
                     scenarios=32,optimism=.52,bandwidth=.5,motion_weight=w,base_after=t,late_bonus=bonus)
                for cap,h,width,cw,bonus,t,w in configs]
    return json.loads((HERE/f"candidates_{attempt}_{case}.json").read_bytes())


def gate(case, level, new, ref):
    at,done=MET[:2]
    delta=new["gate"][at]-ref["gate"][at]
    conditions = dict(official_floor=new["official"][at]+1e-12 >= (.8708 if case=="S2" else (.83 if level==1 else .835)),
        official_delta=new["official"][at]+1e-12 >= ref["official"][at],
        extra_gain=delta+1e-12 >= (.006 if level==1 else .005),
        official_completion=new["official"][done]+1e-12 >= (83/90 if case=="S2" else .9),
        extra_completion=new["gate"][done]+.005+1e-12 >= ref["gate"][done])
    return {k:bool(v) for k,v in conditions.items()}


def attempt(case, num):
    OUT.mkdir(exist_ok=True)
    available = sorted(OUT.glob(f"{case}_promotion_*.json"))
    if len(available) >= (1 if case=="S2" else 2):return
    level=len(available)+1
    target=OUT/f"{case}_attempt_{num}"
    if (target/"verdict.json").exists():return
    target.mkdir(exist_ok=True)
    bench=Bench(case)
    before=json.loads(available[-1].read_bytes())["candidate"] if available else {"kind":"base"}
    candidates=grid(num,case)
    rows=[]
    for i,model in enumerate(candidates):
        m=bench.run(model,range(73000,73120 if num>=5 else 73048))[MET].mean().to_dict()
        rows.append(dict(id=i,**m,model=json.dumps(model)))
    table=pd.DataFrame(rows)
    table["rank_score"]=table[MET[0]]+.25*table[MET[1]]-.000025*table[MET[2]]
    if num>=5:
        ref = bench.run(before,range(73000,73120))[MET[1]].mean()
        table["rank_score"] -= 2.0*(ref-.003-table[MET[1]]).clip(lower=0)
    table.to_csv(target/"screen.csv",index=False)
    candidate=candidates[int(table.rank_score.idxmax())]
    save(target/"nominee.json",candidate)
    start=430000+level*1000
    results=[]; means={}
    for method,model in [("previous",before),("nominee",candidate)]:
        means[method]={}
        for split,seeds in [("official",OFFICIAL),("gate",range(start,start+180))]:
            d=bench.run(model,seeds)
            results.append(d.assign(method=method,split=split))
            means[method][split]=d[MET].mean().to_dict()
    pd.concat(results).to_csv(target/"trials.csv")
    checks=gate(case,level,means["nominee"],means["previous"])
    state=dict(case=case,attempt=num,level=level,passed=all(checks.values()),conditions=checks,previous=before,candidate=candidate,means=means)
    save(target/"verdict.json",state)
    if state["passed"]:
        save(OUT/f"{case}_promotion_{level}.json",state)
    print(case,num,level,state["passed"],checks,means,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("attempt",type=int)
    parser.add_argument("cases",nargs="+",choices=["S2","S3"])
    a=parser.parse_args()
    for case in a.cases:attempt(case,a.attempt)
