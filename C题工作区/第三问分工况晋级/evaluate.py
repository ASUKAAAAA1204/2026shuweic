"""Paired, causal evaluation; train() runs separately before this entry point."""
import argparse
import gzip
import hashlib
import json
import random
import sys
import numpy as np
import pandas as pd
from search import ROOT,OLD,CFG,OUT,GATES,settings
import dispatch
sys.path.insert(0,str(OLD/"code"))
from engine import Engine,independent_metrics
from baseline import action
from deadline_policy import choose,pressure
from support import jobs_for
import family_prior
import rolling


def sample_hidden(case,seed):
    # Official row order is part of the sampling protocol.
    jobs,pool=case
    rng=random.Random(seed)
    return {j["job_id"]:dict(duration=r.duration_s,unit_id=r.unit_id,features={str(t):[] for t in (.2,.6,1.)})
            for j in jobs for r in [rng.choice(pool[j["family_id"]])]}


def replay(c,jobs,zones,pools,model,hidden):
    sim=Engine(jobs,hidden,zones,c["horizon_s"],c["cost_budget"],c["observation_limit"],1.,.5)
    med={f:np.median(x) for f,x in pools.items()}
    history={}
    while not sim.terminal:
        o=sim.observe()
        if model is None:
            a=choose(o,pools,history) if pressure(o,pools,history) else action(o,med,"density","minimum")
        elif model["kind"]=="family_prior":
            a=family_prior.dispatch(o,pools)
        elif model["kind"]=="rolling":
            a=rolling.choose(o,model,pools,history)
        else:
            a=dispatch.choose(o,model)
        e=sim.step(a)
        if a["type"]=="execute" and e.get("completion") is not None:
            history.setdefault(sim.jobs[a["job_id"]]["family_id"],[]).append(e["completion"]-e["service_start"]-.5)
    m=sim.metrics()
    check=independent_metrics(jobs,sim.events,sim.H)
    assert not m["invalid_actions"] and max(abs(check[k]-m[k]) for k in m)<1e-9
    return m,sim.events


def evaluate(case,round_id,force=False):
    locked=OUT/f"frozen_{case}.json"
    if locked.exists() and not force:return
    run=OUT/f"round_{round_id}"
    model_path=run/f"{case}.json"
    model=json.loads(model_path.read_bytes())
    c,tasks,zones,pools=settings(case)
    jobs=jobs_for(tasks,case)
    segments=pd.read_csv(CFG/"source_segment_index.csv").query("split=='validation' and eligible==1")
    pool={f:list(g.itertuples()) for f,g in segments.groupby("family_id")}
    seeds=json.loads((CFG/"common_conditions.json").read_bytes())["validation_seeds"]
    ref=pd.read_csv(OLD/"audit/original_frozen_trials.csv").query("case_id==@case and alpha==1 and beta==.5 and method=='pressure_gated'").set_index("seed")
    rows=[]
    with gzip.open(run/f"{case}_events.jsonl.gz","wt",encoding="utf8") as log:
        for split,ss in [("official",seeds),("extra",range(73000,73120))]:
            for seed in ss:
                hidden=sample_hidden((jobs,pool),seed)
                for method,param in [("old",None),("new",model)]:
                    m,events=replay(c,jobs,zones,pools,param,hidden)
                    if method=="old" and split=="official":
                        for k in ["weighted_on_time_rate","weighted_completion_rate","cost","weighted_tardiness_s"]:
                            assert abs(m[k]-ref.at[seed,k])<1e-9
                    tag=dict(case_id=case,split=split,seed=seed,method=method)
                    rows.append(dict(**tag,**m))
                    if method=="new":log.write(json.dumps(dict(**tag,events=events))+"\n")
    d=pd.DataFrame(rows)
    d.to_csv(run/f"{case}_trials.csv",index=False)
    t=d.groupby(["split","method"]).mean(numeric_only=True)
    old=t.loc["official","old"];new=t.loc["official","new"]
    extra=t.loc["extra","new"].weighted_on_time_rate-t.loc["extra","old"].weighted_on_time_rate
    gate_on,gate_done=GATES[case]
    passed=bool(new.weighted_on_time_rate+1e-12>=gate_on and new.weighted_completion_rate+1e-12>=gate_done and extra>=-1e-12)
    state=dict(case=case,round=round_id,passed=passed,official_on=new.weighted_on_time_rate,
               official_done=new.weighted_completion_rate,extra_on_gain=extra,
               official_on_gain=new.weighted_on_time_rate-old.weighted_on_time_rate,
               model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
               dispatch_sha256=hashlib.sha256((ROOT/{"family_prior":"family_prior.py","priority":"dispatch.py","rolling":"rolling.py"}[model["kind"]]).read_bytes()).hexdigest())
    if model["kind"]=="rolling":
        state["rolling_cpp_sha256"]=hashlib.sha256((ROOT/"rolling.cpp").read_bytes()).hexdigest()
    print(state,flush=True)
    (run/f"{case}_gate.json").write_text(json.dumps(state,indent=2),encoding="utf8")
    if passed and not locked.exists():locked.write_text(json.dumps(state,indent=2),encoding="utf8")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("round",type=int);ap.add_argument("cases",nargs="*",default=list(GATES))
    args=ap.parse_args()
    for case in args.cases:evaluate(case,args.round)
