"""Repeat the previous formal nominal dispatcher on the final model's fixed instances."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
ROOT = RESULTS.parent
SPACE = ROOT/"tmp/final_v1_20261122/C题工作区"
sys.path.insert(0,str(ROOT/"tmp/previous_nominal_e3c1885/C题工作区/正式方案/code"))
from nominal_policy import dispatcher
import rolling
from policy import Route
sys.path.append(str(SPACE/"第三问B最终方案/code"))
from engine import Engine
from independent import independent_rebuild

DEFAULT = 20261122
ROLL, BEAM = rolling.choose, Route.beam
MET = ["weighted_on_time_rate","weighted_completion_rate","weighted_tardiness_s","cost","observations",
       "elapsed_s","observed_overrun_s","reserved_idle_s","censored_units","invalid_actions","valid_protocol","steps"]


def shifted(fn, base, delta):
    if delta==0:
        return fn
    cs = list(fn.__code__.co_consts)
    assert cs.count(base)==1
    cs[cs.index(base)] = (base+delta) % 2**32
    return FunctionType(fn.__code__.replace(co_consts=tuple(cs)),fn.__globals__,fn.__name__,fn.__defaults__,fn.__closure__)


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def replay(args):
    f,seeds = args
    assert digest({k:v for k,v in f.items() if k!="instance_hash"})==f["instance_hash"]
    case,jobs,zones,hidden = [f[k] for k in ("case","jobs","zones","hidden")]
    rows = []
    for batch,seed in seeds:
        rolling.choose = shifted(ROLL,881,seed-DEFAULT)
        Route.beam = shifted(BEAM,1799,seed-DEFAULT)
        policy = dispatcher(case["case_id"])
        e = Engine(jobs,hidden,zones,case["horizon_s"],case["cost_budget"],case["observation_limit"],1,.5)
        actions = []
        while not e.terminal:
            action = policy.choose(e.observe())
            actions.append(action)
            event = e.step(action)
            if action["type"]=="execute":
                assert abs(event["realized_duration"]-hidden[action["job_id"]]["duration"]-.5)<1e-9
            if event.get("completion") is not None:
                policy.record_completion(e.jobs[action["job_id"]]["family_id"],event["service_start"],event["completion"])
        scores = e.metrics()
        check = independent_rebuild(jobs,e.events,zones,e.H,e.B,e.K)
        diff = max(abs(scores[k]-v) for k,v in check.items() if k!="violations")
        assert not check["violations"] and diff<1e-9 and scores["valid_protocol"] and scores["invalid_actions"]==0
        rows.append(dict(case_id=case["case_id"],seed=f["seed"],alpha=1.,beta=.5,batch=batch,internal_seed=seed,
                         instance_hash=f["instance_hash"],path_hash=digest(actions),steps=len(actions),independent_diff=diff,**scores))
    rolling.choose,Route.beam = ROLL,BEAM
    return rows


def main(smoke):
    registry = RESULTS/"final_v1_stability/fixed_instances.json"
    audit = json.loads((registry.parent/"seed_audit.json").read_bytes())
    assert hashlib.sha256(registry.read_bytes()).hexdigest()==audit["registry_sha256"]
    instances = [f for f in json.loads(registry.read_bytes()) if (f["alpha"],f["beta"])==(1,.5)]
    second = json.loads((RESULTS/"final_v1_evidence_extension/scheduling_audit.json").read_bytes())["new_internal_seeds"]
    first = audit["internal_seeds"]
    assert len(set(first))==len(set(second))==20 and set(first).isdisjoint(second) and len(instances)==90
    streams = [("original",s) for s in first]+[("replication",s) for s in second]
    groups = [(f,[("original",DEFAULT)] if smoke else streams) for f in instances]
    if smoke:
        groups = [(f,s) for f,s in groups if f["seed"]==min(x["seed"] for x in instances)]
        rows = [row for group in groups for row in replay(group)]
    else:
        # Compile native dependencies once before Windows workers load the DLLs.
        for case in ("S1","S2","S3"):
            fixed = next(f for f in instances if f["case"]["case_id"]==case)
            replay((fixed,[("original",DEFAULT)]))
        with ProcessPoolExecutor(max_workers=8) as pool:
            rows = []
            for i,part in enumerate(pool.map(replay,groups),1):
                rows.extend(part)
                if i%15==0:
                    print(f"{i}/90 instances",flush=True)
    data = pd.DataFrame(rows)
    ref = pd.read_csv(SPACE/"正式方案/reference/q3_official_B_and_previous.csv.gz").query("method=='previous_local' and alpha==1 and beta==.5")
    idx = ["case_id","seed"]
    default = data.query("internal_seed==@DEFAULT").set_index(idx).sort_index()
    expected = ref.set_index(idx).loc[default.index]
    err = np.abs(default[MET].to_numpy(float)-expected[MET].to_numpy(float)).max()
    assert err<1e-9 and default.index.is_unique
    print("Matched defaults",len(default),"max metric error",err,flush=True)
    if not smoke:
        assert len(default)==90 and len(data)==3600 and not data.duplicated(idx+["internal_seed"]).any()
        data.to_csv(OUT/"previous_nominal_trials.csv.gz",index=False)
        print(data.groupby(["batch","case_id"]).weighted_on_time_rate.mean(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    main(p.parse_args().smoke)
