"""Frozen B_K64 replay. Only the per-episode scenario RNG stream varies."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
TAG = OUT.parents[1]/"tmp/final_v1_20261122"
SPACE = TAG/"C题工作区"
FINAL = SPACE/"正式方案"
sys.path.insert(0, str(FINAL/"code"))
from b_replay import Engine, KEY, METRICS, dispatcher, independent_rebuild, jobs_for

DEFAULT = 20261122
SEEDS = list(range(DEFAULT, DEFAULT+20))
COMMIT = "9f8bc05fae9d8ce32da7fa7c5e28a82960085819"


def check_tag():
    for ref in ("HEAD", "v1.0.0-final^{}"):
        assert subprocess.check_output(["git","-C",str(TAG),"rev-parse",ref],text=True).strip() == COMMIT
    assert not subprocess.check_output(["git","-C",str(TAG),"status","--porcelain"],text=True).strip()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def instances():
    cfg = TAG/"赛题分析报告"
    cond = json.loads((cfg/"common_conditions.json").read_bytes())
    tasks = pd.read_csv(cfg/"tasks.csv").fillna("")
    zones = pd.read_csv(cfg/"service_zones.csv").set_index("zone_id")[["x","y"]].T.to_dict("list")
    units = pd.read_csv(cfg/"source_segment_index.csv").query("split=='validation' and eligible==1")
    pools = {f: list(g.itertuples()) for f,g in units.groupby("family_id")}
    svc = cond["service"]
    a,b = svc["alpha"],svc["beta_s"]
    mappings = [(a,b)]+[(v,b) for v in svc["alpha_sensitivity"]]+[(a,v) for v in svc["beta_sensitivity"]]
    for case in cond["cases"]:
        jobs = jobs_for(tasks, case["case_id"])
        for seed in cond["validation_seeds"]:
            rng = random.Random(seed)
            hidden = {}
            for job in jobs:
                unit = rng.choice(pools[job["family_id"]])
                hidden[job["job_id"]] = dict(duration=unit.duration_s, unit_id=unit.unit_id,
                                             features={"0.2":[], "0.6":[], "1.0":[]})
            for alpha,beta in mappings:
                fixed = dict(case=case,jobs=jobs,zones=zones,hidden=hidden,seed=seed,alpha=alpha,beta=beta)
                yield dict(**fixed, instance_hash=digest(fixed))


def run_group(args):
    fixed,seeds = args
    case,jobs,hidden,zones = (fixed[k] for k in ("case","jobs","hidden","zones"))
    a,b = fixed["alpha"],fixed["beta"]
    rows = []
    for seed in seeds:
        policy = dispatcher(case["case_id"])
        p = policy.policy
        assert (p.K,p.strength,p.feedback_strength,p.base,p.stratified,p.seed)==(64,20,3,"survival",True,DEFAULT)
        if seed != DEFAULT:
            # Fresh dispatcher, unchanged weights and decision code. Replace just its RNG before step 1.
            p.rng = np.random.default_rng(seed)
        engine = Engine(jobs,hidden,zones,case["horizon_s"],case["cost_budget"],case["observation_limit"],a,b)
        actions = []
        while not engine.terminal:
            action = policy.choose(engine.observe())
            actions.append(action)
            event = engine.step(action)
            if action["type"] == "execute":
                actual = a*hidden[action["job_id"]]["duration"]+b
                assert abs(actual-event["realized_duration"]) < 1e-9
            if event.get("completion") is not None:
                policy.record_completion(engine.jobs[action["job_id"]]["family_id"],event["service_start"],event["completion"])
        scores = engine.metrics()
        check = independent_rebuild(jobs,engine.events,zones,engine.H,engine.B,engine.K)
        diff = max(abs(check[k]-scores[k]) for k in check if k!="violations")
        assert not check["violations"] and diff<1e-9 and scores["valid_protocol"] and not scores["invalid_actions"]
        first = actions[0]
        rows.append(dict(case_id=case["case_id"],seed=fixed["seed"],alpha=a,beta=b,internal_seed=seed,
                         instance_hash=fixed["instance_hash"],first_action=json.dumps(first,sort_keys=True),
                         path_hash=digest(actions),steps=len(actions),independent_diff=diff,**scores))
    assert digest({k:v for k,v in fixed.items() if k!="instance_hash"})==fixed["instance_hash"]
    return rows


def save(frame,name):
    path=OUT/name
    temp=path.with_name(path.stem+".partial"+path.suffix)
    frame.to_csv(temp,index=False)
    temp.replace(path)


def main(workers):
    check_tag()
    jobs=list(instances())
    assert len(jobs)==len({j["instance_hash"] for j in jobs})==450
    (OUT/"fixed_instances.json").write_text(json.dumps(jobs,ensure_ascii=False,separators=(",",":")),encoding="utf8")
    ref=pd.read_csv(FINAL/"results/q3/b_k64/trials.csv.gz").set_index(KEY).sort_index()
    assert set(ref.method)=={"B_K64"}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        nominal=pd.DataFrame([r for group in pool.map(run_group,[(j,[DEFAULT]) for j in jobs]) for r in group])
        obs=nominal.set_index(KEY).sort_index()
        cols=METRICS+["steps"]
        pd.testing.assert_frame_equal(obs[cols],ref[cols],check_dtype=False,check_exact=False,atol=1e-9,rtol=0)
        archive_diff=np.abs(obs[cols].to_numpy(float)-ref[cols].to_numpy(float)).max()
        save(nominal,"default_replay.csv.gz")
        print(f"450 default-seed runs matched all archived outcome columns; max difference {archive_diff:.3g}",flush=True)
        rows=[]
        for i,group in enumerate(pool.map(run_group,[(j,SEEDS[1:]) for j in jobs]),1):
            rows.extend(group)
            if i%50==0: print(f"Changed streams: {i}/450 instances, {len(rows)} runs",flush=True)
    frame=pd.concat([nominal,pd.DataFrame(rows)],ignore_index=True)
    assert len(frame)==9000 and not frame.duplicated(KEY+["internal_seed"]).any()
    assert frame.groupby(KEY).instance_hash.nunique().eq(1).all()
    assert frame.groupby(KEY).internal_seed.nunique().eq(20).all()
    check_tag()
    save(frame.sort_values(KEY+["internal_seed"]),"seed_trials.csv.gz")
    audit=dict(tag="v1.0.0-final",commit=COMMIT,clean_before_and_after=True,K=64,internal_seeds=SEEDS,
               instances=len(jobs),replays=len(frame),default_archive_max_diff=archive_diff,
               invalid_actions=int(frame.invalid_actions.sum()),independent_max_diff=frame.independent_diff.max(),
               registry_sha256=hashlib.sha256((OUT/"fixed_instances.json").read_bytes()).hexdigest())
    (OUT/"seed_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    print("Complete",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=10)
    main(parser.parse_args().workers)
