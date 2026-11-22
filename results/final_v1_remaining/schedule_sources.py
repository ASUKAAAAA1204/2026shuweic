"""Cross the available validation duration sources with all policy settings fixed."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import json
import random
import sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/"final_v1_stability"))
import seed_replay as released


def main():
    released.check_tag()
    cfg=released.TAG/"赛题分析报告"
    units=pd.read_csv(cfg/"source_segment_index.csv").query("split=='validation' and eligible==1")
    # Shared uniform ranks compare short/long realizations at the same empirical percentile.
    pools={(f,e): list(g.sort_values(["duration_s","unit_id"]).itertuples())
           for (f,e),g in units.groupby(["family_id","environment_id"])}
    env={f:sorted(e for ff,e in pools if ff==f) for f in ["A","P","L"]}
    assert [len(env[f]) for f in ("A","P","L")]==[2,2,1]
    schedule=[];manifest=[];labels=[]
    for fixed in released.instances():
        if (fixed["alpha"],fixed["beta"])!=(1.,.5): continue
        rng=random.Random(fixed["seed"])
        ranks={j["job_id"]:rng.random() for j in fixed["jobs"]}
        for ia,ip in product(range(2),repeat=2):
            pick={"A":env["A"][ia],"P":env["P"][ip],"L":env["L"][0]}
            hidden={}
            for j in fixed["jobs"]:
                jf,jid=j["family_id"],j["job_id"]
                pool=pools[jf,pick[jf]]
                unit=pool[int(ranks[jid]*len(pool))]
                hidden[jid]=dict(duration=unit.duration_s,unit_id=unit.unit_id,features={"0.2":[],"0.6":[],"1.0":[]})
                manifest.append(dict(case_id=fixed["case"]["case_id"],seed=fixed["seed"],job_id=jid,
                                     a=ia,p=ip,family=jf,environment=pick[jf],u=ranks[jid],unit_id=unit.unit_id,duration_s=unit.duration_s))
            scenario={k:v for k,v in fixed.items() if k not in ("hidden","instance_hash")}
            scenario["hidden"]=hidden
            schedule.append((dict(scenario,instance_hash=released.digest(scenario)),[released.DEFAULT]))
            labels.append(dict(a=ia,p=ip,source_A=pick["A"],source_P=pick["P"],source_L=pick["L"]))
    assert len(schedule)==360
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=[dict(**group[0],**label) for group,label in zip(pool.map(released.run_group,schedule),labels,strict=True)]
    trials=pd.DataFrame(rows)
    time="weighted_on_time_rate"
    assert trials.valid_protocol.all() and trials.invalid_actions.eq(0).all()
    m=pd.DataFrame(manifest)
    assert m.groupby(["case_id","seed","job_id"]).u.nunique().eq(1).all()
    assert m.groupby(["case_id","seed","job_id","environment"]).unit_id.nunique().eq(1).all()
    m.to_csv(OUT/"source_draws.csv.gz",index=False)
    trials.to_csv(OUT/"source_trials.csv",index=False)
    summary=trials.groupby(["a","p","source_A","source_P","source_L","case_id"])[time].mean().unstack()
    summary["overall"]=summary.mean(axis=1)
    summary.to_csv(OUT/"source_summary.csv")
    t=trials.pivot(index="seed",columns=["a","p","case_id"],values=time).sort_index(axis=1)
    assert t.shape==(30,12) and t.notna().all().all()
    rates={(a,p):t[a,p].mean(axis=1) for a,p in product(range(2),repeat=2)}
    delta={"A1-A0_P0":rates[1,0]-rates[0,0],"A1-A0_P1":rates[1,1]-rates[0,1],
           "P1-P0_A0":rates[0,1]-rates[0,0],"P1-P0_A1":rates[1,1]-rates[1,0]}
    bootstrap=np.random.default_rng(20261122).integers(0,30,(10000,30))
    effects=[]
    for name,v in delta.items():
        v=v.to_numpy()*100
        low,high=np.quantile(v[bootstrap].mean(axis=1),[.025,.975])
        effects.append(dict(contrast=name,delta_pp=v.mean(),low_pp=low,high_pp=high))
    pd.DataFrame(effects).to_csv(OUT/"source_effects.csv",index=False)
    released.check_tag()
    (OUT/"source_audit.json").write_text(json.dumps({"commit":released.COMMIT,"K":64,"internal_seed":released.DEFAULT,
            "trials":len(trials),"per_cell":90,"source_units":{f"{f}/{e}":len(p) for (f,e),p in pools.items()},
            "verified_durations":True,"invalid_actions":int(trials.invalid_actions.sum()),
            "independent_max_diff":trials.independent_diff.max()},indent=2),encoding="utf8")


if __name__=="__main__": main()
