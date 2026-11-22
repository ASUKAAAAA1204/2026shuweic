"""Factorial duration stress test and replication of internal-stream variability."""
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent/"final_v1_stability"
sys.path.insert(0, str(PRIOR))
import seed_replay as frozen

EXTERNAL = range(918400000, 918400300)
NEW_INTERNAL = [int(v) for v in np.random.SeedSequence(941203).generate_state(20)]
GRID = list(itertools.product((.75, 1., 1.5), (0., .5, 1.)))


def build_inputs():
    original = list(frozen.instances())
    held = [f for f in original if f["alpha"] in (1., 1.5) and f["beta"] == .5]
    templates = {f["case"]["case_id"]: f for f in held}
    cfg = frozen.TAG/"赛题分析报告"
    units = pd.read_csv(cfg/"source_segment_index.csv").query("split=='validation' and eligible==1")
    pools = {family: list(g.itertuples()) for family,g in units.groupby("family_id")}
    grid = []
    draws = []
    for seed in EXTERNAL:
        for case,f in templates.items():
            rng = random.Random(seed)
            hidden = {}
            for job in f["jobs"]:
                unit = rng.choice(pools[job["family_id"]])
                hidden[job["job_id"]] = dict(duration=unit.duration_s, unit_id=unit.unit_id,
                                             features={"0.2":[],"0.6":[],"1.0":[]})
                draws.append(dict(case_id=case,seed=seed,job_id=job["job_id"],
                                  unit_id=unit.unit_id,source_duration_s=unit.duration_s))
            for a,b in GRID:
                fixed = dict(case=f["case"],jobs=f["jobs"],zones=f["zones"],hidden=hidden,seed=seed,alpha=a,beta=b)
                grid.append(dict(**fixed,instance_hash=frozen.digest(fixed)))
    pd.DataFrame(draws).to_csv(OUT/"external_draws.csv.gz",index=False)
    assert len(grid)==8100 and len(held)==180
    assert len(set(NEW_INTERNAL))==20 and not set(NEW_INTERNAL)&set(frozen.SEEDS)
    assert not set(EXTERNAL)&{f["seed"] for f in held}
    return grid,held


def run_batch(pool, jobs, seeds, name):
    rows=[]
    for i,group in enumerate(pool.map(frozen.run_group,((j,seeds) for j in jobs),chunksize=3),1):
        rows.extend(group)
        if i%1000==0: print(f"{name}: {i}/{len(jobs)}",flush=True)
    d=pd.DataFrame(rows)
    assert len(d)==len(jobs)*len(seeds) and not d.duplicated(frozen.KEY+["internal_seed"]).any()
    assert d.valid_protocol.all() and not d.invalid_actions.any() and d.independent_diff.max()<1e-9
    d.to_csv(OUT/(name+".csv.gz"),index=False)
    return d


if __name__=="__main__":
    frozen.check_tag()
    grid,held=build_inputs()
    # Check old inputs match the stored per-instance fingerprints before running new streams.
    old=pd.read_csv(PRIOR/"default_replay.csv.gz").set_index(frozen.KEY)
    assert all(old.loc[(f["case"]["case_id"],f["seed"],f["alpha"],f["beta"]),"instance_hash"]==f["instance_hash"] for f in held)
    with ProcessPoolExecutor(max_workers=12) as pool:
        factorial=run_batch(pool,grid,[frozen.DEFAULT],"duration_grid")
        seedrun=run_batch(pool,held,NEW_INTERNAL,"internal_replication")
    frozen.check_tag()
    audit=dict(commit=frozen.COMMIT,K=64,validation_pool_only=True,crossed_runs=len(factorial),
               fixed_instance_replications=len(seedrun),new_internal_seeds=NEW_INTERNAL,external_seed_start=EXTERNAL.start,
               archive_hashes_matched=len(held),external_seed_end=EXTERNAL.stop-1,invalid_actions=0,
               max_independent_score_error=max(factorial.independent_diff.max(),seedrun.independent_diff.max()))
    (OUT/"scheduling_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    print("Both batches complete",flush=True)
