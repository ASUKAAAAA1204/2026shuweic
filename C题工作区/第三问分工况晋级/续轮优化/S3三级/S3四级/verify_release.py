"""Repeat the passing pair through both published entries and compare every action."""
import gzip
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from deploy import dispatcher,selected
from study import HERE,OUT,MET,bench,measure,previous,digest,save
from engine import independent_metrics


def main():
    selected()
    rounds=sorted(OUT.glob("round_*/verdict.json"),key=lambda p:int(p.parent.name.split("_")[-1]))
    history=[]
    for r,path in enumerate(rounds,1):
        f=path.parent
        v=json.loads(path.read_bytes())
        before=json.loads((f/"before.json").read_bytes())
        assert before["candidate_hash"]==digest(f/"nominee.json")
        assert before["files"]["fourth.py"]==digest(f/"policy_snapshot.txt")
        subprocess.run([sys.executable,"-B",str(HERE/"audit/validate.py"),str(f/"paired.csv"),
                        "--round",str(r),"--output",str(f/"independent.json")],check=True)
        rep=json.loads((f/"independent.json").read_bytes())
        assert rep["passed"]==v["passed"]==(r==len(rounds))
        np.testing.assert_allclose(rep["on_time"]["candidate_ci95"],v["metrics"][MET[0]]["absolute_ci"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(rep["on_time"]["gain_ci95"],v["metrics"][MET[0]]["gain_ci"],atol=1e-12,rtol=0)
        assert abs(rep["completion"]["gain"]-v["metrics"][MET[1]]["gain"])<1e-12
        history.append(dict(round=r,design=v["design"],passed=rep["passed"]))
    seeds=list(range(rep["start"],rep["start"]+2048))
    b=bench()
    scores=pd.read_csv(f/"paired.csv").set_index(["method","seed"]).sort_index()
    old=measure(b,None,seeds)
    class Audit:
        def __init__(self,stream):self.stream=stream
        def write(self,text):
            entry=json.loads(text)
            assert entry==json.loads(self.stream.readline())
            seed=entry["seed"]
            for e in entry["events"]:
                if e["action"]["type"]=="execute":
                    sample=b.hidden[seed][e["action"]["job_id"]]
                    assert e["source_reference"]==sample["unit_id"] and abs(e["realized_duration"]-sample["duration"]-.5)<1e-9
            independent=independent_metrics(b.jobs,entry["events"],b.c["horizon_s"])
            assert all(abs(independent[m]-scores.loc[("candidate",seed),m])<1e-9 for m in MET)
    with gzip.open(f/"events.jsonl.gz","rt",encoding="utf8") as log:
        new=b.run(None,seeds,Audit(log),lambda:dispatcher("S3"))
        assert not log.readline()
    for method,df in [("candidate",new),("tier3",old)]:
        np.testing.assert_allclose(df[MET],scores.loc[method].loc[seeds,MET],atol=1e-9,rtol=0)
    for case in ("S1","S2"):
        assert type(dispatcher(case)) is type(previous(case))
    save(OUT/"release_audit.json",dict(passed=True,rounds=history,unique_new_seeds=len(rounds)*2048,
        final_paired_seeds=2048,final_trajectories_reproduced=len(new),invalid_actions=int(new.invalid_actions.sum()),
        paid_reviews=int(new.observations.sum()),baseline_reproduced=True,frozen_S1_S2=True))
    print((OUT/"release_audit.json").read_text(),flush=True)


if __name__=="__main__":main()
