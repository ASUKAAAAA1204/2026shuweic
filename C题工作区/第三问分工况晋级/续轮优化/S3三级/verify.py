"""Rebuild both paired estimates and reproduce every published decision on the passing block."""
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from published import ROOT,dispatcher,locked_candidate
from run import OUT, MET,bench,measure
from experiment import save
from engine import independent_metrics


def main():
    locked_candidate()
    history=[]
    seen=set()
    finals=sorted(OUT.glob("round_*/verdict.json"),key=lambda p:int(p.parent.name.split("_")[-1]))
    for ix,path in enumerate(finals,1):
        v=json.loads(path.read_bytes())
        start=6810000+1000*(ix-1);seeds=list(range(start,start+640))
        assert not seen.intersection(seeds)
        seen.update(seeds)
        d=pd.read_csv(path.parent/"paired_trials.csv")
        assert len(d)==1280 and set(d.method)=={"tier2","candidate"} and not d.duplicated(["seed","method"]).any()
        assert d.invalid_actions.max()==0 and d.valid_protocol.all() and d.cost.max()<=40+1e-9 and d.observations.max()<=8
        tables={k:d[d.method==k].set_index("seed").sort_index() for k in ("tier2","candidate")}
        assert all(list(t.index)==seeds for t in tables.values())
        assert hashlib.sha256((path.parent/"nominee.json").read_bytes()).hexdigest()==v["nominee_sha256"]
        diff=(tables["candidate"][MET[0]]-tables["tier2"][MET[0]]).to_numpy()
        bootstrap=np.random.default_rng(66517).integers(640,size=(10000,640))
        low,high=np.percentile(diff[bootstrap].mean(axis=1),[2.5,97.5])
        rate=tables["candidate"][MET[0]].mean()
        completed=(tables["candidate"][MET[1]]-tables["tier2"][MET[1]]).mean()
        decision=bool(rate>=.85 and low>0 and completed>=0)
        assert decision==v["passed"]==(ix==len(finals))
        np.testing.assert_allclose([rate,diff.mean(),low,high,completed],[v["metrics"][MET[0]][k] for k in ("new","gain","low","high")]+[v["metrics"][MET[1]]["gain"]],atol=1e-12,rtol=0)
        for k,t in tables.items():
            suffix="old" if k=="tier2" else "new"
            for m in MET:assert abs(t[m].mean()-v["metrics"][m][suffix])<1e-9
        history.append(dict(round=ix,start=start,count=640,passed=decision))
    b=bench()
    previous=measure(b,None,seeds)
    class ActionAudit:
        def __init__(self,file):self.file=file
        def write(self,line):
            stored=json.loads(self.file.readline())
            actual=json.loads(line)
            assert stored==actual
            for e in actual["events"]:
                if e["action"]["type"]=="execute":
                    sample=b.hidden[actual["seed"]][e["action"]["job_id"]]
                    assert e["source_reference"]==sample["unit_id"]
                    assert abs(e["realized_duration"]-sample["duration"]-.5)<1e-9
            score=independent_metrics(b.jobs,actual["events"],b.c["horizon_s"])
            assert all(abs(score[m]-tables["candidate"].loc[actual["seed"],m])<1e-9 for m in MET)
    with gzip.open(path.parent/"actions.jsonl.gz","rt",encoding="utf8") as f:
        new=b.run(None,seeds,ActionAudit(f),lambda:dispatcher("S3"))
        assert not f.readline()
    for current,method in [(new,"candidate"),(previous,"tier2")]:
        np.testing.assert_allclose(current[MET],tables[method][MET],rtol=0,atol=1e-9)
    for case in ("S1","S2"):
        # Published wrapper goes straight to the previous implementation.
        from release import dispatcher as old_dispatcher
        assert type(dispatcher(case)) is type(old_dispatcher(case))
    absolute=np.quantile(tables["candidate"][MET[0]].to_numpy()[bootstrap].mean(axis=1),[.025,.975])
    save(OUT/"audit.json",dict(verified=True,rounds=history,unique_gate_seeds=len(seen),final_actions_replayed=len(seeds),
        absolute_on_time_ci95=absolute.tolist(),paid_reviews=int(new.observations.sum()),invalid_actions=int(new.invalid_actions.sum())))
    print((OUT/"audit.json").read_text(),flush=True)


if __name__=="__main__":main()
