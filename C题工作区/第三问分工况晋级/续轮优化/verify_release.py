"""Check every nomination and replay the published versions on the final holdout."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from experiment import Bench, HERE, MET, OFFICIAL, OUT, save
from release import dispatcher, verified

CONFIRM = list(range(440000,440480))


def audit_attempts():
    summaries=[]
    for case, expected in (("S2",1),("S3",2)):
        promotions=0
        for num in range(1,6):
            folder=OUT/f"{case}_attempt_{num}"
            v=json.loads((folder/"verdict.json").read_bytes())
            d=pd.read_csv(folder/"trials.csv")
            level=promotions+1
            assert v["level"]==level
            ref={"kind":"base"} if level==1 else json.loads((OUT/f"{case}_promotion_1.json").read_bytes())["candidate"]
            assert ref == v["previous"]
            assert v["candidate"] == json.loads((folder/"nominee.json").read_bytes())
            assert len(d)==420 and not d.duplicated(["method","split","seed"]).any()
            assert d.valid_protocol.all() and d.invalid_actions.max()==0 and d.cost.max()<=({"S2":34,"S3":40}[case])+1e-9
            assert d.observations.max()<= {"S2":6,"S3":8}[case]
            for method in ("previous","nominee"):
                assert set(d.query("split=='official' and method==@method").seed)==set(OFFICIAL)
                start=430000+1000*level
                assert set(d.query("split=='gate' and method==@method").seed)==set(range(start,start+180))
            s=d.groupby(["method","split"])[MET].mean()
            base=s.loc["previous"]; new=s.loc["nominee"]
            gain=new-base
            floor=.8708 if case=="S2" else .83 if level==1 else .835
            passed=bool(new.loc["official",MET[0]]>=floor-1e-12 and gain.loc["official",MET[0]]>=-1e-12
                    and gain.loc["gate",MET[0]] >= (.006 if level==1 else .005)-1e-12
                    and new.loc["official",MET[1]]>=(83/90 if case=="S2" else .9)-1e-12
                    and gain.loc["gate",MET[1]]>=-.005-1e-12)
            assert passed==v["passed"]
            if passed:
                assert json.loads((OUT/f"{case}_promotion_{level}.json").read_bytes())==v
                promotions+=1
            summaries.append(dict(case=case,attempt=num,level=level,passed=passed,
                official_old=base.loc["official",MET[0]],official_new=new.loc["official",MET[0]],
                official_completion=new.loc["official",MET[1]],gate_old=base.loc["gate",MET[0]],
                gate_new=new.loc["gate",MET[0]],gate_gain=gain.loc["gate",MET[0]],gate_completion_gain=gain.loc["gate",MET[1]]))
        assert promotions==expected
    return pd.DataFrame(summaries)


def main(output):
    output.mkdir(exist_ok=True,parents=True)
    levels=verified()
    audit_attempts().to_csv(output/"attempts.csv",index=False)
    initial=pd.read_csv(HERE.parent/"results/final/trials.csv").query("method=='new'")
    summary=[]; all_trials=[]; rng=np.random.default_rng(220366)
    with gzip.open(output/"actions.jsonl.gz","wt",encoding="utf8") as actions:
        for case,max_level in levels.items():
            bench=Bench(case)
            for split,seeds in [("official",OFFICIAL),("confirmation",CONFIRM)]:
                scores={}
                for level in range(max_level+1):
                    # The same public entry point is used for both confirmation and deployment.
                    frame=bench.run(None,seeds,policy_factory=lambda:dispatcher(case,level))
                    scores[level]=frame
                    if split=="official":
                        if level==0:
                            ref=initial.query("case_id==@case and split=='official'").set_index("seed")
                        else:
                            lock=json.loads((OUT/f"{case}_promotion_{level}.json").read_bytes())
                            ref=pd.read_csv(OUT/f"{case}_attempt_{lock['attempt']}"/"trials.csv").query("split=='official' and method=='nominee'").set_index("seed")
                        np.testing.assert_allclose(frame[MET],ref.loc[frame.index,MET],atol=1e-9,rtol=0)
                    all_trials.append(frame.assign(case=case,split=split,level=level))
                # Reproduce deployed actions against their hidden input and all computed metrics.
                class TaggedLog:
                    def write(self,line):
                        record=json.loads(line)
                        for e in record["events"]:
                            if e["action"]["type"]=="execute":
                                private=bench.hidden[record["seed"]][e["action"]["job_id"]]
                                assert e["source_reference"]==private["unit_id"]
                                assert abs(e["realized_duration"]-private["duration"]-.5)<1e-9
                        actions.write(json.dumps(dict(split=split,level=max_level,**record))+"\n")
                repeated=bench.run(None,seeds,log=TaggedLog(),policy_factory=lambda:dispatcher(case))
                np.testing.assert_allclose(repeated[MET],scores[max_level][MET],atol=1e-9,rtol=0)
                for left,right in ([(0,0)] if max_level==0 else [(0,1)] if max_level==1 else [(0,1),(1,2),(0,2)]):
                    ix=rng.integers(len(seeds),size=(5000,len(seeds)))
                    for k in MET:
                        diff=(scores[right][k]-scores[left][k]).to_numpy()
                        low,high=np.quantile(diff[ix].mean(axis=1),[.025,.975])
                        summary.append(dict(case=case,split=split,previous_level=left,new_level=right,metric=k,seeds=len(seeds),
                            old=scores[left][k].mean(),new=scores[right][k].mean(),gain=diff.mean(),low=low,high=high))
                print(case,split,"verified",flush=True)
    pd.concat(all_trials).to_csv(output/"trials.csv")
    pd.DataFrame(summary).to_csv(output/"comparison.csv",index=False)
    save(output/"audit.json",dict(passed=True,levels=levels,attempts_per_optimized_case=5,holdout_seeds=len(CONFIRM),
        official_seeds=len(OFFICIAL),action_logs=3*(len(CONFIRM)+len(OFFICIAL)),all_attempts_recalculated=True,
        official_versions_match_archives=True,source_draw_checked=True,illegal_actions=0))


if __name__=="__main__":
    import policy
    import rolling
    from numerical_backend import configure
    configure(rolling, policy)
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=OUT/"final")
    main(p.parse_args().output)
