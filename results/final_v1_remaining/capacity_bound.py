"""Offline feasibility ceiling, computed from fixed jobs. No alternative policy is simulated."""
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PRIOR=OUT.parent/"final_v1_stability"


def subset_sums(values):
    array=np.ones(1,dtype=float)*0
    for value in values: array=np.concatenate([array,array+value])
    return array


class Ceiling:
    def __init__(self,scenario):
        jobs=scenario["jobs"]
        zones=scenario["zones"]
        self.n=len(jobs)
        ids=[j["job_id"] for j in jobs]
        @lru_cache(None)
        def closure(i):
            mask=1<<i
            for p in jobs[i]["predecessors"]: mask |= closure(ids.index(p))
            return mask
        names=sorted({j["zone_id"] for j in jobs})
        zones_of=np.array([0])
        required=np.array([0])
        for i,j in enumerate(jobs):
            zones_of=np.r_[zones_of,zones_of|(1<<names.index(j["zone_id"]))]
            required=np.r_[required,required|closure(i)]
        routes=[0.]
        for mask in range(1,2**len(names)):
            selected=[z for i,z in enumerate(names) if mask&(1<<i)]
            routes.append(min(sum(math.dist(zones[a],zones[b]) for a,b in zip(("Z0",)+order,order))
                              for order in itertools.permutations(selected)))
        self.travel=np.asarray(routes)[zones_of]
        self.full=np.arange(2**self.n)
        self.weight=subset_sums([j["priority_weight"] for j in jobs])
        count=subset_sums([1]*self.n)
        self.budget_ok=(2.02*count[required]+.2*self.travel[required]<=scenario["case"]["cost_budget"]+1e-9)
        ends=np.array([min(j["due_s"],scenario["case"]["horizon_s"]) for j in jobs])
        self.by_deadline=[]
        for t in sorted(set(ends)):
            mask=sum(1<<i for i,d in enumerate(ends) if d<=t)
            self.by_deadline.append((t,required[self.full&mask]))

    def evaluate(self,durations):
        # Every selected job and its ancestors must finish by its deadline.
        # Omitting release times and the actual visiting order makes these necessary conditions.
        time=subset_sums([.5+v for v in durations])+self.travel
        feasible=self.budget_ok.copy()
        for t,required in self.by_deadline: feasible &= time[required]<=t+1e-9
        return self.weight[feasible].max()/self.weight.max()


def main():
    observed=pd.read_csv(PRIOR/"default_replay.csv.gz").set_index(["case_id","seed","alpha","beta"])
    cache={}; rows=[]
    for s in json.loads((PRIOR/"fixed_instances.json").read_bytes()):
        a,b=s["alpha"],s["beta"]
        if b!=.5 or a not in (1,1.5):continue
        c=s["case"]["case_id"]
        if c not in cache:cache[c]=Ceiling(s)
        v=observed.loc[(c,s["seed"],a,b)]
        assert s["instance_hash"]==v.instance_hash
        ceiling=cache[c].evaluate([a*s["hidden"][j["job_id"]]["duration"]+b for j in s["jobs"]])
        assert v.weighted_on_time_rate<=ceiling+1e-12
        rows.append(dict(case_id=c,seed=s["seed"],alpha=a,beta=b,observed=v.weighted_on_time_rate,ceiling=ceiling,
                         unavoidable_loss_floor=1-ceiling,unexplained_gap_ceiling=ceiling-v.weighted_on_time_rate))
    out=pd.DataFrame(rows)
    assert len(out)==180
    out.to_csv(OUT/"capacity_bounds.csv",index=False)
    out.groupby(["alpha","case_id"])[["observed","ceiling","unavoidable_loss_floor","unexplained_gap_ceiling"]].mean().to_csv(OUT/"capacity_summary.csv")


if __name__=="__main__":main()
