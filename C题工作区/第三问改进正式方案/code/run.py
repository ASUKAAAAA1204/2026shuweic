"""One frozen candidate versus baseline, all official validation seeds/mappings."""
from pathlib import Path
import json,hashlib,random,time,gzip,copy
import numpy as np
import pandas as pd
from support import load,jobs_for,Engine,independent_metrics,action
from deadline_policy import choose,pressure
ROOT=Path(__file__).resolve().parents[1]
MAPPINGS=[(1.,.5),(.75,.5),(1.5,.5),(1.,0.),(1.,1.)]

def main():
    cond,tasks,zones,train=load();out=ROOT/'results/validation_frozen';out.mkdir(parents=True,exist_ok=True)
    freeze=ROOT/'selection_before_validation.json'
    frozen=json.loads(freeze.read_text(encoding='utf8'))
    assert hashlib.sha256((ROOT/'model/training_family_samples.json').read_bytes()).hexdigest()==frozen['model_hash']
    for f,h in frozen['hashes'].items():assert hashlib.sha256((ROOT/'code'/f).read_bytes()).hexdigest()==h
    seg=pd.read_csv(ROOT.parent/'处理结果/v1/segments_private.csv');pool=seg[(seg.split=='validation')&(seg.eligible==1)]
    pools={f:g.duration_s.to_numpy() for f,g in train.groupby('family_id')};med=train.groupby('family_id').duration_s.median().to_dict()
    rows=[];times=[];invariance=0;maxdiff=0
    with gzip.open(out/'actions.jsonl.gz','wt',encoding='utf8') as log:
        for case in cond['cases']:
            jobs=jobs_for(tasks,case['case_id'])
            for seed in cond['validation_seeds']:
                rng=random.Random(seed);hidden={}
                for j in jobs:
                    candidates=pool[pool.family_id==j['family_id']];r=candidates.iloc[rng.randrange(len(candidates))]
                    hidden[j['job_id']]=dict(duration=float(r.duration_s),unit_id=r.unit_id,features={str(v):[] for v in [.2,.6,1.]})
                for alpha,beta in MAPPINGS:
                    for method in ['density','pressure_gated']:
                        tag=dict(case_id=case['case_id'],seed=seed,alpha=alpha,beta=beta,method=method)
                        e=Engine(jobs,copy.deepcopy(hidden),zones,case['horizon_s'],case['cost_budget'],case['observation_limit'],alpha,beta);history={};began=time.perf_counter();used=0
                        while not e.terminal:
                            o=e.observe();start=time.perf_counter();gated=method=='pressure_gated' and pressure(o,pools,history)
                            a=choose(o,pools,history) if gated else action(o,med,'density','minimum');used+=gated
                            times.append(dict(**tag,seconds=time.perf_counter()-start))
                            # Unknown future durations and unrevealed prefix changes must not affect action.
                            if method=='pressure_gated':
                                other=copy.deepcopy(e)
                                for key,v in other.hidden.items():
                                    if not other.status[key]['attempted']:v['duration']=99999.;v['features']={str(p):[] for p in [.2,.6,1.]}
                                oo=other.observe();aa=choose(oo,pools,history) if pressure(oo,pools,history) else action(oo,med,'density','minimum')
                                assert aa==a;invariance+=1
                            event=e.step(a);log.write(json.dumps(dict(**tag,history=history,pressure=gated,event=event),ensure_ascii=False)+'\n')
                            if a['type']=='execute' and event.get('completion') is not None:
                                family=next(j['family_id'] for j in jobs if j['job_id']==a['job_id']);history.setdefault(family,[]).append((event['completion']-event['service_start']-beta)/alpha)
                        m=e.metrics();ind=independent_metrics(jobs,e.events,e.H)
                        maxdiff=max(maxdiff,max(abs(float(m[key])-float(ind[key])) for key in m));assert not m['invalid_actions'] and maxdiff<1e-9
                        rows.append(dict(**tag,seconds=time.perf_counter()-began,pressure_decisions=used,**m))
            pd.DataFrame(rows).to_csv(out/'trials.csv',index=False);print(case['case_id'],'complete',len(rows),flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'trials.csv',index=False);pd.DataFrame(times).to_csv(out/'decision_times.csv',index=False)
    summary=d.groupby(['case_id','alpha','beta','method'],as_index=False).mean(numeric_only=True);summary.to_csv(out/'summary.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(trials=len(rows),test_performance_reads=0,invalid_actions=0,independent_metrics_max_error=maxdiff,future_duration_invariance_checks=invariance,code_frozen=True),indent=2),encoding='utf8')
    print(summary[(summary.alpha==1)&(summary.beta==.5)][['case_id','method','weighted_on_time_rate','cost','seconds']].to_string(index=False))
if __name__=='__main__':main()
