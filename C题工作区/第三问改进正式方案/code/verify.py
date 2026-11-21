"""Independent source-to-action reconstruction. Does not import the simulator."""
from pathlib import Path
import json,gzip,random,math
import pandas as pd
from deadline_policy import choose,pressure
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

def main():
    cfg=WORK/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints'
    tasks=pd.read_csv(cfg/'tasks.csv').fillna('');conditions=json.loads((cfg/'common_conditions.json').read_text());cases={x['case_id']:x for x in conditions['cases']}
    source=pd.read_csv(cfg/'source_segment_index.csv');source=source[(source.split=='validation')&(source.eligible==1)]
    train=pd.DataFrame(json.loads((ROOT/'model/training_family_samples.json').read_text()))
    pools={f:g.duration_s.to_numpy() for f,g in train.groupby('family_id')};med=train.groupby('family_id').duration_s.median().to_dict()
    zones={r.zone_id:[r.x,r.y] for r in pd.read_csv(cfg/'service_zones.csv').itertuples()}
    path=ROOT/'results/validation_frozen';trials=pd.read_csv(path/'trials.csv');groups={}
    with gzip.open(path/'actions.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);key=tuple(r[k] for k in ['case_id','seed','alpha','beta','method']);groups.setdefault(key,[]).append(r)
    maximum=0.;decisions=0;feedback_samples=0
    for (case,seed,alpha,beta,method),events in groups.items():
        raw=tasks[tasks.case_id==case].sort_values('job_id');jobs={r.job_id:r for r in raw.itertuples()};rng=random.Random(seed);assigned={}
        for j in jobs.values():
            subset=source[source.family_id==j.family_id];assigned[j.job_id]=subset.iloc[rng.randrange(len(subset))]
        H=cases[case]['horizon_s'];B=cases[case]['cost_budget'];t=0.;cost=0.;zone='Z0';history={};attempted=set();done={};over=idle=censored=0
        for row in events:
            e=row['event'];a=e['action'];assert len(events)<=200
            assert abs(e['before']['time']-t)<1e-9 and abs(e['before']['cost']-cost)<1e-9 and e['before']['zone']==zone
            assert row['history'].keys()==history.keys()
            for f,v in history.items():assert len(v)==len(row['history'][f]) and all(abs(x-y)<1e-9 for x,y in zip(v,row['history'][f]))
            public=[]
            for j in jobs.values():
                public.append(dict(job_id=j.job_id,family=j.family_id,release_s=j.release_s,due_s=j.due_s,weight=j.priority_weight,zone_xy=zones[j.zone_id],predecessors=j.predecessors.split('|') if j.predecessors else [],attempted=j.job_id in attempted,completed=j.job_id in done))
            o=dict(jobs=public,time=t,H=H,B=B,cost=cost,zone_xy=zones[zone],alpha=alpha,beta=beta)
            if method=='pressure_gated':
                gate=pressure(o,pools,history);assert gate==row['pressure']
                if gate:assert choose(o,pools,history)==a
                else:
                    # Independently implement frozen density, rather than calling it.
                    options=[]
                    for j in public:
                        if j['attempted'] or j['release_s']>t+1e-12 or not set(j['predecessors'])<=set(done):continue
                        D=math.dist(zones[zone],j['zone_xy']);estimate=max(.2,alpha*med[j['family']]+beta)
                        if t+D+.7<=H+1e-12 and cost+2.02+.2*D<=B+1e-12:options.append((-j['weight']/(D+.5+estimate),j['due_s'],j['job_id']))
                    if options:expected=dict(type='execute',job_id=min(options)[2],reserve=.2)
                    else:
                        future=[j['release_s'] for j in public if not j['attempted'] and t+1e-12<j['release_s']<=H]
                        expected=dict(type='wait',duration=min(future)-t) if future else dict(type='stop')
                    assert expected==a
            decisions+=1
            if a['type']=='execute':
                j=jobs[a['job_id']];assert j.job_id not in attempted and j.release_s<=t+1e-12
                parents=j.predecessors.split('|') if j.predecessors else [];assert set(parents)<=set(done)
                D=math.dist(zones[zone],zones[j.zone_id]);s=t+D+.5;b=a['reserve'];p=alpha*assigned[j.job_id].duration_s+beta
                assert b>=.2 and s+b<=H+1e-12 and cost+2+.1*b+.2*D<=B+1e-12
                assert e['source_reference']==assigned[j.job_id].unit_id and abs(e['service_start']-s)<1e-9
                assert abs(e['realized_duration']-p)<1e-9
                cost+=2+.1*b+.2*D;zone=j.zone_id;attempted.add(j.job_id);over+=max(min(p,H-s)-b,0)
                if s+p<=H+1e-12:
                    done[j.job_id]=min(s+p,H);idle+=max(b-p,0);history.setdefault(j.family_id,[]).append((s+p-s-beta)/alpha);feedback_samples+=1
                else:censored+=1
                t=min(H,s+max(p,b))
            elif a['type']=='wait':assert a['duration']>0 and t+a['duration']<=H+1e-12;t+=a['duration']
            else:assert a['type']=='stop'
            assert abs(t-e['after']['time'])<1e-9 and abs(cost-e['after']['cost'])<1e-9 and zone==e['after']['zone']
        W=sum(j.priority_weight for j in jobs.values())
        metrics=dict(weighted_completion_rate=sum(j.priority_weight for j in jobs.values() if j.job_id in done)/W,weighted_on_time_rate=sum(j.priority_weight for j in jobs.values() if j.job_id in done and done[j.job_id]<=j.due_s+1e-12)/W,weighted_tardiness_s=sum(j.priority_weight*max(done.get(j.job_id,H)-j.due_s,0) for j in jobs.values()),cost=cost,elapsed_s=t,observed_overrun_s=over,reserved_idle_s=idle,censored_units=censored,observations=0,invalid_actions=0)
        match=trials[(trials.case_id==case)&(trials.seed==seed)&(trials.alpha==alpha)&(trials.beta==beta)&(trials.method==method)];assert len(match)==1
        maximum=max(maximum,max(abs(v-float(match.iloc[0][k])) for k,v in metrics.items()))
    assert len(groups)==900 and maximum<1e-9
    audit=dict(passed=True,trials=len(groups),decisions=decisions,feedback_samples=feedback_samples,maximum_metric_error=maximum,source_assignment_from_official_index=True,feedback_reconstructed_before_each_action=True,simulator_imported=False,test_performance_read=False)
    (path/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(audit)
if __name__=='__main__':main()
