"""Deadline survival and one-service-delay loss, with free completed-duration feedback."""
import math,zlib
import numpy as np

def choose(o,pools,history=None,deadline=True,delay=True,closure=True):
    history=history or {}
    samples={f:o['alpha']*np.asarray(history.get(f) or list(v))+o['beta'] for f,v in pools.items()}
    med={f:float(np.median(v)) for f,v in samples.items()}
    done={j['job_id'] for j in o['jobs'] if j['completed']}
    available=[j for j in o['jobs'] if not j['attempted'] and j['release_s']<=o['time']+1e-12 and set(j['predecessors'])<=done]
    mean_step=float(np.mean([med[j['family']]+.5+math.dist(o['zone_xy'],j['zone_xy']) for j in available])) if available else 1.
    candidates=[]
    for j in available:
        D=math.dist(o['zone_xy'],j['zone_xy']);start=o['time']+D+.5
        if start+.2>o['H']+1e-12 or o['cost']+2.02+.2*D>o['B']+1e-12:continue
        p=samples[j['family']];slack=min(j['due_s'],o['H'])-start
        survival=float(np.mean(p<=slack+1e-12));delay_loss=survival-float(np.mean(p<=slack-mean_step+1e-12))
        value=j['weight']*(survival+(delay_loss if delay else 0.)) if deadline else j['weight']
        score=value/(D+.5+med[j['family']])
        # Preserve a positive value for a late ancestor only if an immediate child
        # can still be made on time; no lookahead cost is charged to reality.
        if deadline and closure:
            for child in o['jobs']:
                if child['attempted'] or j['job_id'] not in child['predecessors'] or not (set(child['predecessors'])-{j['job_id']})<=done:continue
                q=samples[child['family']];travel=math.dist(j['zone_xy'],child['zone_xy'])+.5
                finish=np.maximum(start+p[:,None],child['release_s'])+travel+q[None,:]
                chance=float(np.mean(finish<=min(child['due_s'],o['H'])+1e-12))
                score=max(score,(j['weight']*survival+child['weight']*chance)/(D+.5+med[j['family']]+travel+med[child['family']]))
        candidates.append((-score,j['due_s'],j['job_id']))
    if candidates:
        _,_,job=min(candidates);return {'type':'execute','job_id':job,'reserve':.2}
    future=[j['release_s'] for j in o['jobs'] if not j['attempted'] and o['time']+1e-12<j['release_s']<=o['H']]
    return {'type':'wait','duration':min(future)-o['time']} if future else {'type':'stop'}

def pressure(o,pools,history=None):
    """90% empirical deadline workload screen, ignoring movement (optimistic screen).

    A failed screen diagnoses congestion; passing does not certify feasibility.
    No case ID, target performance or hidden duration is used.
    """
    history=history or {};jobs=[j for j in o['jobs'] if not j['attempted']]
    if not jobs:return False
    costs=[]
    for j in jobs:
        values=np.sort(history.get(j['family']) or list(pools[j['family']]))
        rng=np.random.default_rng(20261123+zlib.crc32(j['job_id'].encode()));u=(np.arange(64)+.5)/64;rng.shuffle(u)
        costs.append(o['alpha']*values[np.minimum((u*len(values)).astype(int),len(values)-1)]+o['beta']+.5)
    costs=np.asarray(costs)
    for due in sorted(set(min(j['due_s'],o['H']) for j in jobs if j['due_s']>o['time'])):
        relevant=[i for i,j in enumerate(jobs) if j['due_s']<=due]
        if relevant and np.mean(o['time']+costs[relevant].sum(axis=0)>due+1e-12)>.1:return True
    return False
