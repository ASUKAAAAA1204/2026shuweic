"""Causal fixed-damping replay, including a last-six-frame-slot availability rule."""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit_position import DATA, OUT, HORIZONS

MODELS=("recent6_hold","hold_two","zero_velocity","decay")
ZERO=np.zeros(3)

@dataclass(frozen=True)
class State:
    last:int=-1
    time:float=-np.inf
    pos:object=None
    pair:int=-1
    speed:object=None
    pair_time:float=-np.inf
    pair_pos:object=None
    pair_speed:object=None
    damp:object=None
    zero:object=None

def receive(st,i,t,x,gap=False):
    if i<=st.last or t<=st.time or not np.isfinite(t) or not np.isfinite(x).all():return st
    if st.last<0:return State(i,t,x,-1,ZERO,t,x,ZERO,ZERO,ZERO)
    if not gap and i==st.last+1:
        v=(x-st.pos)/(t-st.time)
        return State(i,t,x,i,v,t,x,v,v,v)
    return replace(st,last=i,time=t,pos=x,zero=ZERO,damp=st.damp*np.exp(-(t-st.time)/.2),
                   pair=-1 if gap else st.pair)

def prediction(st,now,query,model):
    if st.last<0:return None,np.nan
    x,t=st.pos,st.time
    if model=="recent6_hold":v=st.speed if st.pair>=0 and st.pair>=now-4 else ZERO
    elif model=="hold_two":x,t,v=st.pair_pos,st.pair_time,st.pair_speed
    elif model=="zero_velocity":v=st.zero
    else:v=st.damp
    return x-.2*np.expm1(-(query-t)/.2)[:,None]*v,query[0]-t

def tests():
    a=receive(State(),0,0.,np.array([0.,0,0]));a=receive(a,1,.1,np.array([1.,0,0]))
    for i,t,x in [(1,.1,ZERO),(3,-1.,ZERO),(2,.2,np.array([np.nan,0,0]))]:
        assert receive(a,i,t,x) is a
    r=receive(a,6,.6,np.array([2.,0,0]))
    np.testing.assert_allclose(r.damp,a.damp*np.exp(-.5/.2))
    assert not prediction(r,9,np.array([1.]),"recent6_hold")[0][0,0]-2
    assert receive(r,7,.7,np.array([3.,0,0])).pair==7
    assert receive(a,2,1.,np.array([2.,0,0]),gap=True).pair==-1
    assert prediction(State(),0,np.array([1.]),"decay")[0] is None

def main():
    tests();dest=OUT/"recovery";dest.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261121);windows=pd.read_csv(OUT/"window_manifest.csv")
    intervals=pd.read_csv(OUT/"interval_usage.csv").query("split!='test'")
    idx=pd.read_csv(DATA/"record_index.csv").query("split!='test'")
    pools=[]
    for (uid,side),g in intervals.query("split=='train'").groupby(["record","side"]):
        with np.load(DATA/"records"/f"{uid}.npz") as z:t=z[side+"__t_s"]
        for r in g.itertuples():
            if r.start_index>0 and r.end_index_exclusive<len(t):
                pools.append((r.env,t[r.end_index_exclusive]-t[r.start_index]))
    pool=pd.DataFrame(pools,columns=["env","duration"])
    masks=[];synt=[];natural=[];events=[];diff=0.;checks=0
    for rec in idx.sort_values("episode_uuid").itertuples():
        uid=rec.episode_uuid
        with np.load(DATA/"records"/f"{uid}.npz") as z:
            for side in ("left_hand","right_hand"):
                t=z[side+"__t_s"];x=z[side+"__position"].mean(1);good=z[side+"__reference_candidate"];gap=z[side+"__gap_before"]
                states=[State()]
                for i in range(len(t)):
                    states.append(receive(states[-1],i,t[i],x[i],gap[i]) if good[i] else states[-1])
                base=dict(record=uid,source=rec.dataset,env=rec.environment_id,side=side,split=rec.split)
                for event in intervals[(intervals.record==uid)&(intervals.side==side)].itertuples():
                    a,b=event.start_index,event.end_index_exclusive
                    events.append(dict(**base,event=a,end=b,frames=b-a,left_censored=a==0,right_censored=b==len(t),
                                   next_contiguous=b+1<len(t) and good[b+1] and not gap[b+1],initialized=states[b].last>=0))
                    for phase,k in [("pre_first",b),("pre_second",b+1)]:
                        valid=k<len(t) and good[k] and (phase=="pre_first" or not gap[k])
                        for model in MODELS:
                            pred,age=(prediction(states[k],k-1,t[k:k+1],model) if valid else (None,np.nan))
                            err=np.linalg.norm(pred[0]-x[k]) if pred is not None else np.nan
                            natural.append(dict(**base,event=a,phase=phase,frame=k,model=model,age=age,error=err,
                                                status="unscorable" if not valid else "no_init" if pred is None else "ok"))
                block=windows[(windows.record==uid)&(windows.side==side)].sort_values("first_hidden")
                durations=pool.loc[pool.env!=rec.environment_id,"duration"].to_numpy()
                for w in block.itertuples():
                    anchor=w.last_visible;i,j=w.first_hidden,w.end_exclusive
                    D=float(rng.choice(durations))
                    # A sampled missing span ends immediately before two reserved recovery slots.
                    stop=anchor-1
                    begin=max(0,min(stop-1,int(np.searchsorted(t,t[stop]-D,"left"))))
                    assert begin<stop and good[anchor-1:anchor+1].all()
                    q=t[i:j];h=q-t[anchor]
                    for scenario,revealed in [("unmasked",None),("no_return",0),("one_return",1),("two_return",2)]:
                        if revealed is None:st=states[i];nmask=0
                        else:
                            st=states[begin]
                            for k in range(i-revealed,i):st=receive(st,k,t[k],x[k],gap[k])
                            nmask=int(good[begin:i-revealed].sum())
                        masks.append(dict(**base,window_id=w.window_id,scenario=scenario,duration=D,start=begin,
                                      first_hidden_query=i,n_masked=nmask,returned=revealed,
                                      last_visible=st.last,age_at_anchor=t[anchor]-st.time if st.last>=0 else np.nan))
                        for model in MODELS:
                            pred,_=prediction(st,anchor,q,model)
                            err=np.linalg.norm(pred-x[i:j],axis=1) if pred is not None else None
                            if scenario=="unmasked":
                                v=(x[anchor]-x[anchor-1])/(t[anchor]-t[anchor-1])
                                diff=max(diff,np.abs(pred-(x[anchor]-.2*np.expm1(-h/.2)[:,None]*v)).max())
                            for H in HORIZONS:
                                take=h<=H+1e-12
                                synt.append(dict(**base,window_id=w.window_id,scenario=scenario,model=model,horizon=H,
                                    availability=err is not None,ade=err[take].mean() if err is not None else np.nan,
                                    fde=err[take][-1] if err is not None else np.nan))
                # Poison unobservable coordinates and every future suffix; re-run to each boundary.
                cutoff=len(t)//2;changed=x.copy();changed[~good]=1e12;changed[cutoff:]=-1e12
                st=State()
                for k in range(cutoff+1):
                    for model in MODELS:
                        a,_=prediction(states[k],k-1,t[k:k+1],model);b,_=prediction(st,k-1,t[k:k+1],model)
                        if a is None:assert b is None
                        else:np.testing.assert_array_equal(a,b)
                    if good[k]:st=receive(st,k,t[k],changed[k],gap[k])
                checks+=1
    assert diff<1e-10
    for name,rows in [("masks.csv.gz",masks),("synthetic_metrics.csv.gz",synt),("natural_points.csv.gz",natural),("events.csv",events)]:
        pd.DataFrame(rows).to_csv(dest/name,index=False)
    (dest/"audit.json").write_text(json.dumps(dict(unmasked_max_error=diff,causal_checked_hand_streams=checks,
        duration_pool=len(pool),seed=20261121,synthetic_rows=len(synt),natural_events=len(events)),indent=2))
if __name__=="__main__":main()
