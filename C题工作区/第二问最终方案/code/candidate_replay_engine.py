# Experimental copy of frozen engine, adding only an injectable state factory.
"""Causal Q2 replay. Policy receives a frozen public view, never reference arrays."""
from dataclasses import dataclass
import math,time
import numpy as np

EPS=1e-9

@dataclass(frozen=True)
class View:
    now: float
    relative: float
    remaining: float
    age: float
    requests: int
    pending_until: float|None
    last_time: float
    last_position: tuple
    velocity: tuple
    successful_returns: int

class FixedPolicy:
    def __init__(self,schedule=(.3,.6)):self.schedule=tuple(schedule);self.used=set()
    def request(self,view):
        due=[a for a in self.schedule if a not in self.used and view.relative+1e-12>=a]
        if not due:return False
        self.used.add(due[0]);return True

class NeverPolicy:
    def request(self,view):return False

class MotionState:
    def __init__(self,history_t,history_x,motion='damped',update='secant'):
        assert len(history_t)>=2 and np.all(np.diff(history_t)>0)
        self.last_time=float(history_t[-1]);self.last=np.array(history_x[-1],float).copy()
        self.velocity=(np.asarray(history_x[-1])-history_x[-2])/(history_t[-1]-history_t[-2])
        self.motion=motion;self.update=update;self.returns=0
    def predict(self,now):
        age=now-self.last_time
        assert age>=-EPS
        factor=0. if self.motion=='hold' else age if self.motion=='cv' else -.2*math.expm1(-age/.2)
        return self.last+factor*self.velocity
    def receive(self,sample_time,position):
        assert sample_time>self.last_time
        if self.update=='secant':new_v=(position-self.last)/(sample_time-self.last_time)
        elif self.update=='decayed':new_v=self.velocity*math.exp(-(sample_time-self.last_time)/.2)
        else:raise ValueError(self.update)
        self.velocity=new_v;self.last=np.asarray(position,float).copy();self.last_time=float(sample_time);self.returns+=1
    def view(self,now,start,end,requests,pending_until):
        return View(float(now),float(now-start),float(end-now),float(now-self.last_time),requests,pending_until,self.last_time,tuple(self.last),tuple(self.velocity),self.returns)

def replay(times,reference,history_times,history_positions,policy,delay=.1,budget=2,motion='damped',update='secant',valid=None,window_end=None,state_factory=None):
    """Known timestamps; evaluator captures payload, reveals it only at due time.
    Events at/before a scoring timestamp are processed before prediction.
    Boundary: an observation arriving exactly on the final scoring frame can improve FDE.
    """
    times=np.asarray(times,float);reference=np.asarray(reference,float)
    assert len(times)>0 and reference.shape==(len(times),3) and np.all(np.diff(times)>0)
    assert delay>=0 and isinstance(budget,int) and budget>=0
    valid=np.ones(len(times),bool) if valid is None else np.asarray(valid,bool)
    assert valid.shape==times.shape
    start=float(times[0]);end=float(times[-1] if window_end is None else window_end)
    assert end>=times[-1]-EPS
    state=(MotionState if state_factory is None else state_factory)(history_times,history_positions,motion,update);pending=None;requests=0;logs=[];points=[];seconds=[]
    for k,now in enumerate(times):
        tic=time.perf_counter()
        if pending is not None and now+1e-12>=pending[0]:
            arrival,sample_t,value,good=pending
            if good:state.receive(sample_t,value)
            logs.append(dict(event='return',time=arrival,processed_at=float(now),sample_time=sample_t,valid=bool(good),request_id=requests))
            pending=None
        deadline=pending[0] if pending is not None else None
        view=state.view(now,start,end,requests,deadline)
        # Evaluator enforces budget/time before asking policy; reference is inaccessible to policy.
        if pending is None and requests<budget and now+delay<=end+1e-12:
            if policy.request(view):
                requests+=1;pending=(float(now+delay),float(now),reference[k].copy(),bool(valid[k]))
                logs.append(dict(event='request',time=float(now),sample_time=float(now),arrival=float(now+delay),request_id=requests))
                if delay==0:
                    if valid[k]:state.receive(float(now),reference[k])
                    logs.append(dict(event='return',time=float(now),processed_at=float(now),sample_time=float(now),valid=bool(valid[k]),request_id=requests));pending=None
        prediction=state.predict(now)
        seconds.append(time.perf_counter()-tic)
        points.append(dict(time=float(now),relative=float(now-start),prediction=prediction.tolist(),reference=reference[k].tolist(),error=float(np.linalg.norm(prediction-reference[k])),age=float(now-state.last_time),last_time=state.last_time,requests=requests,returns=state.returns,pending_until=pending[0] if pending is not None else None))
    errors=np.array([p['error'] for p in points]);duration=end-start
    intervals=[(a['time'],min(a['arrival'],end)) for a in logs if a['event']=='request']
    waiting=sum(max(0.,b-a) for a,b in intervals)
    assert requests<=budget and waiting<=duration+EPS and np.isfinite(errors).all()
    metrics=dict(ade=float(errors.mean()),fde=float(errors[-1]),p95=float(np.quantile(errors,.95)),requests=requests,cost=requests,waiting_s=waiting,no_request_wait_s=max(0.,duration-waiting),successful_returns=state.returns,points=len(points),strategy_compute_s=sum(seconds),step_compute_p95_s=float(np.quantile(seconds,.95)))
    return dict(metrics=metrics,points=points,actions=logs)
