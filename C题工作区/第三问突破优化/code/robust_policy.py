import numpy as np
from deadline_policy import choose,pressure
from baseline import action

def select(o,pools,history=None,strength=10,feedback_strength=0,use_mean=False,always=False):
    history=history or {};global_values=np.concatenate(list(pools.values()));updated={}
    # A fixed deterministic quantile representation preserves each mixture weight.
    u=(np.arange(128)+.5)/128
    for f,v in pools.items():
        v=np.asarray(v,float);count=int(round(128*strength/(len(v)+strength)))
        values=np.r_[np.quantile(global_values,(np.arange(count)+.5)/count) if count else [],np.quantile(v,(np.arange(128-count)+.5)/(128-count))]
        if history.get(f):
            h=np.asarray(history[f],float)
            if feedback_strength==0:values=h
            else:
                count=int(round(128*len(h)/(len(h)+feedback_strength)))
                values=np.r_[np.quantile(h,(np.arange(count)+.5)/count),np.quantile(values,(np.arange(128-count)+.5)/(128-count)) if count<128 else []]
        if use_mean:
            # Winsorized expected service occupancy, represented as a deterministic assessment.
            values=np.array([np.mean(np.minimum(values,np.quantile(values,.9)))])
        updated[f]=values
    med={f:float(np.median(v)) for f,v in updated.items()}
    return choose(o,updated,{}) if (always or pressure(o,updated,{})) else action(o,med,'density','minimum')
