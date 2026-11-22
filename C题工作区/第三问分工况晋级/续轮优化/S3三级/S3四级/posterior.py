"""Training-environment prototypes, updated only from completed services in this episode."""
from functools import lru_cache
import numpy as np
import pandas as pd
from search import CFG


@lru_cache(maxsize=1)
def prototypes():
    train=pd.read_csv(CFG/"source_segment_index.csv",usecols=["family_id","split","eligible","environment_id","dataset","duration_s"])
    train=train.query("split=='train' and eligible==1")
    output={}
    q=(np.arange(16)+.5)/16
    for f,group in train.groupby("family_id"):
        base=np.quantile(np.log(group.duration_s),q)
        sizes=[];shapes=[]
        for _,part in group.groupby(["environment_id","dataset"]):
            count=len(part)
            shape=(count*np.quantile(np.log(part.duration_s),q)+2*base)/(count+2)
            sizes.append(count**.5);shapes.append(shape)
        weights=np.array(sizes)/sum(sizes)
        output[f]=np.stack(shapes),weights
    return output


def quantiles(shapes,w,q):
    flat=shapes.ravel();weight=np.repeat(w/shapes.shape[1],shapes.shape[1])
    order=np.argsort(flat);v=flat[order];mid=np.cumsum(weight[order])-weight[order]/2
    return np.interp(q,mid,v)


@lru_cache(maxsize=32768)
def morph(f,history,power,bw):
    if not history:return 1.
    shapes,initial=prototypes()[f]
    q=(np.arange(100)+.5)/100
    log_weights=np.log(initial)
    for d in history:
        densities=np.exp(-.5*((np.log(d)-shapes)/bw)**2).mean(axis=1)
        log_weights+=power*np.log(np.maximum(densities,.008))
    posterior=np.exp(log_weights-log_weights.max());posterior/=posterior.sum()
    change=quantiles(shapes,posterior,q)-quantiles(shapes,initial,q)
    return np.exp(np.clip(change,-.65,.65))
