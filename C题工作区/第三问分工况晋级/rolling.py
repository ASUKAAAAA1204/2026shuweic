"""Replan using hypothetical durations independent of the evaluation draw."""
import ctypes as ct
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

@lru_cache(maxsize=1)
def library():
    source=Path(__file__).with_suffix(".cpp")
    binary=Path(tempfile.gettempdir())/("q3_roll_"+hashlib.sha256(source.read_bytes()).hexdigest()[:14]+".dll")
    if not binary.exists():
        temp=binary.with_name(binary.stem+str(os.getpid())+".dll")
        subprocess.run(["g++","-std=c++17","-O3","-shared","-static","-o",str(temp),str(source)],check=True)
        os.replace(temp,binary)
    dll=ct.CDLL(str(binary))
    arr=np.ctypeslib.ndpointer(np.float64,flags="C_CONTIGUOUS")
    ints=np.ctypeslib.ndpointer(np.int32,flags="C_CONTIGUOUS")
    dll.rolling.argtypes=[ct.c_int]*3+[ct.c_double]*4+[arr]*5+[ints,arr,ct.c_int,ct.c_uint32]
    dll.rolling.restype=ct.c_int
    return dll

def choose(o,model,pools,history):
    jobs=o["jobs"];n=len(jobs)
    ids=[j["job_id"] for j in jobs]
    done=sum(1<<i for i,j in enumerate(jobs) if j["completed"])
    parents=np.array([sum(1<<ids.index(v) for v in j["predecessors"]) for j in jobs],dtype=np.int32)
    dist=np.array([[math.dist(x,j["zone_xy"]) for j in jobs] for x in [j["zone_xy"] for j in jobs]+[o["zone_xy"]]])
    seed=(881+done*13+n*17)&0xffffffff
    rng=np.random.default_rng(seed)
    ns=model["scenarios"]
    samples=np.empty((ns,n));typical=np.empty(n)
    for i,j in enumerate(jobs):
        f=j["family"];prior=pools[f];hist=history.get(f,[])
        if model.get("stratified"):
            values=np.r_[prior,hist]
            mass=np.r_[np.full(len(prior),model["prior_count"]/len(prior)),np.ones(len(hist))]
            order=np.argsort(values)
            cdf=np.cumsum(mass[order])/mass.sum()
            q=(np.arange(ns)+rng.random(ns))/ns
            rng.shuffle(q)
            sim=values[order[np.searchsorted(cdf,q)]]
        else:
            sim=rng.choice(prior,ns)
            if hist:
                mix=rng.random(ns)<len(hist)/(len(hist)+model["prior_count"])
                sim[mix]=rng.choice(hist,mix.sum())
        samples[:,i]=o["alpha"]*sim+o["beta"]
        typical[i]=np.median(samples[:,i])
    args=[np.array([j[k] for j in jobs],float) for k in ("release_s","due_s","weight")]
    j=library().rolling(n,done,ns,o["time"],o["B"]-o["cost"],o["H"],model["completion_weight"],
        *args,dist,samples,parents,typical,model["steps"],seed)
    if j>=0:return dict(type="execute",job_id=ids[j],reserve=.2)
    future=[j["release_s"] for j in jobs if not j["attempted"] and j["release_s"]>o["time"]+1e-12]
    return dict(type="wait",duration=min(future)-o["time"]) if future else dict(type="stop")
