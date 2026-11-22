"""S3-only prospective policies. All history is restricted to finished services."""
import ctypes as ct
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from kinematics import ROOT, estimate

sys.path.insert(0,str(ROOT.parent))
from policy import calibrated_prior,beam_library
from release import dispatcher as frozen_dispatcher


@lru_cache(maxsize=1024)
def motion_factor(features):
    return float(np.exp(estimate(forest(),features))) if features else 1.


@lru_cache(maxsize=1)
def forest():
    return json.loads((ROOT/"kinematics.json").read_bytes())["trees"]


@lru_cache(maxsize=1)
def recourse_library():
    src=ROOT/"rollout/recourse.cpp"
    path=Path(tempfile.gettempdir())/("sw_recourse_"+hashlib.sha256(src.read_bytes()).hexdigest()[:16]+".dll")
    if not path.exists():
        tmp=path.with_name(path.stem+str(os.getpid())+".dll")
        subprocess.run(["g++","-O3","-std=c++17","-shared","-static",str(src),"-o",str(tmp)],check=True)
        os.replace(tmp,path)
    lib=ct.CDLL(str(path))
    ptr=np.ctypeslib.ndpointer(np.float64,flags="C_CONTIGUOUS")
    ints=np.ctypeslib.ndpointer(np.int32,flags="C_CONTIGUOUS")
    lib.recourse.argtypes=[ct.c_int]*3+[ct.c_double]*3+[ptr]*5+[ints]+[ct.c_double]*3
    lib.recourse.restype=ct.c_int
    return lib


class Adaptive:
    def __init__(self,p):
        self.p=p
        self.base=frozen_dispatcher("S3",2)
        self.pool=calibrated_prior("S3",p.get("cap",.99),1.,.05)
        self.observed={f:[] for f in self.pool}
        self.pending=None
        self.geometry=None

    def record_completion(self,family,start,end):
        self.base.record_completion(family,start,end)
        # The decision stores only a coarse motion estimate for the job just executed.
        factor=self.pending if self.pending is not None else 1.
        self.observed[family].append((end-start-.5,factor))
        self.pending=None

    def choose(self,o):
        p=self.p;jobs=o["jobs"];n=len(jobs)
        if (o["alpha"],o["beta"])!=(1.,.5) or tuple(j["job_id"] for j in jobs)!=self.base.base.job_ids:
            raise ValueError("Expected published S3 nominal conditions")
        if self.geometry is None:
            ids=[j["job_id"] for j in jobs]
            xy=[j["zone_xy"] for j in jobs]
            self.geometry=(np.array([[math.dist(a,b) for b in xy] for a in xy]),
                [np.array([j[k] for j in jobs],float) for k in ("release_s","due_s","weight")],
                np.array([sum(1<<ids.index(v) for v in j["predecessors"]) for j in jobs],np.int32))
        distances,attrs,parents=self.geometry
        dist=np.vstack((distances,[math.dist(o["zone_xy"],j["zone_xy"]) for j in jobs]))
        done=sum(1<<i for i,j in enumerate(jobs) if j["completed"])
        ns=32
        rng=np.random.default_rng(1799+17*done)
        samples=np.empty((ns+1,n))
        factors=[motion_factor(tuple(j["features"])) if j["features"] else 1. for j in jobs]
        for i,j in enumerate(jobs):
            f=j["family"];g=factors[i]
            pool=self.pool[f];history=self.observed[f]
            strength=p["motion"]
            scale=np.clip(g**strength,.4,2.5)
            samples_prior=pool*scale
            h=[v*np.clip((g/k)**strength,.5,2.) for v,k in history]
            v=np.r_[samples_prior,h]
            weight=np.r_[np.full(len(pool),p["hist"]/len(pool)),np.ones(len(h))]
            order=np.argsort(v);cdf=np.cumsum(weight[order])/weight.sum()
            q=(np.arange(ns)+.5)/ns
            rng.shuffle(q)
            sim=v[order[np.searchsorted(cdf,q)]]
            samples[:-1,i]=sim+.5
            samples[-1,i]=np.median(sim)+.5
        cw=p["cw"]+p.get("bonus",.5)*(o["time"]/o["H"])**2
        if p.get("kind","beam")=="recourse":
            first=recourse_library().recourse(n,done,ns,o["time"],o["B"]-o["cost"],o["H"],*attrs,dist,
                                samples,parents,cw,p["urgency"],p["travel"])
        else:
            first=beam_library(True).stochastic_route(n,done,ns,p["width"],o["time"],o["B"]-o["cost"],o["H"],
                                *attrs,dist,samples,parents,cw,p.get("optimism",.52),p.get("margin",2.))
        if first>=0:
            self.pending=factors[first]
            return dict(type="execute",job_id=jobs[first]["job_id"],reserve=.2)
        future=[j["release_s"] for j in jobs if not j["attempted"] and j["release_s"]>o["time"]+1e-12]
        if first==-2 and future:return dict(type="wait",duration=min(future)-o["time"])
        action=self.base.choose(o)
        if action["type"]=="execute":
            self.pending=next(factors[i] for i,j in enumerate(jobs) if j["job_id"]==action["job_id"])
        return action
