"""A train-only value table with causal service distributions at dispatch time."""
import ctypes as ct
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from adaptive import ROOT, calibrated_prior, frozen_dispatcher, motion_factor
from search import settings

FAMS=("A","P","L")


@lru_cache(maxsize=1)
def library():
    src=ROOT/"dynamic.cpp"
    path=Path(tempfile.gettempdir())/("sw_bellman_"+hashlib.sha256(src.read_bytes()).hexdigest()[:16]+".dll")
    if not path.exists():
        tmp=path.with_name(path.stem+str(os.getpid())+".dll")
        subprocess.run(["g++","-O3","-std=c++17","-shared","-static",str(src),"-o",str(tmp)],check=True)
        os.replace(tmp,path)
    lib=ct.CDLL(str(path))
    doubles=np.ctypeslib.ndpointer(np.float64,flags="C_CONTIGUOUS")
    ints=np.ctypeslib.ndpointer(np.int32,flags="C_CONTIGUOUS")
    lib.fit_value.argtypes=[ct.c_double]*4+[ints]*3+[doubles]*4+[ct.c_int,doubles]
    lib.fit_value.restype=ct.c_void_p
    lib.refit_value.argtypes=lib.fit_value.argtypes+[ct.c_int]
    lib.refit_value.restype=ct.c_void_p
    lib.best_job.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_int,doubles,ct.c_double,ct.c_double]
    lib.best_job.restype=ct.c_int
    lib.free_value.argtypes=[ct.c_void_p]
    lib.free_value.restype=None
    return lib


class Table:
    def __init__(self,cap,scale,cw,dt,mix,tail):
        c,t,z,_=settings("S3")
        names=sorted(z)
        self.locations={tuple(z[v]):i for i,v in enumerate(names)}
        dist=np.array([[math.dist(z[a],z[b]) for b in names] for a in names])
        self.ids=t.job_id.tolist()
        families=np.array([FAMS.index(f) for f in t.family_id],np.int32)
        zones=np.array([names.index(v) for v in t.zone_id],np.int32)
        pred=np.array([sum(1<<self.ids.index(v) for v in a.split("|") if v) for a in t.predecessors],np.int32)
        assert len(self.ids)==16 and (2.02+.2*dist.max())*16<=c["cost_budget"]
        self.pools=calibrated_prior("S3",cap,1.,mix)
        self.pools={f:scale*(np.median(v)+(v-np.median(v))*tail) for f,v in self.pools.items()}
        self.pools={f:np.maximum(.2,v) for f,v in self.pools.items()}
        support=np.vstack([np.quantile(self.pools[f],(np.arange(16)+.5)/16)+.5 for f in FAMS])
        self.lib=library()
        self.args=(c["horizon_s"],dt,cw,.0008,families,zones,pred,
            *[np.ascontiguousarray(t[k],dtype=float) for k in ("release_s","due_s","priority_weight")],
            dist)
        self.handle=self.lib.fit_value(*self.args,16,support)
        if not self.handle:raise ValueError("Invalid Bellman table")
    def __del__(self):
        if getattr(self,"handle",None):self.lib.free_value(self.handle)


@lru_cache(maxsize=2)
def get_table(cap,scale,cw,dt,mix,tail):
    return Table(cap,scale,cw,dt,mix,tail)


class Bellman:
    def __init__(self,p):
        self.p=p
        self.base=frozen_dispatcher("S3",2)
        self.table=get_table(p["cap"],p["scale"],p["cw"],p["dt"],p.get("mix",.05),p.get("tail",1.))
        self.pools=self.table.pools
        self.history={f:[] for f in FAMS}
        self.pending=1.
        self.local=None
        self.last_count=-1

    def __del__(self):
        if getattr(self,"local",None):self.table.lib.free_value(self.local)

    def record_completion(self,family,start,end):
        self.base.record_completion(family,start,end)
        self.history[family].append((end-start-.5,self.pending))

    def choose(self,o):
        jobs=o["jobs"];p=self.p;table=self.table
        if (o["alpha"],o["beta"])!=(1.,.5) or [j["job_id"] for j in jobs]!=table.ids:raise ValueError("Expected nominal S3")
        done=sum(1<<i for i,j in enumerate(jobs) if j["completed"])
        ratios=[np.clip(np.log(v/np.median(self.pools[f])),-1.,1.) for f in FAMS for v,_ in self.history[f]]
        pace=np.exp(sum(ratios)/(len(ratios)+6)*p.get("adapt",0.))
        factors=[motion_factor(tuple(j["features"]))**p.get("motion",0.) if p.get("motion") and j["features"] else 1. for j in jobs]
        samples=np.empty((16,32));q=(np.arange(32)+.5)/32
        for i,j in enumerate(jobs):
            f=j["family"];factor=factors[i];v=self.pools[f]*factor
            past=self.history[f]
            values=np.r_[v,[d*np.clip(factor/a,.6,1.8) for d,a in past]]
            weights=np.r_[np.full(len(v),p["hist"]/len(v)),np.ones(len(past))]
            order=np.argsort(values)
            cdf=np.cumsum(weights[order])/weights.sum()
            samples[i]=values[order[np.searchsorted(cdf,q)]]+.5
        count=done.bit_count()
        if count in p.get("refresh",[]) and self.last_count!=count:
            args=list(table.args)
            args[1]=p.get("local_dt",1.)
            args[2]=p.get("local_cw",p["cw"])
            if self.local:table.lib.free_value(self.local)
            self.local=table.lib.refit_value(*args,16,np.ascontiguousarray(samples[:,1::2]),done)
            self.last_count=count
        idx=table.lib.best_job(self.local or table.handle,done,table.locations[tuple(o["zone_xy"])],o["time"],o["B"]-o["cost"],32,
                               samples,1 if self.local else pace,p.get("shift",0.))
        if idx>=0:
            self.pending=factors[idx]
            return dict(type="execute",job_id=jobs[idx]["job_id"],reserve=.2)
        future=[j["release_s"] for j in jobs if not j["attempted"] and j["release_s"]>o["time"]+1e-12]
        if idx==-2 and future:
            return dict(type="wait",duration=min(future)-o["time"])
        a=self.base.choose(o)
        if a["type"]=="execute":self.pending=factors[table.ids.index(a["job_id"])]
        return a
