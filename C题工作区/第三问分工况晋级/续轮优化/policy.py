"""Online route planning from published tasks and observed completions."""
import ctypes
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from final_solution import NominalDispatcher, load_case
from search import settings
import rolling
from motion import conditional


@lru_cache(maxsize=2)
def prior(case):
    return settings(case)[3]


@lru_cache(maxsize=16)
def calibrated_prior(case, cap, scale, mix):
    pools = prior(case)
    pooled = np.concatenate(list(pools.values()))
    rank = (np.arange(100) + .5) / 100
    return {family: scale * ((1-mix) * np.quantile(values, np.minimum(rank, cap))
                             + mix * np.quantile(pooled, np.minimum(rank, cap)))
            for family, values in pools.items()}


@lru_cache(maxsize=2)
def beam_library(stochastic=False):
    source = HERE / ("stochastic.cpp" if stochastic else "beam/beam.cpp")
    path = Path(tempfile.gettempdir()) / ("sw_beam_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16] + ".dll")
    if not path.exists():
        temp = path.with_name(path.stem + str(os.getpid()) + ".dll")
        subprocess.run(["g++", "-std=c++17", "-O3", "-static", "-shared", str(source), "-o", str(temp)], check=True)
        os.replace(temp, path)
    lib = ctypes.CDLL(str(path))
    vec = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    ints = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    if stochastic:
        lib.stochastic_route.argtypes = [ctypes.c_int]*4+[ctypes.c_double]*3+[vec]*5+[ints]+[ctypes.c_double]*3
        lib.stochastic_route.restype = ctypes.c_int
    else:
        lib.beam.argtypes = [ctypes.c_int]*2 + [ctypes.c_double]*3 + [vec]*5 + [ints, ctypes.c_int] + [ctypes.c_double]*2
        lib.beam.restype = ctypes.c_int
    return lib


class Route:
    def __init__(self, case, model):
        self.case = case
        self.model = model
        self.base = NominalDispatcher(case)
        self.history = self.base.history
        self.pools = calibrated_prior(case, model.get("cap", .85), model.get("scale", 1.), model.get("mix", .3))

    def record_completion(self, family, start, end):
        self.base.record_completion(family, start, end)

    def choose(self, o):
        p = self.model
        if p["kind"] == "base":
            return self.base.choose(o)
        if o["time"]/o["H"] >= p.get("base_after",2.):
            return self.base.choose(o)
        if p["kind"] == "robust":
            return rolling.choose(o, p, self.pools, self.history)
        if p["kind"] == "motion_roll":
            priors = {}
            history = {}
            jobs = []
            for j in o["jobs"]:
                f = j["family"]
                key = j["job_id"]
                values = self.pools[f]
                if j["features"]:
                    local = conditional(f,tuple(j["features"]),p["bandwidth"])
                    local = np.minimum(local,np.quantile(local,p["cap"]))
                    values = (1-p["motion_weight"])*values + p["motion_weight"]*local*p["scale"]
                priors[key] = values
                history[key] = self.history.get(f,[])
                jobs.append(dict(j,family=key))
            return rolling.choose(dict(o,jobs=jobs), p, priors, history)
        return self.beam(o)

    def beam(self, o):
        p = self.model
        jobs = o["jobs"]
        ids = [j["job_id"] for j in jobs]
        coords = [j["zone_xy"] for j in jobs]
        distances = np.array([[math.dist(a,b) for b in coords] for a in coords+[o["zone_xy"]]])
        parents = np.array([sum(1<<ids.index(v) for v in j["predecessors"]) for j in jobs], dtype=np.int32)
        done = sum(1<<i for i,j in enumerate(jobs) if j["completed"])
        quantile = p.get("quantile",.5)
        history_weight = p["hist"]
        typical = {}
        for family, pool in self.pools.items():
            obs = self.history.get(family, [])
            typical[family] = (np.quantile(pool,quantile)*history_weight +
                               (np.quantile(obs,quantile)*len(obs) if obs else 0))/(history_weight+len(obs))
        durations = np.array([typical[j["family"]]+.5 for j in jobs])
        attrs = [np.array([j[k] for j in jobs], float) for k in ("release_s","due_s","weight")]
        if p["kind"] == "stochastic":
            ns = p["scenarios"]
            rng = np.random.default_rng(1799 + 17*done)
            samples = np.empty((ns+1,len(jobs)))
            for i,j in enumerate(jobs):
                f = j["family"]
                values = self.pools[f]
                if j["features"] and p["motion_weight"]:
                    local = conditional(f,tuple(j["features"]),p["bandwidth"])
                    values = (1-p["motion_weight"])*values+p["motion_weight"]*p["scale"]*np.minimum(local,np.quantile(local,p["cap"]))
                hist = self.history.get(f,[])
                obs = np.r_[values,hist]
                mass = np.r_[np.full(len(values),history_weight/len(values)),np.ones(len(hist))]
                order = np.argsort(obs)
                cdf = np.cumsum(mass[order])/mass.sum()
                q = (np.arange(ns)+.5)/ns
                rng.shuffle(q)
                samples[:-1,i] = obs[order[np.searchsorted(cdf,q)]]+.5
                samples[-1,i] = np.median(samples[:-1,i])
            completion_weight = p["cw"] + p.get("late_bonus",0.)*(o["time"]/o["H"])**2
            first = beam_library(True).stochastic_route(len(jobs),done,ns,p["width"],o["time"],
                o["B"]-o["cost"],o["H"],*attrs,distances,samples,parents,completion_weight,p["optimism"],p["margin"])
        else:
            first = beam_library().beam(len(jobs), done, o["time"], o["B"]-o["cost"], o["H"], *attrs,
                                        distances,durations,parents,p["width"],p["cw"],p["margin"])
        if first >= 0:
            return dict(type="execute",job_id=ids[first],reserve=.2)
        future = [j["release_s"] for j in jobs if not j["attempted"] and j["release_s"]>o["time"]+1e-12]
        if first == -2 and future and min(future) <= o["H"]:
            return dict(type="wait",duration=min(future)-o["time"])
        return self.base.choose(o)
