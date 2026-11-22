"""Acceptance adapters: explicit per-episode state and canonical public order."""
from copy import deepcopy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frozen"))
import acceptance_run as acc
import b_policy
import c_policy
import c_original
import e_policy


def context(fit, vpool):
    pools = {f: g.duration_s.to_numpy(float) for f, g in fit.groupby("family_id")}
    # A family missing in an INNER training fold uses only that fold's global
    # training observations; no held-out observation enters its fallback.
    missing = sorted(set(("A", "L", "P")) - set(pools))
    glob = fit.duration_s.to_numpy(float)
    for f in missing:
        pools[f] = glob.copy()
    pools = {k: pools[k] for k in sorted(pools)}
    med = {f: float(np.median(v)) for f, v in pools.items()}
    records = [(env, str(env).split("_x_")[0], fam, g.duration_s.to_numpy(float))
               for (env, fam), g in fit.groupby(["environment_id", "family_id"])]
    prior = e_policy.build_priors(records, strength=40., balance="env")
    if missing:
        # E is not selected in the B/C nested experiment; this explicit fallback
        # only keeps shared setup well-defined without fabricating env records.
        generic = e_policy._combine([e_policy._empirical_support(glob)], [1.])
        for f in missing:
            prior[f] = generic
    return {"vpool": vpool, "pools": pools, "med": med, "missing_train_families": missing,
            "e": {"e_prior": {"env_s40_fb3": prior},
                  "e_cfgs": {"env_s40_fb3": {"feedback_strength": 3., "grid": 64, "nq_root": 5}}}}


def adapter(method, K=64, samples=256, dt=1., repaired=True):
    if method == "B":
        def select(o, pools, history, ctx):
            obs = dict(o, jobs=sorted(o["jobs"], key=lambda j: j["job_id"]))
            if "pol" not in ctx:
                ctx["pol"] = b_policy.JointScenarioPolicy(K=K, strength=20, feedback_strength=3,
                                                          seed=20261122, base="survival", stratified=True)
            return ctx["pol"].select(obs, pools, history)
        return select
    if method == "C":
        module = c_policy if repaired else c_original
        def select(o, pools, history, ctx):
            obs = dict(o, jobs=sorted(o["jobs"], key=lambda j: j["job_id"]))
            return module.select(obs, pools, history, mode="mc", topk=4, mc_samples=samples,
                                 timeline="quantile", scenarios=(.2, .5, .8), tail_quantiles=3,
                                 dt=dt, orders=("edf", "wdd", "slack"), use_dp=True,
                                 use_greedy=True, force_first=False)
        return select
    if method == "E":
        def select(o, pools, history, ctx):
            obs = dict(o, jobs=sorted(o["jobs"], key=lambda j: j["job_id"]))
            return e_policy.select(obs, ctx["e_prior"]["env_s40_fb3"], history,
                                   feedback_strength=3., grid=64, nq_root=5)
        return select
    return acc.ar.METHODS["baseline"]["adapter"]


def register(name, method, **kwargs):
    acc.ar.METHODS[name] = {"adapter": adapter(method, **kwargs), "kind": "acceptance"}


def run(case, jobs, zones, shared, name, method, seed, alpha=1., beta=.5, **kwargs):
    # acc.make_context has a narrow original-E special case. All E-based adapters
    # receive their immutable prior explicitly; every call gets new mutable state.
    register(name, method, **kwargs)
    original = acc.make_context
    def make_context(_, __):
        return {"med": shared["med"], "pools": shared["pools"], **shared["e"]}
    acc.make_context = make_context
    try:
        return acc.run_episode(case, jobs, zones, shared["pools"], shared, name, seed, alpha, beta)
    finally:
        acc.make_context = original
