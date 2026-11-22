"""Role G unified audit runner.

Re-runs every candidate policy under one frozen protocol:

  * 3 cases S1/S2/S3 x 30 validation seeds x 5 mappings, paired with the
    official pressure_gated baseline;
  * hidden durations sampled verbatim like confirm_lookahead.hidden_for:
        rng = random.Random(seed)
        for j in jobs_for(tasks, case):  # original order, never changed
            c = validation & eligible rows of j's family
            draw c.iloc[rng.randrange(len(c))]
  * official engine.Engine and engine.independent_metrics re-scoring;
  * in-trial feedback d_hat = (completion - service_start - beta) / alpha per family;
  * decision budget 0.2 s per action is recorded separately (see timing pass).

Every (case, seed, mapping, method) is independent, so multiprocessing does not
break pairing: the hidden draw depends only on (case, seed).
"""
import argparse
import json
import math
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROLE = HERE.parent
BREAK = HERE.parents[2]
WORK = HERE.parents[3]
FORMAL = WORK / "第三问改进正式方案" / "code"
A_CODE = BREAK / "parallel" / "A_多步前瞻" / "code"
B_CODE = BREAK / "parallel" / "B_联合场景前瞻" / "code"
C_CODE = BREAK / "parallel" / "C_尾部价值DP" / "code"
D_CODE = BREAK / "parallel" / "D_信息价值" / "code"
E_CODE = BREAK / "parallel" / "E_稳健分布与选参" / "code"
F_CODE = BREAK / "parallel" / "F_上界与可行性" / "code"
for _p in (str(FORMAL), str(BREAK / "code"), str(A_CODE), str(B_CODE), str(C_CODE),
           str(D_CODE), str(E_CODE), str(F_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAPPINGS = [(1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.)]
NOMINAL = (1., .5)
PREFIXES = (.2, .6, 1.)


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #
def load_all():
    from support import load
    cond, tasks, zones, train = load()
    seg = pd.read_csv(WORK / "处理结果" / "v1" / "segments_private.csv")
    return cond, tasks, zones, train, seg


def jobs_of(tasks, case_id):
    from support import jobs_for
    return jobs_for(tasks, case_id)


def hidden_for(jobs, seed, pool):
    """Verbatim confirm_lookahead.hidden_for; returns durations only."""
    rng = random.Random(seed)
    out = {}
    for j in jobs:
        c = pool[pool.family_id == j["family_id"]]
        out[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
    return out


def hidden_for_with_features(jobs, seed, pool, feature_lookup):
    """Same RNG order and same values; additionally attaches recorded prefixes (role D)."""
    rng = random.Random(seed)
    out = {}
    for j in jobs:
        c = pool[pool.family_id == j["family_id"]]
        row = c.iloc[rng.randrange(len(c))]
        out[j["job_id"]] = (float(row.duration_s), row.unit_id)
    return out


def wrap(dur):
    return {k: dict(duration=v, features={str(p): [] for p in PREFIXES}) for k, v in dur.items()}


# --------------------------------------------------------------------------- #
# policy registry (each adapter: (o, pools, history, ctx) -> action)
# --------------------------------------------------------------------------- #
def _baseline(o, pools, history, ctx):
    from deadline_policy import choose, pressure
    from baseline import action
    return choose(o, pools, history) if pressure(o, pools, history) else action(o, ctx["med"], "density", "minimum")


def _audit_ref(o, pools, history, ctx):
    import lookahead_policy_audit_variants as av
    return av.select(o, pools, history, strength=40, feedback_strength=3, quantiles=5,
                     fix_clock=True, fix_chain=True, fix_budget=True)


def _a_v2(o, pools, history, ctx):
    import multi_step_policy as ms
    return ms.select(o, pools, history, strength=20, feedback_strength=3, depth=1,
                     nq_root=5, nq=2, beam=3)


def _a_d2b4(o, pools, history, ctx):
    import multi_step_policy as ms
    return ms.select(o, pools, history, strength=20, feedback_strength=3, depth=2,
                     nq_root=5, nq=2, beam=4)


def _a_d2b8(o, pools, history, ctx):
    import multi_step_policy as ms
    return ms.select(o, pools, history, strength=20, feedback_strength=3, depth=2,
                     nq_root=5, nq=2, beam=8)


def _a_d3b4(o, pools, history, ctx):
    import multi_step_policy as ms
    return ms.select(o, pools, history, strength=20, feedback_strength=3, depth=3,
                     nq_root=5, nq=2, beam=4)


def _make_d(use_free, allow_paid):
    def fn(o, pools, history, ctx):
        import info_policy
        return info_policy.select(o, pools, history, model=ctx["d_model"], use_free=use_free,
                                  allow_paid=allow_paid, strength=20, feedback_strength=3,
                                  depth=1, nq_root=5, nq=2, beam=3,
                                  prefix_cache=ctx["d_cache"])
    return fn


def _make_e(config_name):
    def fn(o, pools, history, ctx):
        import robust_chain_policy as rcp
        return rcp.select(o, ctx["e_prior"][config_name], history,
                          feedback_strength=ctx["e_cfgs"][config_name].get("feedback_strength", 3.0),
                          winsor_feedback=ctx["e_cfgs"][config_name].get("winsor_feedback"),
                          nq_root=ctx["e_cfgs"][config_name].get("nq_root", 5),
                          grid=ctx["e_cfgs"][config_name].get("grid", 64))
    return fn


def _make_b(K):
    def fn(o, pools, history, ctx):
        import joint_policy
        pol = ctx.get("b_pol")
        if pol is None:
            pol = joint_policy.JointScenarioPolicy(K=K, strength=20, feedback_strength=3,
                                                   seed=20261122, base="survival", stratified=True)
            ctx["b_pol"] = pol
        return pol.select(o, pools, history)
    return fn


def _c_plan(o, pools, history, ctx):
    import dp_tail_policy as dp
    return dp.select(o, pools, history)


def _c_plan_mean(o, pools, history, ctx):
    import dp_tail_policy as dp
    return dp.select(o, pools, history, mode="plan_mean")


def _c_onestep(o, pools, history, ctx):
    import dp_tail_policy as dp
    return dp.select(o, pools, history, mode="onestep")


def _c_mc(topk, samples, dt):
    def fn(o, pools, history, ctx):
        import dp_tail_policy as dp
        return dp.select(o, pools, history, mode="mc", topk=topk, mc_samples=samples,
                         timeline="quantile", scenarios=(.2, .5, .8), tail_quantiles=3,
                         dt=dt, orders=("edf", "wdd", "slack"),
                         use_dp=True, use_greedy=True, force_first=False)
    return fn


def _official_s40(o, pools, history, ctx):
    """Official (pre-audit) lookahead at strength=40, for paired A/B against audit_ref."""
    import lookahead_policy as LP
    return LP.select(o, pools, history, strength=40, feedback_strength=3, quantiles=5)


METHODS = {
    "baseline": dict(adapter=_baseline, kind="base"),
    "audit_ref": dict(adapter=_audit_ref, kind="base"),
    "official_s40": dict(adapter=_official_s40, kind="base"),
    "A_v2": dict(adapter=_a_v2, kind="base"),
    "A_d2b4": dict(adapter=_a_d2b4, kind="base"),
    "A_d2b8": dict(adapter=_a_d2b8, kind="base"),
    "A_d3b4": dict(adapter=_a_d3b4, kind="base"),
    "D_v2_base": dict(adapter=_make_d(False, False), kind="D"),
    "D_v2_free": dict(adapter=_make_d(True, False), kind="D"),
    "D_v2_paid": dict(adapter=_make_d(False, True), kind="D"),
    "D_v2_both": dict(adapter=_make_d(True, True), kind="D"),
    "B_K16": dict(adapter=_make_b(16), kind="base"),
    "B_K32": dict(adapter=_make_b(32), kind="base"),
    "B_K64": dict(adapter=_make_b(64), kind="base"),
    "B_K128": dict(adapter=_make_b(128), kind="base"),
    "C_plan": dict(adapter=_c_plan, kind="base"),
    "C_plan_mean": dict(adapter=_c_plan_mean, kind="base"),
    "C_onestep": dict(adapter=_c_onestep, kind="base"),
    "C_mc_t4_s64_dt05": dict(adapter=_c_mc(4, 64, .5), kind="base"),
    "C_mc_t4_s64_dt10": dict(adapter=_c_mc(4, 64, 1.), kind="base"),
    "C_mc_t4_s256_dt05": dict(adapter=_c_mc(4, 256, .5), kind="base"),
    "C_mc_t4_s256_dt10": dict(adapter=_c_mc(4, 256, 1.), kind="base"),
}


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
_G = {}


def _init():
    cond, tasks, zones, train, seg = load_all()
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
    _G.update(cond=cond, tasks=tasks, zones=zones, train=train, seg=seg, vpool=vpool,
              pools=pools, med={f: float(np.median(v)) for f, v in pools.items()},
              jobs={c["case_id"]: jobs_of(tasks, c["case_id"]) for c in cond["cases"]},
              cases={c["case_id"]: c for c in cond["cases"]})


def _ctx(method, extra):
    base = dict(med=_G["med"], pools=_G["pools"])
    base.update(extra or {})
    return base


def _run_one(task):
    case_id, seed, alpha, beta, method, extra = task
    from engine import Engine, independent_metrics
    if method not in METHODS and method.startswith("E_"):
        METHODS[method] = dict(adapter=_make_e(method[2:]), kind="E")
    case = _G["cases"][case_id]
    jobs = _G["jobs"][case_id]
    kind = METHODS[method]["kind"]
    try:
        if kind == "D":
            sampled = hidden_for_with_features(jobs, seed, _G["vpool"], extra["d_feat"])
            hidden = {k: dict(duration=v[0],
                              features={str(p): extra["d_feat"].get((v[1], p), []) for p in PREFIXES})
                      for k, v in sampled.items()}
        else:
            dur = hidden_for(jobs, seed, _G["vpool"])
            hidden = wrap(dur)
        e = Engine(jobs, hidden, _G["zones"], case["horizon_s"], case["cost_budget"],
                   case["observation_limit"], alpha, beta)
        history = {}
        adapter = METHODS[method]["adapter"]
        ctx = _ctx(method, extra)
        steps = 0
        while not e.terminal:
            o = e.observe()
            a = adapter(o, _G["pools"], history, ctx)
            ev = e.step(a)
            steps += 1
            if a["type"] == "execute" and ev.get("completion") is not None:
                f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                history.setdefault(f, []).append((ev["completion"] - ev["service_start"] - beta) / alpha)
        m = e.metrics()
        ind = independent_metrics(jobs, e.events, e.H)
        diff = max(abs(float(m[k]) - float(ind[k])) for k in m if isinstance(m[k], (int, float)))
        return dict(case=case_id, seed=seed, alpha=alpha, beta=beta, method=method,
                    weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                    weighted_completion_rate=float(m["weighted_completion_rate"]),
                    cost=float(m["cost"]), observations=float(m["observations"]),
                    invalid_actions=int(m["invalid_actions"]), independent_diff=float(diff),
                    steps=steps)
    except Exception:
        return dict(case=case_id, seed=seed, alpha=alpha, beta=beta, method=method, error=traceback.format_exc())


def _feature_lookup(units):
    """Read recorded 0.2/0.6/1.0 prefix features for the given unit ids (role D)."""
    import prefix_model
    return prefix_model.load_features(units)


def build_extra(methods, init_state=None):
    """Assemble per-method shared context that must be fitted on train."""
    extra = {}
    cond, tasks, zones, train, seg = load_all()
    if any(m.startswith("D_") for m in methods):
        import prefix_model
        tr = seg[(seg.split == "train") & (seg.eligible == 1)]
        units = list(tr.unit_id) + list(seg[(seg.split == "validation") & (seg.eligible == 1)].unit_id)
        feat = _feature_lookup(units)
        model = prefix_model.PrefixModel(tr, feat, kind="tree")
        for m in methods:
            if m.startswith("D_"):
                extra[m] = dict(d_feat=feat, d_model=model, d_cache={})
    if any(m.startswith("E_") for m in methods):
        import robust_chain_policy as rcp
        from configs import CONFIGS
        fit = seg[(seg.split == "train") & (seg.eligible == 1)]
        records = []
        for (env, fam), g in fit.groupby(["environment_id", "family_id"]):
            records.append((env, str(env).split("_x_")[0], fam, g.duration_s.to_numpy(float)))
        for m in methods:
            if m.startswith("E_"):
                cfg_name = m[2:]
                prior = rcp.build_priors(records, strength=CONFIGS[cfg_name]["strength"],
                                         balance=CONFIGS[cfg_name]["balance"],
                                         winsor_q=CONFIGS[cfg_name].get("winsor_q"))
                extra[m] = dict(e_prior={cfg_name: prior}, e_cfgs={cfg_name: CONFIGS[cfg_name]})
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="baseline,audit_ref,A_v2,A_d2b4")
    ap.add_argument("--mappings", choices=["nominal", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--tag", default="audit_full")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cond, tasks, zones, train, seg = load_all()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m.startswith("E_") and m[2:] not in __import__("configs").CONFIGS:
            raise SystemExit(f"unknown E config {m}")
        if m not in METHODS:
            METHODS[m] = dict(adapter=_make_e(m[2:]), kind="E")
    maps = MAPPINGS if args.mappings == "all" else [NOMINAL]
    seeds = cond["validation_seeds"][:args.seeds]

    extra = build_extra(methods)
    tasks_list = [(c["case_id"], s, a, b, m, extra.get(m))
                  for c in cond["cases"] for s in seeds for (a, b) in maps for m in methods]

    out = ROLE / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        futs = [ex.submit(_run_one, t) for t in tasks_list]
        done = 0
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(tasks_list)} {time.time()-t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    errs = d[d.get("error").notna()] if "error" in d else d.iloc[0:0]
    if len(errs):
        print(errs[["case", "seed", "method", "error"]].to_string())
        raise SystemExit(f"{len(errs)} runs failed")
    d = d.sort_values(["method", "case", "seed", "alpha", "beta"]).reset_index(drop=True)
    d.to_csv(out / "trials.csv", index=False)
    print(d.groupby("method")[["weighted_on_time_rate", "weighted_completion_rate", "cost", "observations"]]
          .mean().round(4).to_string())
    print("max independent diff", float(d.independent_diff.max()),
          "invalid", int(d.invalid_actions.sum()))
    print("elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
