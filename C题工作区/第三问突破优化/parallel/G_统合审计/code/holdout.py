"""Environment-group holdout (two folds), replicating code/holdout_lookahead.py.

Pools are fitted on one half of the 16 training environments and evaluated on
the held half with seeds 2026120100..2026120119 and the nominal mapping.  This
is an environment-level stress test, not the official validation split.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import audit_runner as ar

ROLE = HERE.parent
ROOT = ar.WORK


def build_method_ctx(method, fit, feat=None, model=None):
    if method.startswith("E_"):
        import robust_chain_policy as rcp
        from configs import CONFIGS
        cfg = method[2:]
        records = []
        for (env, fam), g in fit.groupby(["environment_id", "family_id"]):
            records.append((env, str(env).split("_x_")[0], fam, g.duration_s.to_numpy(float)))
        prior = rcp.build_priors(records, strength=CONFIGS[cfg]["strength"], balance=CONFIGS[cfg]["balance"],
                                 winsor_q=CONFIGS[cfg].get("winsor_q"))
        return dict(e_prior={cfg: prior}, e_cfgs={cfg: CONFIGS[cfg]})
    if method.startswith("D_"):
        return dict(d_feat=feat, d_model=model, d_cache={})
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="baseline,audit_ref,A_v2,E_env_s40_fb3_w95")
    ap.add_argument("--tag", default="holdout")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ar.METHODS and m.startswith("E_"):
            ar.METHODS[m] = dict(adapter=ar._make_e(m[2:]), kind="E")

    cond, tasks, zones, train, seg = ar.load_all()
    groups = json.loads((ROOT / "第三问突破优化" / "results" / "grouped_training" / "registration.json")
                        .read_text())["groups"]
    needD = any(m.startswith("D_") for m in methods)
    feat = model = None
    if needD:
        import prefix_model
        tr = seg[(seg.split == "train") & (seg.eligible == 1)]
        units = list(tr.unit_id) + list(seg[(seg.split == "validation") & (seg.eligible == 1)].unit_id)
        feat = ar._feature_lookup(units)
        model = prefix_model.PrefixModel(tr, feat, kind="tree")

    from engine import Engine
    rows = []
    for fold, held in enumerate(groups):
        fit = seg[(seg.split == "train") & (seg.eligible == 1) & (~seg.environment_id.isin(held))]
        evp = seg[(seg.split == "train") & (seg.eligible == 1) & (seg.environment_id.isin(held))]
        pools = {f: g.duration_s.to_numpy(float) for f, g in fit.groupby("family_id")}
        med = {f: float(np.median(v)) for f, v in pools.items()}
        shared = {m: build_method_ctx(m, fit, feat, model) for m in methods}
        for case in cond["cases"]:
            jobs = ar.jobs_of(tasks, case["case_id"])
            for seed in range(2026120100, 2026120120):
                rng = random.Random(seed)
                dur = {}
                for j in jobs:  # verbatim order
                    c = evp[evp.family_id == j["family_id"]]
                    dur[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
                hidden = ar.wrap(dur)
                for m in methods:
                    # fresh ctx per trial: stateful policies must not leak across seeds
                    ctx = dict(med=med, pools=pools)
                    for k, v in shared[m].items():
                        ctx[k] = dict(v) if isinstance(v, dict) else v
                    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                               case["observation_limit"], 1., .5)
                    history = {}
                    while not e.terminal:
                        o = e.observe()
                        a = ar.METHODS[m]["adapter"](o, pools, history, ctx)
                        ev = e.step(a)
                        if a["type"] == "execute" and ev.get("completion") is not None:
                            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
                            history.setdefault(f, []).append(ev["completion"] - ev["service_start"] - .5)
                    mm = e.metrics()
                    assert not mm["invalid_actions"]
                    rows.append(dict(fold=fold, case=case["case_id"], seed=seed, method=m,
                                     weighted_on_time_rate=float(mm["weighted_on_time_rate"]),
                                     weighted_completion_rate=float(mm["weighted_completion_rate"]),
                                     cost=float(mm["cost"]), observations=float(mm["observations"])))
        print("finished fold", fold, flush=True)
    d = pd.DataFrame(rows)
    out = ROLE / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "trials.csv", index=False)
    base = d[d.method == "baseline"].set_index(["fold", "case", "seed"]).weighted_on_time_rate
    piv = d.pivot_table(index="method", columns="case", values="weighted_on_time_rate", aggfunc="mean")
    piv["avg3"] = piv.mean(axis=1)
    summary = {"by_case_pct": (piv * 100).round(4).to_dict(), "by_fold": {}}
    for m in methods:
        if m == "baseline":
            continue
        x = d[d.method == m].set_index(["fold", "case", "seed"]).weighted_on_time_rate
        delta = (x - base) * 100
        summary["by_fold"][m] = dict(
            mean_pp=float(delta.mean()),
            fold0_pp=float(delta[delta.index.get_level_values(0) == 0].mean()),
            fold1_pp=float(delta[delta.index.get_level_values(0) == 1].mean()),
            pooled_mean3_pct=float(piv.loc[m, "avg3"] * 100),
            baseline_mean3_pct=float(piv.loc["baseline", "avg3"] * 100),
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
