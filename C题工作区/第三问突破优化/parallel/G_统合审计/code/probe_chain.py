"""Probe: is the claimed ancestor-chain probability gating actually active?

`_ancestors` is used *only* inside the leaf gating loop in A/D/E.  If replacing
it with an empty set changes nothing, the gating is a no-op (and descendants are
implicitly treated as certainly completed once their parent is executed).

Reproducible: same hidden_for protocol, 3 cases x 3 seeds x nominal mapping.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import audit_runner as ar


def episode(select, method, case_id, seed, patch):
    from engine import Engine
    cond, tasks, zones, train, seg = ar.load_all()
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    case = next(c for c in cond["cases"] if c["case_id"] == case_id)
    jobs = ar.jobs_of(tasks, case_id)
    dur = ar.hidden_for(jobs, seed, vpool)
    hidden = ar.wrap(dur)
    pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"], case["observation_limit"], 1., .5)
    history = {}
    while not e.terminal:
        o = e.observe()
        a = select(o, pools, history, dict(med={}))
        ev = e.step(a)
        if a["type"] == "execute" and ev.get("completion") is not None:
            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
            history.setdefault(f, []).append(ev["completion"] - ev["service_start"] - .5)
    return e.metrics()["weighted_on_time_rate"]


def main():
    import robust_chain_policy as rcp
    import multi_step_policy as ms
    from configs import CONFIGS

    cond, tasks, zones, train, seg = ar.load_all()
    fit = seg[(seg.split == "train") & (seg.eligible == 1)]
    records = [(env, str(env).split("_x_")[0], fam, g.duration_s.to_numpy(float))
               for (env, fam), g in fit.groupby(["environment_id", "family_id"])]
    cfg = CONFIGS["env_s40_fb3"]
    prior = rcp.build_priors(records, strength=cfg["strength"], balance=cfg["balance"],
                             winsor_q=cfg.get("winsor_q"))
    seeds = cond["validation_seeds"][:3]
    rows = []
    for name, sel, getter in [
        ("E_env_s40_fb3", lambda o, p, h, c: rcp.select(o, prior, h, feedback_strength=3), lambda: rcp._ancestors),
        ("A_v2", lambda o, p, h, c: ms.select(o, p, h, strength=20, feedback_strength=3, depth=1,
                                              nq_root=5, nq=2, beam=3), lambda: ms._ancestors),
    ]:
        mod = rcp if name.startswith("E_") else ms
        orig = mod._ancestors
        diffs = []
        for case in cond["cases"]:
            for seed in seeds:
                a = episode(sel, name, case["case_id"], seed, False)
                mod._ancestors = lambda *args, **kw: ()      # disable gating
                b = episode(sel, name, case["case_id"], seed, True)
                mod._ancestors = orig
                diffs.append(abs(a - b))
                rows.append(dict(method=name, case=case["case_id"], seed=seed,
                                 normal=a, gating_off=b, abs_diff=abs(a - b)))
        print(f"{name}: max |normal - gating_off| = {max(diffs):.3e} over {len(diffs)} episodes")
    out = HERE.parent / "results" / "chain_probe"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)


if __name__ == "__main__":
    main()
