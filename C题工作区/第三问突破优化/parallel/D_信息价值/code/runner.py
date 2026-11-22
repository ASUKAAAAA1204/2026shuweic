"""Episode runner for all methods (paired by construction)."""
import math

import numpy as np

from protocol import ROOT, build_hidden, hidden_for, medians, pools_from
from support import jobs_for  # noqa: E402  (protocol inserted the formal code path)
from engine import Engine, independent_metrics  # noqa: E402
from deadline_policy import choose as base_choose, pressure as base_pressure  # noqa: E402
from baseline import action as base_action  # noqa: E402
import info_policy  # noqa: E402

METHODS = {
    "pressure_gated": dict(kind="baseline"),
    "v2_s20": dict(kind="info", use_free=False, allow_paid=False),
    "v2_free": dict(kind="info", use_free=True, allow_paid=False),
    "v2_paid": dict(kind="info", use_free=False, allow_paid=True),
    "v2_both": dict(kind="info", use_free=True, allow_paid=True),
}


def gated_arm(model, min_gain=3.0):
    """Rational train-only gate: open a channel only if it beats the prior by >=3%."""
    d = model.summary()["prefixes"]
    free = d["0.2"]["improvement_pct"] >= min_gain
    paid = max(d["0.6"]["improvement_pct"], d["1.0"]["improvement_pct"]) >= min_gain
    return dict(kind="info", use_free=free, allow_paid=paid)


def run_episode(jobs, hidden, zones, case, alpha, beta, method, pools, med, model, known=None):
    cfg = METHODS[method]
    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
               case["observation_limit"], alpha, beta)
    history = {}
    cache = {}
    steps = 0
    override = None
    if known is not None:
        override = {jid: np.array([alpha * d + beta]) for jid, d in known.items()}
    while not e.terminal:
        o = e.observe()
        if cfg["kind"] == "baseline":
            a = base_choose(o, pools, history) if base_pressure(o, pools, history) else base_action(o, med, "density", "minimum")
        else:
            a = info_policy.select(o, pools, history, model=model, use_free=cfg["use_free"],
                                   allow_paid=cfg["allow_paid"], strength=20, feedback_strength=3,
                                   depth=1, nq_root=5, nq=2, beam=3, prefix_cache=cache,
                                   job_dists_override=override)
        ev = e.step(a)
        steps += 1
        if a["type"] == "execute" and ev.get("completion") is not None:
            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
            history.setdefault(f, []).append((ev["completion"] - ev["service_start"] - beta) / alpha)
    m = e.metrics()
    ind = independent_metrics(jobs, e.events, e.H)
    diff = max(abs(float(m[k]) - float(ind[k])) for k in m if isinstance(m[k], (int, float)))
    m["actions"] = steps
    return m, ind, diff


def make_pools(train_frame):
    pools = pools_from(train_frame)
    return pools, medians(pools)
