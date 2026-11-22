"""Shared protocol helpers for role E (robust belief + parameter selection).

Protocol (must not be changed, otherwise results are not mergeable):
  * 3 cases S1/S2/S3; 30 validation seeds; 5 mappings; paired baseline/candidate;
  * hidden durations sampled exactly like `confirm_lookahead.py`:
        rng = random.Random(seed); iterate jobs_for(tasks, case) in order;
        sample uniformly (with replacement) from validation & eligible rows of
        the same family; never change the traversal order;
  * official `engine.Engine` plus `engine.independent_metrics` re-scoring;
  * in-trial feedback d_hat = (completion - service_start - beta) / alpha,
    accumulated per family;
  * train split only for fitting; validation only for the final confirmation;
    test split is never read.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROLE = HERE.parent
BREAK = HERE.parents[2]
WORK = HERE.parents[3]
FORMAL = WORK / "第三问改进正式方案" / "code"
if str(FORMAL) not in sys.path:
    sys.path.insert(0, str(FORMAL))

from support import load, jobs_for  # noqa: E402
from engine import Engine, independent_metrics  # noqa: E402
from deadline_policy import choose as base_choose, pressure as base_pressure  # noqa: E402
from baseline import action as base_action  # noqa: E402

MAPPINGS = [(1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
NOMINAL = (1.0, 0.5)


def load_all():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(WORK / "处理结果" / "v1" / "segments_private.csv")
    return cond, tasks, zones, train, seg


def pools_from(seg_rows):
    return {f: g.duration_s.to_numpy(float) for f, g in seg_rows.groupby("family_id")}


def records_from(seg_rows):
    """[(environment_id, region, family_id, duration array)] used by the belief."""
    out = []
    for (env, fam), g in seg_rows.groupby(["environment_id", "family_id"]):
        out.append((env, str(env).split("_x_")[0], fam, g.duration_s.to_numpy(float)))
    return out


def train_rows(seg, environments=None):
    rows = seg[(seg.split == "train") & (seg.eligible == 1)]
    if environments is not None:
        rows = rows[rows.environment_id.isin(environments)]
    return rows


def hidden_for(jobs, seed, pool_frame):
    rng = random.Random(seed)
    out = {}
    for j in jobs:
        c = pool_frame[pool_frame.family_id == j["family_id"]]
        out[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
    return out


def wrap_hidden(dur):
    return {k: dict(duration=v, features={str(p): [] for p in (0.2, 0.6, 1.0)})
            for k, v in dur.items()}


def make_decider(method, ctx):
    """Return f(observation, history) -> action for the requested method."""
    pools, med, prior, params = ctx["pools"], ctx["med"], ctx["prior"], ctx.get("params", {})

    if method == "baseline":
        def decide(o, history):
            return base_choose(o, pools, history) if base_pressure(o, pools, history) \
                else base_action(o, med, "density", "minimum")
    elif method == "v2_s20":
        import multi_step_policy as ms

        def decide(o, history):
            return ms.select(o, pools, history, strength=20, feedback_strength=3,
                             depth=1, nq_root=5, nq=2, beam=3)
    elif method == "role_e":
        import robust_chain_policy as rcp

        def decide(o, history):
            return rcp.select(o, prior, history,
                              feedback_strength=params.get("feedback_strength", 3.0),
                              winsor_feedback=params.get("winsor_feedback"),
                              nq_root=params.get("nq_root", 5),
                              grid=params.get("grid", 64))
    else:
        raise ValueError(method)
    return decide


def replay(jobs, hidden, zones, case, alpha, beta, decide, record_steps=False):
    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
               case["observation_limit"], alpha, beta)
    history = {}
    steps = 0
    while not e.terminal:
        o = e.observe()
        a = decide(o, history)
        ev = e.step(a)
        steps += 1
        if a["type"] == "execute" and ev.get("completion") is not None:
            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
            history.setdefault(f, []).append(
                (ev["completion"] - ev["service_start"] - beta) / alpha)
    m = e.metrics()
    ind = independent_metrics(jobs, e.events, e.H)
    diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
    return m, diff, steps


def add_v2_path():
    a_code = BREAK / "parallel" / "A_多步前瞻" / "code"
    if str(a_code) not in sys.path:
        sys.path.insert(0, str(a_code))


def run_grid(cases, jobs_by_case, seeds, mappings, zones, hidden_pool,
             methods, ctx_for, log_every=None):
    """Paired grid: every (case, seed, mapping) is replayed under every method.

    ctx_for(method) -> context dict for `make_decider`.  Returns the row list and
    the maximum |official - independent| metric difference.
    """
    rows = []
    max_diff = 0.0
    n = 0
    for case in cases:
        jobs = jobs_by_case[case["case_id"]]
        for seed in seeds:
            hidden = wrap_hidden(hidden_for(jobs, seed, hidden_pool))
            for alpha, beta in mappings:
                for method in methods:
                    decide = make_decider(method, ctx_for(method))
                    m, diff, steps = replay(jobs, hidden, zones, case, alpha, beta, decide)
                    assert not m["invalid_actions"], (case["case_id"], seed, alpha, beta, method)
                    assert diff < 1e-9, (case["case_id"], seed, alpha, beta, method, diff)
                    max_diff = max(max_diff, diff)
                    rows.append(dict(case=case["case_id"], seed=seed, alpha=alpha, beta=beta,
                                     method=method,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                                     weighted_completion_rate=float(m["weighted_completion_rate"]),
                                     cost=float(m["cost"]), observations=float(m["observations"]),
                                     actions=steps))
                    n += 1
            if log_every and n % log_every == 0:
                print("replays", n, flush=True)
    return rows, max_diff
