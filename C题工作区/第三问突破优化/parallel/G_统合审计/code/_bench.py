"""Timing benchmark for candidate policies (not a deliverable)."""
import sys, time, random, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"d:\CQUPT\self_file\competition\2026shuwei\C题工作区")
FORMAL = ROOT / "第三问改进正式方案" / "code"
BREAK = ROOT / "第三问突破优化"
ROLEA = BREAK / "parallel" / "A_多步前瞻" / "code"
for p in (str(FORMAL), str(BREAK / "code"), str(ROLEA)):
    sys.path.insert(0, p)

from support import load, jobs_for
from engine import Engine, independent_metrics
from deadline_policy import choose, pressure
from baseline import action
import multi_step_policy as ms
import lookahead_policy_audit_variants as audit

cond, tasks, zones, train = load()
seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
pools = {f: g.duration_s.to_numpy(float) for f, g in train.groupby("family_id")}
med = {f: float(np.median(v)) for f, v in pools.items()}


def hidden_for(jobs, seed):
    rng = random.Random(seed)
    out = {}
    for j in jobs:
        c = vpool[vpool.family_id == j["family_id"]]
        out[j["job_id"]] = float(c.iloc[rng.randrange(len(c))].duration_s)
    return out


def run(case, seed, method):
    jobs = jobs_for(tasks, case["case_id"])
    dur = hidden_for(jobs, seed)
    hidden = {k: dict(duration=v, features={str(p): [] for p in (.2, .6, 1.)}) for k, v in dur.items()}
    e = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"], case["observation_limit"], 1., .5)
    history = {}
    ndec = 0
    t0 = time.perf_counter()
    while not e.terminal:
        o = e.observe(); ndec += 1
        if method == "baseline":
            a = choose(o, pools, history) if pressure(o, pools, history) else action(o, med, "density", "minimum")
        elif method == "audit":
            a = audit.select(o, pools, history, strength=40, feedback_strength=3, quantiles=5,
                             fix_clock=True, fix_chain=True, fix_budget=True)
        elif method == "v2":
            a = ms.select(o, pools, history, strength=20, feedback_strength=3, depth=1,
                          nq_root=5, nq=2, beam=3)
        elif method == "d2b4":
            a = ms.select(o, pools, history, strength=20, feedback_strength=3, depth=2,
                          nq_root=5, nq=2, beam=4)
        elif method == "d3b4":
            a = ms.select(o, pools, history, strength=20, feedback_strength=3, depth=3,
                          nq_root=5, nq=2, beam=4)
        ev = e.step(a)
        if a["type"] == "execute" and ev.get("completion") is not None:
            f = next(j["family_id"] for j in jobs if j["job_id"] == a["job_id"])
            history.setdefault(f, []).append((ev["completion"] - ev["service_start"] - .5) / 1.)
    return time.perf_counter() - t0, ndec, e.metrics()["weighted_on_time_rate"]


if __name__ == "__main__":
    for method in ("baseline", "audit", "v2", "d2b4", "d3b4"):
        tot = 0.; nd = 0; rows = []
        for case in cond["cases"]:
            dt, n, ot = run(case, cond["validation_seeds"][0], method)
            tot += dt; nd += n; rows.append(f"{case['case_id']}={ot:.4f}/{dt:.2f}s")
        print(f"{method:9s} total={tot:7.2f}s decisions={nd:4d}  " + " ".join(rows), flush=True)
