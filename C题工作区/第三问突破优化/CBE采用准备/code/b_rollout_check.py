"""Compare B vectorized root values with independent scalar Engine rollouts."""
import json
import math
from pathlib import Path

import numpy as np
import runtime as rt
from engine import Engine


def base_action(e, beliefs, jobs):
    obs = e.observe()
    done = {j["job_id"] for j in obs["jobs"] if j["completed"]}
    attempted = {j["job_id"] for j in obs["jobs"] if j["attempted"]}
    best = None
    for i, j in enumerate(jobs):
        jid = j["job_id"]
        if jid in attempted or j["release_s"] > e.time + 1e-12 or not set(j["predecessors"]) <= done:
            continue
        D = math.dist(e.zones[e.zone], e.zones[j["zone_id"]])
        s = e.time + D + .5
        if s + .2 > e.H + 1e-12 or e.cost + 2.02 + .2 * D > e.B + 1e-12:
            continue
        p = beliefs[j["family_id"]]
        mean = float(p.mean())
        value = j["priority_weight"] * float(np.mean(p <= min(j["due_s"], e.H) - s + 1e-12))
        for child in jobs:
            if jid in child["predecessors"] and child["job_id"] not in attempted:
                travel = math.dist(e.zones[j["zone_id"]], e.zones[child["zone_id"]]) + .5
                q = beliefs[child["family_id"]]
                chance = float(np.mean(q <= min(child["due_s"], e.H) - (s + mean + travel) + 1e-12))
                value += child["priority_weight"] * chance
        score = value / (D + .5 + mean)
        key = (-score, i)
        if score > 0 and (best is None or key < best[0]):
            best = (key, {"type": "execute", "job_id": jid, "reserve": .2})
    if best is not None:
        return best[1]
    future = [j["release_s"] for j in jobs if j["job_id"] not in attempted
              and set(j["predecessors"]) <= done and e.time + 1e-12 < j["release_s"] <= e.H + 1e-12]
    return {"type": "wait", "duration": min(future) - e.time} if future else {"type": "stop"}


def main():
    rng = np.random.default_rng(2026112301)
    zones = {"Z0": (0., 0.), "Z1": (2., 0.), "Z2": (0., 3.)}
    errors = []
    for trial in range(64):
        n, K = int(rng.integers(2, 7)), 8
        jobs = [dict(job_id=f"J{i}", family_id="A" if i % 2 else "P",
                     release_s=0. if i == 0 else float(rng.integers(0, 15)),
                     due_s=float(rng.integers(4, 30)), priority_weight=float(rng.integers(1, 7)),
                     zone_id=list(zones)[int(rng.integers(0, 3))],
                     predecessors=[f"J{i-1}"] if i and rng.random() < .3 else []) for i in range(n)]
        H, B = float(rng.integers(15, 35)), float(rng.integers(5, 22))
        pools = {"A": rng.uniform(.03, 20., size=12), "P": rng.uniform(.03, 25., size=16)}
        beliefs = {f: np.sort(rt.b_policy.mixture(f, pools, {}, 20, 3)) for f in pools}
        pol = rt.b_policy.JointScenarioPolicy(K=K, strength=20, feedback_strength=3,
                                              seed=20261122, base="survival", stratified=True)
        srng = np.random.default_rng(20261122)
        samples = {}
        for j in jobs:
            vals = beliefs[j["family_id"]]
            u = (np.arange(K) + srng.random(K)) / K
            idx = np.minimum((u * len(vals)).astype(int), len(vals) - 1)
            samples[j["job_id"]] = vals[srng.permutation(idx)]
        e = Engine(jobs, rt.acc.ar.wrap({j["job_id"]: 1. for j in jobs}), zones, H, B, 0, 1., 0.)
        pol.select(e.observe(), pools, {})
        cands = []
        for i, j in enumerate(jobs):
            D = math.dist(zones["Z0"], zones[j["zone_id"]])
            if not j["predecessors"] and j["release_s"] == 0 and D + .5 + .2 <= H and 2.02 + .2 * D <= B:
                cands.append((i, {"type": "execute", "job_id": j["job_id"], "reserve": .2}))
        future = [j["release_s"] for j in jobs if 0 < j["release_s"] <= H]
        if future:
            cands.append((-1, {"type": "wait", "duration": min(future)}))
        scalar_values = []
        for _, first in cands:
            outcomes = []
            for k in range(K):
                hidden = rt.acc.ar.wrap({jid: float(draw[k]) for jid, draw in samples.items()})
                e = Engine(jobs, hidden, zones, H, B, 0, 1., 0.)
                e.step(first)
                while not e.terminal:
                    e.step(base_action(e, beliefs, jobs))
                outcomes.append(e.metrics()["weighted_on_time_rate"] * sum(j["priority_weight"] for j in jobs))
            scalar_values.append(float(np.mean(outcomes)))
        assert [c[0] for c in cands] == pol.last_candidates
        errors.extend(abs(x - y) for x, y in zip(scalar_values, pol.last_values))
    assert max(errors) < 1e-9
    out = Path(__file__).resolve().parents[1] / "results" / "b_rollout"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"synthetic_instances": 64, "root_candidates_checked": len(errors),
               "scenarios_per_root": 8, "max_absolute_value_error": max(errors),
               "scalar_oracle": "formal Engine; independently computed survival density tail"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
