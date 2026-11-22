"""Role E: depth-1 expected lookahead with a hierarchical robust duration belief.

Decision core (fixed across all belief variants, so that any difference is
attributable to the duration belief / parameter choice):

    score(j) = mean_q [ 1{j on time} * w_j
                        + greedy_tail(t = start + p_j(q), zone_j, done U {j}) ]

Correctness points required by the audit:
  * clock advances to the completion instant `start + p` (never `start + D + .5 + p`);
  * chain value is gated per job by its *ancestor* completion probability, so
    parallel branches are not multiplied together;
  * budget feasibility is screened before ranking, never used to abort a tail;
  * the belief lives on a fixed discrete support with **exact float mixture
    weights**; survival is exact on that support and therefore a discrete
    approximation of the continuous prior (no "exact weight preservation" claim);
  * `feedback_strength <= 0` means "do not use feedback" (prior only).

Belief hierarchy (train split only; no case id, hidden duration or target value):
    family component  --shrink strength-->  global prior component
  `balance` sets how each level aggregates training units:
    pooled : equal weight per sample
    env    : equal weight per environment
    region : equal weight per region (top-level domain of the environment id)
  `winsor_q` clips each component's upper tail at that train quantile.

No case id, hidden duration, target value or paid observation is read.  `alpha`
and `beta` are public in `observe()` and are used legally.
"""
import math

import numpy as np

GRID = 64


class Belief:
    """Discrete weighted distribution over service durations (source seconds)."""

    __slots__ = ("values", "weights")

    def __init__(self, values, weights):
        values = np.asarray(values, float)
        weights = np.asarray(weights, float)
        weights = weights / weights.sum()
        order = np.argsort(values, kind="mergesort")
        self.values = values[order]
        self.weights = weights[order]

    def survival(self, slack):
        return float(self.weights[self.values <= slack + 1e-12].sum())

    def mean(self):
        return float((self.values * self.weights).sum())

    def quantiles(self, qs):
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0
        idx = np.searchsorted(cdf, np.asarray(qs, float), side="left")
        return self.values[np.clip(idx, 0, len(self.values) - 1)].copy()

    def mapped(self, alpha, beta):
        return Belief(alpha * self.values + beta, self.weights)


def _empirical_support(values, grid=GRID, winsor_q=None):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("empty duration component")
    if winsor_q is not None:
        v = np.minimum(v, float(np.quantile(v, winsor_q)))
    qs = (np.arange(grid) + 0.5) / grid
    return np.quantile(v, qs), np.full(grid, 1.0 / grid)


def _combine(parts, weights):
    vals, wts = [], []
    for part, wt in zip(parts, weights):
        if wt <= 0:
            continue
        if isinstance(part, Belief):
            v, w = part.values, part.weights
        else:
            v, w = part
        v = np.asarray(v, float)
        w = np.asarray(w, float)
        vals.append(v)
        wts.append(w * (wt / w.sum()))
    return Belief(np.concatenate(vals), np.concatenate(wts))


def _unit_parts(records, family, unit, grid, winsor_q):
    """Components per environment / region; each unit contributes equally."""
    buckets = {}
    for env, region, fam, vals in records:
        if family is not None and fam != family:
            continue
        key = env if unit == "env" else region
        buckets.setdefault(key, []).append(np.asarray(vals, float))
    out = []
    for key in sorted(buckets):
        vals = np.concatenate(buckets[key])
        if vals.size:
            out.append(_empirical_support(vals, grid, winsor_q))
    return out


def build_priors(records, strength=20.0, balance="pooled", winsor_q=None, grid=GRID):
    """Return {family: Belief} fitted on one training split only.

    records: iterable of (environment_id, region, family_id, values_array).
    """
    families = sorted({r[2] for r in records})
    if balance == "env":
        parts = _unit_parts(records, None, "env", grid, winsor_q)
        global_component = _combine(parts, [1.0] * len(parts))
    elif balance == "region":
        parts = _unit_parts(records, None, "region", grid, winsor_q)
        global_component = _combine(parts, [1.0] * len(parts))
    elif balance == "pooled":
        global_component = _combine([_empirical_support(np.concatenate(
            [np.asarray(r[3], float) for r in records]), grid, winsor_q)], [1.0])
    else:
        raise ValueError(balance)

    priors = {}
    for fam in families:
        fam_values = np.concatenate([np.asarray(r[3], float) for r in records if r[2] == fam])
        if balance == "pooled":
            family_component = _combine([_empirical_support(fam_values, grid, winsor_q)], [1.0])
        else:
            parts = _unit_parts(records, fam, balance, grid, winsor_q)
            family_component = _combine(parts, [1.0] * len(parts))
        n = fam_values.size
        w = strength / (n + strength)
        priors[fam] = _combine([family_component, global_component], [1.0 - w, w])
    return priors


def refine(belief, samples, feedback_strength, grid=GRID, winsor_q=None):
    """Fold completed in-trial durations into the belief with weight n/(n+fb)."""
    if feedback_strength is None or feedback_strength <= 0 or not samples:
        return belief
    h = _empirical_support(samples, grid, winsor_q)
    w = len(samples) / (len(samples) + feedback_strength)
    return _combine([(belief.values, belief.weights), h], [1.0 - w, w])


def _ancestors(job_id, preds, cache):
    if job_id in cache:
        return cache[job_id]
    out = set()
    stack = list(preds.get(job_id, ()))
    while stack:
        a = stack.pop()
        if a in out:
            continue
        out.add(a)
        stack.extend(preds.get(a, ()))
    cache[job_id] = out
    return out


def _greedy_tail(t, xy, done, remaining, jobs, beliefs, H, B, cost, preds, anc_cache):
    """Mean-time greedy rollout; per-job value gated by ancestor completion probability."""
    got = 0.0
    remaining = set(remaining)
    done = set(done)
    xy = np.asarray(xy, float)
    pcomp = {d: 1.0 for d in done}
    for _ in range(64):
        if not remaining:
            break
        best = None
        for jid in remaining:
            j = jobs[jid]
            if not preds[jid] <= done or j["release_s"] > t + 1e-12:
                continue
            attempted = 1.0
            for a in _ancestors(jid, preds, anc_cache):
                if a in remaining:
                    attempted *= pcomp.get(a, 0.0)
            if attempted <= 1e-9:
                continue
            jz = np.asarray(j["zone_xy"], float)
            D = math.dist(xy, jz)
            start = t + D + 0.5
            b = beliefs[j["family"]]
            mp = b.mean()
            if start + mp > H + 1e-12:
                continue
            s = b.survival(min(j["due_s"], H) - start)
            if s <= 1e-9:
                continue
            inc = D + 0.5 + mp
            fee = 2 + 0.1 * 0.2 + 0.2 * D
            if cost + fee > B + 1e-12:
                continue
            cand = (-attempted * j["weight"] * s / inc, j["due_s"], jid, inc, s,
                    b.survival(H - start), fee, attempted)
            if best is None or cand < best:
                best = cand
        if best is None:
            future = [jobs[k]["release_s"] for k in remaining
                      if preds[k] <= done and jobs[k]["release_s"] > t + 1e-12
                      and jobs[k]["release_s"] < H]
            if future:
                t = min(future)
                continue
            break
        _, _, jid, inc, s, comp, fee, attempted = best
        got += attempted * jobs[jid]["weight"] * s
        pcomp[jid] = attempted * comp
        cost += fee
        t += inc
        xy = np.asarray(jobs[jid]["zone_xy"], float)
        done.add(jid)
        remaining.discard(jid)
    return got


def select(o, prior, history=None, feedback_strength=3.0, winsor_feedback=None,
           nq_root=5, grid=GRID):
    """Return the next action under the fixed depth-1 expected-lookahead rule."""
    history = history or {}
    beliefs = {}
    for fam, b in prior.items():
        bb = refine(b, history.get(fam), feedback_strength, grid=grid, winsor_q=winsor_feedback)
        beliefs[fam] = bb.mapped(o["alpha"], o["beta"])
    jobs = {j["job_id"]: j for j in o["jobs"]}
    preds = {jid: set(j["predecessors"]) for jid, j in jobs.items()}
    anc_cache = {}
    done = {j["job_id"] for j in o["jobs"] if j["completed"]}
    avail = [j for j in o["jobs"]
             if not j["attempted"] and j["release_s"] <= o["time"] + 1e-12
             and set(j["predecessors"]) <= done]
    if not avail:
        future = [j["release_s"] for j in o["jobs"]
                  if not j["attempted"] and o["time"] + 1e-12 < j["release_s"] <= o["H"]]
        return {"type": "wait", "duration": min(future) - o["time"]} if future else {"type": "stop"}

    remaining_all = [j["job_id"] for j in o["jobs"] if not j["attempted"]]
    qs = (np.arange(nq_root) + 0.5) / nq_root
    best = None
    for j in avail:
        D = math.dist(o["zone_xy"], j["zone_xy"])
        start = o["time"] + D + 0.5
        b = beliefs[j["family"]]
        fee = 2 + 0.02 + 0.2 * D
        if start + 0.2 > o["H"] + 1e-12 or o["cost"] + fee > o["B"] + 1e-12:
            continue
        after = [r for r in remaining_all if r != j["job_id"]]
        total = 0.0
        for pq in b.quantiles(qs):
            pq = max(float(pq), 0.2)
            comp = start + pq <= o["H"] + 1e-12
            t2 = min(o["H"], start + pq)
            imm = j["weight"] if (comp and start + pq <= j["due_s"] + 1e-12) else 0.0
            done2 = done | {j["job_id"]} if comp else done
            total += imm + _greedy_tail(t2, j["zone_xy"], done2, after, jobs, beliefs,
                                        o["H"], o["B"], o["cost"] + fee, preds, anc_cache)
        score = total / len(qs)
        cand = (-score, j["due_s"], j["job_id"])
        if best is None or cand < best:
            best = cand
    if best is not None:
        return {"type": "execute", "job_id": best[2], "reserve": 0.2}
    future = [j["release_s"] for j in o["jobs"]
              if not j["attempted"] and o["time"] + 1e-12 < j["release_s"] <= o["H"]]
    return {"type": "wait", "duration": min(future) - o["time"]} if future else {"type": "stop"}
