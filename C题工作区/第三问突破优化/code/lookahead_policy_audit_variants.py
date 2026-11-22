"""Risk-integrated one-step lookahead with a greedy expected-time tail critic.

Decision model
--------------
For every currently available job j the policy scores the *whole remaining
horizon* rather than a single index:

    score(j) = w_j * P(j on time) + E_tail(after j)

`E_tail` is a greedy rollout of the remaining jobs under the current duration
belief, accumulating only the weight that this rollout expects to finish on
time.  The expectation over j's own duration is taken by integrating over a
fixed quantile grid, so a long j correctly shrinks the value of its successors
instead of being replaced by a point estimate.

Duration belief
---------------
Family empirical distribution, shrunk toward a global prior by `strength` and
updated from in-trial completed durations with `feedback_strength`.  All
mixtures are expressed on a fixed 128-point quantile grid so that every weight
in the mixture is preserved exactly.  Durations are mapped to scenario seconds
with the public load `p = alpha * d + beta` before any survival check.

No case id, hidden duration, paid observation, target value, or validation
outcome is used.  `observe()` exposes `alpha`/`beta`, so conditioning on them is
legal.
"""
import math
import numpy as np

_GRID = 128


def _mixture(family, pools, history, strength, feedback_strength):
    """Fixed-grid mixture of the family pool, global prior and completed feedback."""
    global_values = np.concatenate([np.asarray(v, float) for v in pools.values()])
    v = np.asarray(pools[family], float)
    prior_count = int(round(_GRID * strength / (len(v) + strength)))
    values = np.r_[
        np.quantile(global_values, (np.arange(prior_count) + .5) / prior_count) if prior_count else [],
        np.quantile(v, (np.arange(_GRID - prior_count) + .5) / (_GRID - prior_count)),
    ]
    h = history.get(family) if history else None
    if h:
        h = np.asarray(h, float)
        if feedback_strength <= 0:
            return h
        count = int(round(_GRID * len(h) / (len(h) + feedback_strength)))
        values = np.r_[
            np.quantile(h, (np.arange(count) + .5) / count),
            np.quantile(values, (np.arange(_GRID - count) + .5) / (_GRID - count)) if count < _GRID else [],
        ]
    return values


def _survival(p, slack):
    return float(np.mean(p <= slack + 1e-12))


def _tail_value(t, xy, done, remaining, jobs, pdists, H, B, cost, fix_clock=False, fix_chain=False, fix_budget=False):
    """Greedy expected-time rollout collecting the on-time weight of the remaining jobs."""
    got = 0.0
    path_probability = 1.0
    remaining = set(remaining)
    done = set(done)
    xy = np.asarray(xy, float)
    for _ in range(64):
        if not remaining or (fix_chain and path_probability < 1e-8):
            break
        best = None
        for jid in remaining:
            j = jobs[jid]
            if not set(j['predecessors']) <= done or j['release_s'] > t + 1e-12:
                continue
            jz = np.asarray(j['zone_xy'], float)
            D = math.dist(xy, jz)
            start = t + D + .5
            p = pdists[j['family']]
            occ = D + .5 + float(np.mean(p))
            duration = max(float(np.mean(p)), .2)
            screen = duration if fix_clock else occ
            if start + screen > H + 1e-12:
                continue
            s = _survival(p, min(j['due_s'], H) - start)
            completion_probability = _survival(p, H - start)
            if s <= 1e-9:
                continue
            fee = 2 + .1 * .2 + .2 * D
            if fix_budget and cost + fee > B + 1e-12:
                continue
            cand = (-j['weight'] * s / occ, j['due_s'], jid, start, occ, s, completion_probability, fee)
            if best is None or cand < best:
                best = cand
        if best is None:
            future = [jobs[k]['release_s'] for k in remaining
                      if set(jobs[k]['predecessors']) <= done
                      and jobs[k]['release_s'] > t + 1e-12 and jobs[k]['release_s'] < H]
            if future:
                t = min(future)
                continue
            break
        _, _, jid, start, occ, s, completion_probability, fee = best
        j = jobs[jid]
        got += (path_probability if fix_chain else 1.0) * j['weight'] * s
        if fix_chain:
            path_probability *= completion_probability
        jz = np.asarray(j['zone_xy'], float)
        cost += fee
        if not fix_budget and cost > B + 1e-12:
            break
        t = start + (max(float(np.mean(pdists[j['family']])), .2) if fix_clock else occ)
        xy = jz
        done.add(jid)
        remaining.discard(jid)
    return got


def select(o, pools, history=None, strength=40, feedback_strength=3, quantiles=5, fix_clock=False, fix_chain=False, fix_budget=False):
    """Return the next action; see the module docstring for the scoring rule."""
    history = history or {}
    raw = {f: _mixture(f, pools, history, strength, feedback_strength) for f in pools}
    pdists = {f: o['alpha'] * raw[f] + o['beta'] for f in pools}
    jobs = {j['job_id']: j for j in o['jobs']}
    done = {j['job_id'] for j in o['jobs'] if j['completed']}
    available = [j for j in o['jobs']
                 if not j['attempted'] and j['release_s'] <= o['time'] + 1e-12
                 and set(j['predecessors']) <= done]
    if not available:
        future = [j['release_s'] for j in o['jobs']
                  if not j['attempted'] and o['time'] + 1e-12 < j['release_s'] <= o['H']]
        return {'type': 'wait', 'duration': min(future) - o['time']} if future else {'type': 'stop'}

    remaining_all = [j['job_id'] for j in o['jobs'] if not j['attempted']]
    qs = [.5] if quantiles <= 1 else list((np.arange(quantiles) + .5) / quantiles)
    best = None
    for j in available:
        D = math.dist(o['zone_xy'], j['zone_xy'])
        start = o['time'] + D + .5
        p = pdists[j['family']]
        if start + .2 > o['H'] + 1e-12 or o['cost'] + 2.02 + .2 * D > o['B'] + 1e-12:
            continue
        immediate = j['weight'] * _survival(p, min(j['due_s'], o['H']) - start)
        future_cost = o['cost'] + 2 + .02 + .2 * D
        after = [r for r in remaining_all if r != j['job_id']]
        tails = [
            _tail_value(start + max(float(np.quantile(p, q)), .2), j['zone_xy'],
                        done | {j['job_id']}, after, jobs, pdists, o['H'], o['B'], future_cost, fix_clock, fix_chain, fix_budget)
            for q in qs
        ]
        score = immediate + float(np.mean(tails))
        cand = (-score, j['due_s'], j['job_id'])
        if best is None or cand < best:
            best = cand
    if best is not None:
        return {'type': 'execute', 'job_id': best[2], 'reserve': .2}
    future = [j['release_s'] for j in o['jobs']
              if not j['attempted'] and o['time'] + 1e-12 < j['release_s'] <= o['H']]
    return {'type': 'wait', 'duration': min(future) - o['time']} if future else {'type': 'stop'}
