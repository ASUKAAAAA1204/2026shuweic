"""Information-value scheduling policy (role D).

Base decision rule is the v2_s20 ancestor-gated multi-step lookahead (verbatim
copy of multi_step_policy.select with depth/strength/beam defaults), extended so
that each job can carry its own conditional duration belief derived from the
disclosed prefix.  Two information channels are controlled separately:

  free  : the 0.2 s prefix, released automatically with the job (no cost);
  paid  : 0.6 s and 1.0 s prefixes, obtained by a review action (1 s, 1 cost).

The purchase rule is fixed a priori and uses only the policy's own beliefs:
it buys one review for the top-ranked job when (a) reviews remain, (b) the job's
on-time probability is uncertain (0.15..0.85), (c) the job's due time falls
inside its predictive 20..80% duration interval, (d) the 1 s delay still leaves
room to execute on time, and (e) the budget still covers review + service.

No hidden duration, case id, target value or validation outcome is read.
"""
import math

import numpy as np

_GRID = 128
PREFIXES = (.2, .6, 1.0)


def mixture(family, pools, history, strength, feedback_strength):
    global_values = np.concatenate([np.asarray(v, float) for v in pools.values()])
    v = np.asarray(pools[family], float)
    prior_count = int(round(_GRID * strength / (len(v) + strength)))
    values = np.r_[
        np.quantile(global_values, (np.arange(prior_count) + .5) / prior_count) if prior_count else [],
        np.quantile(v, (np.arange(_GRID - prior_count) + .5) / (_GRID - prior_count)),
    ]
    h = history.get(family) if history else None
    if h and feedback_strength > 0:
        h = np.asarray(h, float)
        count = int(round(_GRID * len(h) / (len(h) + feedback_strength)))
        values = np.r_[
            np.quantile(h, (np.arange(count) + .5) / count),
            np.quantile(values, (np.arange(_GRID - count) + .5) / (_GRID - count)) if count < _GRID else [],
        ]
    return values


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


def _greedy_tail(t, xy, done, remaining, jobs, pdists, job_dists, H, B, cost, preds, anc_cache):
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
            if not preds[jid] <= done or j['release_s'] > t + 1e-12:
                continue
            attempted = 1.0
            for a in _ancestors(jid, preds, anc_cache):
                if a in remaining:
                    attempted *= pcomp.get(a, 0.0)
            if attempted <= 1e-9:
                continue
            jz = np.asarray(j['zone_xy'], float)
            D = math.dist(xy, jz)
            start = t + D + .5
            p = job_dists.get(jid, pdists[j['family']])
            mp = float(np.mean(p))
            if start + mp > H + 1e-12:
                continue
            s = float(np.mean(p <= min(j['due_s'], H) - start + 1e-12))
            if s <= 1e-9:
                continue
            inc = D + .5 + mp
            fee = 2 + .1 * .2 + .2 * D
            if cost + fee > B + 1e-12:
                continue
            cand = (-attempted * j['weight'] * s / inc, j['due_s'], jid, inc, s,
                    float(np.mean(p <= H - start + 1e-12)), fee, attempted)
            if best is None or cand < best:
                best = cand
        if best is None:
            future = [jobs[k]['release_s'] for k in remaining
                      if preds[k] <= done and jobs[k]['release_s'] > t + 1e-12 and jobs[k]['release_s'] < H]
            if future:
                t = min(future)
                continue
            break
        _, _, jid, inc, s, comp, fee, attempted = best
        got += attempted * jobs[jid]['weight'] * s
        pcomp[jid] = attempted * comp
        cost += fee
        t += inc
        xy = np.asarray(jobs[jid]['zone_xy'], float)
        done.add(jid)
        remaining.discard(jid)
    return got


def _value(t, xy, done, remaining, cost, depth, jobs, pdists, job_dists, H, B, nq, beam, preds, anc_cache, is_root):
    remaining = set(remaining)
    if depth <= 0 or not remaining:
        return _greedy_tail(t, xy, done, remaining, jobs, pdists, job_dists, H, B, cost, preds, anc_cache)
    cands = []
    for jid in remaining:
        j = jobs[jid]
        if not preds[jid] <= done or j['release_s'] > t + 1e-12:
            continue
        jz = np.asarray(j['zone_xy'], float)
        D = math.dist(xy, jz)
        start = t + D + .5
        p = job_dists.get(jid, pdists[j['family']])
        mp = float(np.mean(p))
        if start + mp > H + 1e-12:
            continue
        fee = 2 + .1 * .2 + .2 * D
        if cost + fee > B + 1e-12:
            continue
        on = float(np.mean(p <= min(j['due_s'], H) - start + 1e-12))
        if on <= 1e-9:
            continue
        rank = j['weight'] * on + _greedy_tail(start + mp, jz, done | {jid}, remaining - {jid},
                                               jobs, pdists, job_dists, H, B, cost + fee, preds, anc_cache)
        cands.append((rank, jid, jz, start, p, fee))
    if not cands:
        future = [jobs[k]['release_s'] for k in remaining
                  if preds[k] <= done and jobs[k]['release_s'] > t + 1e-12 and jobs[k]['release_s'] < H]
        if future:
            t2 = min(future)
            if t2 < H:
                return _value(t2, xy, done, remaining, cost, depth, jobs, pdists, job_dists,
                              H, B, nq, beam, preds, anc_cache, is_root)
        return 0.0
    cands.sort(key=lambda c: -c[0])
    if not is_root:
        cands = cands[:beam]
    grid = (np.arange(nq) + .5) / nq
    best = 0.0
    for _, jid, jz, start, p, fee in cands:
        j = jobs[jid]
        qvals = np.quantile(p, grid)
        total = 0.0
        for pq in qvals:
            pq = max(float(pq), .2)
            comp = start + pq <= H + 1e-12
            t2 = min(H, start + pq)
            imm = j['weight'] if (comp and start + pq <= j['due_s'] + 1e-12) else 0.0
            done2 = done | {jid} if comp else done
            total += imm + _value(t2, jz, done2, remaining - {jid}, cost + fee, depth - 1,
                                  jobs, pdists, job_dists, H, B, nq, beam, preds, anc_cache, False)
        val = total / len(qvals)
        if val > best:
            best = val
    return best


def _job_dists(o, pools, history, model, use_free, strength, feedback_strength):
    """Scenario-seconds belief per job from whatever information is in `o`."""
    base = {f: o['alpha'] * mixture(f, pools, history, strength, feedback_strength) + o['beta'] for f in pools}
    fam_med = {f: float(np.median(pools[f])) for f in pools}
    out = {}
    for j in o['jobs']:
        # Free 0.2 s information exists once the job is released; paid prefixes only
        # count in the "paid-only" arm after at least one review.
        visible = (o['time'] + 1e-12 >= j['release_s'])
        if not visible:
            continue
        if not (use_free or j['reviews'] > 0):
            continue
        feats = j.get('features')
        if feats is None or len(feats) == 0:
            continue
        prefix = PREFIXES[min(j['reviews'], 2)]
        d = model.duration_dist_from_features(np.asarray(feats, float), prefix, j['family'], fam_med[j['family']])
        out[j['job_id']] = o['alpha'] * d + o['beta']
    return base, out


def purchase(o, pools, history, model, use_free, allow_paid, strength=20, feedback_strength=3):
    """Return a review action or None.  Rule is fixed in the module docstring."""
    if not allow_paid or o['reviews'] >= o['K'] or o['time'] + 1 > o['H'] - 1e-12:
        return None
    base, job_dists = _job_dists(o, pools, history, model, use_free, strength, feedback_strength)
    done = {j['job_id'] for j in o['jobs'] if j['completed']}
    cands = []
    for j in o['jobs']:
        if j['attempted'] or j['release_s'] > o['time'] + 1e-12 or not set(j['predecessors']) <= done:
            continue
        if j['reviews'] >= 2:
            continue
        D = math.dist(o['zone_xy'], j['zone_xy'])
        start = o['time'] + D + .5
        p = job_dists.get(j['job_id'], base[j['family']])
        mp = float(np.mean(p))
        if start + .2 > o['H'] + 1e-12 or o['cost'] + 1 + 2.02 + .2 * D > o['B'] + 1e-12:
            continue
        on = float(np.mean(p <= min(j['due_s'], o['H']) - start + 1e-12))
        score = j['weight'] * on / (D + .5 + mp)
        cands.append((score, j['due_s'], j['job_id'], j, D, start, p))
    if not cands:
        return None
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    score, _, jid, j, D, start, p = cands[0]
    if len(cands) > 1:
        second = cands[1][0]
        ambiguity = score <= 1e-12 or second / score > .85
    else:
        ambiguity = False
    on = float(np.mean(p <= min(j['due_s'], o['H']) - start + 1e-12))
    if not (ambiguity or .35 <= on <= .65):
        return None
    if not (.15 <= on <= .85):
        return None
    slack = min(j['due_s'], o['H']) - start
    lo = float(np.quantile(p, .2))
    hi = float(np.quantile(p, .8))
    if not (lo <= slack <= hi):
        return None
    start1 = o['time'] + 1 + D + .5
    if start1 + .2 > o['H'] + 1e-12 or o['cost'] + 1 + 2.02 + .2 * D > o['B'] + 1e-12:
        return None
    return {'type': 'review', 'job_id': jid}


def select(o, pools, history=None, model=None, use_free=False, allow_paid=False,
           strength=20, feedback_strength=3, depth=1, nq_root=5, nq=2, beam=3,
           prefix_cache=None, job_dists_override=None):
    """Return the next action; optionally purchase a review before deciding.

    `job_dists_override` exists only for the analysis-only perfect-information
    bound (see run_oracle.py); deployable arms never pass it.
    """
    history = history or {}
    if allow_paid:
        review = purchase(o, pools, history, model, use_free, allow_paid, strength, feedback_strength)
        if review is not None:
            return review
    if job_dists_override is not None:
        pdists = {f: o['alpha'] * mixture(f, pools, history, strength, feedback_strength) + o['beta']
                  for f in pools}
        job_dists = job_dists_override
    elif prefix_cache is not None:
        key = (o['time'], tuple((j['job_id'], j['reviews']) for j in o['jobs']))
        if key in prefix_cache:
            pdists, job_dists = prefix_cache[key]
        else:
            pdists, job_dists = _job_dists(o, pools, history, model, use_free, strength, feedback_strength)
            prefix_cache[key] = (pdists, job_dists)
    else:
        pdists, job_dists = _job_dists(o, pools, history, model, use_free, strength, feedback_strength)

    jobs = {j['job_id']: j for j in o['jobs']}
    preds = {jid: set(j['predecessors']) for jid, j in jobs.items()}
    anc_cache = {}
    done = {j['job_id'] for j in o['jobs'] if j['completed']}
    avail = [j for j in o['jobs']
             if not j['attempted'] and j['release_s'] <= o['time'] + 1e-12 and set(j['predecessors']) <= done]
    if not avail:
        future = [j['release_s'] for j in o['jobs']
                  if not j['attempted'] and o['time'] + 1e-12 < j['release_s'] <= o['H']]
        return {'type': 'wait', 'duration': min(future) - o['time']} if future else {'type': 'stop'}

    remaining_all = [j['job_id'] for j in o['jobs'] if not j['attempted']]
    grid = (np.arange(nq_root) + .5) / nq_root
    best = None
    for j in avail:
        D = math.dist(o['zone_xy'], j['zone_xy'])
        start = o['time'] + D + .5
        p = job_dists.get(j['job_id'], pdists[j['family']])
        fee = 2 + .02 + .2 * D
        if start + .2 > o['H'] + 1e-12 or o['cost'] + fee > o['B'] + 1e-12:
            continue
        after = [r for r in remaining_all if r != j['job_id']]
        total = 0.0
        for pq in np.quantile(p, grid):
            pq = max(float(pq), .2)
            comp = start + pq <= o['H'] + 1e-12
            t2 = min(o['H'], start + pq)
            imm = j['weight'] if (comp and start + pq <= j['due_s'] + 1e-12) else 0.0
            done2 = done | {j['job_id']} if comp else done
            total += imm + _value(t2, j['zone_xy'], done2, after, o['cost'] + fee, depth - 1,
                                  jobs, pdists, job_dists, o['H'], o['B'], nq, beam, preds, anc_cache, False)
        score = total / len(grid)
        cand = (-score, j['due_s'], j['job_id'])
        if best is None or cand < best:
            best = cand
    if best is not None:
        return {'type': 'execute', 'job_id': best[2], 'reserve': .2}
    future = [j['release_s'] for j in o['jobs']
              if not j['attempted'] and o['time'] + 1e-12 < j['release_s'] <= o['H']]
    return {'type': 'wait', 'duration': min(future) - o['time']} if future else {'type': 'stop'}
