"""Vendored online reference policy for the role-F feasibility study.

Provenance: verbatim copy of
``C题工作区/第三问突破优化/parallel/A_多步前瞻/code/multi_step_policy.py``
taken on 2026-11-22, used only as a legal *online lower-bound reference*
(the "v2_s20" configuration: depth=1, strength=20, feedback_strength=3,
nq_root=5, nq=2, beam=3).  It is not a contribution of role F and all numbers
reported by role F come from re-running it under the shared protocol.

Depth-D expected lookahead.  At every decision the policy maximises

    V(t, xy, done, remaining, cost, D)
      = max over feasible j of  E_{p_j}[ 1{j on time} * w_j
                                       + V(t', xy_j, done', remaining\\{j}, cost', D-1) ]

with `t' = min(H, start + p_j)` and `done' = done ∪ {j}` only when
`start + p_j <= H` (a censored job never unlocks its successors).  The leaf
value is a greedy expected-time rollout whose per-job value is gated by the
*ancestor* completion probability, so parallel branches are not penalised.
"""
import math

import numpy as np

_GRID = 128


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


def survival(p, slack):
    return float(np.mean(p <= slack + 1e-12))


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


def _greedy_tail(t, xy, done, remaining, jobs, pdists, H, B, cost, preds, anc_cache):
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
            p = pdists[j['family']]
            mp = float(np.mean(p))
            if start + mp > H + 1e-12:
                continue
            s = survival(p, min(j['due_s'], H) - start)
            if s <= 1e-9:
                continue
            inc = D + .5 + mp
            fee = 2 + .1 * .2 + .2 * D
            if cost + fee > B + 1e-12:
                continue
            cand = (-attempted * j['weight'] * s / inc, j['due_s'], jid, inc, s,
                    survival(p, H - start), fee, attempted)
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


def _value(t, xy, done, remaining, cost, depth, jobs, pdists, H, B, nq, beam, preds, anc_cache, is_root):
    remaining = set(remaining)
    if depth <= 0 or not remaining:
        return _greedy_tail(t, xy, done, remaining, jobs, pdists, H, B, cost, preds, anc_cache)
    cands = []
    for jid in remaining:
        j = jobs[jid]
        if not preds[jid] <= done or j['release_s'] > t + 1e-12:
            continue
        jz = np.asarray(j['zone_xy'], float)
        D = math.dist(xy, jz)
        start = t + D + .5
        p = pdists[j['family']]
        mp = float(np.mean(p))
        if start + mp > H + 1e-12:
            continue
        fee = 2 + .1 * .2 + .2 * D
        if cost + fee > B + 1e-12:
            continue
        on = survival(p, min(j['due_s'], H) - start)
        if on <= 1e-9:
            continue
        rank = j['weight'] * on + _greedy_tail(start + mp, jz, done | {jid}, remaining - {jid},
                                               jobs, pdists, H, B, cost + fee, preds, anc_cache)
        cands.append((rank, jid, jz, start, p, fee))
    if not cands:
        future = [jobs[k]['release_s'] for k in remaining
                  if preds[k] <= done and jobs[k]['release_s'] > t + 1e-12 and jobs[k]['release_s'] < H]
        if future:
            t2 = min(future)
            if t2 < H:
                return _value(t2, xy, done, remaining, cost, depth, jobs, pdists, H, B, nq, beam, preds, anc_cache, is_root)
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
                                  jobs, pdists, H, B, nq, beam, preds, anc_cache, False)
        val = total / len(qvals)
        if val > best:
            best = val
    return best


def select(o, pools, history=None, strength=20, feedback_strength=3, depth=2,
           nq_root=5, nq=2, beam=3):
    history = history or {}
    pdists = {f: o['alpha'] * mixture(f, pools, history, strength, feedback_strength) + o['beta'] for f in pools}
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
        p = pdists[j['family']]
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
                                  jobs, pdists, o['H'], o['B'], nq, beam, preds, anc_cache, False)
        score = total / len(grid)
        cand = (-score, j['due_s'], j['job_id'])
        if best is None or cand < best:
            best = cand
    if best is not None:
        return {'type': 'execute', 'job_id': best[2], 'reserve': .2}
    future = [j['release_s'] for j in o['jobs']
              if not j['attempted'] and o['time'] + 1e-12 < j['release_s'] <= o['H']]
    return {'type': 'wait', 'duration': min(future) - o['time']} if future else {'type': 'stop'}
