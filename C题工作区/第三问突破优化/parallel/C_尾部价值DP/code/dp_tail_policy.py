"""Role C: DP tail value function for the Q3 stochastic single-worker scheduling task.

What this module replaces
-------------------------
The audited lookahead (v2_s20) valued the tail with a *density-greedy* rollout:
at every tail step it served the job with the largest ``w * P(on time) / occupancy``.
A greedy index cannot see that serving a cheap low-weight job now can block a heavy
tight-deadline job later, and it cannot trade total time against total weight.
Role C replaces that critic with a **restricted DP**:

    V(clock, zone, chain-mask) = best expected on-time weight obtainable from the
                                 remaining jobs when they are served in a fixed
                                 topological order.

The DP is a time-bucketed, precedence-aware 0/1 knapsack over the remaining jobs.
Each job is either *skipped* or *served*; serving advances the clock to
``start + p`` and earns ``w_j * P(p_j <= due_j - start)`` where the probability is
always taken from the **full family distribution**, never from a point estimate.
The clock speed is a separate, explicitly named *scenario* (``mean`` or a
duration quantile); the reported method integrates the plan value over several
clock scenarios (``timeline='quantile'``), which is the "mean -> quantile
integration" ablation axis.

Correctness points demanded by the audit
----------------------------------------
* Clock: a job served at ``start`` completes at ``start + p``; travel and the
  0.5 s changeover enter once, inside ``start``.  Never ``start + D + .5 + p``.
* Chain probability: a successor's value is multiplied only by the completion
  probability of *its own ancestors*, ``P(p_a <= H - s_a)``.  Parallel branches
  are never multiplied into each other.  Gating is applied in the plan
  evaluation that ranks plans, i.e. in the value that drives decisions.
* Budget: unaffordable jobs are screened out inside the plan walk *before*
  being accepted, and the walk continues with the next job instead of aborting.
* Mixture: the family/global/feedback mixture lives on a fixed 128-point
  quantile grid; that is a **discrete approximation**, not exact weight
  preservation.
* ``feedback_strength <= 0`` means *do not use in-trial feedback* (prior only).
* No hidden duration, case id, target value or validation outcome is read;
  ``alpha``/``beta`` come from the public ``observe()`` payload.

Reduction and its bound
-----------------------
A subset-exact DP over ``(clock, zone, done-set)`` costs ``2^n * buckets * zones``
and is infeasible here (n up to 16, 250 buckets).  This module restricts the
policy class to *topologically ordered selection*; the restriction can only
lower the achievable value, so the DP value is a lower bound on the unrestricted
stochastic optimum, while the greedy rollout is not ordered.  ``report.md``
quantifies both sides with a fractional-knapsack upper relaxation.
"""
import math

import numpy as np

_GRID = 128


# --------------------------------------------------------------------------- #
# duration belief
# --------------------------------------------------------------------------- #
def mixture(family, pools, history, strength, feedback_strength):
    """Fixed 128-point grid mixture of family pool, global prior and feedback.

    Discrete approximation: quantile positions are rounded onto the grid, so
    nominal mixture weights are preserved only to one grid cell (~1/128).
    """
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
    """P(duration <= slack) on the discrete grid; zero for a negative slack."""
    if slack < -1e-12:
        return 0.0
    return float(np.mean(p <= slack + 1e-12))


def _surv_vec(p, slack):
    return np.mean(p[None, :] <= slack[:, None] + 1e-12, axis=1)


def _zk(xy):
    return (round(float(xy[0]), 6), round(float(xy[1]), 6))


def _ancestors(jid, preds, cache):
    if jid in cache:
        return cache[jid]
    out, stack = set(), list(preds.get(jid, ()))
    while stack:
        a = stack.pop()
        if a in out:
            continue
        out.add(a)
        stack.extend(preds.get(a, ()))
    cache[jid] = out
    return out


# --------------------------------------------------------------------------- #
# clock scenarios
# --------------------------------------------------------------------------- #
def clock_scenarios(pdists, timeline, scenarios):
    """List of ``{family: scalar advance}`` clock scenarios."""
    if timeline == 'mean':
        return [{f: float(np.mean(p)) for f, p in pdists.items()}]
    return [{f: max(float(np.quantile(p, q)), .2) for f, p in pdists.items()} for q in scenarios]


# --------------------------------------------------------------------------- #
# order restrictions
# --------------------------------------------------------------------------- #
def priority_order(jobs, remaining, done, kind, mean_of):
    """Topological list order of ``remaining`` by a static priority."""
    ordered, placed, left = [], set(done), list(remaining)
    while left:
        cands = [j for j in left if set(jobs[j]['predecessors']) <= placed]
        if not cands:
            ordered.extend(left)
            break
        if kind == 'edf':
            key = lambda j: (jobs[j]['due_s'], jobs[j]['release_s'], j)
        elif kind == 'wdd':
            key = lambda j: (jobs[j]['due_s'] / max(jobs[j]['weight'], 1e-9), jobs[j]['due_s'], j)
        elif kind == 'slack':
            key = lambda j: ((jobs[j]['due_s'] - jobs[j]['release_s'] - mean_of[jobs[j]['family']])
                             / max(jobs[j]['weight'], 1e-9), jobs[j]['due_s'], j)
        elif kind == 'wdens':
            key = lambda j: (mean_of[jobs[j]['family']] / max(jobs[j]['weight'], 1e-9), jobs[j]['due_s'], j)
        elif kind == 'release':
            key = lambda j: (jobs[j]['release_s'], jobs[j]['due_s'], j)
        else:
            raise ValueError(kind)
        best = min(cands, key=key)
        ordered.append(best)
        placed.add(best)
        left.remove(best)
    return ordered


# --------------------------------------------------------------------------- #
# restricted DP
# --------------------------------------------------------------------------- #
def _context(o):
    jobs = {j['job_id']: j for j in o['jobs']}
    keys = sorted({_zk(j['zone_xy']) for j in o['jobs']} | {_zk(o['zone_xy'])})
    zindex = {k: i for i, k in enumerate(keys)}
    return jobs, keys, zindex, zindex[_zk(o['zone_xy'])]


def dp_table(order, t0, z0, jobs, pdists, clock, H, dt, keys, zindex, need_of, bit_of):
    """Forward time-bucketed knapsack DP over a fixed topological order.

    ``clock`` gives the scalar duration used to advance the clock per family;
    ``pdists`` gives the full distributions used for on-time probabilities.
    Returns the value tables after 0..n jobs plus the initial bucket.
    """
    nb = int(math.floor(H / dt + 1e-9)) + 3
    nz = len(keys)
    nm = 1 << max(1, (max(bit_of.values()) if bit_of else 0).bit_length())
    bt = np.arange(nb) * dt
    V = np.full((nb, nz, nm), -np.inf)
    b0 = min(nb - 1, int(math.ceil(t0 / dt - 1e-9)))
    V[b0, z0, 0] = 0.0
    hist = [V.copy()]
    zxy = {i: np.asarray(k, float) for k, i in zindex.items()}
    for jid in order:
        j = jobs[jid]
        p = pdists[j['family']]
        mp = clock[j['family']]
        due = min(j['due_s'], H)
        rel = j['release_s']
        w = j['weight']
        need = need_of.get(jid, 0)
        bit = bit_of.get(jid, 0)
        zj = zindex[_zk(j['zone_xy'])]
        valid = [m for m in range(nm) if (m & need) == need and (m & bit) == 0]
        src_m = np.asarray(valid, int)
        dst_m = np.asarray([m | bit for m in valid], int)
        Vnew = V.copy()
        for zi in range(nz):
            col = V[:, zi, :]
            if not np.isfinite(col).any():
                continue
            Dz = math.dist(zxy[zi], zxy[zj])
            start = np.maximum(bt + Dz + .5, rel)
            gain = w * _surv_vec(p, due - start)
            end = start + mp
            ok = end <= H + 1e-12
            if not ok.any():
                continue
            A = col + gain[:, None]
            A[~ok] = -np.inf
            bdest = np.clip(np.ceil(end / dt - 1e-9).astype(int), 0, nb - 1)
            udest, first = np.unique(bdest, return_index=True)
            grp = np.maximum.reduceat(A, first, axis=0)
            sub = Vnew[udest[:, None], zj, dst_m[None, :]]
            Vnew[udest[:, None], zj, dst_m[None, :]] = np.maximum(sub, grp[:, src_m])
        V = Vnew
        hist.append(V.copy())
    return hist, b0


def extract_plan(hist, order, b0, z0, jobs, pdists, clock, cost0, B, H, dt, keys, zindex, need_of, bit_of):
    """Greedy policy extraction under the DP value function.

    Unaffordable jobs are screened *before* acceptance; the walk continues with
    the next job instead of terminating the tail.
    """
    nb = int(math.floor(H / dt + 1e-9)) + 3
    zxy = {i: np.asarray(k, float) for k, i in zindex.items()}
    b, z, m, cost = b0, z0, 0, cost0
    plan = []
    for k, jid in enumerate(order):
        need = need_of.get(jid, 0)
        if (m & need) != need:
            continue
        j = jobs[jid]
        zj = zindex[_zk(j['zone_xy'])]
        D = math.dist(zxy[z], zxy[zj])
        fee = 2 + .1 * .2 + .2 * D
        if cost + fee > B + 1e-12:
            continue
        start = max(b * dt + D + .5, j['release_s'])
        end = start + clock[j['family']]
        if end > H + 1e-12:
            continue
        b2 = min(nb - 1, int(math.ceil(end / dt - 1e-9)))
        z2, m2 = zj, m | bit_of.get(jid, 0)
        V1 = hist[k + 1]
        if V1[b2, z2, m2] > V1[b, z, m] + 1e-9:
            plan.append(jid)
            b, z, m = b2, z2, m2
            cost += fee
    return plan


# --------------------------------------------------------------------------- #
# plan evaluation (ancestor-gated, distribution based, scenario timelines)
# --------------------------------------------------------------------------- #
def eval_plan(plan, t0, zxy0, cost0, jobs, pdists, H, B, anc, clocks, gate0=None):
    """Mean expected on-time weight of ``plan`` over the supplied clock scenarios.

    For each scenario the plan is replayed on that scenario's clock; every job's
    contribution is multiplied by the completion-by-H probability of its own
    ancestors and by its own on-time probability from the full distribution.
    ``gate0`` seeds the gate of jobs served before the plan (outer layer).
    """
    vals = []
    for clock in clocks:
        t, z, cost = t0, np.asarray(zxy0, float), cost0
        gate, val = dict(gate0 or {}), 0.0
        for jid in plan:
            j = jobs[jid]
            D = math.dist(z, np.asarray(j['zone_xy'], float))
            s = max(t + D + .5, j['release_s'])
            fee = 2 + .1 * .2 + .2 * D
            if cost + fee > B + 1e-12:
                break
            step = clock[j['family']]
            if s + step > H + 1e-12:
                break
            p = pdists[j['family']]
            g = 1.0
            for a in anc[jid]:
                if a in gate:
                    g *= gate[a]
            val += g * j['weight'] * survival(p, min(j['due_s'], H) - s)
            gate[jid] = survival(p, H - s)
            cost += fee
            t = s + step
            z = np.asarray(j['zone_xy'], float)
        vals.append(val)
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# public policy
# --------------------------------------------------------------------------- #
DEFAULT_CFG = dict(mode='plan', strength=20, feedback_strength=3, dt=.5,
                   orders=('edf', 'wdd', 'slack'), timeline='quantile',
                   scenarios=(.2, .5, .8), tail_quantiles=3, topk=4, repair=False,
                   mc_samples=64, mc_seed=20261122)


def select(o, pools, history=None, mode=None, strength=None, feedback_strength=None, dt=None,
           orders=None, timeline=None, scenarios=None, tail_quantiles=None, topk=None, repair=None,
           mc_samples=None, mc_seed=None, use_dp=True, use_greedy=True, force_first=False, **_):
    """Return the next action.  See ``DEFAULT_CFG`` for the tunables."""
    cfg = dict(DEFAULT_CFG)
    for k, v in (('mode', mode), ('strength', strength), ('feedback_strength', feedback_strength),
                 ('dt', dt), ('orders', orders), ('timeline', timeline),
                 ('scenarios', scenarios), ('tail_quantiles', tail_quantiles), ('topk', topk),
                 ('repair', repair), ('mc_samples', mc_samples), ('mc_seed', mc_seed)):
        if v is not None:
            cfg[k] = v
    history = history or {}
    raw = {f: mixture(f, pools, history, cfg['strength'], cfg['feedback_strength']) for f in pools}
    pdists = {f: o['alpha'] * raw[f] + o['beta'] for f in pools}
    mean_of = {f: float(np.mean(v)) for f, v in pdists.items()}

    jobs, keys, zindex, z0 = _context(o)
    preds = {jid: set(j['predecessors']) for jid, j in jobs.items()}
    done = {jid for jid, j in jobs.items() if j['completed']}
    H, B = o['H'], o['B']

    remaining = [jid for jid, j in jobs.items() if not j['attempted']]
    available = [jid for jid in remaining
                 if jobs[jid]['release_s'] <= o['time'] + 1e-12 and preds[jid] <= done]
    if not remaining:
        return {'type': 'stop'}
    if not available:
        return _wait_or_stop(o)

    tracked = sorted({p for jid in remaining for p in preds[jid] if p in remaining})
    bit_of = {jid: 1 << k for k, jid in enumerate(tracked)}
    need_of = {jid: sum(bit_of[p] for p in preds[jid] if p in remaining) for jid in remaining}
    anc = {jid: _ancestors(jid, preds, {}) for jid in jobs}
    clocks = clock_scenarios(pdists, cfg['timeline'], cfg['scenarios'])

    if cfg['mode'] == 'plan':
        seen, scored = set(), []
        for kind in cfg['orders']:
            order = priority_order(jobs, remaining, done, kind, mean_of)
            for clock in clocks:
                hist, b0 = dp_table(order, o['time'], z0, jobs, pdists, clock, H, cfg['dt'],
                                    keys, zindex, need_of, bit_of)
                plan = extract_plan(hist, order, b0, z0, jobs, pdists, clock, o['cost'], B, H,
                                    cfg['dt'], keys, zindex, need_of, bit_of)
                key = tuple(plan)
                if not plan or key in seen:
                    continue
                seen.add(key)
                scored.append((eval_plan(plan, o['time'], o['zone_xy'], o['cost'], jobs, pdists, H, B,
                                         anc, clocks), plan))
        if not scored:
            return _wait_or_stop(o)
        scored.sort(key=lambda x: -x[0])
        best_val, best_plan = scored[0]
        if cfg['repair']:
            for _, plan in scored[:2]:
                rp, rv = repair_front(plan, jobs, preds, available, o['time'], o['zone_xy'], o['cost'],
                                      pdists, H, B, anc, clocks)
                if rv > best_val + 1e-12:
                    best_val, best_plan = rv, rp
        jid = _first_actionable(best_plan, jobs, o, B, H)
        if jid is not None:
            return {'type': 'execute', 'job_id': jid, 'reserve': .2}
        return _wait_or_stop(o)

    if cfg['mode'] == 'hybrid':
        # one-step candidate scoring from the audited v2 policy, but the tail of
        # the top-K candidates is re-valued by the restricted DP.
        scores, detail, _ = select_v2(o, pools, history, strength=cfg['strength'],
                                     feedback_strength=cfg['feedback_strength'],
                                     quantiles=cfg['tail_quantiles'], scores_only=True)
        if not scores:
            return _wait_or_stop(o)
        cands = sorted(scores, key=lambda j: (-scores[j], jobs[j]['due_s'], j))[:cfg['topk']]
        best = None
        for jid in cands:
            d = detail[jid]
            rest = [r for r in remaining if r != jid]
            need2 = {r: sum(bit_of[x] for x in preds[r] if x in rest) for r in rest}
            zj = zindex[_zk(jobs[jid]['zone_xy'])]
            gate0 = {jid: survival(d['p'], H - d['start'])}
            tails = []
            for q in cfg['scenarios']:
                step = max(float(np.quantile(d['p'], q)), .2)
                if d['start'] + step > H + 1e-12:
                    tails.append(0.0)
                    continue
                clock_q = {f: max(float(np.quantile(v, q)), .2) for f, v in pdists.items()}
                best_tail = 0.0
                for kind in cfg['orders'][:2]:
                    order = priority_order(jobs, rest, done | {jid}, kind, mean_of)
                    if not order:
                        continue
                    hist, b0 = dp_table(order, d['start'] + step, zj, jobs, pdists, clock_q, H, cfg['dt'],
                                        keys, zindex, need2, bit_of)
                    plan = extract_plan(hist, order, b0, zj, jobs, pdists, clock_q, o['cost'] + d['fee'],
                                        B, H, cfg['dt'], keys, zindex, need2, bit_of)
                    best_tail = max(best_tail, eval_plan(plan, d['start'] + step, jobs[jid]['zone_xy'],
                                                         o['cost'] + d['fee'], jobs, pdists, H, B, anc,
                                                         [clock_q], gate0))
                tails.append(best_tail)
            score = d['imm'] + float(np.mean(tails))
            cand = (-score, jobs[jid]['due_s'], jid)
            if best is None or cand < best[0]:
                best = (cand, jid)
        if best is not None:
            return {'type': 'execute', 'job_id': best[1], 'reserve': .2}
        return _wait_or_stop(o)

    if cfg['mode'] == 'mc':
        # Portfolio: restricted-DP plans plus audited greedy plans, all ranked by
        # the same Monte-Carlo, distribution-exact evaluator with common random
        # numbers.  The evaluator is the value function; the DP supplies the
        # global subset allocation the greedy index cannot see.
        K = cfg['mc_samples']
        rng = np.random.default_rng(cfg['mc_seed'])
        samples = {jid: pdists[jobs[jid]['family']][rng.integers(0, len(pdists[jobs[jid]['family']]), K)]
                   for jid in jobs}
        Dmat = np.asarray([[float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))
                            for b in keys] for a in keys])
        zk = {jid: zindex[_zk(jobs[jid]['zone_xy'])] for jid in jobs}
        plans, seen = [], set()

        def push(pl):
            if pl and tuple(pl) not in seen:
                seen.add(tuple(pl))
                plans.append(pl)

        for kind in (cfg['orders'] if use_dp else ()):
            order = priority_order(jobs, remaining, done, kind, mean_of)
            for clock in clocks:
                hist, b0 = dp_table(order, o['time'], z0, jobs, pdists, clock, H, cfg['dt'],
                                    keys, zindex, need_of, bit_of)
                push(extract_plan(hist, order, b0, z0, jobs, pdists, clock, o['cost'], B, H,
                                  cfg['dt'], keys, zindex, need_of, bit_of))
        gscores, gdetail, _ = select_v2(o, pools, history, strength=cfg['strength'],
                                        feedback_strength=cfg['feedback_strength'],
                                        quantiles=cfg['tail_quantiles'], scores_only=True)
        ranked = sorted(gscores, key=lambda x: (-gscores[x], jobs[x]['due_s'], x))
        if force_first:
            # Plans with a *forced* first job whose tail is DP-optimal over the
            # rest.  The MC evaluator then compares first jobs on their full,
            # tail-optimised expected on-time weight instead of a greedy index.
            for jid in ranked[:5]:
                j = jobs[jid]
                D = math.dist(np.asarray(o['zone_xy'], float), np.asarray(j['zone_xy'], float))
                fee = 2 + .1 * .2 + .2 * D
                s = max(o['time'] + D + .5, j['release_s'])
                if s + .2 > H + 1e-12 or o['cost'] + fee > B + 1e-12:
                    continue
                rest = [r for r in remaining if r != jid]
                # mask over predecessors that are still in ``rest`` only; jid is
                # served by the prefix, so it must not gate its own successors.
                need2 = {r: sum(bit_of[x] for x in preds[r] if x in rest) for r in rest}
                for kind in cfg['orders'][:2]:
                    order = priority_order(jobs, rest, done | {jid}, kind, mean_of)
                    if not order:
                        continue
                    for clock in clocks[:2]:
                        t1 = s + clock[j['family']]
                        if t1 > H + 1e-12:
                            continue
                        hist, b0 = dp_table(order, t1, zindex[_zk(j['zone_xy'])], jobs, pdists, clock, H,
                                            cfg['dt'], keys, zindex, need2, bit_of)
                        tail = extract_plan(hist, order, b0, zindex[_zk(j['zone_xy'])], jobs, pdists, clock,
                                            o['cost'] + fee, B, H, cfg['dt'], keys, zindex, need2, bit_of)
                        push([jid] + tail)
        for jid in (ranked[:cfg['topk']] if use_greedy else ()):
            d = gdetail[jid]
            rest = [r for r in remaining if r != jid]
            tail = greedy_tail(d['start'] + float(np.mean(d['p'])), jobs[jid]['zone_xy'],
                               done | {jid}, rest, o['cost'] + d['fee'],
                               {jid: survival(d['p'], H - d['start'])},
                               jobs, pdists, preds, anc, H, B)
            push([jid] + tail)
        if not plans:
            return _wait_or_stop(o)
        best = None
        for pl in plans:
            jid = _first_actionable(pl, jobs, o, B, H)
            if jid is None:
                continue
            v = mc_eval(pl, o['time'], o['zone_xy'], o['cost'], jobs, pdists, preds, H, B,
                        samples, K, zk, Dmat, z0, done)
            cand = (-v, jobs[jid]['due_s'], jid)
            if best is None or cand < best[0]:
                best = (cand, jid)
        if best is not None:
            return {'type': 'execute', 'job_id': best[1], 'reserve': .2}
        return _wait_or_stop(o)

    if cfg['mode'] == 'plan_mean':          # ablation: single mean clock, one plan per order
        best = None
        for kind in cfg['orders']:
            order = priority_order(jobs, remaining, done, kind, mean_of)
            clock = {f: v for f, v in mean_of.items()}
            hist, b0 = dp_table(order, o['time'], z0, jobs, pdists, clock, H, cfg['dt'],
                                keys, zindex, need_of, bit_of)
            plan = extract_plan(hist, order, b0, z0, jobs, pdists, clock, o['cost'], B, H,
                                cfg['dt'], keys, zindex, need_of, bit_of)
            jid = _first_actionable(plan, jobs, o, B, H)
            if jid is None:
                continue
            val = eval_plan(plan, o['time'], o['zone_xy'], o['cost'], jobs, pdists, H, B, anc, clocks)
            cand = (-val, jobs[jid]['due_s'], jid)
            if best is None or cand < best[0]:
                best = (cand, jid)
        if best is not None:
            return {'type': 'execute', 'job_id': best[1], 'reserve': .2}
        return _wait_or_stop(o)

    # mode == 'onestep': immediate on-time value plus a DP tail over the rest
    qs = [.5] if cfg['tail_quantiles'] <= 1 else list((np.arange(cfg['tail_quantiles']) + .5) / cfg['tail_quantiles'])
    best = None
    for jid in available:
        j = jobs[jid]
        D = math.dist(np.asarray(o['zone_xy'], float), np.asarray(j['zone_xy'], float))
        start = o['time'] + D + .5
        p = pdists[j['family']]
        fee = 2 + .1 * .2 + .2 * D
        if start + .2 > H + 1e-12 or o['cost'] + fee > B + 1e-12:
            continue
        rest = [r for r in remaining if r != jid]
        zj = zindex[_zk(j['zone_xy'])]
        need2 = {r: sum(bit_of[x] for x in preds[r] if x in rest) for r in rest}
        tail = 0.0
        for q in qs:
            step = max(float(np.quantile(p, q)), .2)
            if start + step > H + 1e-12:
                continue
            best_tail = 0.0
            for kind in cfg['orders']:
                order = priority_order(jobs, rest, done | {jid}, kind, mean_of)
                if not order:
                    continue
                for clock in clocks:
                    hist, b0 = dp_table(order, start + step, zj, jobs, pdists, clock, H, cfg['dt'],
                                        keys, zindex, need2, bit_of)
                    plan = extract_plan(hist, order, b0, zj, jobs, pdists, clock, o['cost'] + fee, B, H,
                                        cfg['dt'], keys, zindex, need2, bit_of)
                    best_tail = max(best_tail, eval_plan(plan, start + step, j['zone_xy'], o['cost'] + fee,
                                                         jobs, pdists, H, B, anc, clocks))
            tail += best_tail
        score = j['weight'] * survival(p, min(j['due_s'], H) - start) + tail / len(qs)
        cand = (-score, j['due_s'], jid)
        if best is None or cand < best[0]:
            best = (cand, jid)
    if best is not None:
        return {'type': 'execute', 'job_id': best[1], 'reserve': .2}
    return _wait_or_stop(o)


def greedy_tail(t, zxy, done, rem, cost, gate, jobs, pdists, preds, anc, H, B):
    """v2-style ancestor-gated density rollout, returned as an ordered job list."""
    t = float(t)
    zxy = np.asarray(zxy, float)
    done, rem, gate = set(done), set(rem), dict(gate)
    plan = []
    for _ in range(64):
        if not rem:
            break
        best = None
        for jid in rem:
            j = jobs[jid]
            if not preds[jid] <= done or j['release_s'] > t + 1e-12:
                continue
            D = math.dist(zxy, np.asarray(j['zone_xy'], float))
            start = t + D + .5
            p = pdists[j['family']]
            mp = float(np.mean(p))
            if start + mp > H + 1e-12:
                continue
            g = 1.0
            for a in anc[jid]:
                g *= gate.get(a, 1.0)
            s = survival(p, min(j['due_s'], H) - start)
            if s <= 1e-9:
                continue
            fee = 2 + .1 * .2 + .2 * D
            if cost + fee > B + 1e-12:
                continue
            cand = (-g * j['weight'] * s / (D + .5 + mp), j['due_s'], jid)
            if best is None or cand < best[0]:
                best = (cand, jid, start, mp, fee, j['zone_xy'])
        if best is None:
            future = [jobs[k]['release_s'] for k in rem
                      if preds[k] <= done and jobs[k]['release_s'] > t + 1e-12 and jobs[k]['release_s'] < H]
            if future:
                t = min(future)
                continue
            break
        _, jid, start, mp, fee, zj = best
        p = pdists[jobs[jid]['family']]
        gate[jid] = survival(p, H - start)
        cost += fee
        t = start + mp
        zxy = np.asarray(zj, float)
        done.add(jid)
        rem.discard(jid)
        plan.append(jid)
    return plan


def mc_eval(plan, t0, zxy0, cost0, jobs, pdists, preds, H, B, samples, K, zk, Dmat, z0, done=()):
    """Monte-Carlo value of a whole plan under the full family distributions.

    Every sample replays the plan with its own duration draws; a job is served
    only if its predecessors really completed in that sample, so ancestor gating
    is exact (no multiplication shortcut).  Cost is charged per sample and an
    unaffordable job is skipped instead of ending the plan.
    """
    if not plan:
        return 0.0
    t = np.full(K, float(t0))
    cost = np.full(K, float(cost0))
    val = np.zeros(K)
    z = np.full(K, z0, int)
    completed = {jid: np.full(K, jid in done, bool) for jid in jobs}
    for jid in plan:
        j = jobs[jid]
        k = zk[jid]
        D = Dmat[z, k]
        fee = 2 + .1 * .2 + .2 * D
        pred_ok = np.ones(K, bool)
        for p in preds[jid]:
            pred_ok &= completed[p]
        s = np.maximum(t + D + .5, j['release_s'])
        end = s + samples[jid]
        served = pred_ok & (cost + fee <= B + 1e-12) & (end <= H + 1e-12)
        val += np.where(served & (end <= min(j['due_s'], H) + 1e-12), j['weight'], 0.0)
        t = np.where(served, end, t)
        z = np.where(served, k, z)
        cost = cost + np.where(served, fee, 0.0)
        completed[jid] = served
    return float(np.mean(val))


def _first_actionable(plan, jobs, o, B, H):
    for jid in plan:
        j = jobs[jid]
        D = math.dist(np.asarray(o['zone_xy'], float), np.asarray(j['zone_xy'], float))
        if j['release_s'] > o['time'] + 1e-12:
            continue
        if o['cost'] + 2 + .02 + .2 * D > B + 1e-12:
            continue
        if o['time'] + D + .5 + .2 > H + 1e-12:
            continue
        return jid
    return None


def repair_front(plan, jobs, preds, available, t0, zxy0, cost0, pdists, H, B, anc, clocks):
    """Move-to-front repair of the order restriction.

    The DP searches only topologically ordered subsets, so its first job can be
    forced by the order rather than by value.  This operator tries every
    currently available job at the front of the plan (pulling its ancestors ahead
    of it) and keeps the best evaluated plan.  Evaluation is the same
    ancestor-gated, distribution-based, multi-clock objective, so the repair can
    only improve the reported plan value.
    """
    best_val = eval_plan(plan, t0, zxy0, cost0, jobs, pdists, H, B, anc, clocks)
    best = list(plan)
    for jid in available:
        head = [a for a in plan if a in anc[jid]] + [jid]
        cand = head + [x for x in plan if x != jid and x not in anc[jid]]
        if cand == best:
            continue
        val = eval_plan(cand, t0, zxy0, cost0, jobs, pdists, H, B, anc, clocks)
        if val > best_val + 1e-12:
            best_val, best = val, cand
    return best, best_val


def _wait_or_stop(o):
    future = [j['release_s'] for j in o['jobs']
              if not j['attempted'] and o['time'] + 1e-12 < j['release_s'] <= o['H']]
    if future:
        return {'type': 'wait', 'duration': min(future) - o['time']}
    return {'type': 'stop'}


# --------------------------------------------------------------------------- #
# reference policy: v2 = one-step + ancestor-gated density-greedy tail (s=20)
# --------------------------------------------------------------------------- #
def select_v2(o, pools, history=None, strength=20, feedback_strength=3, quantiles=5, scores_only=False):
    """Reproduction of the reported v2_s20 policy (one-step + greedy density tail).

    Kept here so the DP comparison is paired inside one protocol file and never
    depends on an external number.
    """
    history = history or {}
    raw = {f: mixture(f, pools, history, strength, feedback_strength) for f in pools}
    pdists = {f: o['alpha'] * raw[f] + o['beta'] for f in pools}
    jobs = {j['job_id']: j for j in o['jobs']}
    preds = {jid: set(j['predecessors']) for jid, j in jobs.items()}
    anc = {jid: _ancestors(jid, preds, {}) for jid in jobs}
    done = {jid for jid, j in jobs.items() if j['completed']}
    H, B = o['H'], o['B']
    remaining = [jid for jid, j in jobs.items() if not j['attempted']]
    available = [jid for jid in remaining
                 if jobs[jid]['release_s'] <= o['time'] + 1e-12 and preds[jid] <= done]
    if not remaining:
        return {'type': 'stop'}
    if not available:
        return _wait_or_stop(o)

    def tail(t, zxy, done2, rem, cost, gate):
        got = 0.0
        t = float(t)
        zxy = np.asarray(zxy, float)
        done2, rem = set(done2), set(rem)
        for _ in range(64):
            if not rem:
                break
            best = None
            for jid in rem:
                j = jobs[jid]
                if not preds[jid] <= done2 or j['release_s'] > t + 1e-12:
                    continue
                D = math.dist(zxy, np.asarray(j['zone_xy'], float))
                start = t + D + .5
                p = pdists[j['family']]
                mp = float(np.mean(p))
                if start + mp > H + 1e-12:
                    continue
                g = 1.0
                for a in anc[jid]:
                    g *= gate.get(a, 1.0)
                s = survival(p, min(j['due_s'], H) - start)
                if s <= 1e-9:
                    continue
                occ = D + .5 + mp
                fee = 2 + .1 * .2 + .2 * D
                if cost + fee > B + 1e-12:
                    continue
                cand = (-g * j['weight'] * s / occ, j['due_s'], jid)
                if best is None or cand < best[0]:
                    best = (cand, jid, start, mp, fee, j['zone_xy'])
            if best is None:
                future = [jobs[k]['release_s'] for k in rem
                          if preds[k] <= done2 and jobs[k]['release_s'] > t + 1e-12 and jobs[k]['release_s'] < H]
                if future:
                    t = min(future)
                    continue
                break
            _, jid, start, mp, fee, zj = best
            p = pdists[jobs[jid]['family']]
            got += jobs[jid]['weight'] * survival(p, min(jobs[jid]['due_s'], H) - start)
            gate[jid] = survival(p, H - start)
            cost += fee
            t = start + mp
            zxy = np.asarray(zj, float)
            done2.add(jid)
            rem.discard(jid)
        return got

    qs = [.5] if quantiles <= 1 else list((np.arange(quantiles) + .5) / quantiles)
    scores, detail = {}, {}
    for jid in available:
        j = jobs[jid]
        D = math.dist(np.asarray(o['zone_xy'], float), np.asarray(j['zone_xy'], float))
        start = o['time'] + D + .5
        p = pdists[j['family']]
        fee = 2 + .1 * .2 + .2 * D
        if start + .2 > H + 1e-12 or o['cost'] + fee > B + 1e-12:
            continue
        imm = j['weight'] * survival(p, min(j['due_s'], H) - start)
        rest = [r for r in remaining if r != jid]
        tails = []
        for q in qs:
            step = max(float(np.quantile(p, q)), .2)
            if start + step > H + 1e-12:
                tails.append(0.0)
                continue
            tails.append(tail(start + step, j['zone_xy'], done | {jid}, rest, o['cost'] + fee,
                              {jid: survival(p, H - start)}))
        scores[jid] = imm + float(np.mean(tails))
        detail[jid] = dict(start=start, p=p, fee=fee, imm=imm)
    if scores_only:
        return scores, detail, dict(pdists=pdists, jobs=jobs, preds=preds, anc=anc, done=done,
                                    remaining=remaining, available=available, H=H, B=B)
    if not scores:
        return _wait_or_stop(o)
    best = min(scores, key=lambda j: (-scores[j], jobs[j]['due_s'], j))
    return {'type': 'execute', 'job_id': best, 'reserve': .2}
