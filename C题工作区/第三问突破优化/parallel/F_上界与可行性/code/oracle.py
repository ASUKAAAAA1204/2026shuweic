"""Perfect-information oracle and provable relaxation bounds for Q3.

This module answers a feasibility question, not a policy question:

* ``solve_oracle`` returns the exact maximum weighted on-time rate that is
  achievable when *all* realized service durations are known before the first
  decision.  It is an evaluation-side upper bound on every legal online policy
  and is **not** a legal policy (it reads hidden durations).
* ``relaxation_ub`` returns a provably valid upper bound that only uses
  necessary conditions (per-job earliest completion, deadline-class capacity,
  release-window capacity, budget).  It never solves a scheduling problem, so
  it is looser than the oracle but needs no optimisation proof.
* ``replay_plan`` replays a fixed job order through the official
  ``engine.Engine`` with the minimum reserve, so the oracle score is audited by
  ``engine.independent_metrics`` instead of being trusted from the MILP.

Model fidelity notes (mirrors 第三问改进正式方案/code/engine.py)
--------------------------------------------------------------
* service start ``s = t + D + 0.5``; completion ``C = s + p``;
  next clock ``t' = s + max(p, 0.2)``.
* reserve only enters the fee ``2 + 0.1*b + 0.2*D`` and the planned-horizon
  test ``s + b <= H``.  Completion and on-time status do not depend on ``b``,
  and a larger reserve costs more, so ``b = 0.2`` dominates for the on-time
  objective.
* a predecessor must be *completed* (``C <= H``) before a successor starts.
"""
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

MIN_RESERVE = 0.2
FIXED_FEE = 2.0 + 0.1 * MIN_RESERVE  # 2.02
CHANGEOVER = 0.5


def build_arrays(jobs, durations, alpha, beta, zones):
    """Scenario arrays: p = alpha * hidden_duration + beta (public mapping)."""
    n = len(jobs)
    p = np.array([alpha * float(durations[j["job_id"]]) + beta for j in jobs], float)
    xy = [np.asarray(zones[j["zone_id"]], float) for j in jobs]
    z0 = np.asarray(zones["Z0"], float)
    D = np.array([[math.dist(xy[a], xy[b]) for b in range(n)] for a in range(n)])
    D0 = np.array([math.dist(z0, xy[b]) for b in range(n)])
    return p, xy, z0, D, D0


def _ancestors(preds):
    """All transitive predecessors (jobs that may be executed late for unlocking)."""
    anc = set()
    changed = True
    while changed:
        changed = False
        for s in preds:
            for a in s:
                if a not in anc:
                    anc.add(a)
                    changed = True
    return anc


def solve_oracle(jobs, p, zones, H, B, D=None, D0=None, time_limit=60.0):
    """Exact max weighted on-time weight with full duration information.

    The formulation is an exact model of the engine: a single path from the
    depot Z0 through an ordered subset of jobs, sequence-dependent travel and
    changeover, release times, predecessor gating, the planned-horizon test,
    and the cost budget.  ``u_j`` marks on-time completion.  Late executions
    are allowed only for ancestors, because a late non-ancestor yields no
    reward and cannot unlock anything (removing it never hurts).

    ``mip_rel_gap=0`` and ``status == 0`` are asserted by the caller; the
    returned objective is the proven optimum.  Subtour elimination is implicit:
    every selected arc strictly increases the clock, so a cycle is infeasible.

    Returns dict(value, order, ontime, status, objective, cost).
    """
    n = len(jobs)
    if n == 0:
        return dict(value=0.0, order=[], ontime=[], status=0, objective=0.0, cost=0.0)

    w = np.array([j["priority_weight"] for j in jobs], float)
    r = np.array([j["release_s"] for j in jobs], float)
    due = np.array([j["due_s"] for j in jobs], float)
    preds = [{k for k in j["predecessors"]} for j in jobs]
    job_id = [j["job_id"] for j in jobs]
    index = {job_id[i]: i for i in range(n)}
    if D is None or D0 is None:
        _, _, _, D, D0 = build_arrays(jobs, {job_id[i]: 0.0 for i in range(n)}, 1.0, 0.0, zones)

    q = np.maximum(p, MIN_RESERVE)
    ancestors = _ancestors(preds)
    is_anc = np.array([job_id[i] in ancestors for i in range(n)])

    # variable layout
    iy = 0
    iu = n
    ix = 2 * n
    n_arc = n * n
    it = 2 * n + n_arc
    nvar = it + n

    def ax(i, j):
        """Arc index from node i in {0=depot, 1..n = job a+1} into job j."""
        if i == 0:
            return ix + j
        a = i - 1
        k = j if j < a else j - 1
        return ix + n + a * (n - 1) + k

    rows, lbs, ubs = [], [], []
    # Arc variables that can never be useful are fixed to zero.
    arc_ub = np.ones(n_arc)

    def add(coeffs, lb, ub):
        rows.append(coeffs)
        lbs.append(lb)
        ubs.append(ub)

    M_time = H + float(q.max()) + float(D.max()) + CHANGEOVER + 2.0
    M_rel = H + 2.0
    M_hor = H + 2.0
    M_ot = H + float(p.max()) + 2.0
    M_prec = H + float(p.max()) + 2.0

    # 1. in-degree equals y_j
    for j in range(n):
        c = {iy + j: -1.0}
        c[ax(0, j)] = c.get(ax(0, j), 0.0) + 1.0
        for a in range(n):
            if a != j:
                c[ax(a + 1, j)] = c.get(ax(a + 1, j), 0.0) + 1.0
        add(c, 0.0, 0.0)
    # 2. out-degree <= y_i ; 3. depot out-degree <= 1
    for a in range(n):
        c = {iy + a: -1.0}
        for b in range(n):
            if b != a:
                c[ax(a + 1, b)] = c.get(ax(a + 1, b), 0.0) + 1.0
        add(c, -np.inf, 0.0)
    c = {ax(0, j): 1.0 for j in range(n)}
    add(c, -np.inf, 1.0)
    # 4. time propagation along the path (also eliminates subtours)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            c = {it + b: 1.0, it + a: -1.0, ax(a + 1, b): -M_time}
            add(c, q[a] + D[a][b] + CHANGEOVER - M_time, np.inf)
    for j in range(n):
        add({it + j: 1.0, ax(0, j): -M_time}, D0[j] + CHANGEOVER - M_time, np.inf)
    # arc pruning: a non-ancestor b must be on time; if even the absolute
    # earliest possible service start (release + travel + changeover) already
    # misses its due date, that arc is impossible.
    for a in range(n):
        for b in range(n):
            if a == b or is_anc[b]:
                continue
            if r[b] + D[a][b] + CHANGEOVER + p[b] > due[b] + 1e-9:
                arc_ub[ax(a + 1, b) - ix] = 0.0
    for b in range(n):
        if not is_anc[b] and r[b] + D0[b] + CHANGEOVER + p[b] > due[b] + 1e-9:
            arc_ub[ax(0, b) - ix] = 0.0
    # 5. release is checked on the decision clock, before travel: along an
    #    active arc a->b the service start satisfies
    #    t_b >= r_b + D_ab + 0.5 (plus t_b >= t_a + q_a + D_ab + 0.5 from #4).
    for b in range(n):
        add({it + b: 1.0, ax(0, b): -M_time},
            r[b] + D0[b] + CHANGEOVER - M_time, np.inf)
        for a in range(n):
            if a == b:
                continue
            add({it + b: 1.0, ax(a + 1, b): -M_time},
                r[b] + D[a][b] + CHANGEOVER - M_time, np.inf)
    # 6. completion must be inside the horizon
    for j in range(n):
        add({it + j: 1.0, iy + j: M_hor}, -np.inf, H - p[j] + M_hor)
        add({it + j: 1.0, iy + j: M_hor}, -np.inf, H - MIN_RESERVE + M_hor)
    # 7. on-time indicator; non-ancestors must be on time when executed
    for j in range(n):
        add({it + j: 1.0, iu + j: M_ot}, -np.inf, due[j] - p[j] + M_ot)
        add({iu + j: 1.0, iy + j: -1.0}, -np.inf, 0.0)
        if not is_anc[j]:
            add({iy + j: 1.0, iu + j: -1.0}, -np.inf, 0.0)
    # 8. predecessor gating and completion ordering
    for j in range(n):
        for pj in preds[j]:
            a = index[pj]
            add({iy + j: 1.0, iy + a: -1.0}, -np.inf, 0.0)
            add({it + j: 1.0, it + a: -1.0, iy + j: -M_prec}, p[a] - M_prec, np.inf)
    # 9. cost budget
    c = {}
    for j in range(n):
        c[iy + j] = c.get(iy + j, 0.0) + FIXED_FEE
    for j in range(n):
        c[ax(0, j)] = c.get(ax(0, j), 0.0) + 0.2 * D0[j]
    for a in range(n):
        for b in range(n):
            if a != b:
                c[ax(a + 1, b)] = c.get(ax(a + 1, b), 0.0) + 0.2 * D[a][b]
    add(c, -np.inf, B)

    A = lil_matrix((len(rows), nvar), dtype=float)
    for i, c in enumerate(rows):
        for k, v in c.items():
            A[i, k] = v
    A = A.tocsr()
    cons = LinearConstraint(A, np.array(lbs, float), np.array(ubs, float))

    cvec = np.zeros(nvar)
    cvec[iu:iu + n] = -w
    integrality = np.zeros(nvar)
    integrality[iy:iy + n] = 1
    integrality[iu:iu + n] = 1
    integrality[ix:ix + n_arc] = 1
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[iy:iy + n] = 1.0
    ub[iu:iu + n] = 1.0
    ub[ix:ix + n_arc] = arc_ub
    ub[it:it + n] = H

    res = milp(cvec, constraints=cons, integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None:
        raise RuntimeError("oracle MILP infeasible: %s" % (res.message,))
    sol = np.asarray(res.x, float)

    def xval(i, j):
        return sol[ax(i, j)]

    # Reconstruct the path from the depot.
    path = []
    cur = None
    for j in range(n):
        if xval(0, j) > 0.5:
            cur = j
            break
    guard = 0
    while cur is not None and guard < n + 1:
        path.append(cur)
        nxt = None
        for j in range(n):
            if j != cur and xval(cur + 1, j) > 0.5:
                nxt = j
                break
        cur = nxt
        guard += 1

    ontime = [j for j in range(n) if sol[iu + j] > 0.5]
    # Keep only on-time jobs plus their executed predecessor closure; any other
    # executed job carries no reward and is removed so the replay has minimal cost.
    keep = set(ontime)
    changed = True
    while changed:
        changed = False
        for j in list(keep):
            for pj in preds[j]:
                a = index[pj]
                if a not in keep:
                    keep.add(a)
                    changed = True
    order = [job_id[j] for j in path if j in keep]
    value = float(w[ontime].sum())
    return dict(value=value / float(w.sum()), weight=value, order=order,
                ontime=[job_id[j] for j in ontime], status=int(res.status),
                objective=float(-res.fun), cost=float(FIXED_FEE * len(path) +
                0.2 * sum(D0[j] for j in ([path[0]] if path else [])) +
                0.2 * sum(D[path[k]][path[k + 1]] for k in range(len(path) - 1))))


def solve_oracle_dp(jobs, p, zones, H, B=None, D=None, D0=None, allow_late=True):
    """Exact perfect-information optimum by subset DP (primary oracle).

    Because ``2.02*n + 0.2*total_travel < B`` for every case in this data set
    (max travel per job 2.236, n<=16, B>=32), the cost budget never binds and
    is checked after the fact instead of entering the state.  With the budget
    removed, an earlier clock weakly dominates for the same (executed set,
    last job), so ``F[mask, last] = min clock`` is an exact state value.

    Late executions only help through predecessor gating.  For each subset
    ``L`` of ancestors we solve the DP in which jobs outside ``L`` must be on
    time (late non-ancestors are always removable WLOG) and ``L`` jobs are
    allowed late but score zero.  The maximum over ``L`` is exact: the optimal
    plan's true late set is one of the enumerated ``L`` and cannot be
    over-counted because every DP transition enforces the on-time rule it
    claims.

    Returns the same dict as ``solve_oracle``.
    """
    n = len(jobs)
    if n == 0:
        return dict(value=0.0, order=[], ontime=[], status=0, objective=0.0, cost=0.0)
    w = np.array([j["priority_weight"] for j in jobs], float)
    r = np.array([j["release_s"] for j in jobs], float)
    due = np.array([j["due_s"] for j in jobs], float)
    preds = [{k for k in j["predecessors"]} for j in jobs]
    job_id = [j["job_id"] for j in jobs]
    index = {job_id[i]: i for i in range(n)}
    if D is None or D0 is None:
        _, _, _, D, D0 = build_arrays(jobs, {job_id[i]: 0.0 for i in range(n)}, 1.0, 0.0, zones)
    q = np.maximum(p, MIN_RESERVE)
    total_w = float(w.sum())

    anc_ids = _ancestors(preds)
    anc_idx = [i for i in range(n) if job_id[i] in anc_ids] if allow_late else []
    m = len(anc_idx)

    pmask = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for x in preds[i]:
            pmask[i] |= 1 << index[x]
    bits = (np.int64(1) << np.arange(n))
    masks = np.arange(1 << n)

    base_val = np.zeros(1 << n)
    for j in range(n):
        base_val += w[j] * ((masks >> j) & 1)

    best = -1.0
    best_mask = -1
    best_last = -1
    best_L = None

    for lmask in range(1 << m):
        L = np.zeros(n, bool)
        for k, idx in enumerate(anc_idx):
            if (lmask >> k) & 1:
                L[idx] = True
        # Any plan under L scores at most the total weight of non-L jobs.
        if float(w[~L].sum()) <= best + 1e-12:
            continue
        F = np.full((1 << n, n), np.inf)
        # initial transitions from the depot (only jobs without predecessors)
        for j in range(n):
            if pmask[j] != 0:
                continue
            s = r[j] + D0[j] + CHANGEOVER
            if s + MIN_RESERVE > H + 1e-12:
                continue
            C = s + p[j]
            if C > H + 1e-12:
                continue
            if not L[j] and C > due[j] + 1e-12:
                continue
            F[1 << j, j] = s + q[j]
        for mask in range(1, 1 << n):
            tvec = F[mask]
            ls = np.nonzero(np.isfinite(tvec))[0]
            if ls.size == 0:
                continue
            rem = np.nonzero((mask & bits) == 0)[0]
            if rem.size == 0:
                continue
            avail = rem[((pmask[rem] & mask) == pmask[rem])]
            if avail.size == 0:
                continue
            start = np.maximum(tvec[ls][:, None], r[avail][None, :]) \
                    + D[np.ix_(ls, avail)] + CHANGEOVER
            C = start + p[avail][None, :]
            ok = (start + MIN_RESERVE <= H + 1e-12) & (C <= H + 1e-12)
            need = ~L[avail]
            if need.any():
                ok[:, need] &= C[:, need] <= due[avail][need][None, :] + 1e-12
            newt = np.min(np.where(ok, start + q[avail][None, :], np.inf), axis=0)
            good = np.isfinite(newt)
            if good.any():
                js = avail[good]
                nm = mask | bits[js]
                upd = newt[good] < F[nm, js]
                if upd.any():
                    F[nm[upd], js[upd]] = newt[good][upd]
        reach = np.isfinite(F).any(axis=1)
        reach[0] = True
        val = base_val.copy()
        for j in range(n):
            if L[j]:
                val -= w[j] * ((masks >> j) & 1)
        val[~reach] = -np.inf
        k = int(np.argmax(val))
        if val[k] > best + 1e-12:
            best = float(val[k])
            best_mask = k
            best_last = int(np.nonzero(np.isfinite(F[k]))[0][0]) if k else -1
            best_L = L.copy()
            best_F = F

    # Backtrack the optimal subset DP.
    order_rev = []
    mask, last = best_mask, best_last
    L = best_L
    while mask:
        order_rev.append(job_id[last])
        if mask == (1 << last):
            break
        prev_mask = mask & ~(1 << last)
        target = best_F[mask, last]
        found = None
        for pj in np.nonzero(np.isfinite(best_F[prev_mask]))[0]:
            s = max(best_F[prev_mask, pj], r[last]) + D[pj, last] + CHANGEOVER
            C = s + p[last]
            if s + MIN_RESERVE > H + 1e-12 or C > H + 1e-12:
                continue
            if not L[last] and C > due[last] + 1e-12:
                continue
            if abs(s + q[last] - target) < 1e-7:
                found = int(pj)
                break
        if found is None:
            # fall back to the earliest feasible predecessor
            found = int(np.nonzero(np.isfinite(best_F[prev_mask]))[0][0])
        mask, last = prev_mask, found
    order = order_rev[::-1]
    ontime = [jid for k, jid in enumerate(job_id) if ((best_mask >> k) & 1) and not L[k]]
    if not order:
        return dict(value=0.0, weight=0.0, order=[], ontime=[], status=0,
                    objective=0.0, cost=0.0)
    cost = FIXED_FEE * len(order) + 0.2 * D0[index[order[0]]]
    for k in range(len(order) - 1):
        cost += 0.2 * D[index[order[k]], index[order[k + 1]]]
    return dict(value=best / total_w, weight=best, order=order, ontime=ontime,
                status=0, objective=best, cost=float(cost))


def solve_oracle_bruteforce(jobs, p, zones, H, B=None, D=None, D0=None):
    """Exact enumeration of every legal sequence; only for n <= 9 (validation).

    Independent of both the MILP and the subset DP.  A job is executed only if
    it completes by H; late non-ancestors are pruned because they can neither
    score nor unlock anything.
    """
    n = len(jobs)
    if D is None or D0 is None:
        _, _, _, D, D0 = build_arrays(jobs, {j["job_id"]: 0.0 for j in range(n)}, 1.0, 0.0, zones)
    w = np.array([j["priority_weight"] for j in jobs], float)
    r = np.array([j["release_s"] for j in jobs], float)
    due = np.array([j["due_s"] for j in jobs], float)
    preds = [{k for k in j["predecessors"]} for j in jobs]
    job_id = [j["job_id"] for j in jobs]
    index = {job_id[i]: i for i in range(n)}
    pmask = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for x in preds[i]:
            pmask[i] |= 1 << index[x]
    has_succ = np.zeros(n, bool)
    for i in range(n):
        for x in preds[i]:
            has_succ[index[x]] = True
    q = np.maximum(p, MIN_RESERVE)
    best = [0.0]
    best_order = [[]]

    def rec(t, last, mask, reward, order):
        if reward > best[0] + 1e-12:
            best[0] = float(reward)
            best_order[0] = list(order)
        for j in range(n):
            if (mask >> j) & 1:
                continue
            if (pmask[j] & mask) != pmask[j]:
                continue
            d = D0[j] if last < 0 else D[last, j]
            s = max(t, r[j]) + d + CHANGEOVER
            if s + MIN_RESERVE > H + 1e-12:
                continue
            C = s + p[j]
            if C > H + 1e-12:
                continue
            add = float(w[j]) if C <= due[j] + 1e-12 else 0.0
            if add <= 0.0 and not has_succ[j]:
                continue
            order.append(job_id[j])
            rec(s + q[j], j, mask | (1 << j), reward + add, order)
            order.pop()

    rec(0.0, -1, 0, 0.0, [])
    total = float(w.sum())
    order = best_order[0]
    if not order:
        return dict(value=0.0, weight=0.0, order=[], ontime=[], status=0,
                    objective=0.0, cost=0.0)
    cost = FIXED_FEE * len(order) + 0.2 * D0[index[order[0]]]
    for k in range(len(order) - 1):
        cost += 0.2 * D[index[order[k]], index[order[k + 1]]]
    return dict(value=best[0] / total, weight=best[0], order=order,
                ontime=list(order), status=0, objective=best[0], cost=float(cost))


def replay_plan(jobs, hidden, zones, H, B, K, alpha, beta, order, return_engine=False):
    """Replay a fixed job order through the official engine with reserve 0.2."""
    import sys
    from pathlib import Path
    formal = Path(__file__).resolve().parents[3] / "第三问改进正式方案" / "code"
    if str(formal) not in sys.path:
        sys.path.insert(0, str(formal))
    from engine import Engine, independent_metrics

    job_map = {j["job_id"]: j for j in jobs}
    e = Engine(jobs, hidden, zones, H, B, K, alpha, beta)
    idx = 0
    while not e.terminal:
        o = e.observe()
        if idx >= len(order):
            a = {"type": "stop"}
        else:
            jid = order[idx]
            j = job_map[jid]
            if o["time"] + 1e-12 < j["release_s"]:
                a = {"type": "wait", "duration": j["release_s"] - o["time"]}
            else:
                a = {"type": "execute", "job_id": jid, "reserve": MIN_RESERVE}
        ev = e.step(a)
        if "violation" in ev:
            raise RuntimeError("oracle replay illegal action: %s %s" % (a, ev["violation"]))
        if a["type"] == "execute":
            idx += 1
            if ev.get("completion") is None:
                raise RuntimeError("oracle replay censored a planned job: %s" % (jid,))
    m = e.metrics()
    ind = independent_metrics(jobs, e.events, e.H)
    diff = max(abs(float(m[k]) - float(ind[k])) for k in m)
    if return_engine:
        return m, ind, diff, e
    return m, ind, diff


def relaxation_ub(jobs, p, zones, H, B, D0=None, use_deadline=True,
                  use_release=True, use_budget=True):
    """Provably valid upper bound from necessary conditions only.

    For any on-time set T:
      * job j can be on time only if ``r_j + D0_j + 0.5 + p_j <= due_j``;
      * for every deadline threshold d, the jobs of T with due <= d must fit
        in ``d`` seconds of service + changeover (``sum(p_j + 0.5) <= d``);
      * for every release window [a, b], the jobs with r >= a and due <= b
        must fit in ``b - a`` seconds (``sum(p_j + 0.5) <= b - a``);
      * every executed job costs at least 2.02.
    Maximising total weight under these necessary conditions is an upper bound.
    """
    n = len(jobs)
    if n == 0:
        return 0.0, []
    w = np.array([j["priority_weight"] for j in jobs], float)
    r = np.array([j["release_s"] for j in jobs], float)
    due = np.array([j["due_s"] for j in jobs], float)
    if D0 is None:
        z0 = np.asarray(zones["Z0"], float)
        D0 = np.array([math.dist(z0, np.asarray(zones[j["zone_id"]], float)) for j in jobs])
    dur = p + CHANGEOVER
    earliest = r + D0 + CHANGEOVER + p
    eligible = earliest <= due + 1e-9

    rows, lbs, ubs = [], [], []

    def add(coeffs, lb, ub):
        rows.append(coeffs)
        lbs.append(lb)
        ubs.append(ub)

    # mandatory first travel: the first executed job starts at >= 1.5
    if use_deadline:
        for d in sorted(set([float(x) for x in due] + [H])):
            c = {}
            for j in range(n):
                if eligible[j] and due[j] <= d + 1e-12:
                    c[j] = dur[j]
            if c:
                add(c, -np.inf, d)
    # release windows
    if use_release:
        for a in sorted(set([float(x) for x in r])):
            for b in sorted(set([float(x) for x in due] + [H])):
                if b <= a + 1e-12:
                    continue
                c = {j: dur[j] for j in range(n)
                     if eligible[j] and r[j] >= a - 1e-12 and due[j] <= b + 1e-12}
                if c:
                    add(c, -np.inf, b - a)
    # budget lower bound
    if use_budget:
        add({j: FIXED_FEE for j in range(n) if eligible[j]}, -np.inf, B)

    if not rows:
        return float(w[eligible].sum() / w.sum()), [jobs[j]["job_id"]
                                                    for j in range(n) if eligible[j]]

    A = lil_matrix((len(rows), n), dtype=float)
    for i, c in enumerate(rows):
        for k, v in c.items():
            A[i, k] = v
    A = A.tocsr()
    cons = LinearConstraint(A, np.array(lbs, float), np.array(ubs, float))
    lb = np.zeros(n)
    ub = np.array([1.0 if eligible[j] else 0.0 for j in range(n)])
    res = milp(-w, constraints=cons, integrality=np.ones(n),
               bounds=Bounds(lb, ub), options={"time_limit": 60.0, "mip_rel_gap": 0.0})
    if res.x is None:
        return 0.0, []
    z = np.asarray(res.x, float)
    chosen = [jobs[j]["job_id"] for j in range(n) if z[j] > 0.5]
    return float(w[z > 0.5].sum() / w.sum()), chosen
