"""Joint-scenario one-step lookahead for the Q3 stochastic scheduling task.

Role B: instead of a quantile grid on the first job's duration with a
point-estimate greedy tail, draw ``K`` *joint* realisations of the durations of
every unfinished job from the current family belief, and evaluate each candidate
first action by rolling a deadline-aware base policy to the terminal state in
each realisation.  The value of a candidate is the mean realised on-time weight.

Correctness contract (audit items)
---------------------------------
* Clock: after executing ``j`` the clock advances to ``start + max(p, .2)`` and
  the completion instant is ``start + p``; movement and preparation are never
  counted twice.
* Chains: a job unlocks its successors only in realisations where
  ``start + p <= H`` (censored jobs stay attempted and unlock nothing).  A
  successor's value is therefore multiplied by the *realised* ancestry, not by
  a probability treated as deterministic completion.
* Budget: infeasible jobs are removed from the candidate set *before* ranking;
  a candidate never aborts the tail because it becomes unaffordable.
* H truncation: every start/clock/completion check uses ``H``; the clock is
  capped at ``H``.
* Mapping: every duration is mapped to scenario seconds as ``p = alpha*d +
  beta`` before any survival/clock check; ``alpha``/``beta`` come from
  ``observe()`` and are legal.
* Belief: 128-point discrete mixture of the family pool, a global prior
  (``strength``) and in-trial completed-duration feedback
  (``feedback_strength``).  It is a discrete approximation, not an exact weight
  preservation.  ``feedback_strength <= 0`` means "do not use feedback".

The policy reads only the public observation, the training family pools and the
in-trial feedback history.  It never reads hidden durations, case ids, target
values or validation/test outcomes.
"""
import math

import numpy as np

GRID = 128
SCENARIO_SEED = 20261122
RESERVE = 0.2
FIXED_FEE = 2.0
RESERVE_RATE = 0.1
DIST_RATE = 0.2


def mixture(family, pools, history, strength=20, feedback_strength=3):
    """128-point discrete mixture: global prior + family pool + feedback."""
    global_values = np.concatenate([np.asarray(v, float) for v in pools.values()])
    v = np.asarray(pools[family], float)
    prior_count = int(round(GRID * strength / (len(v) + strength)))
    values = np.r_[
        np.quantile(global_values, (np.arange(prior_count) + .5) / prior_count)
        if prior_count else np.array([]),
        np.quantile(v, (np.arange(GRID - prior_count) + .5) / (GRID - prior_count)),
    ]
    h = history.get(family) if history else None
    if h and feedback_strength > 0:
        h = np.asarray(h, float)
        count = int(round(GRID * len(h) / (len(h) + feedback_strength)))
        if count > 0:
            values = np.r_[
                np.quantile(h, (np.arange(count) + .5) / count),
                np.quantile(values, (np.arange(GRID - count) + .5) / (GRID - count))
                if count < GRID else np.array([]),
            ]
    return values


class JointScenarioPolicy:
    """One-step lookahead with K joint duration scenarios and a rollout tail."""

    def __init__(self, K=32, strength=20, feedback_strength=3, seed=SCENARIO_SEED,
                 base="deadline", stratified=True):
        self.K = int(K)
        self.strength = int(strength)
        self.feedback_strength = int(feedback_strength)
        self.seed = int(seed)
        self.base = base
        self.stratified = bool(stratified)
        self.rng = np.random.default_rng(self.seed)
        self.last_values = []
        self.last_candidates = []
        self.last_se = 0.0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _dist(a, b):
        return math.dist(a, b)

    def _score_arrays(self, start, D, due, weight, legal, belief_s, fam, med,
                      children, zones, attempted, mean_step, n):
        """Per-job expected value / occupancy, vectorised over (A, K)."""
        A, K = start.shape[0], start.shape[1]
        score = np.full((A, K, n), -np.inf)
        for j in range(n):
            vj = belief_s[fam[j]]
            mj = med[j]
            slack = due[j] - start[:, :, j]
            sv = np.searchsorted(vj, slack + 1e-12, side="right") / len(vj)
            if self.base == "deadline":
                svd = np.searchsorted(vj, slack - mean_step + 1e-12, side="right") / len(vj)
                val = weight[j] * (sv + (sv - svd))
                # Closure: a late ancestor keeps value if an immediate child can
                # still be on time.  Point-estimate parent duration, probability
                # for the child (never "probability as certain completion").
                for c in children[j]:
                    vc = belief_s[fam[c]]
                    travel = self._dist(zones[j], zones[c]) + .5
                    slackc = due[c] - (start[:, :, j] + mj + travel)
                    chance = np.searchsorted(vc, slackc + 1e-12, side="right") / len(vc)
                    val = val + weight[c] * chance * (~attempted[:, :, c])
            elif self.base == "survival":
                val = weight[j] * sv
                for c in children[j]:
                    vc = belief_s[fam[c]]
                    travel = self._dist(zones[j], zones[c]) + .5
                    slackc = due[c] - (start[:, :, j] + mj + travel)
                    chance = np.searchsorted(vc, slackc + 1e-12, side="right") / len(vc)
                    val = val + weight[c] * chance * (~attempted[:, :, c])
            else:  # density control: no deadline information
                val = np.full_like(sv, weight[j])
            sc = val / (D[:, :, j] + .5 + mj)
            score[:, :, j] = np.where(legal[:, :, j], sc, -np.inf)
        return score

    # ------------------------------------------------------------------- policy
    def select(self, o, pools, history=None):
        history = history or {}
        K = self.K
        jobs = o["jobs"]
        n = len(jobs)
        H = float(o["H"])
        B = float(o["B"])
        t0 = float(o["time"])
        cost0 = float(o["cost"])
        alpha = float(o["alpha"])
        beta = float(o["beta"])

        zones = np.asarray([j["zone_xy"] for j in jobs], float)
        cur = np.asarray(o["zone_xy"], float)
        due = np.asarray([min(float(j["due_s"]), H) for j in jobs], float)
        rel = np.asarray([float(j["release_s"]) for j in jobs], float)
        weight = np.asarray([float(j["weight"]) for j in jobs], float)
        fam = [j["family"] for j in jobs]
        att0 = np.asarray([bool(j["attempted"]) for j in jobs], bool)
        done0 = np.asarray([bool(j["completed"]) for j in jobs], bool)
        id2idx = {j["job_id"]: i for i, j in enumerate(jobs)}
        preds = [[id2idx[p] for p in j["predecessors"]] for j in jobs]
        children = [[] for _ in range(n)]
        for c in range(n):
            for p in preds[c]:
                children[p].append(c)

        # Root feasibility (screen before ranking; never abort a tail later).
        avail = []
        for i, j in enumerate(jobs):
            if att0[i] or rel[i] > t0 + 1e-12:
                continue
            if not all(done0[p] for p in preds[i]):
                continue
            D = self._dist(cur, zones[i])
            s = t0 + D + .5
            fee = FIXED_FEE + RESERVE_RATE * RESERVE + DIST_RATE * D
            if s + RESERVE > H + 1e-12 or cost0 + fee > B + 1e-12:
                continue
            avail.append(i)
        future = [rel[i] for i in range(n)
                  if not att0[i] and rel[i] > t0 + 1e-12 and rel[i] <= H + 1e-12]
        if not avail and not future:
            self.last_values, self.last_candidates, self.last_se = [], [], 0.0
            return {"type": "stop"}
        next_release = min(future) if future else None

        # Duration belief in scenario seconds, and K joint draws per job.
        belief_s = {}
        med = {}
        for f in pools:
            d = mixture(f, pools, history, self.strength, self.feedback_strength)
            vals = alpha * d + beta
            belief_s[f] = np.sort(vals)
            med[f] = float(np.mean(vals))
        p_samples = np.zeros((n, K))
        for i in range(n):
            if att0[i]:
                continue
            vals = belief_s[fam[i]]
            if self.stratified:
                # One draw per equal-probability stratum, independently permuted
                # per job: stratified marginals, no artificial cross-job coupling.
                u = (np.arange(K) + self.rng.random(K)) / K
                idx = np.minimum((u * len(vals)).astype(int), len(vals) - 1)
                idx = self.rng.permutation(idx)
            else:
                idx = self.rng.integers(0, len(vals), size=K)
            p_samples[i] = vals[idx]

        # Baseline scalar delay for the deadline-aware tail (root availability).
        med_job = np.asarray([med[fam[i]] for i in range(n)], float)
        if avail:
            mean_step = float(np.mean([med_job[i] + .5 + self._dist(cur, zones[i]) for i in avail]))
        else:
            mean_step = 1.0

        cands = [(i, "exec") for i in avail]
        if next_release is not None:
            cands.append((-1, "wait"))
        A = len(cands)

        # Root-parallel rollout state.
        t = np.zeros((A, K))
        cost = np.zeros((A, K))
        pos = np.zeros((A, K, 2))
        att = np.broadcast_to(att0, (A, K, n)).copy()
        done = np.broadcast_to(done0, (A, K, n)).copy()
        reward = np.zeros((A, K))
        alive = np.ones((A, K), bool)
        for a, (i, kind) in enumerate(cands):
            if kind == "exec":
                D0 = self._dist(cur, zones[i])
                s0 = t0 + D0 + .5
                p0 = p_samples[i]
                comp = s0 + p0 <= H + 1e-12
                reward[a] = weight[i] * comp * (s0 + p0 <= due[i] + 1e-12)
                t[a] = np.minimum(H, s0 + np.maximum(p0, RESERVE))
                cost[a] = cost0 + FIXED_FEE + RESERVE_RATE * RESERVE + DIST_RATE * D0
                pos[a] = zones[i]
                att[a, :, i] = True
                done[a, :, i] = comp
            else:
                t[a] = next_release
                cost[a] = cost0
                pos[a] = cur
            alive[a] = t[a] < H - 1e-12

        # Roll the deadline-aware base policy to the terminal state per scenario.
        for _ in range(2 * n + 4):
            if not alive.any():
                break
            D = np.linalg.norm(pos[:, :, None, :] - zones[None, None, :, :], axis=3)
            start = t[:, :, None] + D + .5
            fee = FIXED_FEE + RESERVE_RATE * RESERVE + DIST_RATE * D
            legal = ((~att)
                     & (rel[None, None, :] <= t[:, :, None] + 1e-12)
                     & (start + RESERVE <= H + 1e-12)
                     & (cost[:, :, None] + fee <= B + 1e-12))
            for j in range(n):
                if preds[j]:
                    legal[:, :, j] &= done[:, :, preds[j]].all(axis=2)
            score = self._score_arrays(start, D, due, weight, legal, belief_s, fam, med_job,
                                       children, zones, att, mean_step, n)
            best = score.max(axis=2)
            choice = score.argmax(axis=2)
            has = alive & np.isfinite(best) & (best > 0)

            idx = np.argwhere(has)
            if len(idx):
                aa, kk = idx[:, 0], idx[:, 1]
                jj = choice[aa, kk]
                pdraw = p_samples[jj, kk]
                s = start[aa, kk, jj]
                C = s + pdraw
                ok = C <= H + 1e-12
                reward[aa, kk] += weight[jj] * ok * (C <= due[jj] + 1e-12)
                att[aa, kk, jj] = True
                done[aa, kk, jj] = ok
                cost[aa, kk] += FIXED_FEE + RESERVE_RATE * RESERVE + DIST_RATE * D[aa, kk, jj]
                t[aa, kk] = np.minimum(H, s + np.maximum(pdraw, RESERVE))
                pos[aa, kk] = zones[jj]
                alive[aa, kk] &= t[aa, kk] < H - 1e-12

            # Cells with no affordable/valuable action wait for the next release.
            nohas = alive & ~has
            if nohas.any():
                relmask = ((~att)
                           & (rel[None, None, :] > t[:, :, None] + 1e-12)
                           & (rel[None, None, :] <= H + 1e-12))
                for j in range(n):
                    if preds[j]:
                        relmask[:, :, j] &= done[:, :, preds[j]].all(axis=2)
                nxt = np.where(relmask, rel[None, None, :], np.inf).min(axis=2)
                move = nohas & np.isfinite(nxt)
                t[move] = nxt[move]
                alive[nohas & ~np.isfinite(nxt)] = False
            alive &= (~att).any(axis=2)

        values = reward.mean(axis=1)
        se = (reward.std(axis=1, ddof=1) / math.sqrt(K)) if K > 1 else np.zeros(A)
        self.last_values = [float(x) for x in values]
        self.last_candidates = [cands[a][0] for a in range(A)]
        self.last_se = float(np.mean(se)) if A else 0.0

        # Deterministic selection: highest value, then execute over wait, then due/id.
        best_a = None
        best_key = None
        for a, (i, kind) in enumerate(cands):
            key = (-values[a], 0 if kind == "exec" else 1,
                   due[i] if i >= 0 else math.inf,
                   jobs[i]["job_id"] if i >= 0 else "")
            if best_key is None or key < best_key:
                best_key = key
                best_a = a
        i, kind = cands[best_a]
        if kind == "exec":
            return {"type": "execute", "job_id": jobs[i]["job_id"], "reserve": RESERVE}
        return {"type": "wait", "duration": float(next_release - t0)}


def select(o, pools, history=None, K=32, strength=20, feedback_strength=3,
           seed=SCENARIO_SEED, base="deadline", policy=None):
    """Functional wrapper; pass ``policy`` to keep one RNG stream across a run."""
    if policy is None:
        policy = JointScenarioPolicy(K=K, strength=strength,
                                     feedback_strength=feedback_strength,
                                     seed=seed, base=base)
    return policy.select(o, pools, history), policy
