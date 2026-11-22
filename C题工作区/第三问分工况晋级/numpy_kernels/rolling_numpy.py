"""NumPy/standard-Python port of the third-question ``rolling.cpp`` kernel.

``rolling`` keeps the C ABI's positional argument order, so NumPy callers can
use this module without changing the call site.  The implementation preserves
the C++ uint32 xorshift stream, stable priority ordering, scoring thresholds,
waiting behaviour and first-feasible-job return code.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rolling"]

_MISS = object()
_U32 = 0xFFFFFFFF
_EPS = 1e-12


def _floats(values):
    return np.asarray(values, dtype=np.float64).ravel().tolist()


def _ints(values):
    return np.asarray(values, dtype=np.int64).ravel().tolist()


def rolling(n, done, ns, now, left, horizon, cw, release, due, weight,
            distance, samples, parent, typical, steps, seed):
    """Return the job chosen by the C++ ``rolling`` kernel, or -1."""
    n = int(n)
    done = int(done)
    active = [i for i in range(n) if not (done >> i) & 1]
    if not active:
        return -1

    ns = int(ns)
    steps = int(steps)
    now = float(now)
    left = float(left)
    horizon = float(horizon)
    cw = float(cw)

    release = _floats(release)
    due = _floats(due)
    weight = _floats(weight)
    distance = _floats(distance)
    samples = _floats(samples)
    parent = _ints(parent)
    typical = _floats(typical)

    bits = [1 << j for j in range(n)]
    due_eps = [d + _EPS for d in due]
    # max(0., horizon - due[j]), fixed for the whole invocation.
    horizon_due = [max(0.0, horizon - d) for d in due]
    # 2.02 + .2 * d for each leg, also fixed across evaluations.
    costs = [2.02 + 0.2 * d for d in distance]

    depot = n * n
    horizon_eps = horizon + _EPS
    horizon_lo = horizon - _EPS
    wake_default = horizon + 1.0
    m = len(active)

    # The C++ evaluator repeatedly scores identical permutations (notably the
    # incumbent at each restart and revisited trials).  Score depends only on
    # the active-job sequence; done jobs in the tail are skipped by next().
    cache = {}

    def score(order):
        key = tuple(order)
        cached = cache.get(key, _MISS)
        if cached is not _MISS:
            return cached

        value = 0.0
        for s in range(ns):
            mask = done
            loc = n
            t = now
            remaining = left
            base = s * n
            while t < horizon_lo:
                cutoff = t + _EPS
                chosen = -1
                chosen_d = 0.0
                # Problem::next() -- keep its candidate order and checks.
                for job in order:
                    if mask & bits[job]:
                        continue
                    rj = release[job]
                    if rj > cutoff:
                        continue
                    if (mask & parent[job]) != parent[job]:
                        continue
                    off = loc * n + job
                    d = distance[off]
                    if (t + d + 0.7 <= horizon_eps
                            and remaining + _EPS >= costs[off]):
                        chosen = job
                        chosen_d = d
                        break

                if chosen < 0:
                    # Wake time is only observed when no candidate exists;
                    # compute the same minimum over future-release jobs.
                    wake = wake_default
                    for job in order:
                        if mask & bits[job]:
                            continue
                        rj = release[job]
                        if rj > cutoff and rj < wake:
                            wake = rj
                    if wake > horizon:
                        break
                    t = wake
                    continue

                off = loc * n + chosen
                d = chosen_d
                start = t + d + 0.5
                service = samples[base + chosen]
                end = start + service
                if end > horizon_eps:
                    break
                value += weight[chosen] * (
                    (1.0 if end <= due_eps[chosen] else 0.0)
                    + cw
                    + 0.00065
                    * (
                        horizon_due[chosen]
                        - (end - due[chosen] if end > due[chosen] else 0.0)
                    )
                )
                t = start + (service if service > 0.2 else 0.2)
                remaining -= costs[off]
                loc = chosen
                mask |= bits[chosen]

        cache[key] = value
        return value

    # Four deterministic restarts, mirroring rolling.cpp.
    best = active.copy()
    best_value = -1e100
    state = int(seed) & _U32

    for restart in range(4):
        cur = best.copy()
        if restart < 3:
            priority = [0.0] * n
            if restart == 0:
                for i in range(n):
                    priority[i] = -due[i] + 0.8 * weight[i]
            elif restart == 1:
                for i in range(n):
                    slack = max(0.0, due[i] - now - typical[i])
                    priority[i] = weight[i] / (3.0 + slack)
            else:
                for i in range(n):
                    # uint32 xorshift: seed ^= seed << 13; seed ^= seed >> 17;
                    # seed ^= seed << 5 (with 32-bit wrapping).
                    state ^= (state << 13) & _U32
                    state ^= state >> 17
                    state ^= (state << 5) & _U32
                    priority[i] = (
                        weight[i] / (1.0 + typical[i]) + 0.08 * (state % 10)
                    )
            # std::stable_sort(..., priority[a] > priority[b]): Python's sort
            # is stable, including with reverse=True.
            cur.sort(key=priority.__getitem__, reverse=True)

        value = score(cur)
        if value > best_value:
            best = cur.copy()
            best_value = value

        k = 0
        while k < steps and m > 1:
            trial = cur.copy()
            state ^= (state << 13) & _U32
            state ^= state >> 17
            state ^= (state << 5) & _U32
            i = state % m
            state ^= (state << 13) & _U32
            state ^= state >> 17
            state ^= (state << 5) & _U32
            j = state % m
            if i != j:
                if k % 3 == 0:
                    trial[i], trial[j] = trial[j], trial[i]
                elif i < j:
                    # std::rotate(begin+i, begin+i+1, begin+j+1)
                    trial[i:j + 1] = trial[i + 1:j + 1] + trial[i:i + 1]
                else:
                    # std::rotate(begin+j, begin+i, begin+i+1)
                    trial[j:i + 1] = trial[i:i + 1] + trial[j:i]
                v = score(trial)
                if v >= value - 1e-10:
                    cur = trial
                    value = v
                if v > best_value + 1e-10:
                    best = trial
                    best_value = v
            k += 1

    # Final Problem::next(best, done, now, n, left, wake).
    cutoff = now + _EPS
    for job in best:
        if done & bits[job]:
            continue
        if release[job] > cutoff:
            continue
        if (done & parent[job]) != parent[job]:
            continue
        off = depot + job
        d = distance[off]
        if now + d + 0.7 <= horizon_eps and left + _EPS >= costs[off]:
            return job
    return -1
