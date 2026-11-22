"""NumPy port of stochastic.cpp. Rows follow the C++ parent-then-job visit order."""
import numpy as np


def _sum_in_order(array):
    # Match the left-to-right double accumulation that determines near-tie rankings.
    return np.add.accumulate(array, axis=1)[:, -1]


def stochastic_route(n, done, ns, width, time, budget, horizon, release, due, weight,
                     dist, service, pred, cw, optimism, softness):
    if not (1 <= n <= 16 and 1 <= ns <= 32 and width >= 1):
        return -1
    jobs = np.arange(n)
    bits = 1 << jobs
    masks, locations = np.array([done]), np.array([n])
    first = np.array([-1])
    costs, values = np.zeros(1), np.zeros(1)
    clocks = np.full((1, ns), time, dtype=float)
    best, action = -1., -1
    for depth in range(n):
        ready = ((masks[:, None] & bits) == 0) & ((masks[:, None] & pred) == pred)
        parent, job = np.nonzero(ready)
        move = dist[locations[parent], job]
        cost = costs[parent] + 2.02 + .2 * move
        keep = cost <= budget + 1e-10
        if depth == 0:
            keep &= np.maximum(time, release[job]) + move + .7 <= horizon + 1e-12
        parent, job, move, cost = parent[keep], job[keep], move[keep], cost[keep]
        if not len(job):
            break
        end = (np.maximum(clocks[parent], release[job, None]) + move[:, None] + .5
               + service[:ns,job].T)
        reached = end <= horizon + 1e-12
        keep = reached.any(axis=1)
        if not np.any(keep):
            break
        parent, job, cost, end, reached = parent[keep], job[keep], cost[keep], end[keep], reached[keep]
        delta = end - due[job, None]
        punctual = end <= due[job, None] if softness <= 0 else np.clip(.5 - delta/(2*softness), 0., 1.)
        earned = weight[job, None] * (punctual + cw + .0008 *
                            (np.maximum(0., horizon-due[job, None]) - np.maximum(0., delta)))
        value = values[parent] + _sum_in_order(np.where(reached, earned, 0.)) / ns
        mean_clock = _sum_in_order(np.minimum(end, horizon+10.)) / ns
        candidate_first = np.where(release[job] > time+1e-12, -2, job) if depth==0 else first[parent]
        winner = np.argmax(value)
        if value[winner] > best:
            best, action = value[winner], int(candidate_first[winner])
        mask = masks[parent] | bits[job]
        projected = (np.maximum(mean_clock[:, None], release) + dist[job] + .5 + service[ns])
        potential = weight * (np.clip(.5+(due-projected)/24., 0., 1.) +
                              cw*np.clip(.5+(horizon-projected)/26., 0., 1.))
        bound = _sum_in_order(np.where((mask[:, None] & bits)==0, potential, 0.))
        rank = value + optimism*bound - .014*mean_clock - .003*cost
        ordered = np.argsort(-rank, kind="stable")
        keys = (mask*17 + job)[ordered]
        _, positions = np.unique(keys, return_index=True)
        selected = ordered[np.sort(positions)[:width]]
        masks, locations, first = mask[selected], job[selected], candidate_first[selected]
        costs, values, clocks = cost[selected], value[selected], end[selected]
    return action
