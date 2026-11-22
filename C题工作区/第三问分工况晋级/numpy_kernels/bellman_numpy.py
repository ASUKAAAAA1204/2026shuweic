"""NumPy evaluation of the published finite-horizon S3 Bellman equations."""
from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=2)
def legal_masks(parents):
    masks = np.arange(1 << len(parents), dtype=np.int32)
    legal = np.ones(len(masks), dtype=bool)
    for job, required in enumerate(parents):
        legal &= ((masks & (1 << job)) == 0) | ((masks & required) == required)
    return masks[legal]


class Value:
    def __init__(self, horizon, dt, cw, late, families, zones, parents, release, due, weights,
                 distances, count, support, root=0, per_job=False):
        self.horizon, self.dt, self.cw, self.late = horizon, dt, cw, late
        self.zones, self.parents, self.release = zones, parents, release
        self.weights, self.distances, self.due = weights, distances, due
        self.bits = 1 << np.arange(len(parents), dtype=np.int32)
        valid = legal_masks(tuple(map(int, parents)))
        masks = valid[(valid & root) == root]
        self.index = np.full(1 << len(parents), -1, dtype=np.int32)
        self.index[masks] = np.arange(len(masks))
        depths = np.bitwise_count(masks)
        ticks = math.ceil(horizon / dt) + 1
        clock = dt * np.arange(ticks)
        self.values = np.zeros((len(masks), len(distances), ticks), np.float32)
        setup = []
        for j, family in enumerate(families):
            end = clock[:, None] + support[j if per_job else family]
            allowed = end <= horizon + 1.e-10
            reward = self.reward(j, end)
            fixed = np.where(allowed, reward, 0.).mean(axis=1)
            left, a, b = self.interpolation(end, ticks)
            kernel = np.zeros((ticks, ticks))
            row = np.arange(ticks)[:, None]
            np.add.at(kernel, (row, left), a / count)
            np.add.at(kernel, (row, left + 1), b / count)
            start = np.maximum(clock, release[j]) + distances[:, zones[j], None] + .5
            begin, ca, cb = self.interpolation(start, ticks)
            feasible = start + .2 <= horizon
            feasible[:, -1] = False
            setup.append((fixed, kernel.T.copy(), begin, ca * feasible, cb * feasible))

        for depth in range(len(parents)-1, int(root).bit_count()-1, -1):
            layer = masks[depths == depth]
            for j, bit in enumerate(self.bits):
                options = layer[((layer & bit) == 0) & ((layer & parents[j]) == parents[j])]
                fixed, kernel, begin, ca, cb = setup[j]
                for start in range(0, len(options), 256):
                    chunk = options[start:start + 256]
                    child = self.values[self.index[chunk | bit], zones[j], :]
                    # C++ stores both the expected reward and the Bellman values in float32.
                    q = (child @ kernel + fixed).astype(np.float32)
                    value = q[:,begin] * ca + q[:,begin + 1] * cb
                    rows = self.index[chunk]
                    self.values[rows] = np.maximum(self.values[rows], value)

    def interpolation(self, times, ticks):
        scaled = np.maximum(times, 0.) / self.dt
        left = np.clip(scaled.astype(np.int32), 0, ticks-2)
        fraction = scaled - left
        return left, (1.-fraction)*(times < self.horizon), fraction*(times < self.horizon)

    def reward(self, j, end):
        return self.weights[j] * ((end <= self.due[j]+1.e-10) + self.cw + self.late *
                    (np.maximum(0., self.horizon-self.due[j])-np.maximum(0.,end-self.due[j])))

    def first(self, mask, loc, now, budget, samples, pace, shift):
        if mask < 0 or mask >= len(self.index) or self.index[mask] < 0:
            return -1
        dist = self.distances[loc,self.zones]
        start = np.maximum(self.release,now) + dist + .5
        eligible = (((mask & self.bits) == 0) & ((mask & self.parents) == self.parents) &
                    (start+.2 <= self.horizon+1.e-12) & (2.02+.2*dist <= budget+1.e-12))
        jobs = np.flatnonzero(eligible)
        if not jobs.size:
            return -1
        end = start[jobs,None] + samples[jobs]
        virtual = np.clip(self.horizon-(self.horizon-end)/pace+shift, 0., self.horizon)
        k,a,b = self.interpolation(virtual,self.values.shape[-1])
        row = self.index[mask | self.bits[jobs],None]
        zone = self.zones[jobs,None]
        future = self.values[row,zone,k]*a + self.values[row,zone,k+1]*b
        scores = np.where(end<=self.horizon+1.e-10, self.reward(jobs[:,None],end)+future, 0.).sum(axis=1)
        best, first = -1., -1
        for j,score in zip(jobs,scores):
            if score > best+1.e-10:
                best, first = score, (-2 if self.release[j]>now+1.e-12 else int(j))
        return first


class ArrayBackend:
    def fit_value(self, *args):
        return Value(*args)

    def refit_value(self, *args):
        return Value(*args,per_job=True)

    def best_job(self, handle, mask, loc, now, budget, count, samples, pace, shift):
        return handle.first(mask,loc,now,budget,samples,pace,shift)

    def free_value(self, handle):
        pass
