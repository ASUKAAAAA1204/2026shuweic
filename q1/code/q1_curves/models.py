"""Along-track functional regression and a fixed-basis damping residual."""
import numpy as np
from compare import basic, vector
from optimizer.cosine import fit

PAST = np.array([-.20, -.15, -.10, -.05])
FUTURE = np.arange(1, 6) / 5
KNOTS = np.r_[0., FUTURE]


def history(obs, direction, scale):
    along = (obs.x - obs.x[-1]) @ direction / scale
    return np.interp(PAST, obs.t - obs.t[-1], along)


def basis(ages, order):
    return np.cos(np.pi * ages[:, None] * np.arange(order))


class Curve:
    def __init__(self, name, order, windows, ids, bank, scale, seed):
        self.name, self.order, self.scale = name, order, scale
        self.threshold = bank.threshold
        self.ready = False
        self.nbytes = self.parameters = 0
        if len(ids) < 5:
            return
        inputs, labels, times = [], [], []
        for i in ids:
            w = windows[i]
            v, f = vector(w.obs, 'M1')
            d = v / f[0]
            if name == 'F':
                inputs.append(history(w.obs, d, scale))
                delta = (w.reference - w.obs.x[-1]) @ d / scale
                labels.append(np.interp(FUTURE, w.ages, delta))
            else:
                ix = np.linspace(1, len(w.ages)-1, 6).round().astype(int)
                h = w.ages[ix]
                correction = w.reference[ix] - w.obs.x[-1] - basic(v, h)
                labels.extend((correction / (scale * -np.expm1(-h/.2))[:, None]) @ d)
                inputs.extend([f] * len(h))
                times.extend(h)
        x, y = np.asarray(inputs), np.asarray(labels)
        self.mu = x.mean(axis=0)
        if name == 'F':
            self.center = y.mean(axis=0)
            xp, yp = x-self.mu, y-self.center
            _, sx, vx = np.linalg.svd(xp, full_matrices=False)
            _, sy, vy = np.linalg.svd(yp, full_matrices=False)
            rx = min(order[0], np.count_nonzero(sx > sx[0]*1e-10))
            ry = min(order[1], np.count_nonzero(sy > sy[0]*1e-10))
            if min(rx, ry) == 0:
                return
            standardized = vx[:rx].T / (sx[:rx]/np.sqrt(len(x)))
            z = xp @ standardized
            mapping = np.linalg.solve(z.T @ z + len(x)*.01*np.eye(rx), z.T @ (yp @ vy[:ry].T))
            self.mapping = standardized @ mapping @ vy[:ry]
            arrays = (self.mu, self.center, self.mapping)
        else:
            self.sd = x.std(axis=0)
            self.sd[self.sd < 1e-12] = 1.
            self.weights = fit((x-self.mu)/self.sd, basis(np.asarray(times), order), y, seed)
            arrays = (self.mu, self.sd, *self.weights)
            self.parameters = sum(a.size for a in self.weights)
        self.nbytes = sum(a.nbytes for a in arrays)
        self.ready = True

    def active(self, f):
        return self.ready and np.isfinite(f).all() and min(f[:3]) > 0 and min(f[:3]) >= self.threshold

    def forecast(self, obs, ages):
        v, f = vector(obs, 'M1')
        if not self.active(f):
            return obs.x[-1] + basic(v, ages)
        d = v / f[0]
        if self.name == 'F':
            history_scores = history(obs, d, self.scale) - self.mu
            knots = self.center + history_scores @ self.mapping
            displacement = self.scale * np.interp(ages, KNOTS, np.r_[0., knots])
            return obs.x[-1] + displacement[:, None]*d
        w1, b1, w2, b2 = self.weights
        coefficients = np.tanh(((f-self.mu)/self.sd) @ w1 + b1) @ w2 + b2
        residual = .5*self.scale * -np.expm1(-ages/.2) * (basis(ages, self.order) @ coefficients)
        return obs.x[-1] + basic(v, ages) + residual[:, None]*d
