"""Prefix-duration model, fitted on the TRAIN split only.

Features are the disclosed relative geometry / motion / quality summaries of the
0.2 s, 0.6 s and 1.0 s recorded prefixes (identical construction to
prefix_duration.features; no absolute timestamps, source identity, description or
end time).  Target is log duration.  Candidate learners are the two models that
were pre-registered on the prefix-information route: Ridge(10) and
HistGradientBoosting(80 iters, 7 leaves, min_leaf 15, l2 10).

Honesty rules enforced here:
  * fitting rows always come from split == "train" and eligible == 1;
  * out-of-fold residuals use leave-one-environment-out inside the fitting rows
    only, so the residual pool used online is never in-sample;
  * validation prefixes are only read at replay time as online observations;
    they never enter fit(), model choice or any parameter.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIVATE = ROOT / "处理结果" / "v1" / "private"
PREFIXES = (.2, .6, 1.0)
FAMILIES = ("A", "L", "P")


def features(z):
    """Same 33-dimensional prefix descriptor as the pre-registered route."""
    result = []
    centers = []
    for side in ["left_hand", "right_hand", "head"]:
        t = z[side + "__t_s"]
        p = z[side + "__position"]
        x = p.mean(1)
        good = z.get(side + "__quality_known", np.ones(len(t), bool)) & ~z.get(side + "__bad", np.zeros(len(t), bool))
        good &= np.isfinite(x).all(1)
        x = x[good]
        t = t[good]
        if len(t) < 2:
            result.extend([np.nan] * 9)
            centers.append(None)
            continue
        dt = np.diff(t)
        v = np.diff(x, axis=0) / dt[:, None]
        s = np.linalg.norm(v, axis=1)
        q = z[side + "__quaternion"][good, 0]
        rotation = 2 * np.arccos(np.clip(abs(np.sum(q[-1] * q[0])), 0, 1))
        shape = np.linalg.norm(p[good] - p[good].mean(1)[:, None, :], axis=-1)
        result.extend([np.median(s), np.std(s), np.max(s), np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1)),
                       np.linalg.norm(x[-1] - x[0]), rotation, np.mean(shape), np.std(shape), np.mean(~good)])
        centers.append((t, x))
    for a, b in [(0, 2), (1, 2), (0, 1)]:
        if centers[a] is None or centers[b] is None:
            result.extend([np.nan] * 3)
            continue
        t, x = centers[a]
        tt, y = centers[b]
        yy = np.stack([np.interp(t, tt, y[:, k]) for k in range(3)], 1)
        d = np.linalg.norm(x - yy, axis=1)
        result.extend([np.median(d), np.std(d), d[-1] - d[0]])
    return np.array(result, float)


def load_features(unit_ids, prefixes=PREFIXES):
    """(unit_id, prefix) -> 33-dim vector, read from the disclosed npz files."""
    out = {}
    for uid in unit_ids:
        for p in prefixes:
            path = PRIVATE / f"{uid}_{p}.npz"
            if not path.exists():
                out[(uid, p)] = np.full(33, np.nan)
                continue
            with np.load(path, allow_pickle=False) as z:
                out[(uid, p)] = features(z)
    return out


def _design(rows, cache, prefix):
    x = np.stack([cache[(uid, prefix)] for uid in rows.unit_id])
    fam = np.stack([(rows.family_id == f).to_numpy(float) for f in FAMILIES], axis=1)
    return np.c_[x, fam]


def _fit_one(design, y, kind):
    if kind == "ridge":
        reg = Ridge(alpha=10)
    else:
        reg = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, min_samples_leaf=15,
                                           l2_regularization=10, random_state=20261122,
                                           early_stopping=False)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), reg)
    model.fit(design, np.log(y))
    return model


def _predict(model, design, lo, hi):
    return np.exp(np.clip(model.predict(design), np.log(lo), np.log(hi)))


class PrefixModel:
    """Train-only model bundle: point predictor + out-of-fold log-residual pools."""

    def __init__(self, rows, cache, kind="tree", prefixes=PREFIXES):
        self.rows = rows.reset_index(drop=True)
        self.cache = cache
        self.kind = kind
        self.prefixes = prefixes
        self.y = self.rows.duration_s.to_numpy(float)
        self.lo, self.hi = float(self.y.min()), float(self.y.max())
        self.models = {}
        self.residuals = {}
        self.diagnostics = {}
        for p in prefixes:
            design = _design(self.rows, cache, p)
            self.models[p] = _fit_one(design, self.y, kind)
            # Leave-one-environment-out out-of-fold predictions for residuals.
            oof = np.full(len(self.rows), np.nan)
            for env in sorted(self.rows.environment_id.unique()):
                te = (self.rows.environment_id == env).to_numpy()
                tr = ~te
                m = _fit_one(design[tr], self.y[tr], kind)
                oof[te] = _predict(m, design[te], self.lo, self.hi)
            resid = np.log(self.y) - np.log(oof)
            self.residuals[p] = {f: resid[(self.rows.family_id == f).to_numpy()] for f in FAMILIES}
            fam_median = self.rows.groupby("family_id").duration_s.median()
            base = self.rows.family_id.map(fam_median).to_numpy(float)
            per_family = {}
            for f in FAMILIES:
                m = (self.rows.family_id == f).to_numpy()
                if m.sum() < 3:
                    continue
                per_family[f] = dict(
                    n=int(m.sum()),
                    model_mae=float(np.mean(np.abs(self.y[m] - oof[m]))),
                    family_mae=float(np.mean(np.abs(self.y[m] - base[m]))),
                    improvement_pct=float(100 * (1 - np.mean(np.abs(self.y[m] - oof[m]))
                                                 / np.mean(np.abs(self.y[m] - base[m])))),
                )
            self.diagnostics[p] = dict(
                model_mae=float(np.mean(np.abs(self.y - oof))),
                family_mae=float(np.mean(np.abs(self.y - base))),
                model_rmse=float(np.sqrt(np.mean((self.y - oof) ** 2))),
                family_rmse=float(np.sqrt(np.mean((self.y - base) ** 2))),
                improvement_pct=float(100 * (1 - np.mean(np.abs(self.y - oof)) / np.mean(np.abs(self.y - base)))),
                spearman=float(np.corrcoef(np.argsort(np.argsort(oof)), np.argsort(np.argsort(self.y)))[0, 1]),
                oof_rows=int(np.isfinite(oof).sum()),
                per_family=per_family,
            )
            self.oof = getattr(self, "oof", {})
            self.oof[p] = oof

    def point_from_features(self, vec, prefix, family, fallback):
        vec = np.asarray(vec, float)
        if vec.size != 33 or not np.isfinite(vec).any():
            return fallback
        design = np.r_[vec, [1.0 if family == f else 0.0 for f in FAMILIES]][None, :]
        return float(_predict(self.models[prefix], design, self.lo, self.hi)[0])

    def point(self, unit_id, prefix, family, fallback):
        vec = self.cache.get((unit_id, prefix))
        return self.point_from_features(vec, prefix, family, fallback)

    def duration_dist_from_features(self, vec, prefix, family, fallback_pool):
        """Conditional source-duration distribution: pred * exp(train OOF residual)."""
        fallback = float(np.median(fallback_pool))
        pred = self.point_from_features(vec, prefix, family, fallback)
        r = self.residuals[prefix][family]
        if len(r) < 8:
            r = np.concatenate([self.residuals[prefix][f] for f in FAMILIES])
        return np.clip(pred * np.exp(r), self.lo, self.hi)

    def duration_dist(self, unit_id, prefix, family, fallback_pool):
        vec = self.cache.get((unit_id, prefix))
        if vec is None:
            vec = np.full(33, np.nan)
        return self.duration_dist_from_features(vec, prefix, family, fallback_pool)

    def summary(self):
        return dict(kind=self.kind, train_rows=int(len(self.rows)),
                    environments=sorted(self.rows.environment_id.unique().tolist()),
                    prefixes={str(p): self.diagnostics[p] for p in self.prefixes})
