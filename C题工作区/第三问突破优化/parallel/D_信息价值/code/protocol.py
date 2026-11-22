"""Shared evaluation protocol for role D (information value).

Replicates the required protocol verbatim:
  * 3 cases S1/S2/S3, 30 public validation seeds, 5 mappings, paired baseline.
  * hidden duration sampling is exactly confirm_lookahead.hidden_for: one
    random.Random(seed), same jobs_for order, one draw per job from the
    validation&eligible rows of its family (with replacement).

Extension (does not change the draw order or the duration values): the sampled
segment row is retained so its recorded 0.2/0.6/1.0 prefix features can be
attached to the engine's observation payload.  Features are the only thing
attached; the policy still receives durations solely through the engine.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FORMAL = ROOT / "第三问改进正式方案" / "code"
if str(FORMAL) not in sys.path:
    sys.path.insert(0, str(FORMAL))

from support import load, jobs_for  # noqa: E402

MAPPINGS = [(1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.)]
PREFIXES = (.2, .6, 1.0)
GROUPS_FILE = ROOT / "第三问突破优化" / "results" / "grouped_training" / "registration.json"


def load_all():
    cond, tasks, zones, train = load()
    seg = pd.read_csv(ROOT / "处理结果" / "v1" / "segments_private.csv")
    return cond, tasks, zones, train, seg


def hidden_for(jobs, seed, pool):
    """Identical RNG usage to confirm_lookahead.hidden_for.

    Returns {job_id: (duration, unit_id)}; the unit id is used only to fetch that
    unit's released prefix features.
    """
    rng = random.Random(seed)
    out = {}
    for j in jobs:
        c = pool[pool.family_id == j["family_id"]]
        row = c.iloc[rng.randrange(len(c))]
        out[j["job_id"]] = (float(row.duration_s), row.unit_id)
    return out


def build_hidden(sampled, feature_lookup):
    """Engine hidden payload with recorded prefix features attached."""
    return {
        k: dict(duration=v[0], features={str(p): feature_lookup.get((v[1], p), []) for p in PREFIXES})
        for k, v in sampled.items()
    }


def pools_from(frame):
    return {f: g.duration_s.to_numpy(float) for f, g in frame.groupby("family_id")}


def medians(pools):
    return {f: float(np.median(v)) for f, v in pools.items()}


def environment_groups():
    return json.loads(GROUPS_FILE.read_text())["groups"]


def bootstrap_ci(delta_by_seed, reps=4000, seed=20261122):
    rng = np.random.default_rng(seed)
    values = np.asarray(delta_by_seed, float)
    draws = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(reps)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
