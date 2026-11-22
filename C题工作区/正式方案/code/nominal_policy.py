"""Public Q3 dispatcher. Create one instance per episode for every case and mapping."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from b_policy import JointScenarioPolicy

MODEL = Path(__file__).resolve().parents[2] / "第三问改进正式方案/model/training_family_samples.json"
METHOD = "B_K64"


@lru_cache(maxsize=1)
def training_pools():
    rows = json.loads(MODEL.read_bytes())
    families = sorted({row["family_id"] for row in rows})
    return {f: np.array([row["duration_s"] for row in rows if row["family_id"] == f]) for f in families}


class Dispatcher:
    def __init__(self):
        self.policy = JointScenarioPolicy(K=64, strength=20, feedback_strength=3,
                                          seed=20261122, base="survival", stratified=True)
        self.history = {}
        self.mapping = None

    def choose(self, observation):
        self.mapping = observation["alpha"], observation["beta"]
        public = dict(observation, jobs=sorted(observation["jobs"], key=lambda job: job["job_id"]))
        return self.policy.select(public, training_pools(), self.history)

    def record_completion(self, family, start, end):
        alpha, beta = self.mapping
        self.history.setdefault(family, []).append((end - start - beta) / alpha)


def dispatcher(case):
    if case not in ("S1", "S2", "S3"):
        raise ValueError(f"Unknown scheduling case {case!r}")
    return Dispatcher()
