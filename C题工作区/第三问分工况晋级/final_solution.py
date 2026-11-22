"""Frozen public dispatcher. Create one instance per nominal episode."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import dispatch
import rolling
import numpy as np

ROOT = Path(__file__).resolve().parent


def checked(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Frozen file has changed: {path}")
    return data


@lru_cache(maxsize=3)
def load_case(case):
    if case not in ("S1", "S2", "S3"):
        raise ValueError("Expected published case S1, S2 or S3")
    folder = ROOT / "results"
    lock = json.loads((folder / f"frozen_{case}.json").read_bytes())
    if not lock["passed"] or lock["case"] != case:
        raise ValueError(f"Invalid lock for {case}")
    model = json.loads(checked(folder / f"round_{lock['round']}" / f"{case}.json", lock["model_sha256"]))
    if model["kind"] == "rolling":
        checked(ROOT / "rolling.py", lock["dispatch_sha256"])
        checked(ROOT / "rolling.cpp", lock["rolling_cpp_sha256"])
        prior = ROOT.parent / "第三问改进正式方案/model/training_family_samples.json"
        data = prior.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != "b9f5ee6fff5d5744d2e7486aece569a866e870572d336336a8c894d064fa42be":
            raise ValueError("Frozen training durations have changed")
        pools = {}
        for row in json.loads(data):
            pools.setdefault(row["family_id"], []).append(row["duration_s"])
        return model, {family: np.asarray(values) for family, values in pools.items()}
    checked(ROOT / "dispatch.py", lock["dispatch_sha256"])
    return model, {}


class NominalDispatcher:
    def __init__(self, case):
        self.model, self.pools = load_case(case)
        count = {"S1": 8, "S2": 12, "S3": 16}[case]
        self.job_ids = tuple(f"{case}_J{i:02d}" for i in range(1, count + 1))
        self.history = {}

    def choose(self, observation):
        if tuple(j["job_id"] for j in observation["jobs"]) != self.job_ids:
            raise ValueError("Expected the published job table in job ID order")
        if (observation["alpha"], observation["beta"]) != (1., .5):
            raise ValueError("NominalDispatcher requires alpha=1, beta=0.5")
        if self.model["kind"] == "rolling":
            return rolling.choose(observation, self.model, self.pools, self.history)
        return dispatch.choose(observation, self.model)

    def record_completion(self, family, service_start, completion):
        """Call once per successful execution, after its public completion."""
        self.history.setdefault(family, []).append(completion - service_start - .5)
