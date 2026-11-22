"""Nominal dispatch for the second release. Keep S1 on its original implementation."""
import hashlib
import json
from functools import lru_cache

from policy import HERE, NominalDispatcher, Route


@lru_cache(maxsize=1)
def verified():
    manifest = json.loads((HERE/"freeze.json").read_bytes())
    for name, expected in manifest["files"].items():
        data = (HERE/name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Frozen dependency changed: {name}")
    return manifest["levels"]


def dispatcher(case, level=None):
    levels = verified()
    if case not in levels:
        raise ValueError("Expected published S1, S2 or S3")
    level = levels[case] if level is None else level
    if level not in range(levels[case]+1):
        raise ValueError("Unpublished promotion level")
    if level == 0:
        return NominalDispatcher(case)
    lock = json.loads((HERE/"results"/f"{case}_promotion_{level}.json").read_bytes())
    return PublishedRoute(case,lock["candidate"])


class PublishedRoute(Route):
    def choose(self, observation):
        if (observation["alpha"],observation["beta"]) != (1.,.5):
            raise ValueError("Expected nominal alpha=1, beta=0.5")
        if tuple(j["job_id"] for j in observation["jobs"]) != self.base.job_ids:
            raise ValueError("Expected official task order")
        return super().choose(observation)
