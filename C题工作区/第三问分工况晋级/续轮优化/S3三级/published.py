"""Published nominal dispatch, with S1 and S2 delegated to their frozen release."""
from functools import lru_cache
import hashlib
import json
from bellman import Bellman
from adaptive import ROOT, frozen_dispatcher
from release import verified


@lru_cache(maxsize=1)
def locked_candidate():
    verified()
    lock=json.loads((ROOT/"published.json").read_bytes())
    for name,expected in lock["hashes"].items():
        digest=hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
        if digest!=expected:raise ValueError(f"Frozen S3 file changed: {name}")
    return json.loads((ROOT/lock["parameters"]).read_bytes())


def dispatcher(case):
    if case in ("S1","S2"):return frozen_dispatcher(case)
    if case=="S3":return Bellman(locked_candidate())
    raise ValueError("Expected nominal S1, S2 or S3")
