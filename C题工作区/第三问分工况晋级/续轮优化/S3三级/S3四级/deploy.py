"""Frozen S3 release; carry forward the published S1 and S2 dispatchers."""
from functools import lru_cache
import hashlib
import json
from fourth import HERE,Policy,previous,locked_candidate
import bellman
import policy
import rolling
from numerical_backend import configure

BACKEND = configure(rolling, policy, bellman)


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


@lru_cache(maxsize=1)
def selected():
    locked_candidate()
    dirs=list((HERE/"results").glob("round_*/passed.lock"))
    if len(dirs)!=1:raise RuntimeError("Expected one verified S3 promotion")
    folder=dirs[0].parent
    if json.loads(dirs[0].read_bytes())!=json.loads((folder/"verdict.json").read_bytes()):
        raise ValueError("Promotion record mismatch")
    record=json.loads((folder/"before.json").read_bytes())
    for name,expected in record["files"].items():
        if digest(HERE/name)!=expected:raise ValueError(f"Frozen dependency changed: {name}")
    if digest(folder/"nominee.json")!=record["candidate_hash"]:raise ValueError("Changed nominee")
    return json.loads((folder/"nominee.json").read_bytes())


def dispatcher(case):
    if case in ("S1","S2"):return previous(case)
    if case=="S3":return Policy(selected())
    raise ValueError("Expected nominal S1, S2 or S3")
