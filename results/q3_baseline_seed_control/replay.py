"""Repeat the archived baseline on the exact B_K64 seed-stability inputs."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
TAG = RESULTS.parent / "tmp/final_v1_20261122"
PACKAGE = TAG / "C题工作区/第三问B最终方案"
sys.path.insert(0, str(PACKAGE / "code"))
import policy
from engine import Engine
from independent import independent_rebuild

REFERENCE_PRESSURE = policy.pressure
COMMIT = "9f8bc05fae9d8ce32da7fa7c5e28a82960085819"
DEFAULT_LABEL = 20261122
BASELINE_SALT = 20261123
KEY = ["case_id", "seed", "alpha", "beta"]
METRICS = ["weighted_on_time_rate", "weighted_completion_rate", "weighted_tardiness_s",
           "cost", "observations", "elapsed_s", "observed_overrun_s", "reserved_idle_s",
           "censored_units", "invalid_actions", "valid_protocol", "steps"]


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_tag():
    for name in ("HEAD", "v1.0.0-final^{}"):
        assert subprocess.check_output(["git", "-C", str(TAG), "rev-parse", name], text=True).strip() == COMMIT
    assert not subprocess.check_output(["git", "-C", str(TAG), "status", "--porcelain"]).strip()


def pressure_for(seed):
    # The released screen has no seed parameter. Rebind just the salt in its
    # existing function in this process, retaining crc32(job_id) and all 64 draws.
    if seed == DEFAULT_LABEL:
        return REFERENCE_PRESSURE
    code = REFERENCE_PRESSURE.__code__
    assert code.co_consts.count(BASELINE_SALT) == 1
    constants = list(code.co_consts)
    constants[constants.index(BASELINE_SALT)] = BASELINE_SALT + seed - DEFAULT_LABEL
    return FunctionType(code.replace(co_consts=tuple(constants)), REFERENCE_PRESSURE.__globals__,
                        REFERENCE_PRESSURE.__name__, REFERENCE_PRESSURE.__defaults__)


def replay_group(args):
    f, batch, seeds = args
    case, jobs, hidden, zones = (f[k] for k in ("case","jobs","hidden","zones"))
    assert digest({k:v for k,v in f.items() if k != "instance_hash"}) == f["instance_hash"]
    alpha, beta = f["alpha"], f["beta"]
    families = {j["job_id"]:j["family_id"] for j in jobs}
    rows = []
    for seed in seeds:
        policy.pressure = pressure_for(seed)
        p = policy.Policy("baseline")
        engine = Engine(jobs,hidden,zones,case["horizon_s"],case["cost_budget"],case["observation_limit"],alpha,beta)
        actions = []
        while not engine.terminal:
            action = p.act(engine.observe())
            actions.append(action)
            event = engine.step(action)
            if action["type"] == "execute":
                duration = alpha * hidden[action["job_id"]]["duration"] + beta
                assert abs(duration-event["realized_duration"])<1e-9
            p.feedback(event, families.get(action.get("job_id")), alpha, beta)
        scores = engine.metrics()
        check = independent_rebuild(jobs, engine.events, zones, engine.H, engine.B, engine.K)
        diff = max(abs(scores[k] - check[k]) for k in check if k != "violations")
        assert diff < 1e-9 and not check["violations"] and scores["valid_protocol"] and not scores["invalid_actions"]
        rows.append(dict(case_id=case["case_id"],seed=f["seed"],alpha=alpha,beta=beta,batch=batch,
                         internal_seed=seed,pressure_seed_base=BASELINE_SALT+seed-DEFAULT_LABEL,
                         instance_hash=f["instance_hash"],first_action=json.dumps(actions[0],sort_keys=True),
                         path_hash=digest(actions),steps=len(actions),independent_diff=diff,**scores))
    policy.pressure = REFERENCE_PRESSURE
    return rows


def main():
    check_tag()
    registry_path = RESULTS/"final_v1_stability/fixed_instances.json"
    seed_audit = json.loads((registry_path.parent/"seed_audit.json").read_bytes())
    assert hashlib.sha256(registry_path.read_bytes()).hexdigest() == seed_audit["registry_sha256"]
    instances = json.loads(registry_path.read_bytes())
    seeds = seed_audit["internal_seeds"]
    second = json.loads((RESULTS/"final_v1_evidence_extension/scheduling_audit.json").read_bytes())["new_internal_seeds"]
    assert len(set(seeds)) == len(set(second)) == 20 and set(seeds).isdisjoint(second)
    groups = [(f,"original",seeds) for f in instances]
    groups += [(f,"replication",second) for f in instances if f["alpha"] in (1,1.5) and f["beta"]==.5]
    assert len(instances) == 450 and len(groups) == 630
    with ProcessPoolExecutor(max_workers=8) as pool:
        data = pd.DataFrame([row for group in pool.map(replay_group, groups, chunksize=2) for row in group])
    ref = pd.read_csv(PACKAGE/"data/expected_metrics.csv").query("method=='baseline'").rename(columns={"case":"case_id"})
    default = data[data.internal_seed.eq(DEFAULT_LABEL)].set_index(KEY).sort_index()
    ref = ref.set_index(KEY).sort_index()
    assert len(default) == len(ref) == 450 and default.index.equals(ref.index)
    archive_diff = np.abs(default[METRICS].to_numpy(float) - ref[METRICS].to_numpy(float)).max()
    assert archive_diff < 1e-9 and len(data)==12600 and not data.duplicated(KEY+["internal_seed"]).any()
    check_tag()
    data.to_csv(OUT/"trials.csv.gz", index=False)
    audit = dict(commit=COMMIT, method="baseline", original_stream_base=BASELINE_SALT,
                 original_label=DEFAULT_LABEL, fixed_instances=450, second_batch_instances=180,
                 default_matches=len(ref), n_replays=len(data), invalid_actions=int(data.invalid_actions.sum()),
                 max_archive_error=archive_diff, max_score_rebuild_error=data.independent_diff.max())
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
