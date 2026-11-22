"""Replay the public B dispatcher and independently verify every official score."""
import gzip
import json
from pathlib import Path
import random
import sys
from time import perf_counter_ns

import numpy as np
import pandas as pd

from nominal_policy import METHOD, dispatcher
from verify_b_events import independent_rebuild

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.append(str(PACKAGE.parent / "第三问改进正式方案/code"))
from support import Engine, jobs_for

KEY = ["case_id", "seed", "alpha", "beta"]
METRICS = ["weighted_on_time_rate", "weighted_completion_rate", "weighted_tardiness_s",
           "cost", "observations", "elapsed_s", "observed_overrun_s", "reserved_idle_s",
           "censored_units", "invalid_actions", "valid_protocol"]


def run(config, out):
    target = out / "q3/b_k64"
    target.mkdir(parents=True, exist_ok=True)
    conditions = json.loads((config / "common_conditions.json").read_bytes())
    tasks = pd.read_csv(config / "tasks.csv").fillna("")
    zones = pd.read_csv(config / "service_zones.csv").set_index("zone_id")[["x", "y"]].T.to_dict("list")
    units = pd.read_csv(config / "source_segment_index.csv").query("split=='validation' and eligible==1")
    pools = {f: list(g.itertuples()) for f, g in units.groupby("family_id")}
    svc = conditions["service"]
    nominal = (svc["alpha"], svc["beta_s"])
    mappings = [nominal] + [(a, nominal[1]) for a in svc["alpha_sensitivity"]] + [
        (nominal[0], b) for b in svc["beta_sensitivity"]]
    rows, timings = [], []
    with gzip.open(target / "events.jsonl.gz", "wt", encoding="utf8") as events:
        for case in conditions["cases"]:
            case_id = case["case_id"]
            jobs = jobs_for(tasks, case_id)
            for seed in conditions["validation_seeds"]:
                rng = random.Random(seed)
                hidden = {}
                for job in jobs:
                    unit = rng.choice(pools[job["family_id"]])
                    hidden[job["job_id"]] = dict(duration=unit.duration_s, unit_id=unit.unit_id,
                                                  features={"0.2": [], "0.6": [], "1.0": []})
                for alpha, beta in mappings:
                    key = dict(case_id=case_id, seed=seed, alpha=alpha, beta=beta, method=METHOD)
                    policy = dispatcher(case_id)
                    engine = Engine(jobs, hidden, zones, case["horizon_s"], case["cost_budget"],
                                    case["observation_limit"], alpha, beta)
                    times = []
                    while not engine.terminal:
                        observation = engine.observe()
                        start = perf_counter_ns()
                        action = policy.choose(observation)
                        times.append((perf_counter_ns() - start) / 1e6)
                        event = engine.step(action)
                        if event.get("completion") is not None:
                            policy.record_completion(engine.jobs[action["job_id"]]["family_id"],
                                                     event["service_start"], event["completion"])
                    scores = engine.metrics()
                    check = independent_rebuild(jobs, engine.events, zones, engine.H, engine.B, engine.K)
                    diff = max(abs(scores[k] - check[k]) for k in check if k != "violations")
                    if check["violations"] or diff > 1e-9 or not scores["valid_protocol"]:
                        raise ValueError((key, check))
                    rows.append(dict(**key, **scores, steps=len(times), decision_total_ms=sum(times), independent_diff=diff))
                    timings.extend(dict(**key, step=i, decision_ms=t) for i, t in enumerate(times, 1))
                    events.write(json.dumps(dict(**key, events=engine.events)) + "\n")
    trials = pd.DataFrame(rows)
    reference = pd.read_csv(PACKAGE / "reference/q3_official_B_and_previous.csv.gz")
    expected = reference[reference.method == METHOD].set_index(KEY).sort_index()
    actual = trials.set_index(KEY).sort_index()
    scores = METRICS + ["steps"]
    pd.testing.assert_frame_equal(actual[scores], expected[scores], check_dtype=False, atol=1e-9, rtol=0)
    if not actual.index.is_unique or len(actual) != 450:
        raise ValueError("Incomplete B replay")
    trials.to_csv(target / "trials.csv.gz", index=False)
    pd.DataFrame(timings).to_csv(target / "decisions.csv.gz", index=False)
    previous = reference.query("method=='previous_local'").set_index(KEY).sort_index()
    if not previous.index.equals(actual.index):
        raise ValueError("Previous local reference does not cover the same episodes")
    pairs = []
    for scope, selected in [("nominal", [nominal]), ("all_five", mappings)]:
        for name in ["S1","S2","S3","overall"]:
            frame = actual.reset_index()
            frame = frame[frame.set_index(["alpha","beta"]).index.isin(selected)]
            if name != "overall":
                frame = frame[frame.case_id == name]
            x = frame.set_index(KEY)[METRICS].astype(float)
            y = previous.loc[x.index, METRICS].astype(float)
            for metric in METRICS:
                pairs.append(dict(scope=scope, case_id=name, metric=metric, episodes=len(x),
                                  previous_local=y[metric].mean(), B_K64=x[metric].mean(),
                                  delta=(x[metric]-y[metric]).mean()))
    comparison = pd.DataFrame(pairs)
    comparison.to_csv(target / "comparison.csv", index=False)
    regression = np.abs(actual[scores].to_numpy(float) - expected[scores].to_numpy(float)).max()
    (target / "verification.json").write_text(json.dumps(dict(passed=True, method=METHOD,
        source_commit="fe7f243d13ffb6f03b3f5f5a25098ba877366adc", cases=3, mappings=5, seeds=30,
        episodes=len(trials), decisions=len(timings), invalid_actions=int(trials.invalid_actions.sum()),
        independent_max_diff=trials.independent_diff.max(), reference_max_diff=regression), indent=2))
    return trials, comparison
