"""Reproduce frozen policies and confirm them on the preregistered unused seeds."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from evaluate import CFG, OUT, OLD, Engine, independent_metrics, jobs_for, replay, sample_hidden, settings
from final_solution import NominalDispatcher

METRICS = ["weighted_on_time_rate", "weighted_completion_rate", "weighted_tardiness_s", "cost"]


def run_new(case, c, jobs, zones, hidden):
    policy = NominalDispatcher(case)
    sim = Engine(jobs, hidden, zones, c["horizon_s"], c["cost_budget"], c["observation_limit"], 1., .5)
    while not sim.terminal:
        action = policy.choose(sim.observe())
        event = sim.step(action)
        if event.get("completion") is not None:
            family = sim.jobs[action["job_id"]]["family_id"]
            policy.record_completion(family, event["service_start"], event["completion"])
    result = sim.metrics()
    independent = independent_metrics(jobs, sim.events, sim.H)
    for key, value in result.items():
        if abs(value - independent[key]) > 1e-9:
            raise ValueError(f"Action reconstruction disagrees on {key}")
    return result, sim.events


def main(out, config):
    out.mkdir(exist_ok=True, parents=True)
    for name in ("tasks.csv", "service_zones.csv", "source_segment_index.csv", "common_conditions.json"):
        if (CFG / name).read_bytes().replace(b"\r\n", b"\n") != (config / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Configuration differs from the registered experiment: {name}")
    official = json.loads((CFG / "common_conditions.json").read_bytes())["validation_seeds"]
    splits = {"official": official, "extra": range(73000, 73120), "confirmation": range(91000, 91240)}
    segments = pd.read_csv(CFG / "source_segment_index.csv").query("split=='validation' and eligible==1")
    pool = {family: list(group.itertuples()) for family, group in segments.groupby("family_id")}
    archive = pd.read_csv(OLD / "audit/original_frozen_trials.csv").query(
        "alpha==1 and beta==.5 and method=='pressure_gated'").set_index(["case_id", "seed"])
    rows = []
    with gzip.open(out / "new_actions.jsonl.gz", "wt", encoding="utf8") as log:
        for case in ("S1", "S2", "S3"):
            c, tasks, zones, pools = settings(case)
            jobs = jobs_for(tasks, case)
            lock = json.loads((OUT / f"frozen_{case}.json").read_bytes())
            previous = pd.read_csv(OUT / f"round_{lock['round']}" / f"{case}_trials.csv").set_index(["split", "seed", "method"])
            for split, seeds in splits.items():
                for seed in seeds:
                    hidden = sample_hidden((jobs, pool), seed)
                    old, _ = replay(c, jobs, zones, pools, None, hidden)
                    new, events = run_new(case, c, jobs, zones, hidden)
                    tag = dict(case_id=case, split=split, seed=seed)
                    log.write(json.dumps(dict(**tag, events=events)) + "\n")
                    for method, result in (("old", old), ("new", new)):
                        if result["invalid_actions"] or result["cost"] > c["cost_budget"] + 1e-9 or result["observations"] > c["observation_limit"]:
                            raise ValueError(f"Resource violation: {tag} {method}")
                        if split != "confirmation":
                            reference = previous.loc[split, seed, method]
                            np.testing.assert_allclose([float(result[k]) for k in result], [float(reference[k]) for k in result], atol=1e-9, rtol=0)
                        if split == "official" and method == "old":
                            np.testing.assert_allclose([result[k] for k in METRICS], archive.loc[(case, seed), METRICS].to_numpy(float), atol=1e-9, rtol=0)
                        rows.append(dict(**tag, method=method, **result))
                print(case, split, len(seeds), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "trials.csv", index=False)
    gate_path = out / "gate_trials.csv"
    frame.query("split!='confirmation'").to_csv(gate_path, index=False)
    subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parent / "核验/verify_trials.py"),
                    "--trials", gate_path.name, "--output", "gate_audit.json"], cwd=out, check=True)
    rng = np.random.default_rng(88234)
    summary = []
    for split, seeds in splits.items():
        wide = frame.loc[frame.split == split].pivot(index=["seed","case_id"],columns="method",values=METRICS)
        for case in ("S1", "S2", "S3", "整体等权"):
            pair = wide.groupby(level="seed").mean() if case == "整体等权" else wide.xs(case, level="case_id")
            assert len(pair) == len(seeds)
            idx = rng.integers(len(pair), size=(5000, len(pair)))
            for metric in METRICS:
                both = pair[metric]
                delta = (both.new - both.old).to_numpy()
                lo, hi = np.quantile(delta[idx].mean(axis=1), [.025, .975])
                summary.append(dict(split=split, case_id=case, metric=metric, seeds=len(pair), old=both.old.mean(),
                                    new=both.new.mean(), gain=delta.mean(), paired_ci_low=lo, paired_ci_high=hi))
    pd.DataFrame(summary).to_csv(out / "comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT / "final")
    parser.add_argument("--constraints", type=Path, default=CFG)
    args = parser.parse_args()
    main(args.output, args.constraints)
