"""Replay both nominal releases from their public entries on identical assignments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nominal_policy import BACKEND, TIER2, TIER4, dispatcher
from published import dispatcher as previous
from run import bench as s3_bench
from experiment import Bench, MET, OFFICIAL

KEY = ["split", "case_id", "seed", "version"]
PROMOTION = range(7832000, 7834048)


class TraceDigests:
    def __init__(self):
        self.rows = []

    def write(self, line):
        record = json.loads(line)
        events = record["events"]
        canonical = json.dumps(events, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.rows.append((record["seed"], len(events), hashlib.sha256(canonical.encode()).hexdigest()))


def replay_case(case):
    bench = s3_bench() if case == "S3" else Bench(case)
    parts = []
    for split, seeds in [("official", OFFICIAL), ("promotion", PROMOTION)]:
        for version, factory in [("tier3", previous), ("tier4", dispatcher)]:
            traces = TraceDigests()
            scores = bench.run(None, seeds, log=traces, policy_factory=lambda: factory(case))
            digests = pd.DataFrame(traces.rows, columns=["seed","event_count","events_sha256"]).set_index("seed")
            if not scores.index.equals(digests.index):
                raise ValueError("Replay trace keys do not match scores")
            parts.append(scores.join(digests).reset_index().assign(case_id=case, split=split, version=version))
    return pd.concat(parts, ignore_index=True)


def summary(trials):
    rows = []
    for split, block in trials.groupby("split"):
        cases = dict(tuple(block.groupby("case_id")))
        cases["overall"] = block.groupby(["seed", "version"], as_index=False)[MET].mean()
        for case, values in cases.items():
            old, new = (values[values.version == v].set_index("seed")[MET].sort_index()
                        for v in ("tier3", "tier4"))
            if not old.index.equals(new.index):
                raise ValueError("Unpaired seed rows")
            on_time = new[MET[0]].to_numpy()
            gain = on_time - old[MET[0]].to_numpy()
            rng = np.random.default_rng(557221)
            boot, paired = [], []
            for _ in range(100):
                indices = rng.integers(len(old), size=(100, len(old)))
                boot.extend(on_time[indices].mean(axis=1))
                paired.extend(gain[indices].mean(axis=1))
            for metric in MET:
                row = dict(split=split, case_id=case, seeds=len(old), metric=metric,
                           tier3=old[metric].mean(), tier4=new[metric].mean(),
                           delta=(new[metric]-old[metric]).mean())
                if metric == MET[0]:
                    row["abs_low"], row["abs_high"] = np.quantile(boot, [.025,.975])
                    row["gain_low"], row["gain_high"] = np.quantile(paired, [.025,.975])
                rows.append(row)
    return pd.DataFrame(rows)


def run(target):
    target.mkdir(parents=True, exist_ok=True)
    print(f"Q3 nominal numerical backend: {BACKEND}", flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        trials = pd.concat(pool.map(replay_case, ["S1","S2","S3"]), ignore_index=True)
    if trials.duplicated(KEY).any() or len(trials) != (30 + 2048) * 3 * 2:
        raise ValueError("Incomplete replay")
    for case in ("S1", "S2"):
        subset = trials[trials.case_id == case].set_index(KEY)
        pd.testing.assert_frame_equal(subset.xs("tier3",level="version"),subset.xs("tier4",level="version"))
    final = pd.read_csv(TIER4/"results/round_5/paired.csv").set_index(["method","seed"]).sort_index()
    gate = trials.query("case_id=='S3' and split=='promotion'").copy()
    gate["method"] = gate.version.map({"tier4":"candidate","tier3":"tier3"})
    actual = gate.set_index(["method","seed"]).sort_index()
    pd.testing.assert_frame_equal(actual[final.columns],final,atol=1e-9,rtol=0)
    history = pd.read_csv(TIER2/"results/final/trials.csv").query("split=='official'").copy()
    history = history[history["level"] == history.groupby("case")["level"].transform("max")]
    history = history.set_index(["case","seed"]).sort_index()
    baseline = []
    for case in ("S1","S2","S3"):
        # Use the original tier-two public prefix and reconstruct its official results.
        from release import dispatcher as old_formal
        original = Bench(case).run(None, OFFICIAL, policy_factory=lambda: old_formal(case))
        pd.testing.assert_frame_equal(original[MET], history.xs(case)[MET], atol=1e-9,rtol=0)
        baseline.append(original.assign(case_id=case).reset_index())
    old_formal = pd.concat(baseline, ignore_index=True)
    reference = old_formal.groupby("case_id")[MET].mean()
    reference.loc["overall"] = reference.mean()
    scores = summary(trials)
    official = scores.query("split=='official'").copy()
    official["tier2"] = [reference.loc[c,m] for c,m in zip(official.case_id,official.metric)]
    official["change_from_tier2"] = official.tier4 - official.tier2
    old_formal.to_csv(target/"official_tier2.csv", index=False)
    traces = trials[KEY + ["event_count","events_sha256"]].sort_values(KEY)
    traces.to_csv(target/"trace_digests.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    trials = trials.drop(columns=["event_count","events_sha256"])
    trials.sort_values(KEY).to_csv(target/"trials.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    scores.to_csv(target/"comparison.csv",index=False)
    official.to_csv(target/"official_versions.csv",index=False)
    (target/"backend.json").write_text(json.dumps({"backend":BACKEND,"trials":len(trials)},indent=2),encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
