"""Fixed two-batch replication of S2 nominal lower-tail on-time performance."""
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "final_v1_stability"))
import seed_replay as frozen

BATCHES = {"A": range(918400300, 918400600), "B": range(918400600, 918400900)}
RATE = "weighted_on_time_rate"


def main():
    frozen.check_tag()
    cfg = frozen.TAG / "赛题分析报告"
    units = pd.read_csv(cfg / "source_segment_index.csv").query("split=='validation' and eligible==1")
    pools = {family: list(group.itertuples()) for family, group in units.groupby("family_id")}
    exemplar = next(f for f in frozen.instances() if f["case"]["case_id"] == "S2"
                    and (f["alpha"], f["beta"]) == (1., .5))
    assert len(exemplar["jobs"]) == 12

    def instance(seed):
        fixed = {k: exemplar[k] for k in ("case", "jobs", "zones", "alpha", "beta")}
        rng = random.Random(seed)
        hidden = {}
        for job in fixed["jobs"]:
            unit = rng.choice(pools[job["family_id"]])
            hidden[job["job_id"]] = dict(duration=unit.duration_s, unit_id=unit.unit_id,
                                          features={"0.2": [], "0.6": [], "1.0": []})
        fixed.update(seed=seed, hidden=hidden)
        return dict(**fixed, instance_hash=frozen.digest(fixed))

    assert instance(exemplar["seed"])["instance_hash"] == exemplar["instance_hash"]
    prior = pd.read_csv(OUT.parent / "final_v1_evidence_extension/duration_grid.csv.gz")
    reference = prior.query("case_id=='S2' and alpha==1 and beta==0.5").sort_values("seed").iloc[0]
    assert instance(int(reference.seed))["instance_hash"] == reference.instance_hash
    previous = set(prior.seed) | set(json.loads((cfg / "common_conditions.json").read_bytes())["validation_seeds"])
    batch_by_seed = {s: batch for batch, seeds in BATCHES.items() for s in seeds}
    assert len(batch_by_seed) == 600 and previous.isdisjoint(batch_by_seed)
    inputs = [(instance(seed), [frozen.DEFAULT]) for seed in batch_by_seed]
    with ProcessPoolExecutor(max_workers=8) as pool:
        data = pd.DataFrame([r for group in pool.map(frozen.run_group, inputs, chunksize=5) for r in group])
    frozen.check_tag()
    assert len(data) == 600 and set(data.seed) == set(batch_by_seed)
    assert data.valid_protocol.all() and data.invalid_actions.eq(0).all()
    assert data.internal_seed.eq(20261122).all() and data[RATE].between(0, 1).all()
    data["batch"] = data.seed.map(batch_by_seed)
    data = data.sort_values(["batch", RATE, "seed"])
    data["rank"] = data.groupby("batch").cumcount() + 1
    rows = []
    for batch, g in data.groupby("batch"):
        assert len(g) == 300
        worst = g.loc[g["rank"] <= 30, RATE]
        assert len(worst) == 30
        rows.append(dict(batch=batch, n=len(g), worst_n=len(worst), tail_pct=100 * worst.mean(),
                         cutoff_pct=100 * worst.max(), overall_pct=100 * g[RATE].mean()))
    summary = pd.DataFrame(rows)
    gap = abs(rows[0]["tail_pct"] - rows[1]["tail_pct"])
    audit = dict(tag="v1.0.0-final", commit=frozen.COMMIT, K=64, internal_seed=frozen.DEFAULT,
                 new_external_seeds=600, gap_pp=gap, pass_5pp=bool(gap <= 5 + 1e-12),
                 invalid_actions=int(data.invalid_actions.sum()), max_score_rebuild_diff=data.independent_diff.max())
    data.to_csv(OUT / "trials.csv.gz", index=False)
    summary.to_csv(OUT / "batches.csv", index=False)
    (OUT / "verdict.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    print(summary.to_string(index=False))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
