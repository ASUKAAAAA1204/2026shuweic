"""Replicate all nine duration mappings on a fresh block of external seeds."""
import argparse
import itertools
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "final_v1_stability"))
import seed_replay as frozen

GRID = list(itertools.product((.75, 1., 1.5), (0., .5, 1.)))
SEEDS = range(918400900, 918401200)


def inputs(seeds, mappings):
    cfg = frozen.TAG / "赛题分析报告"
    units = pd.read_csv(cfg / "source_segment_index.csv").query("split=='validation' and eligible==1")
    pools = {f: list(g.itertuples()) for f, g in units.groupby("family_id")}
    templates = {}
    for row in frozen.instances():
        templates.setdefault(row["case"]["case_id"], row)
        assert row["seed"] not in seeds
    assert set(templates) == {"S1", "S2", "S3"}
    for seed in seeds:
        for template in templates.values():
            fixed = {k: template[k] for k in ("case", "jobs", "zones")}
            rng = random.Random(seed)
            hidden = {}
            for job in fixed["jobs"]:
                unit = rng.choice(pools[job["family_id"]])
                hidden[job["job_id"]] = dict(duration=unit.duration_s, unit_id=unit.unit_id,
                                             features={"0.2": [], "0.6": [], "1.0": []})
            for alpha, beta in mappings:
                instance = dict(**fixed, seed=seed, hidden=hidden, alpha=alpha, beta=beta)
                yield dict(**instance, instance_hash=frozen.digest(instance)), [frozen.DEFAULT]


if __name__ == "__main__":
    frozen.check_tag()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--complete-grid", action="store_true")
    args = parser.parse_args()
    seeds = SEEDS
    mappings = GRID
    filename = "fresh_trials.csv.gz"
    if args.local or args.complete_grid:
        seeds = [*range(918400000, 918400300), *SEEDS]
        offsets = (0., 1.) if args.complete_grid else (.5,)
        mappings = list(itertools.product((.95, 1.05), offsets))
        filename = "grid_completion_trials.csv.gz" if args.complete_grid else "local_trials.csv.gz"
    parts = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for group in pool.map(frozen.run_group, inputs(seeds, mappings), chunksize=9):
            parts.extend(group)
            if len(parts) % 1350 == 0:
                print(len(parts), flush=True)
    df = pd.DataFrame(parts)
    assert len(df) == 3 * len(seeds) * len(mappings) and not df.duplicated(frozen.KEY).any()
    assert df.valid_protocol.all() and df.invalid_actions.eq(0).all() and df.independent_diff.max() < 1e-9
    frozen.check_tag()
    df.to_csv(OUT / filename, index=False)


