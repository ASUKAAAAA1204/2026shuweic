"""Check the published site inventory against the frozen split and prior uses."""
import argparse
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

HERE = Path(__file__).resolve().parent
TAG = HERE.parents[1] / "tmp/final_v1_20261122"
SPACE = TAG / "C题工作区"


def check(raw_dir=None):
    commit = subprocess.check_output(["git", "-C", str(TAG), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "9f8bc05fae9d8ce32da7fa7c5e28a82960085819"
    records = pd.read_csv(SPACE / "处理结果/v1/record_index.csv")
    split = pd.read_csv(TAG / "赛题分析报告/record_split.csv")
    keys = ["episode_uuid", "dataset", "environment_id", "split"]
    pd.testing.assert_frame_equal(
        records[keys].sort_values(keys).reset_index(drop=True),
        split[keys].sort_values(keys).reset_index(drop=True))
    assert len(records) == records.episode_uuid.nunique() == 69
    assert records.groupby("environment_id").split.nunique().eq(1).all()
    if raw_dir is not None:
        for source, group in records.groupby("dataset"):
            with ZipFile(raw_dir / (source + ".zip")) as archive:
                folders = {n.filename.split("/")[1] for n in archive.infolist()
                           if not n.is_dir() and n.filename.startswith(source + "/")}
            assert folders == set(group.folder), source

    train = pd.read_csv(SPACE / "阶段交付/stage2_optimization/E1/train_environments.csv")
    prior = pd.read_csv(HERE.parent / "final_v1_evidence_extension/trajectory/per_window.csv.gz")
    used = set(train.environment_id) | set(prior.environment_id)
    assert set(train.environment_id) == set(split.loc[split.split == "train", "environment_id"])
    assert set(prior.episode_uuid) == set(split.loc[split.split != "train", "episode_uuid"])
    sites = split.groupby(["split", "environment_id"]).agg(
        episodes=("episode_uuid", "nunique"),
        sources=("dataset", lambda values: "|".join(sorted(set(values))))).reset_index()
    sites["prior_evidence"] = sites.split.map({
        "train": "E1/train_environments.csv", "validation": "trajectory/per_window.csv.gz",
        "test": "trajectory/per_window.csv.gz"})
    sites["prior_q2_validation_or_test_windows"] = sites.environment_id.map(
        prior.groupby("environment_id").window_id.nunique())
    sites["eligible_new_site"] = ~sites.environment_id.isin(used)
    pd.testing.assert_frame_equal(sites, pd.read_csv(HERE / "site_inventory.csv"), check_dtype=False)
    print(f"{len(sites)} sites, {sites.episodes.sum()} episodes, "
          f"{sites.eligible_new_site.sum()} eligible new sites")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="folder containing the three official source ZIPs")
    check(parser.parse_args().raw_dir)
