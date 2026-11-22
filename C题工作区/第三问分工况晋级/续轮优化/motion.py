"""Train a duration prior using only public 0.2 s motion. The policy receives aggregates."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "赛题分析报告/source_segment_index.csv"
MODEL = HERE / "motion_prior.json"


def describe(z):
    values = []
    for side in ("left_hand", "right_hand"):
        t = z[side+"__t_s"]
        points = z[side+"__position"]
        good = z[side+"__quality_known"] & ~z[side+"__bad"] & np.isfinite(points).all(axis=(1,2))
        dt = np.diff(t)
        pair = good[1:] & good[:-1] & (dt>0)
        x = points.mean(axis=1)
        speed = np.linalg.norm(np.diff(x,axis=0)[pair],axis=1)/dt[pair]
        spread = np.linalg.norm(points[good]-x[good,None,:],axis=2) if good.any() else []
        values.extend([np.median(dt) if len(dt) else 0., good.mean() if len(t) else 0.,
                       np.log1p(np.median(speed)) if len(speed) else 0.,
                       np.mean(spread) if len(spread) else 0.])
    return [round(float(v),5) for v in values]


def prepare(data):
    rows = pd.read_csv(SOURCE).query("eligible==1 and split in ['train','validation']")
    training = []
    public = {}
    for r in rows.itertuples():
        with np.load(data / "private" / (r.unit_id+"_0.2.npz"),allow_pickle=False) as z:
            features = describe(z)
        if r.split == "train":
            training.append((r.family_id, features, r.duration_s))
        else:
            public[r.unit_id] = features
    # Standardization and conditional distributions use only train rows.
    all_x = np.array([row[1] for row in training])
    sd = np.maximum(all_x.std(axis=0),[.003,.12,.10,.008]*2)
    bundle = {"scale":sd.tolist(), "families":{}}
    for family in ("P","A","L"):
        bundle["families"][family]=[[x,y] for f,x,y in training if f==family]
    for path, value in [(MODEL,bundle),(HERE/"replay_public_prefix.json",public)]:
        path.write_bytes(json.dumps(value,separators=(",",":"),allow_nan=False).encode())


@lru_cache(maxsize=1)
def trained():
    model = json.loads(MODEL.read_bytes())
    groups = {f:(np.array([v[0] for v in rows]), np.array([v[1] for v in rows]))
              for f, rows in model["families"].items()}
    return np.array(model["scale"]),groups


@lru_cache(maxsize=1024)
def conditional(family,features,bandwidth):
    scale, groups = trained()
    x, y = groups[family]
    d = np.square(np.clip((x - features)/scale,-4,4)).mean(axis=1)
    w = np.exp(-(d-d.min())/(2*bandwidth**2))
    # Six training observations worth of unconditional mass regularize sparse neighborhoods.
    w += 6/len(w)
    order = np.argsort(y)
    cdf = np.cumsum(w[order])/w.sum()
    q = (np.arange(100)+.5)/100
    return y[order[np.searchsorted(cdf,q)]]


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    prepare(p.parse_args().data)
