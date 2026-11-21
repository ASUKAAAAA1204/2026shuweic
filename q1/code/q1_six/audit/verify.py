#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np, pandas as pd

ATOL, MODELS = 1e-9, ("B", "A", "L", "T", "K", "K0")

def trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

def close(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{name}: shape mismatch"
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert d <= ATOL, f"{name}: max abs delta {d:.3e}"
    return d

def hierarchy(df, value):
    g = df.groupby(["model", "horizon", "source", "env", "record", "side"], as_index=False)[value].mean()
    for keys in (["model", "horizon", "source", "env", "record"],
                 ["model", "horizon", "source", "env"],
                 ["model", "horizon", "source"], ["model", "horizon"]):
        g = g.groupby(keys, as_index=False)[value].mean()
    return g.groupby("model", as_index=False)[value].mean()

def main():
    assert len(sys.argv) == 2, "usage: python verify.py RESULT_DIRECTORY"
    root = Path(sys.argv[1])
    z = np.load(root / "predictions.npz", allow_pickle=False)
    w = pd.read_csv(root / "windows.csv")
    met = pd.read_csv(root / "metrics.csv")
    summ = pd.read_csv(root / "summary.csv")
    need = {"offsets", "t", "ref", *MODELS}
    assert need <= set(z.files), f"missing arrays: {sorted(need - set(z.files))}"
    off = np.asarray(z["offsets"])
    t, ref = np.asarray(z["t"], float), np.asarray(z["ref"], float)
    pred = {m: np.asarray(z[m], float) for m in MODELS}
    W, P = len(w), len(t)
    assert W and P and off.ndim == 1 and off.size == W + 1
    assert np.isfinite(off).all() and np.all(off == np.floor(off))
    off = off.astype(np.int64)
    assert off[0] == 0 and off[-1] == P and np.all(np.diff(off) > 0)
    assert t.ndim == 1 and ref.shape == (P, 3) and all(p.shape == (P, 3) for p in pred.values())
    assert np.isfinite(t).all() and np.isfinite(ref).all() and all(np.isfinite(p).all() for p in pred.values())
    assert np.all(t >= 0) and t[0] == 0 and np.all(t[off[:-1]] == 0)
    for a, b in zip(off[:-1], off[1:]):
        assert np.all(np.diff(t[a:b]) > 0), "times must strictly increase per window"
    wcols = ["window_id", "source", "env", "record", "side", "scale"]
    mcols = ["window_id", "model", "horizon", "ade", "fde", "nade"]
    assert set(wcols) <= set(w.columns) and set(mcols) <= set(met.columns) and {"model", "score"} <= set(summ.columns)
    # Low-speed windows intentionally omit directional diagnostics; all primary
    # outputs and the reference/forecast coordinates must remain finite.
    assert not (w[wcols].isna().any().any() or met[mcols].isna().any().any() or summ.isna().any().any())
    assert w.window_id.is_unique and summ.model.is_unique
    assert not met.duplicated(["window_id", "model", "horizon"]).any()
    assert np.isfinite(met[["horizon", "ade", "fde", "nade"]].to_numpy()).all()
    assert np.isfinite(w.scale).all() and np.all(w.scale > 0)
    assert np.all(met.horizon > 0) and np.isfinite(summ.score).all()
    assert set(met.model) == set(MODELS) == set(summ.model)
    horizons = sorted(set(met.horizon))
    assert horizons == [0.2,0.6,1.0]
    keys = {(wid, m, h) for wid in w.window_id for m in MODELS for h in horizons}
    assert {tuple(x) for x in met[["window_id", "model", "horizon"]].to_numpy()} == keys, "metrics group coverage"
    wpos = {wid: i for i, wid in enumerate(w.window_id)}
    rows = list(met.itertuples(index=False))
    calc = {}
    for r in rows:
        a, b = off[wpos[r.window_id]], off[wpos[r.window_id] + 1]
        err, tt = np.linalg.norm(pred[r.model][a:b] - ref[a:b], axis=1), t[a:b]
        mask = tt <= r.horizon + ATOL
        assert mask.any(), "no observation within horizon"
        tt, ee = tt[mask], err[mask]
        assert tt[-1] > 0, "zero-duration window at horizon"
        ade = float(trapz(ee, tt) / tt[-1])
        calc[(r.window_id, r.model, r.horizon)] = (ade, float(ee[-1]))
    ade_x = np.array([calc[(r.window_id, r.model, r.horizon)][0] for r in rows])
    fde_x = np.array([calc[(r.window_id, r.model, r.horizon)][1] for r in rows])
    scale = met.window_id.map(w.set_index("window_id")["scale"]).to_numpy(float)
    assert np.isfinite(scale).all()
    nade_x = ade_x / scale
    da, df = close(met.ade, ade_x, "ADE"), close(met.fde, fde_x, "FDE")
    dn = max(close(met.nade, met.ade.to_numpy() / scale, "NADE identity"),
             close(met.nade, nade_x, "NADE recomputation"))
    mw = met.assign(nade_x=nade_x)[["model", "horizon", "window_id", "nade_x"]]
    mw = mw.merge(w[wcols], on="window_id", validate="many_to_one")
    expected = hierarchy(mw, "nade_x").set_index("model")["nade_x"]
    ds = close(summ.set_index("model").score, expected.reindex(summ.model), "summary score")
    print(f"windows={W} points={P} models={len(MODELS)} horizons={len(horizons)} metrics={len(met)} summary={len(summ)}")
    print(f"max_abs_delta ade={da:.3e} fde={df:.3e} nade={dn:.3e} score={ds:.3e}\nOK")

if __name__ == "__main__":
    main()
