"""Replay the original forecast geometry and check all published development errors."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from style import DATA, read
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"code/q1_revised"))
from recovery_fit import State, receive, prediction


def development_points():
    chunks=[]
    anchors=read("window_manifest.csv").set_index("window_id").last_visible
    for kind in ("continuous","pre_first","pre_second"):
        df=read(f"cal_{kind}_points.csv.gz").query("split=='validation'").copy()
        for (uid,side),g in df.groupby(["record","side"]):
            with np.load(DATA/"records"/f"{uid}.npz") as z:
                t,xyz,good,gap=[z[f"{side}__{v}"] for v in ("t_s","position","reference_candidate","gap_before")]
            pos=xyz.mean(axis=1)
            if kind=="continuous":
                end=g.frame_index.to_numpy()
                a=g.window_id.map(anchors).to_numpy()
                speed=(pos[a]-pos[a-1])/(t[a]-t[a-1])[:,None]
                centers=pos[a]-.2*np.expm1(-(t[end]-t[a])/.2)[:,None]*speed
            else:
                end=g.frame.to_numpy()
                states=[State()]
                for j in range(max(end)+1):
                    states.append(receive(states[-1],j,t[j],pos[j],gap[j]) if good[j] else states[-1])
                centers=np.array([prediction(states[j],j-1,t[j:j+1],"decay")[0][0] for j in end])
            residuals=pos[end]-centers
            norms=np.linalg.norm(residuals,axis=1)
            np.testing.assert_allclose(norms,g.error,atol=1e-10,rtol=0)
            valid=g.radius.notna()
            np.testing.assert_array_equal(norms[valid]<=g.radius[valid],g.coverage[valid].astype(bool))
            g=g.copy()
            g[["dx","dy","dz"]]=residuals
            g[["cx","cy","cz"]]=centers
            chunks.append(g.assign(interface=kind))
    all_points=pd.concat(chunks,ignore_index=True)
    assert len(all_points)==6420 and all_points.radius.isna().sum()==8
    return all_points
