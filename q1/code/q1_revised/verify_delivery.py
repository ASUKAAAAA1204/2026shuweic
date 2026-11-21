"""Independent simple-predictor scores, cluster sizes, and deliverable consistency checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_position import DATA, OUT, HORIZONS,ROOT

def main():
    new=pd.read_csv(OUT/"curve/window_metrics.csv.gz")
    sim=new[new.model.isin(["Hold","CV","B"])].drop_duplicates(["window_id","horizon","model"])
    idx=sim.set_index(["window_id","horizon","model"]).sort_index()
    assert idx.index.is_unique and len(idx)==1739*3*3
    manifest=pd.read_csv(OUT/"window_manifest.csv")
    maxerr=0.
    for (uid,side),g in manifest.groupby(["record","side"]):
        with np.load(DATA/"records"/f"{uid}.npz") as z:
            t=z[side+"__t_s"]; x=z[side+"__position"].mean(1)
        for w in g.itertuples():
            i,j=w.first_hidden,w.end_exclusive
            age=t[i:j]-t[i-1];v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2])
            for k,delta in [("CV",age),("Hold",0*age),("B",-.2*np.expm1(-age/.2))]:
                err=np.linalg.norm(x[i:j]-(x[i-1]+delta[:,None]*v),axis=1)
                for h in HORIZONS:
                    keep=age<=h+1e-12;ref=idx.loc[(w.window_id,h,k)]
                    maxerr=max(maxerr,abs(err[keep].mean()-ref.ade),abs(err[keep][-1]-ref.fde))
    assert maxerr<=1e-10
    qs=pd.read_csv(OUT/"quality_predictions.csv.gz")
    assert len(qs)==17724 and not qs.duplicated(["episode_uuid","side","index"]).any()
    assert set(new.split)==set(qs.split)=={"train","validation"}
    sha=pd.read_csv(OUT/"inputs.csv")
    if sha.path.str.startswith("baseline-source-review/").any():
        # The archived run used full local NPZs; verify its published counterparts separately.
        sha = pd.read_csv(OUT/"inputs_checkout.csv")
        data_dir = ROOT/"C_Problem_Data/v1/v1"
    else:
        data_dir = DATA
    for r in sha.itertuples():
        prefix = "C_Problem_Data/v1/v1/"
        path = data_dir/r.path[len(prefix):] if r.path.startswith(prefix) else ROOT/r.path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r.sha256, r.path
    scores=pd.read_csv(OUT/"position_overall.csv")
    assert len(scores)==8
    q=json.loads((OUT/"audit_quality.json").read_text())
    p={"simple_window_horizon_model_rows":len(idx),"max_independent_ade_fde_difference":maxerr,
       "input_hashes_verified":len(sha),"reliability_max_probability_differences":q,"test_performance_rows":0}
    natural=pd.read_csv(OUT/"recovery/natural_points.csv.gz").query("model=='decay'")
    ref=pd.read_csv(OUT/"natural_reference.csv")
    check=natural.merge(ref,left_on=["record","side","event","phase"],right_on=["record","side","start","phase"],validate="one_to_one")
    assert len(check)==len(ref)==len(natural)
    np.testing.assert_array_equal(check.error.isna(),check.decay_error.isna())
    np.testing.assert_allclose(check.error,check.decay_error,atol=1e-10,rtol=0,equal_nan=True)
    p["recovery_candidate_event_points"]=len(check)
    p["max_recovery_error_replay_difference"]=float(abs(check.error-check.decay_error).max())
    dev=new.query("model=='C' and seed==20261120 and split=='validation'").groupby("window_id").nade.mean()
    other=pd.read_csv(OUT/"curve_dev_independent.csv.gz").query("fold=='development'").set_index("window_id").nade
    assert len(dev)==len(other)==212 and set(dev.index)==set(other.index)
    p["max_mlp_dev_window_difference"]=float(abs(dev-other).max())
    assert p["max_mlp_dev_window_difference"]<1e-10
    (OUT/"delivery_check.json").write_text(json.dumps(p,indent=2))
    print(p)
if __name__=="__main__":main()
