"""Public inference interface for the adopted Q1 and fixed-cost Q2 models."""
from pathlib import Path
import numpy as np
import pandas as pd
from trajectory import DELIVERY, PACKAGE, forecast_position, load_json, replay_adopted
from quality_recovery import quality_state


class AdoptedModel:
    def __init__(self, results=None):
        results=Path(results) if results is not None else Path(__file__).resolve().parents[1]/"results"
        with np.load(PACKAGE/"第一问/model/model_parameters.npz",allow_pickle=False) as f:
            self.parameters=dict(f)
        table=pd.read_csv(results/"q1_range_model.csv").query("model=='B'")
        self.radii=table.set_index(["dataset","bin"]).radius
        quality=pd.read_csv(results/"q1_quality_model.csv").query("held=='full'")
        self.quality=quality.set_index("state").probability
        self.paid=load_json(DELIVERY/"model/model.json")
        self.prior=load_json(DELIVERY/"model/prior_models.json")

    def predict_position(self,raw,side,anchor,ages,source):
        """Absolute centroid and empirical radius on supported Q1 forecast ages."""
        ages=np.asarray(ages,dtype=float)
        if ages.ndim!=1 or not len(ages) or not np.all((ages>0)&(ages<=1+1e-12)):
            raise ValueError("Q1 forecast ages must be a vector within (0, 1] seconds")
        pred=forecast_position(self.parameters,raw,side,anchor,ages,source)
        bins=np.clip(np.ceil(ages*30),1,30).astype(int)
        return pred,self.radii.loc[source].loc[bins].to_numpy()

    def quality_probability(self,known_bad_history):
        """Use known flags in the observed 0.2 s including the current sample."""
        flags=np.asarray(known_bad_history)
        if flags.ndim!=1 or not flags.size or flags.dtype!=np.dtype(bool):
            raise ValueError("expected a nonempty vector of observed boolean quality flags")
        return float(self.quality[quality_state(flags)])

    def replay_paid_window(self,raw,side,start,end,source):
        """Evaluate a paid window. Only the replay engine receives hidden positions."""
        t=raw[side+"__t_s"]
        x=raw[side+"__position"].mean(1)
        ages=t[start:end]-t[start-1]
        initial=forecast_position(self.parameters,raw,side,start-1,ages,source)
        return replay_adopted(t[start:end],x[start:end],t[start-2:start],x[start-2:start],
                              initial,self.paid["scale"][source],source,self.paid,self.prior)
