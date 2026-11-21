"""Independently check Brier aggregation, frozen contrasts and decision gates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2]/'results/q1_reliability_candidates'
ALL = ['State','State3','Pool','Chain','Motion','Video']


def main():
    out = pd.read_csv(OUT/'predictions.csv.gz',float_precision='round_trip')
    score = pd.read_csv(OUT/'summary.csv').set_index(['split','subset'])
    paired = pd.read_csv(OUT/'paired.csv').query("reference=='State3'").set_index('candidate')
    maxdiff = 0.
    for split,g in out.groupby('split'):
        for subset in ['all',*g.state.unique()]:
            x = g if subset=='all' else g[g.state==subset]
            size = x.groupby(['dataset','environment_id','episode_uuid']).label.transform('size')
            records = x.groupby(['dataset','environment_id']).episode_uuid.transform('nunique')
            envs = x.groupby('dataset').environment_id.transform('nunique')
            w = 1/(size*records*envs*x.dataset.nunique())
            np.testing.assert_allclose(w.sum(),1.,atol=1e-12)
            y = x.label.to_numpy()[:,None]
            truth = w.to_numpy() @ np.square(x[ALL].to_numpy()-y)
            maxdiff = max(maxdiff,abs(truth-score.loc[(split,subset),ALL].to_numpy()).max())
    assert maxdiff<1e-12
    result = []
    for m,row in paired.iterrows():
        train = score.loc[('train','all')]
        dev = score.loc[('validation','all')]
        gain = 100*(1-train[m]/train.State3)
        np.testing.assert_allclose(gain,row.gain_pct,atol=1e-10)
        dev_gain = 100*(1-dev[m]/dev.State3)
        result.append(dict(model=m,train=train[m],development=dev[m],gain_train=gain,gain_development=dev_gain,
                           lo9875=row.lo9875,hi9875=row.hi9875,qualifies=bool(row.lo9875>0 and dev_gain>0)))
    results = pd.DataFrame(result).sort_values('train')
    results.to_csv(OUT/'decision.csv',index=False)
    with (OUT/'audit.json').open() as f:
        audit = json.load(f)
    audit['max_independent_brier_error'] = maxdiff
    audit['qualified'] = results.loc[results.qualifies,'model'].tolist()
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(results.to_string(index=False))


if __name__=='__main__':
    main()
