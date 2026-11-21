"""Recompute scores from saved coordinates, aggregate all seeds and pair environments."""
import json
import numpy as np
import pandas as pd
from compare import OLD, OUT, SEEDS, HORIZONS, EPS, NAMES
from model.paired import summarize


def check_points(metrics,meta):
    """Independent integration using segment weights, with all cached predictors."""
    blocks=[]
    with np.load(OLD/'predictions.npz') as baseline,np.load(OUT/'predictions.npz') as model:
        off=baseline['offsets']
        t=baseline['t']
        truth=baseline['ref']
        assert np.array_equal(off,model['offsets'])
        for key in model.files:
            if key=='offsets':
                continue
            name,_,suffix=key.partition('_')
            seed=int(suffix) if suffix else SEEDS.start
            pred=model[key]
            assert np.isfinite(pred).all() and np.array_equal(pred[off[:-1]],truth[off[:-1]])
            errors=np.sqrt(((pred-truth)**2).sum(axis=1))
            for i,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
                age=t[lo:hi]
                values=errors[lo:hi]
                for h in HORIZONS:
                    count=np.searchsorted(age,h+EPS,side='right')
                    dt=np.diff(age[:count])
                    e=values[:count]
                    ade=np.dot(dt,e[:-1]+e[1:])/(2*dt.sum())
                    blocks.append((meta.window_id.iat[i],name,seed,h,ade,e[-1],ade/meta.scale.iat[i]))
    keys=['window_id','model','seed','horizon']
    computed=pd.DataFrame(blocks,columns=keys+['ade','fde','nade']).set_index(keys).sort_index()
    actual=metrics.set_index(keys).sort_index()
    assert actual.index.is_unique and actual.index.equals(computed.index)
    deltas=(actual[computed.columns]-computed).abs().max()
    assert deltas.max()<1e-10
    return deltas.to_dict()


def run():
    metrics=pd.read_csv(OUT/'metrics.csv')
    meta=pd.read_csv(OUT/'window_manifest.csv')
    audit=check_points(metrics,meta)
    primary=metrics[metrics.seed==SEEDS.start]
    paired=[]
    for kind in ('M3','M1'):
        data=primary[primary.model.isin(['B','K','K0',kind])].copy()
        data['model']=data.model.replace({kind:'MLP'})
        env,sc,comp=summarize(data)
        comp[['candidate','reference']]=comp[['candidate','reference']].replace({'MLP':kind})
        comp['comparison_set']=kind
        paired.append(comp)
    comparisons=pd.concat(paired,ignore_index=True)
    comparisons.to_csv(OUT/'paired.csv',index=False)
    keys=['seed','model','horizon','source','env','record','side']
    stats=metrics.groupby(keys)[['nade','ade','fde']].mean()
    for _ in range(2):
        keys=keys[:-1]
        stats=stats.groupby(level=keys).mean()
    stats.to_csv(OUT/'environments.csv')
    source=stats.groupby(level=keys[:-1]).mean()
    source.to_csv(OUT/'sources.csv')
    scores=source.nade.groupby(level=['seed','model','horizon']).mean().groupby(level=['seed','model']).mean().rename('score').reset_index()
    scores.to_csv(OUT/'seed_scores.csv',index=False)
    summary=scores.query('seed==@SEEDS.start').copy().set_index('model')
    for r in comparisons.itertuples():
        assert abs(r.gain-(1-summary.at[r.candidate,'score']/summary.at[r.reference,'score']))<1e-10
    original=pd.read_csv(OLD/'summary.csv').set_index('model')
    for m in ('K0','K','B'):
        assert abs(summary.at[m,'score']-original.at[m,'score'])<1e-12
    latency=pd.read_csv(OUT/'timing.csv')
    assert len(latency)==len(meta)*len(NAMES) and not latency.duplicated(['window_id','model']).any()
    summary['gain_vs_B']=1-summary.score/summary.at['B','score']
    summary['median_us']=latency.groupby('model').us.median()
    summary['p95_us']=latency.groupby('model').us.quantile(.95)
    summary.to_csv(OUT/'summary.csv')
    (OUT/'score_audit.json').write_text(json.dumps(dict(verified_rows=len(metrics),max_abs_difference=audit),indent=2))
    print(summary.to_string())
    print(comparisons.to_string(index=False))
    print(scores[scores.model.isin(['M3','M1'])].to_string(index=False))


if __name__=='__main__':
    run()
