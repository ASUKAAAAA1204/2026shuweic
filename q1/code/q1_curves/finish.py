"""Audit saved coordinates and compare the candidates with matched global environments."""
import json
from experiment import np, pd, OLD, OUT, PREVIOUS, PRIMARY, HORIZONS, EPS


def verify(metrics):
    meta = pd.read_csv(PREVIOUS/'window_manifest.csv',float_precision='round_trip')
    maxima = dict(ade=0.,fde=0.,nade=0.)
    with np.load(OLD/'predictions.npz') as old, np.load(OUT/'predictions.npz') as new:
        offsets, times, truth = old['offsets'], old['t'], old['ref']
        np.testing.assert_array_equal(offsets,new['offsets'])
        for key in new.files:
            if key=='offsets':
                continue
            name,_,suffix = key.partition('_')
            seed = int(suffix) if suffix else PRIMARY
            subset = metrics[(metrics.model==name)&(metrics.seed==seed)]
            arr = new[key]
            np.testing.assert_array_equal(arr[offsets[:-1]],truth[offsets[:-1]])
            errors = np.sqrt(np.sum((arr-truth)**2,axis=1))
            for h in HORIZONS:
                computed = []
                for start,stop in zip(offsets[:-1],offsets[1:]):
                    t = times[start:stop]
                    count = np.searchsorted(t,h+EPS,side='right')
                    e = errors[start:start+count]
                    dt = np.diff(t[:count])
                    computed.append((np.dot(dt,e[:-1]+e[1:])/(2*dt.sum()),e[-1]))
                ade,fde = np.array(computed).T
                rows = subset[subset.horizon==h].sort_values('i')
                np.testing.assert_array_equal(rows.window_id,meta.window_id)
                for column,values in [('ade',ade),('fde',fde),('nade',ade/meta.scale.to_numpy())]:
                    maxima[column] = max(maxima[column],np.max(np.abs(rows[column]-values)))
    assert len(metrics)==1523*3*12 and max(maxima.values())<1e-10
    return maxima


def paired(env, scores):
    columns = ['B','M1','F','C']
    tab = env.groupby(['env','source','model']).nade.mean().unstack(['source','model'])
    by_source = [tab[source][columns].to_numpy() for source in sorted(set(tab.columns.get_level_values(0)))]
    assert len(tab)==16 and len(by_source)==3
    rng = np.random.default_rng(PRIMARY)
    samples,accepted,rejected = [],0,0
    while accepted<5000:
        draws = rng.integers(len(tab),size=(5000-accepted,len(tab)))
        valid = np.logical_and.reduce([np.isfinite(a[draws,0]).any(axis=1) for a in by_source])
        rejected += int((~valid).sum())
        counts = draws[valid]
        samples.append(np.mean([np.nanmean(a[counts],axis=1) for a in by_source],axis=0))
        accepted += len(counts)
    boot = pd.DataFrame(np.vstack(samples),columns=columns)
    means = np.mean([np.nanmean(a,axis=0) for a in by_source],axis=0)
    np.testing.assert_allclose(means,scores.loc[columns],atol=1e-12,rtol=0)
    rows = []
    for cand,ref in [('F','B'),('C','B'),('C','M1'),('M1','B')]:
        gain = 1-scores[cand]/scores[ref]
        lo,hi,cl,ch = np.quantile(1-boot[cand]/boot[ref],[.025,.975,.0125,.9875])
        rows.append((cand,ref,gain,lo,hi,cl,ch,ref=='B' and cand in ('F','C') and gain>=.03 and cl>0))
    return pd.DataFrame(rows,columns=['candidate','reference','gain','lo95','hi95','lo975','hi975','advance']),rejected


def main():
    metrics = pd.read_csv(OUT/'metrics.csv')
    errors = verify(metrics)
    keys = ['seed','model','horizon','source','env','record','side']
    stats = metrics.groupby(keys)[['nade','ade','fde','along','lateral']].mean()
    for _ in range(2):
        keys = keys[:-1]
        stats = stats.groupby(level=keys).mean()
    stats.to_csv(OUT/'environments.csv')
    sources = stats.groupby(level=keys[:-1]).mean()
    sources.to_csv(OUT/'sources.csv')
    horizons = sources.groupby(level=['seed','model','horizon']).mean()
    horizons.to_csv(OUT/'horizons.csv')
    scores = horizons.nade.groupby(level=['seed','model']).mean()
    scores.to_csv(OUT/'seed_scores.csv')
    previous = pd.read_csv(PREVIOUS/'seed_scores.csv').set_index(['seed','model']).score
    for k in scores.index:
        if k[1] in ('B','M1'):
            assert abs(scores[k]-previous[k])<1e-12
    comparisons,rejected = paired(stats.loc[PRIMARY].reset_index(),scores.loc[PRIMARY])
    comparisons.to_csv(OUT/'paired.csv',index=False)
    times = pd.read_csv(OUT/'timing.csv')
    assert len(times)==1523*4 and not times.duplicated(['window_id','model']).any()
    summary = scores.loc[PRIMARY].to_frame('nade')
    summary['gain_vs_B'] = 1-summary.nade/summary.at['B','nade']
    summary['p50_us'] = times.groupby('model').us.median()
    summary['p95_us'] = times.groupby('model').us.quantile(.95)
    summary.to_csv(OUT/'summary.csv')
    actions = pd.read_csv(OUT/'actions.csv')
    fallback = (~actions.active).groupby([actions.model,actions.source]).agg(['sum','count','mean'])
    fallback.to_csv(OUT/'fallback.csv')
    report = dict(checked_rows=len(metrics),max_abs_error=errors,accepted=5000,rejected=rejected)
    (OUT/'score_audit.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string())
    print(comparisons.to_string(index=False))


if __name__=='__main__':
    main()
