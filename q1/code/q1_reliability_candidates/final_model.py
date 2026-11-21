"""Export the motion-tree extension comparator and verify raw-history predictions."""
import json
import joblib
import sklearn
from compare import (DATA, ROOT, OUT, KIN, KEY, np, pd, fit_clean_tree,
                     three_state_rates, recovery_state, threadpool_limits)
from run import observed


def predict(model,t,xyz,bad,known):
    """One hand, timestamps in seconds, 21 raw positions and observed quality masks."""
    t = np.asarray(t,dtype=float)
    xyz,bad,known = np.asarray(xyz),np.asarray(bad),np.asarray(known)
    if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or not np.isfinite(t).all():
        raise ValueError('Need at least two strictly increasing finite timestamps')
    if xyz.shape!=(len(t),21,3) or bad.shape!=t.shape or known.shape!=t.shape:
        raise ValueError('History arrays must be aligned')
    lo = np.searchsorted(t,t[-1]-.2)
    if len(t)-lo<2 or not known[lo:].all() or not np.isfinite(xyz[lo:]).all():
        raise ValueError('Visible history needs known marks and finite keypoints')
    if bad[lo:].dtype.kind not in 'buif' or not np.isin(bad[lo:],[0,1]).all() or not np.isin(known[lo:],[1]).all():
        raise ValueError('Expected binary observed quality masks')
    f = observed(t[lo:],xyz[lo:],bad[lo:].astype(bool))
    state = 'flagged' if f['current_bad'] else 'recent_recovery' if f['bad_fraction']>0 else 'clean_history'
    p = model['rates'][state]
    if state=='clean_history':
        x = pd.DataFrame([{k:f[k] for k in model['features']}])
        p = .5*p+.5*model['tree'].predict_proba(x)[0,1]
    return float(p)


def export():
    decisions = pd.read_csv(OUT/'decision.csv')
    assert list(decisions.loc[decisions.qualifies,'model'])==['Motion']
    train = pd.read_csv(ROOT/'results/q1_reliability/predictions.csv.gz',float_precision='round_trip').query("split=='train'").copy()
    train['state'] = recovery_state(train)
    model = dict(features=KIN,rates=three_state_rates(train).p.to_dict(),
                 tree=fit_clean_tree(train,KIN),sklearn_version=sklearn.__version__)
    target = OUT/'three_state_motion.joblib'
    joblib.dump(model,target)
    loaded = joblib.load(target)
    dev = pd.read_csv(OUT/'predictions.csv.gz',float_precision='round_trip').query("split=='validation'")
    errors = []
    for (uid,side),block in dev.groupby(['episode_uuid','side']):
        with np.load(DATA/'records'/f'{uid}.npz',allow_pickle=False) as raw:
            t,xyz,bad,known = [raw[side+'__'+suffix] for suffix in ('t_s','position','bad','quality_known')]
        for row in block.itertuples():
            end = row.index+1
            lo = np.searchsorted(t,t[row.index]-.2)
            p = predict(loaded,t[lo:end],xyz[lo:end],bad[lo:end],known[lo:end])
            errors.append(abs(p-row.Motion))
    assert max(errors)<1e-12 and len(errors)==2236
    (OUT/'export_check.json').write_text(json.dumps(dict(development_replays=len(errors),
        max_probability_difference=max(errors),rates=model['rates'],features=KIN,
        sklearn=sklearn.__version__),indent=2))
    print('exported and verified',len(errors),max(errors))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        export()
