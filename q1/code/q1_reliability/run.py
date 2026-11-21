"""Causal 0.2 s quality-event prediction on the archived train/development anchors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'C_Problem_Data/v1/v1'
ARCHIVE = ROOT/'results/q1/quality_predictions.csv'
OUT = ROOT/'results/q1_reliability'
KEY = ['episode_uuid','side','index']
GROUP = ['dataset','environment_id','episode_uuid']
BASE = ['median_speed','std_speed','bad_fraction','current_bad']
QUAL = ['bad_fraction','recent_change','elapsed']
MOTION = ['median_speed','std_speed','last_speed','joint_spread']
MODELS = ['Constant','State','Source','LR','Q','D50','D','HQ','HM','State3']
STRENGTHS = [50,200,10]


def observed(t,xyz,bad):
    """All arguments end at the current anchor, including its quality mark."""
    centers = xyz.mean(axis=1)
    speed = np.linalg.norm(np.diff(centers,axis=0),axis=1)/np.diff(t)
    v = (xyz[-1]-xyz[-2])/(t[-1]-t[-2])
    current = bool(bad[-1])
    others = np.flatnonzero(bad!=current)
    start = others[-1]+1 if len(others) else 0
    elapsed = t[-1]-t[start]
    cell = 'stable' if not len(others) else 'short' if elapsed<.1 else 'long'
    return dict(current_bad=current,bad_fraction=bad.mean(),median_speed=np.median(speed),
                std_speed=speed.std(),recent_change=bool(len(others)),elapsed=elapsed,
                last_speed=speed[-1],joint_spread=np.median(np.linalg.norm(v-v.mean(axis=0),axis=1)),
                cell=cell)


def rebuild():
    archived = pd.read_csv(ARCHIVE,float_precision='round_trip')
    official = pd.read_csv(next((ROOT/'C_Problem_Data').rglob('record_split.csv'))).set_index('episode_uuid')
    assert not archived.duplicated(KEY).any()
    rows = []
    for (uid,side),group in archived.groupby(['episode_uuid','side'],sort=False):
        split = official.at[uid,'split']
        assert split in ('train','validation') and all(group.split==('train_oof' if split=='train' else split))
        assert all(group.environment_id==official.at[uid,'environment_id']) and all(group.dataset==official.at[uid,'dataset'])
        with np.load(DATA/'records'/f'{uid}.npz',allow_pickle=False) as z:
            t,xyz,bad,known = [z[side+'__'+k] for k in ('t_s','position','bad','quality_known')]
        for row in group.itertuples():
            i = row.index
            lo = np.searchsorted(t,t[i]-.2)
            end = np.searchsorted(t,t[i]+.2,side='right')
            assert known[lo:end].all() and end>i+1 and t[-1]>=t[i]+.2-1e-9
            f = observed(t[lo:i+1],xyz[lo:i+1],bad[lo:i+1])
            y = int(bad[i+1:end].any())
            assert y==row.label and f['current_bad']==row.current_bad
            np.testing.assert_allclose([f[k] for k in BASE[:3]],[getattr(row,k) for k in BASE[:3]],atol=1e-10)
            rows.append(dict(episode_uuid=uid,side=side,index=i,split=split,label=y,dataset=row.dataset,
                             environment_id=row.environment_id,**f))
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(KEY).any() and frame.groupby('environment_id').split.nunique().max()==1
    return frame,archived


def probabilities(fit):
    state = fit.groupby('current_bad').label.agg(['count','sum'])
    return ((state['sum']+1)/(state['count']+2)).to_dict()


def recovery_state(rows):
    """Visible quality in the last 0.2 s, with current bad taking precedence."""
    return pd.Series(np.select([rows.current_bad, rows.bad_fraction.gt(0)],
                               ['flagged','recent_recovery'],default='clean_history'),
                     index=rows.index,name='quality_state')


def three_state_rates(fit):
    cells = fit.groupby(recovery_state(fit)).label.agg(events='sum',samples='size')
    cells['p'] = (cells.events+1)/(cells.samples+2)
    return cells


def shrunk(fit,x,base,cols,alpha):
    table = fit.groupby(cols).label.agg(['sum','count']).reset_index()
    table['prob'] = (table['sum']+alpha*table.current_bad.map(base))/(table['count']+alpha)
    preds = x[cols].merge(table[cols+['prob']],how='left',on=cols,sort=False,validate='many_to_one').prob
    return np.where(preds.isna(),x.current_bad.map(base),preds)


def mean_brier(frame,p):
    rec = frame[GROUP].assign(error=(p-frame.label.to_numpy())**2).groupby(GROUP).error.mean()
    return rec.groupby(level=GROUP[:2]).mean().groupby(level='dataset').mean().mean()


def tune(fit):
    envs = sorted(fit.environment_id.unique())
    p = {s:pd.Series(index=fit.index,dtype=float) for s in STRENGTHS}
    for j in range(4):
        mask = fit.environment_id.isin(envs[j::4])
        sub,x = fit[~mask],fit[mask]
        base = probabilities(sub)
        for s in STRENGTHS:
            p[s].loc[x.index] = shrunk(sub,x,base,['current_bad','cell'],s)
    assert all(v.notna().all() for v in p.values())
    scores = {s:mean_brier(fit,v.to_numpy()) for s,v in p.items()}
    return min(scores,key=scores.get),scores


def logistic(fit,x,cols):
    if fit.label.nunique()<2:
        return np.full(len(x),(fit.label.sum()+1)/(len(fit)+2))
    a,b = fit[cols].to_numpy(float),x[cols].to_numpy(float)
    mu,sd = a.mean(axis=0),a.std(axis=0)
    sd[sd<1e-10] = 1
    model = LogisticRegression(C=1.,tol=1e-11,max_iter=500).fit((a-mu)/sd,fit.label)
    return model.predict_proba((b-mu)/sd)[:,1]


def forecast(fit,x,strength):
    base = probabilities(fit)
    p = pd.DataFrame(index=x.index)
    p['Constant'] = fit.label.mean()
    p['State'] = x.current_bad.map(base)
    p['State3'] = recovery_state(x).map(three_state_rates(fit).p).fillna(p.State)
    np.testing.assert_array_equal(p.loc[x.current_bad,'State3'],p.loc[x.current_bad,'State'])
    p['Source'] = shrunk(fit,x,base,['dataset','current_bad'],50)
    p['LR'] = logistic(fit,x,BASE)
    p['Q'] = logistic(fit,x,['bad_fraction','current_bad'])
    for name,s in [('D50',50),('D',strength)]:
        p[name] = shrunk(fit,x,base,['current_bad','cell'],s)
    good = ~x.current_bad
    for name,cols in [('HQ',QUAL),('HM',QUAL+MOTION)]:
        p[name] = p.D.copy()
        p.loc[good,name] = .5*(p.loc[good,'D']+logistic(fit[~fit.current_bad],x[good],cols))
    assert np.isfinite(p).all().all() and p.ge(0).all().all() and p.le(1).all().all()
    return p


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    frame,archived = rebuild()
    train = frame[frame.split=='train']
    val = frame[frame.split=='validation']
    assert train.environment_id.nunique()==16 and val.environment_id.nunique()==4
    params,parts,state3_rates = [],[],[]
    for env in [*sorted(train.environment_id.unique()),'development']:
        fit,x = (train,val) if env=='development' else (train[train.environment_id!=env],train[train.environment_id==env])
        assert set(fit.environment_id).isdisjoint(x.environment_id)
        strength,scores = tune(fit)
        # Prediction routine never receives the held labels or future coordinates.
        visible = x.drop(columns=['label'])
        p = forecast(fit,visible,strength)
        state3_rates.append(three_state_rates(fit).reset_index().assign(held=env))
        parts.append(x.join(p))
        params.append(dict(held=env,training_envs=fit.environment_id.nunique(),n=len(fit),strength=strength,
                           **{f'inner_{s}':v for s,v in scores.items()}))
    result = pd.concat(parts).sort_index()
    assert len(result)==len(frame) and not result.duplicated(KEY).any()
    paired = result.merge(archived[KEY+['constant','shared_state','source_state','logistic']],on=KEY,validate='one_to_one')
    delta = {m:float((paired[m]-paired[b]).abs().max()) for m,b in
             [('Constant','constant'),('State','shared_state'),('Source','source_state'),('LR','logistic')]}
    assert max(delta.values())<1e-5
    result.to_csv(OUT/'predictions.csv.gz',index=False)
    pd.DataFrame(params).to_csv(OUT/'folds.csv',index=False)
    pd.concat(state3_rates,ignore_index=True).to_csv(OUT/'state3_rates.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(baseline_max_abs_difference=delta,
        anchor_archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),rows=len(frame),test_records_read=0),indent=2))
    print('Baseline check',delta,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
