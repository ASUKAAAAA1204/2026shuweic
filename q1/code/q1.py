"""Question 1: environment-held-out displacement prediction and quality audit."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'C_Problem_Data/v1/v1'
OLD = ROOT.parent / 'C题工作区/阶段交付'
BASE = OLD / 'stage1_review/corrected_results'
OUT = ROOT / 'results/q1'
TAU = .2
PRIOR = np.array([1., 0., 0., 0., 0., 0.])
KEYS = ['window_id','episode_uuid','dataset','environment_id','split','side']


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def design(age, last, robust):
    decay = -np.expm1(-age / TAU)
    a = TAU * decay
    basis = np.stack((a, a * decay, age - a), axis=1)
    return (basis[:, :, None, None] * np.stack((last, robust))[None,None,:,:]).reshape(-1, 6, 3)


@dataclass
class Window:
    meta: dict
    age: np.ndarray
    rel: np.ndarray
    delta: np.ndarray
    features: np.ndarray
    gram: np.ndarray
    rhs: np.ndarray


def inputs():
    records = pd.read_csv(DATA / 'record_index.csv').set_index('episode_uuid')
    split_file = next((ROOT / 'C_Problem_Data').rglob('record_split.csv'))
    official = pd.read_csv(split_file).set_index('episode_uuid')
    assert official.index.is_unique and records.index.is_unique
    assert records.index.symmetric_difference(official.index).empty
    for col in ('split','environment_id'):
        assert records[col].equals(official.loc[records.index,col])
    assert records.groupby('environment_id').split.nunique().max() == 1
    catalog = pd.read_csv(BASE/'data/windows.csv').query("split in ['train','validation']")
    quality = pd.read_csv(BASE/'q12/reliability_predictions.csv')
    assert set(quality.split) == {'train','validation'}
    assert not quality.duplicated(['episode_uuid','side','index']).any()
    assert quality.label.isin([0,1]).all()
    assert quality[['probability','constant_probability']].apply(lambda p:p.between(0,1).all()).all()
    quality['current_bad'] = False
    segments = pd.read_csv(DATA/'segments_private.csv')
    windows = []
    for uid, group in catalog.groupby('episode_uuid'):
        rec = records.loc[uid]
        assert rec.split in ('train','validation') and (group.split == rec.split).all()
        with np.load(DATA/'records'/f'{uid}.npz', allow_pickle=False) as z:
            for side, subset in group.groupby('side'):
                t = z[side+'__t_s']
                x = z[side+'__position'].mean(axis=1)
                valid = z[side+'__reference_candidate']
                bad = z[side+'__bad']
                rows = quality.index[(quality.episode_uuid == uid) & (quality.side == side)]
                idx = quality.loc[rows,'index'].to_numpy()
                quality.loc[rows,'current_bad'] = bad[idx]
                end = np.searchsorted(t,t[idx]+.2,side='right')
                counts = np.r_[0,bad.cumsum()]
                assert np.array_equal(quality.loc[rows,'label'], counts[end] > counts[idx+1])
                history = np.searchsorted(t,t[idx]-.2)
                speed = np.linalg.norm(np.diff(x,axis=0),axis=1)/np.diff(t)
                quality.loc[rows,'median_speed']=[np.median(speed[h:k]) for h,k in zip(history,idx)]
                quality.loc[rows,'std_speed']=[np.std(speed[h:k]) for h,k in zip(history,idx)]
                quality.loc[rows,'bad_fraction']=(counts[idx+1]-counts[history])/(idx+1-history)
                for w in subset.itertuples():
                    i,j = w.start_index,w.end_index_exclusive
                    lo = max(w.history_start, i-6)
                    assert i-lo >= 2 and valid[lo:j].all() and np.all(np.diff(t[lo:j]) > 0)
                    a,b = np.triu_indices(i-lo,1)
                    hist = x[lo:i]
                    slopes = (hist[b]-hist[a])/(t[lo+b]-t[lo+a])[:,None]
                    age = t[i:j]-t[i-1]
                    f = design(age,slopes[-1],np.median(slopes,axis=0))
                    delta = x[i:j]-x[i-1]
                    mat = f.transpose(0,2,1).reshape(-1,6)
                    meta = {k:getattr(w,k) for k in KEYS}
                    hits = segments[(segments.episode_uuid == uid) & (segments.start_s <= t[i]) & (t[i] < segments.end_s)]
                    meta.update(family='unannotated', phase='unannotated')
                    if len(hits) == 1:
                        s = hits.iloc[0]
                        meta.update(family=s.family_id,phase=['early','middle','late'][min(2,int(3*(t[i]-s.start_s)/s.duration_s))])
                    windows.append(Window(meta,age,t[i:j]-t[i],delta,f,mat.T@mat/len(age),mat.T@delta.ravel()/len(age)))
    assert set(quality.episode_uuid) == {w.meta['episode_uuid'] for w in windows}
    assert quality[['std_speed','median_speed','bad_fraction']].notna().all().all()
    return windows, quality


def fit(ws):
    meta = pd.DataFrame([w.meta for w in ws])
    by_record = meta.groupby('episode_uuid').episode_uuid.transform('size')
    by_env = meta.groupby('environment_id').episode_uuid.transform('nunique')
    weights = 1. / (by_record * by_env)
    weights /= weights.sum()
    gram = sum(a*w.gram for a,w in zip(weights,ws))
    rhs = sum(a*w.rhs for a,w in zip(weights,ws))
    penalty = .01*np.trace(gram)/6 + 1e-12
    return np.linalg.solve(gram + penalty*np.eye(6),rhs + penalty*PRIOR)


def aggregate(df, groups, metrics):
    levels = groups+['environment_id','episode_uuid']
    rec = df.groupby(levels)[metrics].mean()
    env = rec.groupby(level=levels[:-1]).mean()
    return env.groupby(level=groups).mean().reset_index(),env.reset_index()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    windows, quality = inputs()
    allrows, params, cal, checks = [], {}, {}, {}
    models = ['hold','constant_velocity','damped','robust','ridge']
    for src in sorted({w.meta['dataset'] for w in windows}):
        train = [w for w in windows if w.meta['dataset']==src and w.meta['split']=='train']
        folds = {env:fit([w for w in train if w.meta['environment_id']!=env]) for env in sorted({w.meta['environment_id'] for w in train})}
        full = fit(train)
        params[src] = dict(full=full.tolist(),leave_one_env_out={k:v.tolist() for k,v in folds.items()})
        for split in ('train','validation'):
            errors = {m:[] for m in models}
            buckets = []
            for w in (w for w in windows if w.meta['dataset']==src and w.meta['split']==split):
                f=w.features
                coef = folds[w.meta['environment_id']] if split=='train' else full
                # a + (h-a) = h, hence columns 0 and 4 recover ordinary constant velocity.
                preds = [np.zeros_like(w.delta), f[:,0]+f[:,4], f[:,0], f[:,1], np.einsum('nfc,f->nc',f,coef)]
                bins = np.clip(np.ceil(w.age*30).astype(int),1,30)-1
                buckets.append(bins)
                for m,p in zip(models,preds):
                    err=np.linalg.norm(p-w.delta,axis=1)
                    errors[m].append(err)
                    for h in (.2,.6,1.):
                        take = w.rel <= h+1e-12
                        row=dict(**w.meta,model=m,horizon_s=h,ade=err[take].mean(),fde=err[take][-1])
                        if split=='validation':
                            radius=np.asarray(cal[src][m])[bins[take]]
                            row.update(coverage=np.mean(err[take]<=radius),radius=radius.mean())
                        allrows.append(row)
            if split=='train':
                b=np.concatenate(buckets)
                cal[src]={}
                for m,parts in errors.items():
                    e=np.concatenate(parts)
                    cal[src][m]=[float(np.quantile(e[b==i] if (b==i).any() else e,.9,method='higher')) for i in range(30)]
    frame=pd.DataFrame(allrows)
    prior=pd.read_csv(OLD/'stage2_optimization/E1/validation_windows.csv')
    prior['model']=prior.model.replace({'0.2':'damped'})
    joined=prior.merge(frame.query("split=='validation'"),on=['window_id','model','horizon_s'],validate='one_to_one',suffixes=('_old',''))
    checks['paired_archive_rows']=len(joined)
    checks['max_abs_ade_diff']=float(abs(joined.ade_old-joined.ade).max())
    assert len(joined)==len(prior) and checks['max_abs_ade_diff']<1e-10
    frame.to_csv(OUT/'motion_windows.csv',index=False)
    groups=['split','dataset','model','horizon_s']
    summary,env=aggregate(frame,groups,['ade','fde','coverage','radius'])
    summary.to_csv(OUT/'motion_summary.csv',index=False)
    env.to_csv(OUT/'motion_environments.csv',index=False)
    ratios=summary.query('horizon_s >= .6').pivot(index=['split','dataset','horizon_s'],columns='model',values='ade')
    ratio=ratios.div(ratios.damped,axis=0).groupby(level='split').mean()
    candidate=ratio.loc['train',['robust','ridge']].idxmin()
    selected=candidate if max(ratio.loc[:,candidate])<1 else 'damped'
    save('motion_model.json',dict(tau=TAU,selected=selected,candidate=candidate,ratios=ratio.to_dict(),coefficients=params,radius90=cal))
    stage=frame.query("split=='validation' and horizon_s==1 and model==@selected")
    phase,_=aggregate(stage,['dataset','family','phase'],['ade','coverage'])
    counts=stage.groupby(['dataset','family','phase']).agg(windows=('ade','size'),records=('episode_uuid','nunique'),environments=('environment_id','nunique'))
    phase.join(counts,on=['dataset','family','phase']).to_csv(OUT/'stage_summary.csv',index=False)
    save('checks.json',checks)
    source_quality(quality)
    from quality_q1 import analyze_quality
    analyze_quality(quality,OUT)


def source_quality(data):
    from sklearn.linear_model import LogisticRegression
    cols=['median_speed','std_speed','bad_fraction','current_bad']
    train=data.query("split=='train'")
    val=data.query("split=='validation'")
    output=[]
    def evaluate(fit_data, evaluation, split):
        X=fit_data[cols].to_numpy(dtype=float)
        mu,sd=X.mean(axis=0),X.std(axis=0)
        sd[sd<1e-10]=1
        model=LogisticRegression(C=1.,tol=1e-11,max_iter=500).fit((X-mu)/sd,fit_data.label)
        rows=evaluation.copy()
        xnew=(rows[cols].to_numpy(dtype=float)-mu)/sd
        rows['logistic']=model.predict_proba(xnew)[:,1]
        global_rates=fit_data.groupby('current_bad').label.agg(['sum','count'])
        base=(global_rates['sum']+1)/(global_rates['count']+2)
        cells=fit_data.groupby(['dataset','current_bad']).label.agg(['sum','count'])
        params={(s,bool(b)):(v['sum']+50*base[b])/(v['count']+50) for (s,b),v in cells.iterrows()}
        rows['source_state']=[params.get((s,b),base[b]) for s,b in zip(rows.dataset,rows.current_bad)]
        rows['shared_state']=rows.current_bad.map(base)
        rows['constant']=fit_data.label.mean()
        if split=='validation':
            assert np.max(np.abs(rows.logistic-rows.probability)) < 1e-5
            save('source_quality_model.json',dict(strength=50,source_state={f'{s}|{b}':p for (s,b),p in params.items()},global_state={str(b):p for b,p in base.items()}))
        rows['split']=split
        output.append(rows)
    for env in sorted(train.environment_id.unique()):
        evaluate(train[train.environment_id!=env],train[train.environment_id==env],'train_oof')
    evaluate(train,val,'validation')
    rows=pd.concat(output,ignore_index=True)
    long=rows.melt(id_vars=['split','episode_uuid','environment_id','dataset','index','side','current_bad','label'],value_vars=['constant','logistic','source_state','shared_state'],var_name='model',value_name='p')
    long['brier']=(long.p-long.label)**2
    for field,subset in [('all',long),('normal',long[~long.current_bad]),('flagged',long[long.current_bad])]:
        scores,_=aggregate(subset,['split','dataset','model'],['brier','label','p'])
        counts=subset.groupby(['split','dataset','model']).label.agg(samples='size',events='sum')
        scores.join(counts,on=['split','dataset','model']).to_csv(OUT/f'quality_{field}.csv',index=False)
    rows.to_csv(OUT/'quality_predictions.csv',index=False)


if __name__ == '__main__':
    run()
