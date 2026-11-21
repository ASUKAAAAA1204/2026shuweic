"""Run the three approved ablations on the frozen development windows."""
import json
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import pandas as pd
from q1 import BASE, DATA, KEYS, ROOT, aggregate
from approved_filters import filtered_velocities

DEST = ROOT/'results/q1_approved'
TAUS = (.05,.1,.2,.4,.8)
HORIZONS = (.6,1.)
MODELS = ['A','B','C','D','H0','H1','H2']


def geometric_median(points):
    # Modified Weiszfeld update handles repeated observations at the iterate.
    x = points.mean(axis=0)
    scale = np.max(np.linalg.norm(points-x,axis=1))
    if scale == 0:
        return x
    for _ in range(2000):
        diff = points-x
        distance = np.linalg.norm(diff,axis=1)
        mask = distance > 1e-13*scale
        if not mask.any():
            return x
        direction = (diff[mask]/distance[mask,None]).sum(axis=0)
        norm = np.linalg.norm(direction)
        if norm <= (~mask).sum():
            return x
        step = direction*(1-(~mask).sum()/norm)/(1/distance[mask]).sum()
        x = x+step
        if np.linalg.norm(step) < 1e-9*scale:
            return x
    raise ArithmeticError('geometric median failed to converge')


@dataclass
class Window:
    meta: dict
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    median: np.ndarray
    head: object
    age: np.ndarray
    delta: np.ndarray
    ends: list


def load():
    catalog = pd.read_csv(BASE/'data/windows.csv').query("split in ['train','validation']")
    official = pd.read_csv(next(DATA.parents[1].rglob('record_split.csv'))).set_index('episode_uuid')
    metadata = json.loads((DATA/'q1_feature_names.json').read_text(encoding='utf8'))
    windows = []
    for (uid,side),g in catalog.groupby(['episode_uuid','side']):
        rec = official.loc[uid]
        assert (g.split==rec.split).all() and (g.environment_id==rec.environment_id).all()
        with np.load(DATA/'records'/f'{uid}.npz',allow_pickle=False) as z:
            t = z[side+'__t_s']
            joints = z[side+'__position']
            pos = joints.mean(1)
            good = z[side+'__reference_candidate']
            gap = z[side+'__gap_before']
            features = metadata[uid]['metadata'].get('info',{}).get('features',{})
            names = ['observation.state.'+n+'_world' for n in ('head','hand_left','hand_right')]
            common = all(n in features and features[n]['names'][-1]==['x','y','z'] for n in names)
            head_pos = z['head__position'][:,0]
            ht = z['head__t_s']
            hg = z['head__finite'] & z['head__quaternion_unit_1e3'] & z['head__time_increasing']
            hgap = z['head__gap_before']
            for w in g.itertuples():
                i,j,lo = w.start_index,w.end_index_exclusive,w.history_start
                if i-lo<2 or not good[i-2:j].all() or gap[i-1:j].any():
                    continue
                while lo < i-2 and (not good[lo:i].all() or gap[lo+1:i].any()):
                    lo += 1
                dt = t[i-1]-t[i-2]
                vel = (pos[i-1]-pos[i-2])/dt
                joint_v = (joints[i-1]-joints[i-2])/dt
                np.testing.assert_allclose(vel,joint_v.mean(0),atol=1e-10,rtol=1e-10)
                median = geometric_median(joint_v)
                head = None
                if common and np.array_equal(ht[i-2:i],t[i-2:i]) and hg[i-2:i].all() and not hgap[i-1]:
                    head = (head_pos[i-1]-head_pos[i-2])/dt
                prev = (pos[i-2]-pos[i-3])/(t[i-2]-t[i-3]) if i-lo>=3 else np.full(3,np.nan)
                speed,previous = np.linalg.norm(vel),np.linalg.norm(prev)
                cosine = prev@vel/(speed*previous) if min(speed,previous)>0 else np.nan
                meta = {k:getattr(w,k) for k in KEYS}
                meta.update(speed=speed,previous=previous,cosine=cosine,history=i-lo,
                            dispersion=np.median(np.linalg.norm(joint_v-median,axis=1)),
                            head_speed=np.linalg.norm(head) if head is not None else np.nan)
                ends = [np.searchsorted(t[i:j]-t[i],h+1e-12,side='right') for h in HORIZONS]
                windows.append(Window(meta,t[lo:i],pos[lo:i],vel,median,head,t[i:j]-t[i-1],pos[i:j]-pos[i-1],ends))
    return windows,catalog


def scores(w, prediction):
    # candidate x future timestamp x coordinate -> candidate x horizon x metric
    error = np.linalg.norm(prediction-w.delta,axis=-1)
    return np.stack([np.stack([error[:,:n].mean(1),error[:,n-1]],axis=-1) for n in w.ends],axis=1)


def damping(age,tau):
    return (-tau*np.expm1(-age/tau))[:,None]


def static(w):
    a = {t:damping(w.age,t) for t in TAUS}
    results = dict(A=scores(w,np.array([a[.2]*w.v])),B=scores(w,np.array([a[.2]*w.median])))
    if w.head is not None:
        results['H0'] = results['A']
        results['H1'] = scores(w,np.array([a[t]*w.v for t in TAUS]))
        pairs = [(s,t) for s in TAUS for t in TAUS]
        results['H2'] = scores(w,np.array([a[s]*w.head+a[t]*(w.v-w.head) for s,t in pairs]))
        np.testing.assert_allclose(results['H2'][pairs.index((.2,.2))],results['A'][0],atol=1e-10)
    return results


def run():
    windows,catalog = load()
    meta = pd.DataFrame([w.meta for w in windows])
    assert meta.groupby('environment_id').split.nunique().max()==1
    frozen = [static(w) for w in windows]
    choices = dict(A=[None],B=[None],C=[(f,0) for f in (.5,1,2,4)],
                   D=[(f,k) for f in (.5,1,2,4) for k in (.5,1,2)],H0=[.2],H1=TAUS,
                   H2=[(s,t) for s in TAUS for t in TAUS])
    rows,params = [],[]
    for source,part in meta.groupby('dataset'):
        train = part[part.split=='train']
        val = part[part.split=='validation']
        folds = [(env,train[train.environment_id!=env],g) for env,g in train.groupby('environment_id')]
        for env,fit,test in folds+[('validation',train,val)]:
            scale = fit.loc[fit.speed>0,'speed'].median()
            cut = fit[['dispersion','speed','head_speed']].quantile(.75)
            low,mid = fit.speed.quantile([.25,.5])
            values = {}
            for idx in list(fit.index)+list(test.index):
                w = windows[idx]
                speeds = filtered_velocities(w.t,w.x,scale)
                values[idx] = frozen[idx].copy()
                for m in ['C','D']:
                    v = np.array([speeds[p] for p in choices[m]])
                    values[idx][m] = scores(w,damping(w.age,.2)*v[:,None,:])
            for m in MODELS:
                ids = [i for i in fit.index if m in values[i]]
                if not ids:
                    continue
                # Equal environment, equal record within environment, equal window within record.
                eligible = fit.loc[ids]
                weights = 1/(eligible.groupby('episode_uuid').window_id.transform('size')*
                             eligible.groupby('environment_id').episode_uuid.transform('nunique'))
                loss = np.average(np.stack([values[i][m][:,:,0].mean(1) for i in ids]),axis=0,weights=weights)
                pick = int(np.argmin(loss))
                params.append(dict(dataset=source,fold=env,model=m,parameter=str(choices[m][pick]),
                                   scale=scale,low=low,median=mid,dispersion75=cut.dispersion,head75=cut.head_speed))
                for i in test.index:
                    if m not in values[i]:
                        continue
                    w = windows[i]
                    info = w.meta
                    for h,(ade,fde) in zip(HORIZONS,values[i][m][pick]):
                        rows.append(dict(**info,horizon_s=h,model=m,ade=ade,fde=fde,
                                         high_disp=info['dispersion']>=cut.dispersion,low_speed=info['speed']<=low,
                                         fast_turn=min(info['previous'],info['speed'])>mid and info['cosine']<=np.cos(np.pi/4),
                                         fast_head=info['head_speed']>=cut.head_speed))
    frame = pd.DataFrame(rows)
    old = pd.read_csv(ROOT/'results/q1/motion_windows.csv').query("model=='damped' and horizon_s>=.6")
    baseline = frame[frame.model=='A'].merge(old,on=['window_id','horizon_s'],validate='one_to_one')
    assert len(baseline)==(frame.model=='A').sum()
    for metric in ['ade','fde']:
        np.testing.assert_allclose(baseline[metric+'_x'],baseline[metric+'_y'],atol=1e-10)
    six = pd.read_csv(ROOT/'results/q1_ablation/velocity_paired.csv')
    prior = set(six.loc[six.split=='validation','window_id'])
    assert len(prior)==152 and prior<=set(frame.window_id)
    DEST.mkdir(parents=True,exist_ok=True)
    frame.to_csv(DEST/'windows.csv.gz',index=False)
    pd.DataFrame(params).to_csv(DEST/'parameters.csv',index=False)
    sums,ratios,envs = [],[],[]
    subsets = dict(all=frame,old152=frame[frame.window_id.isin(prior)],
                   high_disp=frame[frame.high_disp],other_disp=frame[~frame.high_disp],
                   low_speed=frame[frame.low_speed],fast_turn=frame[frame.fast_turn],
                   fast_head=frame[frame.fast_head])
    for name,sub in subsets.items():
        if sub.empty: continue
        groups = ['split','dataset','horizon_s','model']
        summary,env = aggregate(sub,groups,['ade','fde'])
        count = sub.groupby(groups).agg(windows=('window_id','size'),records=('episode_uuid','nunique'),envs=('environment_id','nunique'))
        sums.append(summary.join(count,on=groups).assign(subset=name))
        envs.append(env.assign(subset=name))
        for metric in ['ade','fde']:
            wide = summary.pivot(index=groups[:-1],columns='model',values=metric)
            for m,base in [('B','A'),('C','A'),('D','A'),('D','C'),('H1','H0'),('H2','H0'),('H2','H1')]:
                if m not in wide or base not in wide: continue
                r=(wide[m]/wide[base]).dropna().groupby(level='split').agg(['mean','count'])
                ratios.append(r.reset_index().assign(model=m,base=base,metric=metric,subset=name))
    pd.concat(sums).to_csv(DEST/'summary.csv',index=False)
    env = pd.concat(envs)
    for m in ['A','C','H0','H1']:
        cols=['subset','split','dataset','horizon_s','environment_id']
        base=env[env.model==m].set_index(cols).ade
        env['ade_minus_'+m]=env.ade-env.set_index(cols).index.map(base)
    env.to_csv(DEST/'environments.csv',index=False)
    ratios = pd.concat(ratios)
    ratios.to_csv(DEST/'ratios.csv',index=False)
    audit = catalog.groupby(['split','dataset']).window_id.size().rename('catalog').to_frame()
    audit['eligible']=meta.groupby(['split','dataset']).window_id.size()
    audit['head_eligible']=meta.groupby(['split','dataset']).head_speed.count()
    audit['history_min']=meta.groupby(['split','dataset']).history.min()
    audit['history_max']=meta.groupby(['split','dataset']).history.max()
    audit.to_csv(DEST/'eligibility.csv')
    print(audit)
    print(ratios.query("subset=='all' and metric=='ade'").to_string(index=False))


if __name__=='__main__':
    run()
