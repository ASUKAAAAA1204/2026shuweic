"""Run the approved six-way training-only comparison (Python, NumPy, pandas)."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[variable]='1'
import hashlib
import json
import platform
from time import perf_counter_ns
import numpy as np
import pandas as pd
from data import OUT, HORIZONS, EPS, average, basic, features, hierarchy, load_windows, turn, Obs

MODELS=('B','A','L','T','K','K0')
COEFS=(-.4,-.2,0.,.2,.4)
SEED=20261120


def scales(meta):
    return meta.groupby('source').endpoint.median().to_dict(),meta.endpoint.median()


def grid_errors(windows):
    rows=[]
    for i,w in enumerate(windows):
        target=w.reference-w.obs.x[-1]
        v=features(w.obs)[0]
        for c in COEFS:
            err=np.linalg.norm(basic(v,w.ages,'L',c)-target,axis=1)
            for h in HORIZONS:
                use=w.ages<=h+EPS
                rows.append((i,c,h,average(w.ages[use],err[use])))
    return pd.DataFrame(rows,columns=['idx','c','horizon','raw'])


def select_c(train,grid):
    # Deterministic, balanced-by-environment-count inner split. L has no fitted
    # coefficients other than the grid choice; fold-specific normalization matters.
    envs=sorted(train.env.unique())
    scores=[]
    folds=[]
    for group in np.array_split(envs,min(4,len(envs))):
        held=train.env.isin(group)
        scale,pool=scales(train.loc[~held])
        current=grid.merge(train.loc[held],left_on='idx',right_index=True)
        current['loss']=current.raw/current.source.map(scale).fillna(pool)
        scores.append(current)
        folds.append(group.tolist())
    grouped=hierarchy(pd.concat(scores),['loss'],['c']).groupby(level='c').loss.mean()
    return min(COEFS,key=lambda c:(grouped[c],abs(c),c)),grouped.to_dict(),folds


class Bank:
    def __init__(self,ids,windows,feature_cache,threshold):
        self.threshold=threshold
        self.ids=[]
        self.times=[]
        self.residual=[]
        fs=[]
        for i in ids:
            v,f,_,_=feature_cache[i]
            if not self.eligible(f):
                continue
            w=windows[i]
            self.ids.append(i)
            fs.append(f)
            self.times.append(w.ages)
            self.residual.append((w.reference-w.obs.x[-1]-basic(v,w.ages))@(v/f[0]))
        self.f=np.asarray(fs).reshape(-1,4)
        self.sd=self.f.std(axis=0) if fs else np.ones(4)
        self.sd[self.sd==0]=np.inf
        self.ids=np.asarray(self.ids,dtype=int)
        self.nbytes=sum(a.nbytes for a in [self.ids,self.f,self.sd,*self.times,*self.residual])

    def eligible(self,f):
        return bool(np.isfinite(f).all() and min(f[:3])>0 and min(f[:3])>=self.threshold)

    def correction(self,f,h,speed_only):
        cols=1 if speed_only else 4
        delta=(self.f[:,:cols]-f[:cols])/self.sd[:cols]
        indices=np.argsort((delta*delta).sum(axis=1),kind='stable')[:5]
        values=np.stack([np.interp(h,self.times[k],self.residual[k],left=np.nan,right=np.nan) for k in indices])
        count=np.isfinite(values).sum(axis=0)
        correction=np.divide(np.nansum(values,axis=0),count,out=np.zeros(len(h)),where=count>=3)
        return correction,self.ids[indices],int((count[1:]<3).sum())


def predict(model,obs,h,bank,c):
    # Only historical positions, requested time offsets and training state enter here.
    info=dict(status='active',neighbors=(),partial=0)
    if model in ('B','A','L'):
        v=(obs.x[-1]-obs.x[-2])/(obs.t[-1]-obs.t[-2])
        pred=basic(v,h,model,c if model=='L' else 0.)
    else:
        v,f,axis,omega=features(obs)
        pred=basic(v,h)
        if not bank.eligible(f):
            info['status']='history_or_speed'
        elif model=='T':
            if np.deg2rad(5)<=f[3]<=np.deg2rad(175) and np.any(axis):
                pred=turn(v,h,axis,omega)
            else:
                info['status']='angle'
        elif len(bank.ids)<5:
            info['status']='small_library'
        else:
            corr,indices,partial=bank.correction(f,h,model=='K0')
            pred+=.5*corr[:,None]*(v/f[0])
            info.update(neighbors=indices,partial=partial)
    return obs.x[-1]+pred,info


def paired_intervals(environments):
    table=environments.groupby(['env','source','model']).nade.mean().unstack(['source','model'])
    rng=np.random.default_rng(SEED)
    n=len(table)
    models=list(MODELS)+['K_vs_K0']
    output=[]
    rejects=0
    arrays=[table[src].reindex(columns=MODELS).to_numpy() for src in sorted(table.columns.levels[0])]
    while sum(len(a) for a in output)<5000:
        picks=rng.integers(n,size=(5000-sum(len(a) for a in output),n))
        valid=np.logical_and.reduce([np.isfinite(a[picks,0]).any(axis=1) for a in arrays])
        rejects+=int((~valid).sum())
        idx=picks[valid]
        if not len(idx):
            continue
        mean=np.mean([np.nanmean(a[idx],axis=1) for a in arrays],axis=0)
        diff=1-mean/mean[:,[0]]
        output.append(np.column_stack([diff,1-mean[:,4]/mean[:,5]]))
    samples=np.vstack(output)
    return {m:np.quantile(samples[:,i],[.025,.975,.00625,.99375]).tolist() for i,m in enumerate(models)},rejects


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    ws=load_windows()
    meta=pd.DataFrame([w.key for w in ws])
    meta['endpoint']=[np.linalg.norm(w.reference[-1]-w.obs.x[-1]) for w in ws]
    assert meta.env.nunique()==16 and meta.record.nunique()==55 and meta.window_id.is_unique
    cache=[features(w.obs) for w in ws]
    grid=grid_errors(ws)
    metric_rows,actions,params,setup=[],[],[],[]
    pred_arrays={m:[None]*len(ws) for m in MODELS}
    no_target_checks=0
    for environment in sorted(meta.env.unique()):
        train=meta.loc[meta.env!=environment]
        held=meta.loc[meta.env==environment]
        c,inner,folds=select_c(train,grid)
        scale,pool=scales(train)
        for src,block in held.groupby('source'):
            start=perf_counter_ns()
            ids=train[train.source==src].sort_values(['record','side','start_s']).index
            threshold=np.quantile([cache[i][1][0] for i in ids],.1) if len(ids) else np.inf
            bank=Bank(ids,ws,cache,threshold)
            build_us=(perf_counter_ns()-start)/1000
            setup.append(dict(env=environment,source=src,library_windows=len(bank.ids),
                              library_numeric_bytes=bank.nbytes,build_us=build_us))
            assert not set(block.index)&set(bank.ids)
            normalizer=scale.get(src,pool)
            assert normalizer>0
            params.append(dict(env=environment,source=src,c=c,speed_q10=float(threshold) if len(ids) else None,
                               scale=normalizer,scale_fallback=src not in scale,inner_scores=inner,inner_groups=folds))
            for i in block.index:
                w=ws[i]
                meta.loc[i,'scale']=normalizer
                v,f,_,_=cache[i]
                direction=v/f[0] if f[0]>0 and f[0]>=threshold else None
                for m in MODELS:
                    p,info=predict(m,w.obs,w.ages,bank,c)
                    # Repeat with independent copies; query contains no future positions.
                    if i==block.index[0]:
                        detached=Obs(w.obs.t.copy(),w.obs.x.copy())
                        replay,_=predict(m,detached,w.ages.copy(),bank,c)
                        np.testing.assert_array_equal(p,replay)
                        no_target_checks+=1
                    times=[]
                    for _ in range(3):
                        start=perf_counter_ns()
                        predict(m,w.obs,w.ages,bank,c)
                        times.append((perf_counter_ns()-start)/1000)
                    pred_arrays[m][i]=p
                    error=p-w.reference
                    dist=np.linalg.norm(error,axis=1)
                    neighbors=meta.loc[list(info['neighbors'])]
                    active=info['status']=='active'
                    actions.append(dict(window_id=w.key['window_id'],model=m,env=environment,source=src,status=info['status'],
                                        partial_fallback_points=info['partial'],neighbors='|'.join(neighbors.window_id),
                                        neighbor_envs=neighbors.env.nunique(),neighbor_records=neighbors.record.nunique(),
                                        latency_us=float(np.median(times))))
                    for h in HORIZONS:
                        use=w.ages<=h+EPS
                        ages=w.ages[use]
                        ade=average(ages,dist[use])
                        along=lateral=np.nan
                        if direction is not None:
                            projection=error[use]@direction
                            along=average(ages,projection)
                            lateral=average(ages,np.linalg.norm(error[use]-projection[:,None]*direction,axis=1))
                        metric_rows.append(dict(**w.key,model=m,horizon=h,ade=ade,fde=dist[use][-1],nade=ade/normalizer,
                                                along=along,lateral=lateral,active=active))
    metrics=pd.DataFrame(metric_rows)
    actions=pd.DataFrame(actions)
    keys=['model','horizon','source','env','record','side']
    side=metrics.groupby(keys)[['nade','ade','fde','along','lateral']].mean()
    env=side.groupby(level=keys[:-1]).mean().groupby(level=keys[:-2]).mean().reset_index()
    source=env.groupby(keys[:3])[['nade','ade','fde','along','lateral']].mean().reset_index()
    score=source.groupby(['model','horizon']).nade.mean().groupby(level='model').mean()
    intervals,rejected=paired_intervals(env)
    summary=[]
    for m in MODELS:
        a=actions[actions.model==m]
        gain=1-score[m]/score['B']
        lo,hi,cl,ch=intervals[m]
        summary.append(dict(model=m,score=score[m],gain=gain,lo95=lo,hi95=hi,lo9875=cl,hi9875=ch,
                            advance=m in ('A','L','T','K') and gain>=.03 and cl>0,
                            active_windows=int((a.status=='active').sum()),median_us=a.latency_us.median(),p95_us=a.latency_us.quantile(.95)))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    meta.to_csv(OUT/'windows.csv',index=False)
    metrics.to_csv(OUT/'metrics.csv',index=False)
    env.to_csv(OUT/'environments.csv',index=False)
    source.to_csv(OUT/'sources.csv',index=False)
    actions.to_csv(OUT/'inference.csv',index=False)
    pd.DataFrame(setup).to_csv(OUT/'library.csv',index=False)
    (OUT/'folds.json').write_text(json.dumps(params,indent=2,allow_nan=False),encoding='utf8')
    np.savez_compressed(OUT/'predictions.npz',offsets=np.r_[0,np.cumsum([len(w.ages) for w in ws])],
                        t=np.concatenate([w.ages for w in ws]),ref=np.vstack([w.reference for w in ws]),
                        **{m:np.vstack(pred_arrays[m]) for m in MODELS})
    audit=dict(windows=len(ws),windows_by_source=meta.source.value_counts().to_dict(),
               paired_bootstrap_accept=5000,rejected=rejected,seed=SEED,K_vs_K0_gain=1-score.K/score.K0,
               K_vs_K0_intervals=intervals['K_vs_K0'],detached_input_checks=no_target_checks,
               test_validation_trajectory_reads=0,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
               script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT.parents[1]/'code/q1_six').glob('*.py')})
    (OUT/'checks.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__':
    run()
