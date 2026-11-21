"""Train-only S3 paired update and calibration experiment. No development replay."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from engine import replay,FixedPolicy,NeverPolicy
import s0_audit_replay as s0
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'results/S3'

def hierarchical(frame,by,metrics):
    levels=[['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]
    g=frame.copy()
    for lev in levels:
        keys=by+lev
        g=g.groupby(keys,as_index=False)[metrics].mean() if keys else g[metrics].mean().to_frame().T
    return g

def quantiles(points):
    result={}
    for src,g in points.groupby('dataset'):
        pooled=float(g.error.quantile(.9,interpolation='higher'))
        for age in range(1,31):
            b=g[g.age_bin==age];result[(src,age)]=float(b.error.quantile(.9,interpolation='higher')) if len(b) else pooled
    return result

def paired_ci(env,a,b,metric='nade',reps=2000):
    aa=env[env.method==a].set_index(['dataset','environment_id'])[metric];bb=env[env.method==b].set_index(['dataset','environment_id'])[metric]
    pair=pd.DataFrame({'a':aa,'b':bb}).reset_index();envs=sorted(pair.environment_id.unique());rng=np.random.default_rng(20261121);values=[]
    for _ in range(reps):
        counts=pd.Series(rng.choice(envs,len(envs),replace=True)).value_counts();w=pair.environment_id.map(counts).fillna(0).to_numpy();g=pair.assign(w=w)
        if (g.groupby('dataset').w.sum()==0).any():continue
        nums=g.assign(aw=g.a*g.w,bw=g.b*g.w).groupby('dataset')[['aw','bw','w']].sum();av=(nums.aw/nums.w).mean();bv=(nums.bw/nums.w).mean()
        values.append(100*(1-bv/av))
    av=pair.groupby('dataset').a.mean().mean();bv=pair.groupby('dataset').b.mean().mean()
    return dict(reference=a,candidate=b,metric=metric,gain_pct=float(100*(1-bv/av)),lo95=float(np.quantile(values,.025)),hi95=float(np.quantile(values,.975)),bootstrap_valid=len(values),global_envs=len(envs))

def main():
    OUT.mkdir(parents=True,exist_ok=True);tic=time.perf_counter();windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'")
    scales=json.loads((BASE/'results/S0/fold_scales.json').read_text());cache={};points=[];metrics=[];actions=[]
    for w in windows.itertuples():
        key=(w.episode_uuid,w.side)
        if key not in cache:
            with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
        t,x=cache[key];i,j=w.start_index,w.end_index_exclusive;meta={k:getattr(w,k) for k in ['window_id','episode_uuid','dataset','environment_id','side']};scale=scales[w.environment_id][w.dataset]
        for method in ['open','secant','decayed']:
            r=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],NeverPolicy() if method=='open' else FixedPolicy(),delay=.1,update='secant' if method=='open' else method,window_end=float(t[i]+1))
            errs=np.array([p['error'] for p in r['points']]);row={**meta,'method':method,**r['metrics'],'nade':r['metrics']['ade']/scale,'nfde':r['metrics']['fde']/scale,'np95':r['metrics']['p95']/scale}
            for frac in [.02,.05,.1]:row[f'exceed_{frac}']=float(np.mean(errs>scale*frac))
            metrics.append(row)
            for p in r['points']:
                points.append({**meta,'method':method,'relative':p['relative'],'age':p['age'],'age_bin':min(30,max(1,int(np.ceil(p['age']*30)))),'error':p['error'],'after':p['returns']>0,'scale':scale})
            for a in r['actions']:actions.append({**meta,'method':method,**a})
    m=pd.DataFrame(metrics);p=pd.DataFrame(points);m.to_csv(OUT/'update_windows.csv',index=False);p.to_csv(OUT/'residual_points.csv.gz',index=False);pd.DataFrame(actions).to_csv(OUT/'actions.csv.gz',index=False)
    cols=['nade','nfde','np95','exceed_0.02','exceed_0.05','exceed_0.1','requests','waiting_s','strategy_compute_s']
    summary=hierarchical(m,['method'],cols);summary.to_csv(OUT/'update_summary.csv',index=False)
    env=m.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[cols].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[cols].mean().groupby(['method','dataset','environment_id'],as_index=False)[cols].mean();env.to_csv(OUT/'update_environments.csv',index=False)
    ci=paired_ci(env,'secant','decayed');src=env.groupby(['method','dataset']).nade.mean().unstack(0);source_gains=100*(1-src.decayed/src.secant)
    selected='decayed' if ci['gain_pct']>=3 and ci['lo95']>0 and source_gains.min()>=-5 else 'secant'
    (OUT/'update_selection.json').write_text(json.dumps(dict(selected=selected,comparison=ci,source_gain_pct=source_gains.to_dict(),development_used=False),indent=2),encoding='utf8')
    # Motion/update have no fitted weights. Every fold calibrates residuals from other global environments only.
    chosen=p[p.method==selected].copy();op=p[p.method=='open'];calrows=[];folds=[];fitted={}
    for held in sorted(windows.environment_id.unique()):
        open_fit=op[op.environment_id!=held];post_fit=chosen[(chosen.environment_id!=held)&chosen.after];ev=chosen[chosen.environment_id==held].copy()
        assert not (open_fit.environment_id==held).any() and not (post_fit.environment_id==held).any()
        prior=quantiles(open_fit);post={}
        for (source,age),g in post_fit.groupby(['dataset','age_bin']):
            if len(g)>=100:post[(source,int(age))]=float(g.error.quantile(.9,interpolation='higher'))
        folds.append(dict(held=held,open_points=len(open_fit),post_points=len(post_fit),evaluation_points=len(ev),post_supported_bins=len(post)))
        fitted[held]={'open':{f'{k[0]}|{k[1]}':v for k,v in prior.items()},'post':{f'{k[0]}|{k[1]}':v for k,v in post.items()}}
        for kind in ['open_radius','separate']:
            e=ev.copy();keys=list(zip(e.dataset,e.age_bin));e['radius']=[post.get(k,prior[k]) if kind=='separate' and aft else prior[k] for k,aft in zip(keys,e.after)]
            e['fallback']=[bool(kind=='separate' and aft and k not in post) for k,aft in zip(keys,e.after)];e['covered']=(e.error<=e.radius).astype(float);e['width']=2*e.radius/e.scale
            diff=e.error-e.radius;e['pinball']=np.maximum(.9*diff,-.1*diff)/e.scale;e['calibrator']=kind
            calrows.append(e)
    cal=pd.concat(calrows,ignore_index=True);cal.to_csv(OUT/'calibration_points.csv.gz',index=False)
    rows=[]
    for region in ['all','before','after']:
        g=cal if region=='all' else cal[cal.after==(region=='after')]
        a=g.groupby(['window_id','episode_uuid','dataset','environment_id','side','calibrator'],as_index=False)[['covered','width','pinball','fallback']].mean();a['region']=region;rows.append(a)
    cw=pd.concat(rows,ignore_index=True);cw.to_csv(OUT/'calibration_windows.csv',index=False)
    cs=hierarchical(cw,['region','calibrator'],['covered','width','pinball','fallback']);cs.to_csv(OUT/'calibration_summary.csv',index=False)
    source=cw.groupby(['region','calibrator','dataset','environment_id','episode_uuid','side'],as_index=False)[['covered','width','pinball','fallback']].mean().groupby(['region','calibrator','dataset','environment_id','episode_uuid'],as_index=False)[['covered','width','pinball','fallback']].mean().groupby(['region','calibrator','dataset','environment_id'],as_index=False)[['covered','width','pinball','fallback']].mean();source.to_csv(OUT/'calibration_environments.csv',index=False)
    source.groupby(['region','calibrator','dataset'])[['covered','width','pinball','fallback']].mean().to_csv(OUT/'calibration_sources.csv')
    (OUT/'fold_calibrators.json').write_text(json.dumps(fitted),encoding='utf8');pd.DataFrame(folds).to_csv(OUT/'calibration_folds.csv',index=False)
    after=cs[cs.region=='after'].set_index('calibrator');gain=100*(1-after.loc['separate','pinball']/after.loc['open_radius','pinball']);growth=100*(after.loc['separate','width']/after.loc['open_radius','width']-1)
    cov=after.loc['separate','covered'];mins=source.query("region=='after' and calibrator=='separate'").groupby('dataset').covered.mean().min()
    ce=source.query("region=='after'").rename(columns={'calibrator':'method'});cal_ci=paired_ci(ce,'open_radius','separate',metric='pinball')
    accepted=bool(gain>=5 and growth<=25 and .87<=cov<=.93 and mins>=.85 and cal_ci['lo95']>0)
    (OUT/'calibration_selection.json').write_text(json.dumps(dict(selected='separate' if accepted else 'open_radius',separate_admitted=accepted,pinball_gain_pct=float(gain),width_growth_pct=float(growth),coverage=float(cov),min_source_coverage=float(mins),pinball_interval=cal_ci,development_used=False,notes='candidate may remain exploratory if coverage constraint fails; empirical not guaranteed coverage'),indent=2),encoding='utf8')
    audit=dict(train_windows=len(windows),train_global_envs=windows.environment_id.nunique(),update_replays=len(m),calibration_folds=len(folds),held_environment_in_fit=0,test_trajectory_reads=0,development_replays=0,elapsed_s=time.perf_counter()-tic)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(cs.to_string(index=False));print(json.dumps(ci));print('selected update',selected,'separate admitted',accepted,flush=True)
if __name__=='__main__':main()
