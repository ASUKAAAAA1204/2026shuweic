"""Train-only paired attribution, fixed Q1 and fixed .3/.6 schedule."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from engine import replay,FixedPolicy,NeverPolicy
from update_calibration_experiment import hierarchical,paired_ci
import s0_audit_replay as s0

def main():
 out=BASE/'results';windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");scales=json.loads((OLD/'results/S0/fold_scales.json').read_text());frozen=hashlib.sha256((OLD/'frozen_model.json').read_bytes()).hexdigest();cache={};rows=[];wm=[]
 for w in windows.itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=w.start_index,w.end_index_exclusive;scale=scales[w.environment_id][w.dataset];meta={k:getattr(w,k) for k in ['window_id','episode_uuid','dataset','environment_id','side']};runs={}
  for name,schedule in [('none',()),('first',(.3,)),('second',(.6,)),('both',(.3,.6))]:
   r=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(schedule),delay=.1,window_end=float(t[i]+1));runs[name]=r;wm.append({**meta,'method':name,'nade':r['metrics']['ade']/scale,'requests':r['metrics']['requests']})
  returns=[a for a in runs['both']['actions'] if a['event']=='return'];lastsample=t[i-1];gaps={}
  for n,a in enumerate(returns,1):gaps[n]=a['sample_time']-lastsample;lastsample=a['sample_time']
  for k,p in enumerate(runs['both']['points']):
   n=p['returns'];sample=p['last_time'];ix=int(np.searchsorted(t,sample));ix=min(ix,len(t)-1);inst=(x[ix]-x[ix-1])/(t[ix]-t[ix-1]);prev=i-1 if n==1 else int(np.searchsorted(t,returns[0]['sample_time'])) if n==2 else i-2
   sec=(x[ix]-x[prev])/(t[ix]-t[prev]) if ix>prev else inst
   # Reference derivative is post-hoc diagnostic only, never a policy input.
   norm=np.linalg.norm(inst)*np.linalg.norm(sec);cos=float(np.dot(inst,sec)/norm) if norm>1e-12 else np.nan
   rows.append({**meta,'relative':p['relative'],'age':p['age'],'returns':n,'region':['before','after_first','after_second'][n],'sample_gap':gaps.get(n,float(t[i-1]-t[i-2])),'age_bin':min(30,max(1,int(np.ceil(p['age']*30)))),'nade':p['error']/scale,'none_error':runs['none']['points'][k]['error']/scale,'first_error':runs['first']['points'][k]['error']/scale,'second_error':runs['second']['points'][k]['error']/scale,'both_error':p['error']/scale,'secant_instant_cosine':cos,'velocity_mismatch':float(np.linalg.norm(sec-inst)/scale),'reference_speed':float(np.linalg.norm(inst)/scale)})
 p=pd.DataFrame(rows);m=pd.DataFrame(wm);p.to_csv(out/'train_paired_points.csv.gz',index=False);m.to_csv(out/'train_paired_windows.csv',index=False)
 hierarchical(m,['method'],['nade','requests']).to_csv(out/'schedule_summary.csv',index=False)
 for factor in ['region','age_bin']:
  a=p.groupby([factor,'window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[['nade','none_error','first_error','second_error','both_error','sample_gap','velocity_mismatch','secant_instant_cosine']].mean()
  hierarchical(a,[factor],['nade','none_error','first_error','second_error','both_error','sample_gap','velocity_mismatch','secant_instant_cosine']).to_csv(out/f'{factor}_summary.csv',index=False)
 # Return-local paired benefits: identical window and time, first vs no query; both vs first.
 events=[]
 for w in windows.itertuples():
  a=p[p.window_id==w.window_id]
  for region,ref in [('after_first','none_error'),('after_second','first_error')]:
   b=a[a.region==region]
   if len(b):events.append({**{k:getattr(w,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},'region':region,'reference_error':b[ref].mean(),'with_return_error':b.both_error.mean(),'benefit':(b[ref]-b.both_error).mean(),'fraction_points_helped':float((b.both_error<b[ref]).mean())})
 ev=pd.DataFrame(events);ev.to_csv(out/'return_benefit_windows.csv',index=False);hierarchical(ev,['region'],['reference_error','with_return_error','benefit','fraction_points_helped']).to_csv(out/'return_benefit_summary.csv',index=False)
 env=m.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[['nade','requests']].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[['nade','requests']].mean().groupby(['method','dataset','environment_id'],as_index=False)[['nade','requests']].mean();pd.DataFrame([paired_ci(env,a,b) for a,b in [('none','first'),('first','both'),('none','second'),('second','both')]]).to_csv(out/'paired_intervals.csv',index=False)
 assert frozen==hashlib.sha256((OLD/'frozen_model.json').read_bytes()).hexdigest()
 audit=dict(train_windows=len(windows),global_environments=windows.environment_id.nunique(),replays=len(m),point_rows=len(p),test_reads=0,development_reads=0,formal_model_unchanged=True,formal_model_sha256=frozen,diagnostic_derivatives='posthoc reference derivatives; not usable hidden features; no policy fitting')
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(pd.read_csv(out/'schedule_summary.csv').to_string(index=False));print(pd.read_csv(out/'return_benefit_summary.csv').to_string(index=False))
if __name__=='__main__':main()
