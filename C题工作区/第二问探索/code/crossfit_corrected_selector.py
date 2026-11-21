import json
import numpy as np,pandas as pd
from request_value import BASE,OLD,weights
from distance_aligned_return import make_stage,predictions
from causal_decay_selector import visible_features
from candidate_replay_engine import replay,FixedPolicy
from update_calibration_experiment import hierarchical,paired_ci
from online_corrected_selector import OnlineCorrectedSelector
import s0_audit_replay as s0

def main():
 out=BASE/'results/crossfit_corrected_selector';out.mkdir(exist_ok=True);(out/'events').mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());folds={f['held']:f for f in json.loads((BASE/'results/distance_aligned_return/folds.json').read_text())};cache={};data={};sources=sorted(w.dataset.unique())
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;tt=t[i:j];run=replay(tt,x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(tt[-1]));ret=[v for v in run['actions'] if v['event']=='return'];data[r.window_id]=dict(t=tt,x=x[i:j],ht=t[i-2:i],hist=x[i-2:i],samples=[int(np.argmin(abs(tt-v['sample_time']))) for v in ret],arrivals=[int(np.argmin(abs(tt-v['processed_at']))) for v in ret])
 pairs={frozenset(f['excluded']):f for f in json.loads((BASE/'results/distance_nested_calibration/pair_fits.json').read_text())};rows=[];fits=[];decisions=[]
 def make_events(frame,c1,c2):
  events=[]
  for r in frame.itertuples():
   d=data[r.window_id];v=(d['hist'][-1]-d['hist'][-2])/(d['ht'][-1]-d['ht'][-2]);last=d['hist'][-1];last_t=d['ht'][-1]
   for stage,c in [(0,c1),(1,c2)]:
    q=d['samples'][stage];gap=d['t'][q]-last_t;sec=(d['x'][q]-last)/gap;feat=visible_features(v,sec,gap,stage);newv=c[0]*sec+c[1]*v*np.exp(-gap/.2);lo=d['arrivals'][stage];hi=d['arrivals'][stage+1] if stage==0 else len(d['t']);age=d['t'][lo:hi]-d['t'][q];loss=[]
    for tau in [.1,.2,.4]:loss.append(float(np.linalg.norm(d['x'][q]-tau*np.expm1(-age/tau)[:,None]*newv-d['x'][lo:hi],axis=1).mean()))
    meta={k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']};events.append(dict(**meta,stage=stage,**{f'f{k}':v for k,v in enumerate(feat)},**{f'loss{k}':v for k,v in enumerate(loss)},fraction=(hi-lo)/len(d['t'])));v=newv;last=d['x'][q];last_t=d['t'][q]
  return pd.DataFrame(events)
 for held in sorted(w.environment_id.unique()):
  frames=[];provenance=[]
  for inner in sorted(set(w.environment_id)-{held}):
   f=pairs[frozenset([held,inner])];assert not {held,inner}&set(f['fit_environments']);z=make_events(w[w.environment_id==inner],np.array(f['first']),np.array(f['second']));frames.append(z);provenance.append(dict(target=inner,excluded=f['excluded'],fit_environments=f['fit_environments']))
  train=pd.concat(frames,ignore_index=True);train.to_csv(out/'events'/f'{held}.csv.gz',index=False);assert held not in set(train.environment_id)
  def matrix(z):
   x=z[[f'f{k}' for k in range(6)]].to_numpy().copy();x[:,:3]/=np.array([sc[held][s] for s in z.dataset])[:,None];return np.column_stack([x,*[(z.dataset==s).astype(float).to_numpy() for s in sources]])
  x=matrix(train);wt=weights(train)*train.fraction.to_numpy();wt/=wt.sum();mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);y=train[['loss0','loss1','loss2']].to_numpy();y=(y-y[:,1,None])/np.array([sc[held][s] for s in train.dataset])[:,None];pen=np.eye(x.shape[1])*.1;pen[0,0]=0;coef=np.linalg.solve(x.T@(wt[:,None]*x)+pen,x.T@(wt[:,None]*y));f=folds[held];fit=dict(held=held,mu=mu.tolist(),sd=sd.tolist(),coefficient=coef.tolist(),first=[f['first_a'],f['first_b']],second=[f['second_a'],f['second_b']],fit_environments=sorted(train.environment_id.unique()),event_provenance=provenance);fits.append(fit)
  for r in w[w.environment_id==held].itertuples():
   d=data[r.window_id];instances=[]
   def factory(*args):
    state=OnlineCorrectedSelector(*args,fit=fit,scale=sc[held][r.dataset],source=r.dataset,sources=sources);instances.append(state);return state
   run=replay(d['t'],d['x'],d['ht'],d['hist'],FixedPolicy(),delay=.1,budget=2,window_end=float(d['t'][-1]),state_factory=factory);e=np.array([p['error'] for p in run['points']])/sc[held][r.dataset];rows.append(dict(**{k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},method='crossfit_corrected',nade=e.mean(),nfde=e[-1],exceed05=float((e>.05).mean()),requests=run['metrics']['requests'],waiting_s=run['metrics']['waiting_s']))
   for v in instances[-1].decisions:decisions.append(dict(window_id=r.window_id,stage=v['stage'],tau=v['tau']))
 result=pd.DataFrame(rows);result.to_csv(out/'windows.csv.gz',index=False);pd.DataFrame(decisions).to_csv(out/'choices.csv',index=False);(out/'fits.json').write_text(json.dumps(fits,indent=2),encoding='utf8');old=pd.read_csv(BASE/'results/corrected_velocity_decay_selector/windows.csv.gz');baseline=pd.read_csv(BASE/'results/causal_decay_selector/windows.csv.gz');combined=pd.concat([result,old,baseline]);metrics=['nade','nfde','exceed05','requests','waiting_s'];summary=hierarchical(combined,['method'],metrics);summary.to_csv(out/'summary.csv',index=False);env=combined.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False);audit=dict(comparison_baseline=paired_ci(env,'baseline','crossfit_corrected'),comparison_in_sample=paired_ci(env,'corrected_selector','crossfit_corrected'),windows=len(w),training_event_provenance_groups=240,formal_model_changed=False,development_reads=0,test_reads=0,scope='cross-fitted base-generated selector training events, 14-to-15 base-fit transfer, actual online evaluation');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
