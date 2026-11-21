"""Freeze train-selected methods before development; then evaluate one named split."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import pandas as pd
from engine import replay,FixedPolicy,NeverPolicy
from policies import RadiusPolicy,BudgetPolicy
from update_calibration_experiment import quantiles,hierarchical
import s0_audit_replay as s0
BASE=Path(__file__).resolve().parents[1]

def catalog():return pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv')
def freeze():
 path=BASE/'frozen_model.json'
 assert not path.exists(),'Freeze already exists; do not silently replace after development.'
 sel=json.loads((BASE/'results/S4_adaptive/selection.json').read_text());rank=pd.read_csv(BASE/'results/S4_adaptive/full_train_ranking.csv');points=pd.read_csv(BASE/'results/S3/residual_points.csv.gz');op=points[points.method=='open'];post=points[(points.method=='secant')&points.after];sources=sorted(op.dataset.unique())
 prior=quantiles(op);posterior={}
 for key,g in post.groupby(['dataset','age_bin']):
  if len(g)>=100:posterior[f'{key[0]}|{key[1]}']=float(g.error.quantile(.9,interpolation='higher'))
 params={f:float(rank[rank.family==f].iloc[0].parameter) for f in rank.family.unique()};scales=json.loads((BASE/'results/S0/fold_scales.json').read_text())['development']
 # Fit no-request comparator radii on training only, same mask references.
 baselines=[];cache={}
 for w in catalog().query("split=='train'").itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=w.start_index,w.end_index_exclusive;age=t[i:j]-t[i-1];v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2])
  for method in ['hold','cv']:
   pred=np.broadcast_to(x[i-1],(j-i,3)) if method=='hold' else x[i-1]+age[:,None]*v
   error=np.linalg.norm(pred-x[i:j],axis=1)
   for h,e in zip(age,error):baselines.append(dict(dataset=w.dataset,method=method,age_bin=min(30,max(1,int(np.ceil(h*30)))),error=e))
 bb=pd.DataFrame(baselines);radii={m:{f'{k[0]}|{k[1]}':v for k,v in quantiles(bb[bb.method==m]).items()} for m in ['hold','cv']};radii['damped']={f'{k[0]}|{k[1]}':v for k,v in prior.items()}
 frozen=dict(q1_commit='bf380c4',selected=sel['selected'],selected_adaptive_family=sel['adaptive_best_family'],parameters=params,fixed_schedule=[.3,.6],scale=scales,radii=radii,post_radius_exploratory=posterior,calibrator_formal='open_radius',calibrator_exploratory='separate not admitted by train confidence gate',code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'code').glob('*.py')},protocol_sha256=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest(),frozen_utc=pd.Timestamp.now(tz='UTC').isoformat(),test_trajectory_reads_before_freeze=0,development_replays_before_freeze=0)
 path.write_text(json.dumps(frozen,indent=2),encoding='utf8');print('FROZEN',frozen['selected'],hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)

def evaluate(split):
 frozen_path=BASE/'frozen_model.json';f=json.loads(frozen_path.read_text());out=BASE/'results'/f'S5_{split}';assert not (out/'audit.json').exists(),'Evaluation already complete; do not resample or overwrite.'
 if split=='test':assert (BASE/'results/S5_validation/audit.json').exists(),'Development confirmation must finish first.'
 out.mkdir(parents=True,exist_ok=True);tic=time.perf_counter();windows=catalog().query('split==@split');records=pd.read_csv(s0.DATA/'record_index.csv').query('split==@split');cache={};inputs=[]
 for r in records.itertuples():
  path=s0.DATA/'records'/f'{r.episode_uuid}.npz';inputs.append(dict(episode_uuid=r.episode_uuid,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  with np.load(path,allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:cache[(r.episode_uuid,side)]=(z[side+'__t_s'],z[side+'__position'].mean(axis=1),z[side+'__reference_candidate'])
 rows=[];allpoints=[];actions=[];methods=['hold','cv','damped','fixed','threshold','delay_aware','budget']
 for w in windows.itertuples():
  t,x,valid=cache[(w.episode_uuid,w.side)];i,j=w.start_index,w.end_index_exclusive;assert valid[w.history_start:j].all();scale=f['scale'][w.dataset];curve=[f['radii']['damped'][f'{w.dataset}|{b}'] for b in range(1,31)];meta={k:getattr(w,k) for k in ['window_id','episode_uuid','dataset','environment_id','side']}
  for length in [.2,.6,1.]:
   idx=np.arange(i,j)[t[i:j]-t[i]<=length+1e-9]
   for delay in [0,.1,.2]:
    for method in methods:
     if method in ['hold','cv','damped'] and delay!=0:continue
     if method in ['hold','cv','damped']:policy=NeverPolicy()
     elif method=='fixed':policy=FixedPolicy(f['fixed_schedule'])
     elif method=='budget':policy=BudgetPolicy(curve,scale,f['parameters'][method],delay)
     else:policy=RadiusPolicy(curve,scale,f['parameters'][method],delay,method=='delay_aware')
     r=replay(t[idx],x[idx],t[i-2:i],x[i-2:i],policy,delay=delay,motion=method if method in ['hold','cv'] else 'damped',window_end=float(t[i]+length))
     pp=pd.DataFrame(r['points']);error=pp.error.to_numpy();ages=np.clip(np.ceil(pp.age*30).astype(int),1,30);radius=np.array([f['radii'][method if method in ['hold','cv'] else 'damped'][f'{w.dataset}|{b}'] for b in ages]);post=np.array([f['post_radius_exploratory'].get(f'{w.dataset}|{b}',rr) if aft else rr for b,rr,aft in zip(ages,radius,pp.returns>0)])
     row={**meta,'method':method,'length_s':length,'delay_s':delay,**r['metrics'],'nade':r['metrics']['ade']/scale,'nfde':r['metrics']['fde']/scale,'np95':r['metrics']['p95']/scale,'coverage':float(np.mean(error<=radius)),'path_covered':bool(np.all(error<=radius)),'width':float(np.mean(2*radius/scale)),'coverage_separate_exploratory':float(np.mean(error<=post)),'width_separate_exploratory':float(np.mean(2*post/scale))}
     for frac in [.02,.05,.1]:row[f'exceed_{frac}']=float(np.mean(error>frac*scale))
     rows.append(row)
     for k,p in enumerate(r['points']):allpoints.append({**meta,'method':method,'length_s':length,'delay_s':delay,'time':p['time'],'relative':p['relative'],'error':p['error'],'age':p['age'],'radius':float(radius[k]),'radius_separate_exploratory':float(post[k]),'requests':p['requests'],'returns':p['returns'],'pred_x':p['prediction'][0],'pred_y':p['prediction'][1],'pred_z':p['prediction'][2],'ref_x':p['reference'][0],'ref_y':p['reference'][1],'ref_z':p['reference'][2]})
     for a in r['actions']:actions.append({**meta,'method':method,'length_s':length,'delay_s':delay,**a})
 m=pd.DataFrame(rows);m.to_csv(out/'windows.csv',index=False);pd.DataFrame(allpoints).to_csv(out/'points.csv.gz',index=False);pd.DataFrame(actions).to_csv(out/'actions.csv.gz',index=False);pd.DataFrame(inputs).to_csv(out/'input_hashes.csv',index=False)
 keys=['method','length_s','delay_s'];cols=['nade','nfde','np95','exceed_0.02','exceed_0.05','exceed_0.1','requests','cost','waiting_s','no_request_wait_s','coverage','path_covered','width','coverage_separate_exploratory','width_separate_exploratory','strategy_compute_s','step_compute_p95_s']
 hierarchical(m,keys,cols).to_csv(out/'summary.csv',index=False)
 # Retain source and environment summaries so sparse groups remain visible.
 g=m.groupby(keys+['dataset','environment_id','episode_uuid','side'],as_index=False)[cols].mean().groupby(keys+['dataset','environment_id','episode_uuid'],as_index=False)[cols].mean().groupby(keys+['dataset','environment_id'],as_index=False)[cols].mean();g.to_csv(out/'environments.csv',index=False);g.groupby(keys+['dataset'],as_index=False)[cols].mean().to_csv(out/'sources.csv',index=False)
 audit=dict(split=split,records_read=len(inputs),windows=len(windows),replay_rows=len(m),points=len(allpoints),actions=len(actions),budget_violations=int((m.requests>2).sum()),nonfinite=int((~np.isfinite(m.nade)).sum()),frozen_model_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),elapsed_s=time.perf_counter()-tic,model_refitted=False,protocol_sha256=f['protocol_sha256'])
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','validation','test']);a=parser.parse_args()
 if a.mode=='freeze':freeze()
 else:evaluate(a.mode)
