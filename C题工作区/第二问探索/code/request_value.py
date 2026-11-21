from pathlib import Path
import sys,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from engine import FixedPolicy
from end_guard import safe_replay
from update_calibration_experiment import hierarchical,paired_ci
import s0_audit_replay as s0
CAND=['early','fixed','late'];SCHED=[(.2,.6),(.3,.6),(.4,.6)]

def features(t,x,good,i,scale):
 a=max(0,i-6);full=i-a==6 and bool(good[a:i].all())
 a=a if full else i-2
 assert good[a:i].all()
 v=np.diff(x[a:i],axis=0)/np.diff(t[a:i])[:,None]/scale
 speed=np.linalg.norm(v,axis=1);last=speed[-1]
 acc=np.linalg.norm(np.diff(v,axis=0),axis=1)/np.diff(t[a+1:i]) if len(v)>1 else np.array([0.])
 cosine=float(np.dot(v[-1],v[-2])/(np.linalg.norm(v[-1])*np.linalg.norm(v[-2])+1e-12)) if len(v)>1 else 0.
 return [last,float(speed.mean()),float(speed.std()),float(acc.mean()),cosine,float(t[i-1]-t[a]),float(not full)]

def weights(frame):
 keys=['dataset','environment_id','episode_uuid','side'];w=np.ones(len(frame))/frame.dataset.nunique()
 for k in range(1,len(keys)):
  counts=frame[keys[:k+1]].drop_duplicates().groupby(keys[:k]).size()
  w/=np.array([counts.loc[tuple(v) if k>1 else v[0]] for v in frame[keys[:k]].itertuples(index=False,name=None)])
 w/=frame.groupby(keys).window_id.transform('size').to_numpy();return w/w.sum()

def main():
 out=BASE/'results/request_value';out.mkdir(exist_ok=True)
 windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'")
 scales=json.loads((OLD/'results/S0/fold_scales.json').read_text());cache={};rows=[];h=[]
 for w in windows.itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1),z[w.side+'__reference_candidate'])
  t,x,good=cache[key];i,j=w.start_index,w.end_index_exclusive
  meta={k:getattr(w,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']}
  f=features(t,x,good,i,1.);h.append(dict(**meta,**{f'f{k}':v for k,v in enumerate(f)},last_feature_index=i-1,prediction_start_index=i))
  for method,schedule in zip(CAND,SCHED):
   r=safe_replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(schedule),delay=.1,budget=2,window_end=float(t[i]+1));e=np.array([p['error'] for p in r['points']])
   rows.append(dict(**meta,method=method,raw_ade=r['metrics']['ade'],raw_fde=r['metrics']['fde'],exceed05=float(np.mean(e>.05*scales[w.environment_id][w.dataset])),requests=r['metrics']['requests'],waiting_s=r['metrics']['waiting_s']))
 raw=pd.DataFrame(rows);hist=pd.DataFrame(h);raw.to_csv(out/'candidate_windows.csv.gz',index=False);hist.to_csv(out/'history_features.csv',index=False)
 pivot=raw.pivot(index='window_id',columns='method',values='raw_ade');selected=[];predictions=[];sources=sorted(hist.dataset.unique())
 for held in sorted(hist.environment_id.unique()):
  fit=hist[hist.environment_id!=held].copy();ev=hist[hist.environment_id==held].copy();wt=weights(fit)
  def matrix(frame):
   xx=frame[[f'f{k}' for k in range(7)]].to_numpy().copy();ss=np.array([scales[held][s] for s in frame.dataset]);xx[:,:4]/=ss[:,None]
   return np.column_stack([xx,*[(frame.dataset==src).astype(float).to_numpy() for src in sources]])
  x=matrix(fit);xe=matrix(ev);mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1.
  x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);xe=np.column_stack([np.ones(len(xe)),(xe-mu)/sd]);pen=np.eye(x.shape[1])*.1;pen[0,0]=0
  yy=pivot.loc[fit.window_id];y=np.column_stack([(yy.fixed-yy[c]).to_numpy() for c in ['early','late']])/np.array([scales[held][s] for s in fit.dataset])[:,None]
  beta=np.linalg.solve(x.T@(wt[:,None]*x)+pen,x.T@(wt[:,None]*y));pr=xe@beta
  simple=np.zeros_like(pr)
  for src in sources:
   mask=(fit.dataset==src).to_numpy();simple[(ev.dataset==src).to_numpy()]=np.average(y[mask],axis=0,weights=wt[mask])
  for method,pp in [('history_ridge',pr),('source_only',simple)]:
   for pos,r in enumerate(ev.itertuples()):
    gains=[pp[pos,0],0.,pp[pos,1]];choice=CAND[int(np.argmax(gains))]
    predictions.append(dict(window_id=r.window_id,held=held,method=method,early_gain=gains[0],late_gain=gains[2],choice=choice))
    selected.append(dict(window_id=r.window_id,method=method,choice=choice))
 for r in hist.itertuples():
  selected.extend([dict(window_id=r.window_id,method='fixed',choice='fixed'),dict(window_id=r.window_id,method='oracle_diagnostic',choice=pivot.loc[r.window_id,CAND].idxmin())])
 sel=pd.DataFrame(selected);a=sel.merge(raw.rename(columns={'method':'choice'}),on=['window_id','choice'],validate='many_to_one')
 for col,rawcol in [('nade','raw_ade'),('nfde','raw_fde')]:a[col]=[v/scales[e][s] for v,e,s in zip(a[rawcol],a.environment_id,a.dataset)]
 metrics=['nade','nfde','exceed05','requests','waiting_s'];a.to_csv(out/'selected_windows.csv.gz',index=False);pd.DataFrame(predictions).to_csv(out/'oof_predictions.csv',index=False)
 summary=hierarchical(a,['method'],metrics);summary.to_csv(out/'summary.csv',index=False)
 env=a.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False)
 pd.DataFrame([paired_ci(env,'fixed',m) for m in ['source_only','history_ridge','oracle_diagnostic']]).to_csv(out/'paired_ci.csv',index=False);a.groupby(['method','choice']).size().to_csv(out/'choices.csv')
 audit=dict(passed=True,train_windows=len(hist),replays=len(raw),global_environments=hist.environment_id.nunique(),future_features=0,development_reads=0,test_reads=0,oracle_policy_legal=False,ridge=.1)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
