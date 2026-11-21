import json
import numpy as np,pandas as pd
from request_value import BASE,OLD
import s0_audit_replay as s0

METHODS=['baseline','endpoint6','ols6','zero']
SCOPES=['all','before_return']

def independent_replay(t,x,i,j,v0):
 last_t=float(t[i-1]);last=x[i-1].copy();v=v0.copy();pending=None;returns=0;requests=0;used=set();points=[];start=float(t[i]);end=float(t[j-1])
 for k in range(i,j):
  now=float(t[k])
  if pending is not None and now+1e-12>=pending[0]:
   _,sample_t,value=pending;v=(value-last)/(sample_t-last_t);last=value.copy();last_t=sample_t;returns+=1;pending=None
  rel=now-start
  due=[q for q in (.3,.6) if q not in used and rel+1e-12>=q]
  if pending is None and requests<2 and due and now+.1<=end+1e-12:
   used.add(due[0]);requests+=1;pending=(now+.1,now,x[k].copy())
  age=now-last_t;pred=last-.2*np.expm1(-age/.2)*v;points.append((float(np.linalg.norm(pred-x[k])),returns))
 return points,requests

def main():
 out=BASE/'results/initial_history_candidates';windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");scale=json.loads((OLD/'results/S0/fold_scales.json').read_text());saved=pd.read_csv(out/'windows.csv.gz');assert len(windows)==1527 and len(saved)==1527*4*2;assert not saved.duplicated(['method','scope','window_id']).any();cache={};rows=[];support=[]
 for r in windows.itertuples():
  if r.episode_uuid not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[r.episode_uuid]={s:(z[s+'__t_s'],z[s+'__position'].mean(axis=1),z[s+'__reference_candidate']) for s in ['left_hand','right_hand']}
  t,x,good=cache[r.episode_uuid][r.side];i,j=r.start_index,r.end_index_exclusive;base=(x[i-1]-x[i-2])/(t[i-1]-t[i-2]);a=max(0,i-6);has=i-a==6 and good[a:i].all() and (np.diff(t[a:i])<=.1).all();vel={'baseline':base,'zero':np.zeros(3),'endpoint6':base,'ols6':base}
  if has:
   ht=t[a:i]-t[i-1];hx=x[a:i];vel['endpoint6']=(hx[-1]-hx[0])/(ht[-1]-ht[0]);center=ht-ht.mean();vel['ols6']=np.sum(center[:,None]*(hx-hx.mean(axis=0)),axis=0)/np.sum(center**2)
  support.append((r.window_id,bool(has)))
  for method,v in vel.items():
   points,requests=independent_replay(t,x,i,j,v);errors=np.array([p[0] for p in points])/scale[r.environment_id][r.dataset];after=np.array([p[1]>0 for p in points])
   for scope,mask in [('all',np.ones(len(points),bool)),('before_return',~after)]:
    z=errors[mask];rows.append(dict(window_id=r.window_id,dataset=r.dataset,environment_id=r.environment_id,episode_uuid=r.episode_uuid,side=r.side,method=method,scope=scope,nade=z.mean(),nfde=z[-1],exceed05=(z>.05).mean(),requests=requests,history6=float(has)))
 replayed=pd.DataFrame(rows);recreated=replayed.copy();columns=['nade','nfde','exceed05','requests','history6'];groups=['dataset','environment_id','episode_uuid','side']
 for depth in range(len(groups),-1,-1):
  keys=['method','scope']+groups[:depth];recreated=recreated.groupby(keys,as_index=False)[columns].mean() if keys else recreated[columns].mean().to_frame().T
 expected=pd.read_csv(out/'summary.csv').set_index(['method','scope'])[columns].sort_index();actual=recreated.set_index(['method','scope'])[columns].sort_index();summary_diff=float(abs(expected-actual).to_numpy().max());assert summary_diff<1e-12
 replaydiff=float(abs(saved.set_index(['method','scope','window_id'])[columns]-replayed.set_index(['method','scope','window_id'])[columns]).to_numpy().max());assert replaydiff<1e-12
 old=pd.read_csv(BASE/'results/causal_decay_selector/windows.csv.gz').query("method=='baseline'").set_index('window_id');base=replayed.query("method=='baseline' and scope=='all'").set_index('window_id');baseline_diff=float(abs(old.nade-base.nade).max());assert baseline_diff<1e-12
 checks=pd.DataFrame(support,columns=['window_id','history6']);saved_support=pd.read_csv(out/'history_support.csv');assert checks.history6.equals(saved_support.set_index('window_id').loc[checks.window_id].history6.reset_index(drop=True))
 audit=dict(passed=True,replayed_windows=len(windows),replayed_methods=len(METHODS),independent_window_max_difference=replaydiff,independent_summary_max_difference=summary_diff,baseline_prior_max_difference=baseline_diff,six_frame_support_matches=True,validation_reads=0,test_reads=0,scope='independent event-loop replay and hierarchical aggregation; still same source dataset')
 (out/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
