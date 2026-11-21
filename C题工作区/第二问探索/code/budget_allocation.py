from pathlib import Path
import sys,json,math
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from engine import FixedPolicy
from end_guard import safe_replay
from update_calibration_experiment import quantiles,hierarchical,paired_ci
import s0_audit_replay as s0

def agebin(age):return min(30,max(1,int(np.ceil(age*30))))
class RadiusPolicy:
 def __init__(self,prior,post,scale,gates=(0.,0.)):self.prior=prior;self.post=post;self.scale=scale;self.gates=gates
 def request(self,v):return v.relative+1e-12>=self.gates[v.requests] and (self.post if v.successful_returns else self.prior)[agebin(v.age)]>.05*self.scale

def main():
 out=BASE/'results/budget_allocation';out.mkdir(exist_ok=True)
 windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'")
 p=pd.read_csv(OLD/'results/S3/residual_points.csv.gz');op=p[p.method=='open'];post=p[(p.method=='secant')&p.after]
 scales=json.loads((OLD/'results/S0/fold_scales.json').read_text());rows=[];cache={};support=[];action_rows=[]
 for held,ww in windows.groupby('environment_id'):
  fit=post[post.environment_id!=held];prior=quantiles(op[op.environment_id!=held]);maps={k:{} for k in ['open_radius','post_pooled','post_env_mean']}
  for (src,age),r in prior.items():
   b=fit[(fit.dataset==src)&(fit.age_bin==age)]
   pooled=float(b.error.quantile(.9,interpolation='higher')) if len(b)>=100 else r
   ee=[float(g.error.quantile(.9,interpolation='higher')) for _,g in b.groupby('environment_id') if len(g)>=20]
   maps['open_radius'][(src,age)]=r;maps['post_pooled'][(src,age)]=pooled;maps['post_env_mean'][(src,age)]=float(np.mean(ee)) if len(ee)>=2 else pooled
   support.append(dict(held=held,dataset=src,age_bin=age,points=len(b),eligible_envs=len(ee)))
  for w in ww.itertuples():
   key=(w.episode_uuid,w.side)
   if key not in cache:
    with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
   t,x=cache[key];i,j=w.start_index,w.end_index_exclusive;scale=scales[held][w.dataset];pr={a:prior[(w.dataset,a)] for a in range(1,31)}
   for method in ['fixed','unrestricted','second_half','quarter_half']:
    mp=maps['post_env_mean'];po={a:mp[(w.dataset,a)] for a in range(1,31)}
    policy=FixedPolicy() if method=='fixed' else RadiusPolicy(pr,po,scale,dict(unrestricted=(0.,0.),second_half=(0.,.5),quarter_half=(.25,.5))[method])
    r=safe_replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],policy,delay=.1,budget=2,window_end=float(t[i]+1))
    errors=np.array([v['error'] for v in r['points']]);rad=np.array([(po if v['returns'] else pr)[agebin(v['age'])] for v in r['points']])
    actions=r['actions'];action_rows.extend(dict(window_id=w.window_id,method=method,relative=a['time']-t[i],event=a['event'],request_id=a['request_id']) for a in actions);assert sum(a['event']=='request' for a in actions)==sum(a['event']=='return' for a in actions)
    rows.append(dict(window_id=w.window_id,dataset=w.dataset,environment_id=held,episode_uuid=w.episode_uuid,side=w.side,method=method,nade=r['metrics']['ade']/scale,nfde=r['metrics']['fde']/scale,exceed05=float(np.mean(errors>.05*scale)),requests=r['metrics']['requests'],waiting_s=r['metrics']['waiting_s'],coverage=float(np.mean(errors<=rad)),width=float(np.mean(2*rad/scale))))
 pd.DataFrame(action_rows).to_csv(out/'actions.csv.gz',index=False)
 m=pd.DataFrame(rows);metrics=['nade','nfde','exceed05','requests','waiting_s','coverage','width'];m.to_csv(out/'windows.csv.gz',index=False)
 summary=hierarchical(m,['method'],metrics);summary.to_csv(out/'summary.csv',index=False)
 e=m.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();e.to_csv(out/'environments.csv',index=False);e.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False)
 pd.DataFrame([paired_ci(e,'fixed',c) for c in ['unrestricted','second_half','quarter_half']]).to_csv(out/'paired_ci.csv',index=False);pd.DataFrame([paired_ci(e,'unrestricted',c) for c in ['second_half','quarter_half']]).to_csv(out/'intervention_ci.csv',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False)
 audit=dict(passed=True,replays=len(m),held_environments=windows.environment_id.nunique(),train_windows=len(windows),development_reads=0,test_reads=0,threshold_search=False,natural_recovery_calibration_used=False,unreturned_requests=0)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()

