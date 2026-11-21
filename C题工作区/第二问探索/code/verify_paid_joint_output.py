import json
import numpy as np,pandas as pd
from distance_aligned_return import BASE
from candidate_replay_integration import DistanceState
from candidate_replay_engine import replay,FixedPolicy
from paid_joint_output import PaidRange
import s0_audit_replay as s0

def main():
 out=BASE/'results/paid_joint_output';out.mkdir(exist_ok=True)
 cal=pd.read_csv(BASE/'results/distance_nested_calibration/inner_residuals.csv.gz').query("model=='distance_return'");support=pd.read_csv(BASE/'results/distance_nested_calibration/support.csv').query("model=='distance_return'");tables={}
 for outer,g in support.groupby('outer'):
  c=cal[cal.outer==outer];assert outer not in set(c.environment_id);table={}
  for r in g.itertuples():
   local=c[(c.dataset==r.dataset)&(c.after==r.after)&(c.age_bin==r.age_bin)]
   if len(local)<100:local=c[(c.dataset==r.dataset)&(c.after==r.after)]
   table[f'{r.dataset}|{int(r.after)}|{r.age_bin}']=dict(radius=float(local.error.quantile(.9,interpolation='higher')),points=len(local),environments=local.environment_id.nunique(),age_fallback=bool(r.age_fallback))
  tables[outer]=table
 (out/'fold_calibration_tables.json').write_text(json.dumps(tables,indent=2),encoding='utf8')
 expected=pd.read_csv(BASE/'results/distance_nested_calibration/outer_points.csv.gz').query("model=='distance_return' and calibration=='pooled'").set_index(['window_id','index']);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");co=pd.read_csv(BASE/'results/distance_aligned_return/coefficients.csv').set_index('held');cache={};rows=[];maxdiff=0.;rejections=0
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;coeff=co.loc[r.environment_id];ranges=PaidRange(tables[r.environment_id]);outputs=[]
  class JointState(DistanceState):
   def predict(self,now):
    position=super().predict(now);outputs.append(ranges.output(position,now=float(now),last_sample_time=self.last_time,returns=self.returns,source=r.dataset));return position
  def factory(*args):return JointState(*args,coefficients=[(coeff.first_a,coeff.first_b),(coeff.second_a,coeff.second_b)])
  run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[j-1]),state_factory=factory)
  for k,(v,p) in enumerate(zip(outputs,run['points'])):
   e=expected.loc[(r.window_id,k)];assert v['status']=='empirical_range' and not v['guaranteed'];assert v['age_bin']==e.age_bin and bool(p['returns']>0)==bool(e.after);assert v['position']==p['prediction'];maxdiff=max(maxdiff,abs(v['radius']-e.radius));rows.append(dict(window_id=r.window_id,index=k,**v))
 assert maxdiff<1e-12 and len(rows)==len(expected)
 unit=PaidRange(next(iter(tables.values())))
 for kwargs in [dict(now=-1,last_sample_time=0,returns=0),dict(now=float('nan'),last_sample_time=0,returns=0),dict(now=1,last_sample_time=0,returns=-1)]:
  try:unit.output([0,0,0],source='EgoDemo',**kwargs)
  except ValueError:rejections+=1
  else:raise AssertionError('invalid input accepted')
 try:PaidRange({},protocol='natural_continuous')
 except ValueError:rejections+=1
 else:raise AssertionError('protocol mismatch accepted')
 assert PaidRange({}).output([0,0,0],now=0,last_sample_time=0,returns=0,source='unknown')['radius'] is None
 pd.DataFrame(rows).to_csv(out/'joint_points.csv.gz',index=False);audit=dict(passed=True,windows=len(w),points=len(rows),radius_max_difference=maxdiff,invalid_rejections=rejections,missing_evidence_returns_none=True,guarantee_claims=0,development_reads=0,test_reads=0,formal_model_changed=False,scope='causal joint output with externally prepared fold evidence; not deployment calibration or formal reliability guarantee');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
