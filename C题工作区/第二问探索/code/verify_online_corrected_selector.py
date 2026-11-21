import json
import numpy as np,pandas as pd
from request_value import BASE,OLD
from update_calibration_experiment import paired_ci
from candidate_replay_engine import replay,FixedPolicy
from online_corrected_selector import OnlineCorrectedSelector
import s0_audit_replay as s0

def main():
 out=BASE/'results/online_corrected_selector';out.mkdir(exist_ok=True);prior=BASE/'results/corrected_velocity_decay_selector';fits={f['held']:f for f in json.loads((prior/'fits.json').read_text())};choices=pd.read_csv(prior/'choices.csv').set_index(['window_id','stage']);saved=pd.read_csv(prior/'windows.csv.gz').query("method=='corrected_selector'").set_index('window_id');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sources=sorted(w.dataset.unique());feature_tables={held:pd.read_csv(prior/'events'/f'{held}.csv.gz').set_index(['window_id','stage']) for held in fits};cache={};rows=[];decision_count=0;maxdiff=0.;fdiff=0.;mutation_checks=0
 for number,r in enumerate(w.itertuples()):
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;tt=t[i:j];instances=[]
  def factory(*args):
   state=OnlineCorrectedSelector(*args,fit=fits[r.environment_id],scale=sc[r.environment_id][r.dataset],source=r.dataset,sources=sources);instances.append(state);return state
  def run(reference):return replay(tt,reference,t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(tt[-1]),state_factory=factory)
  original=run(x[i:j]);decisions=instances[-1].decisions
  for d in decisions:
   idx=(r.window_id,d['stage']);assert d['tau']==choices.loc[idx,'tau'];expected_features=feature_tables[r.environment_id].loc[idx,[f'f{k}' for k in range(6)]].to_numpy(dtype=float);fdiff=max(fdiff,float(np.max(abs(np.array(d['features'])-expected_features))));decision_count+=1
  e=np.array([p['error'] for p in original['points']])/sc[r.environment_id][r.dataset];maxdiff=max(maxdiff,abs(e.mean()-saved.loc[r.window_id,'nade']),abs(e[-1]-saved.loc[r.window_id,'nfde']));rows.append(dict(window_id=r.window_id,nade=e.mean(),nfde=e[-1],exceed05=float((e>.05).mean())))
  if number%95==0:
   ret=[v for v in original['actions'] if v['event']=='return'];reference=x[i:j].copy();firstsample=int(np.argmin(abs(tt-ret[0]['sample_time'])));reference[firstsample+1:]+=np.random.default_rng(number).normal(0,1000,reference[firstsample+1:].shape);mutated=run(reference);stop=int(np.argmin(abs(tt-ret[1]['processed_at'])));assert np.array_equal(np.array([p['prediction'] for p in original['points']])[:stop],np.array([p['prediction'] for p in mutated['points']])[:stop]);assert decisions[0]==instances[-1].decisions[0];mutation_checks+=1
 assert maxdiff<1e-12 and fdiff<1e-10 and decision_count==3054
 pd.DataFrame(rows).to_csv(out/'windows.csv',index=False)
 # Direct paired comparison: same saved global-environment evaluation units.
 candidate=pd.read_csv(prior/'environments.csv').query("method=='corrected_selector'");distance=pd.read_csv(BASE/'results/distance_aligned_return/environments.csv').query("method=='distance_return'");direct=paired_ci(pd.concat([candidate,distance]),'distance_return','corrected_selector');audit=dict(passed=True,windows=len(w),online_choices_verified=decision_count,visible_feature_max_difference=fdiff,metric_max_difference=maxdiff,hidden_future_mutation_windows=mutation_checks,first_return_decision_unchanged=True,pre_second_return_predictions_unchanged=True,direct_comparison_to_distance=direct,development_reads=0,test_reads=0,formal_model_changed=False);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
