import json
import numpy as np,pandas as pd
from request_value import BASE,OLD
from candidate_replay_engine import replay
from online_second_purchase import SecondPurchasePolicy
import s0_audit_replay as s0

def main():
 out=BASE/'results/second_purchase_online';out.mkdir(exist_ok=True);prior=BASE/'results/second_request_value';fits={f['held']:f for f in json.loads((prior/'fits.json').read_text())};picks=pd.read_csv(prior/'picks.csv').set_index('window_id');saved=pd.read_csv(prior/'windows.csv.gz').query("method=='adaptive_buy'").set_index('window_id');features=pd.read_csv(BASE/'results/causal_decay_selector/visible_features_and_labels.csv.gz').query('stage==0').set_index('window_id');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sources=sorted(w.dataset.unique());cache={};rows=[];maximum=0.;feature_diff=0.;mutation_checks=0
 for number,r in enumerate(w.itertuples()):
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;tt=t[i:j]
  def execute(reference):
   policy=SecondPurchasePolicy(initial_velocity=(x[i-1]-x[i-2])/(t[i-1]-t[i-2]),initial_time=t[i-1],fit=fits[r.environment_id],scale=sc[r.environment_id][r.dataset],source=r.dataset,sources=sources);run=replay(tt,reference,t[i-2:i],x[i-2:i],policy,delay=.1,budget=2,window_end=float(tt[-1]));return policy,run
  policy,run=execute(x[i:j]);assert policy.second_decided;assert policy.buy==(picks.loc[r.window_id,'choice']=='fixed2');assert abs(policy.gain-picks.loc[r.window_id,'predicted_gain'])<1e-12;feature_diff=max(feature_diff,float(np.max(abs(policy.features-features.loc[r.window_id,[f'f{k}' for k in range(5)]].to_numpy(dtype=float)))));e=np.array([p['error'] for p in run['points']])/sc[r.environment_id][r.dataset];maximum=max(maximum,abs(e.mean()-saved.loc[r.window_id,'nade']),abs(e[-1]-saved.loc[r.window_id,'nfde']));assert run['metrics']['requests']==saved.loc[r.window_id,'requests'];rows.append(dict(window_id=r.window_id,buy=policy.buy,predicted_gain=policy.gain,nade=e.mean(),requests=run['metrics']['requests']))
  if number%95==0:
   first=[a for a in run['actions'] if a['event']=='request'][0];index=int(np.argmin(abs(tt-first['sample_time'])));mutated=x[i:j].copy();mutated[index+1:]+=np.random.default_rng(number).normal(0,1000,mutated[index+1:].shape);other,changed=execute(mutated);assert other.buy==policy.buy and other.gain==policy.gain;assert [a for a in run['actions'] if a['event']=='request']==[a for a in changed['actions'] if a['event']=='request'];mutation_checks+=1
 assert maximum<1e-12 and feature_diff<1e-10
 pd.DataFrame(rows).to_csv(out/'windows.csv',index=False);audit=dict(passed=True,windows=len(w),decisions_verified=len(w),metric_max_difference=maximum,visible_feature_max_difference=feature_diff,hidden_future_mutation_checks=mutation_checks,formal_model_changed=False,development_reads=0,test_reads=0,scope='actual public-view policy replay for qualified windows; invalid first return not validated');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
