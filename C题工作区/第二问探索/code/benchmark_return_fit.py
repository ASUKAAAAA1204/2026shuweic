import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import time,json,sys,platform
import numpy as np,pandas as pd
from distance_aligned_return import fit_coeff,BASE,OLD
import s0_audit_replay as s0
from end_guard import safe_replay
from engine import FixedPolicy

def main():
 out=BASE/'results/return_fit_timing';out.mkdir(exist_ok=True);start=time.perf_counter()
 frame=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());saved=pd.read_csv(BASE/'results/distance_aligned_return/coefficients.csv').set_index('held');cache={};data={}
 for w in frame.itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=w.start_index,w.end_index_exclusive;tt=t[i:j];r=safe_replay(tt,x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[i]+1));ret=[a for a in r['actions'] if a['event']=='return'];data[w.window_id]=dict(t=tt,x=x[i:j],ht=t[i-2:i],hist=x[i-2:i],samples=[int(np.argmin(abs(tt-a['sample_time']))) for a in ret],arrivals=[int(np.argmin(abs(tt-a['processed_at']))) for a in ret])
 preparation=time.perf_counter()-start;rows=[];maxdiff=0.
 for repetition in range(3):
  envs=sorted(frame.environment_id.unique())
  if repetition%2:envs.reverse()
  for held in envs:
   fit=frame[frame.environment_id!=held];t0=time.perf_counter();c1=fit_coeff(data,fit,sc[held],0,np.array([1.,0.]));t1=time.perf_counter();c2=fit_coeff(data,fit,sc[held],1,c1);t2=time.perf_counter();expected=saved.loc[held,['first_a','first_b','second_a','second_b']].to_numpy();maxdiff=max(maxdiff,float(np.max(abs(np.r_[c1,c2]-expected))));rows.append(dict(repetition=repetition,held=held,train_windows=len(fit),first_fit_s=t1-t0,second_fit_s=t2-t1,total_fit_s=t2-t0))
 assert maxdiff<1e-10
 z=pd.DataFrame(rows);z.to_csv(out/'fold_timings.csv',index=False);audit=dict(passed=True,folds=16,repetitions=3,preparation_s=preparation,fit_total_s=float(z.total_fit_s.sum()),median_two_stage_fit_s=float(z.total_fit_s.median()),p95_two_stage_fit_s=float(z.total_fit_s.quantile(.95)),coefficient_max_difference=maxdiff,development_reads=0,test_reads=0,formal_model_changed=False,scope='training window prepared-array construction and both 20-step IRLS stages; no final all-training deployment fit, no double-nested calibration fit',python=sys.version,platform=platform.platform());(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
