from pathlib import Path
import sys,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import time,json,platform,hashlib
import numpy as np,pandas as pd
from distance_aligned_return import BASE,OLD,predictions
import s0_audit_replay as s0
from engine import FixedPolicy
from end_guard import safe_replay

def secant(last,new,old,gap,a,b):return (new-last)/gap
def candidate(last,new,old,gap,a,b):return a*((new-last)/gap)+b*old*np.exp(-gap/.2)
def main():
 out=BASE/'results/return_update_timing';out.mkdir(exist_ok=True)
 windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'")
 coeff=pd.read_csv(BASE/'results/distance_aligned_return/coefficients.csv').set_index('held');cache={};events=[];maxdiff=0.
 for w in windows.itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=w.start_index,w.end_index_exclusive;tt=t[i:j];run=safe_replay(tt,x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[i]+1));ret=[v for v in run['actions'] if v['event']=='return'];c=coeff.loc[w.environment_id];last=x[i-1];sample=t[i-1];v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2]);online=last+(-.2*np.expm1(-(tt-sample)/.2))[:,None]*v
  for k,r in enumerate(ret):
   qi=int(np.argmin(abs(tt-r['sample_time'])));ai=int(np.argmin(abs(tt-r['processed_at'])));new=x[i+qi];gap=tt[qi]-sample;a=float(c['first_a' if k==0 else 'second_a']);b=float(c['first_b' if k==0 else 'second_b']);events.append((last.copy(),new.copy(),v.copy(),float(gap),a,b));v=candidate(last,new,v,gap,a,b);last=new;sample=tt[qi];online[ai:]=last+(-.2*np.expm1(-(tt[ai:]-sample)/.2))[:,None]*v
  d=dict(t=tt,x=x[i:j],ht=t[i-2:i],hist=x[i-2:i],samples=[int(np.argmin(abs(tt-r['sample_time']))) for r in ret],arrivals=[int(np.argmin(abs(tt-r['processed_at']))) for r in ret]);expected=predictions(d,[c.first_a,c.first_b],[c.second_a,c.second_b]);maxdiff=max(maxdiff,float(np.max(abs(online-expected))))
 assert maxdiff<1e-12
 # Same actual candidate-state inputs isolate kernel arithmetic; secant ignores old velocity.
 for fn in [secant,candidate]:
  for e in events:fn(*e)
 rows=[];checksum=0.
 for rep in range(40):
  order=[('secant',secant),('distance_return',candidate)]
  if rep%2:order.reverse()
  for method,fn in order:
   start=time.perf_counter_ns()
   for e in events:v=fn(*e)
   elapsed=time.perf_counter_ns()-start;checksum+=float(v.sum());rows.append(dict(repetition=rep,method=method,events=len(events),elapsed_ns=elapsed,ns_per_update=elapsed/len(events)))
 pd.DataFrame(rows).to_csv(out/'batches.csv',index=False)
 stats={}
 for method,g in pd.DataFrame(rows).groupby('method'):
  v=g.ns_per_update.to_numpy()/1000;stats[method]=dict(median_us=float(np.median(v)),p95_batch_mean_us=float(np.quantile(v,.95)),min_us=float(v.min()),max_us=float(v.max()))
 audit=dict(scope='Python NumPy return velocity kernel only; not end-to-end latency or per-event tail',training_windows=len(windows),actual_return_events=len(events),repetitions=40,warmup_passes=1,alternating_order=True,online_reconstruction_max_difference=maxdiff,development_reads=0,test_reads=0,offline_fit_timed=False,formal_model_changed=False,python=sys.version,platform=platform.platform(),thread_environment={n:os.environ[n] for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},checksum=checksum,statistics=stats)
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
