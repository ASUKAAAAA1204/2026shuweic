"""Reproduce the adopted Q2 model only; no candidate selection or parameter search."""
from pathlib import Path
import argparse,gzip,hashlib,json,time
import numpy as np
import pandas as pd
from continuous_range_state import fixed_joint_replay

BASE=Path(__file__).resolve().parents[1]
def sha(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aggregate(frame,keys,metrics):
 g=frame
 for level in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:
  cols=list(dict.fromkeys(keys+level))
  g=g.groupby(cols,as_index=False)[metrics].mean() if cols else g[metrics].mean().to_frame().T
 return g
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--windows',type=Path,required=True);args=ap.parse_args()
 out=BASE/'results';out.mkdir(exist_ok=True);started=time.time()
 cfg=json.loads((BASE/'model/model.json').read_text(encoding='utf8'))
 folds=json.loads((BASE/'model/fold_scales.json').read_text(encoding='utf8'))
 pri=json.loads((BASE/'model/oof_prior.json').read_text(encoding='utf8'));ret=json.loads((BASE/'model/oof_return.json').read_text(encoding='utf8'))
 hashes={str(p.relative_to(BASE)):sha(p) for folder in ['code','model'] for p in sorted((BASE/folder).glob('*')) if p.is_file()}
 windows=pd.read_csv(args.windows);rows=[];actions=[];inputs={};cache={};npoints=0;checks=0
 with gzip.open(out/'joint_points.jsonl.gz','wt',encoding='utf8') as stream:
  for index,r in enumerate(windows.itertuples()):
   if r.episode_uuid not in cache:
    path=args.data_root/'records'/f'{r.episode_uuid}.npz';inputs[path.name]=sha(path)
    with np.load(path,allow_pickle=False) as z:
     cache[r.episode_uuid]={s:(z[s+'__t_s'],z[s+'__position'].mean(axis=1)) for s in ['left_hand','right_hand']}
   times,pos=cache[r.episode_uuid][r.side];i=int(r.start_index);j=int(r.end_index_exclusive)
   assert i>=2 and times[i-1]<times[i]
   if r.split=='train':
    pm={v['dataset']:v for v in pri if v['held']==r.environment_id};rm={v['dataset']:v for v in ret if v['held']==r.environment_id};scale=folds[r.environment_id][r.dataset]
    assert r.environment_id not in pm[r.dataset]['fit_environments'] and r.environment_id not in rm[r.dataset]['fit_environments']
   else:pm=cfg['prior_models'];rm=cfg['return_models'];scale=cfg['scale'][r.dataset]
   run=fixed_joint_replay(times[i:j],pos[i:j],times[i-2:i],pos[i-2:i],source=r.dataset,scale=scale,prior_models=pm,return_models=rm)
   meta={k:getattr(r,k) for k in ['window_id','episode_uuid','dataset','environment_id','split','side']}
   cover=[];width=[];loss=[];exceed=[]
   for p,q in zip(run['points'],run['ranges'],strict=True):
    assert q['time']==p['time'] and p['last_time']<=p['time'] and q['guaranteed'] is False
    err=p['error']/scale;radius=q['radius']/scale;delta=err-radius
    cover.append(float(delta<=0));width.append(2*radius);loss.append(max(.9*delta,-.1*delta));exceed.append(float(err>.05))
    stream.write(json.dumps(dict(**meta,**p,range=q,scale=scale),ensure_ascii=False)+'\n');npoints+=1
   for a in run['actions']:
    if a['event']=='return':assert a['processed_at']+1e-12>=a['time'] and a['processed_at']<=times[j-1]+1e-12
    actions.append(dict(**meta,**a))
   assert run['metrics']['requests']<=2 and len(cover)==j-i
   m=run['metrics'];rows.append(dict(**meta,nade=m['ade']/scale,nfde=m['fde']/scale,coverage=np.mean(cover),width=np.mean(width),pinball=np.mean(loss),exceed_005=np.mean(exceed),**m));checks+=1
   if (index+1)%250==0:print('Completed',index+1,'/',len(windows),flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(out/'window_metrics.csv',index=False);pd.DataFrame(actions).to_csv(out/'actions.csv.gz',index=False)
 metrics=['nade','nfde','coverage','width','pinball','exceed_005','requests','waiting_s','strategy_compute_s']
 summary=aggregate(frame,['split'],metrics);summary.to_csv(out/'summary.csv',index=False)
 aggregate(frame,['split','dataset'],metrics).to_csv(out/'sources.csv',index=False)
 aggregate(frame,['split','dataset','environment_id'],metrics).to_csv(out/'environments.csv',index=False)
 # Reproduction regression: these are already viewed validation values, not selection targets.
 val=summary.set_index('split').loc['validation'];expected={'nade':.09164589544559261,'coverage':.8805208789471123,'width':.3450493656560285,'pinball':.015070895980127631}
 regression={k:float(abs(val[k]-v)) for k,v in expected.items()};assert max(regression.values())<1e-10,regression
 assert all(sha(BASE/k)==v for k,v in hashes.items())
 audit=dict(windows=len(frame),points=npoints,records=len(inputs),split_counts=frame.groupby('split').size().to_dict(),checked_windows=checks,validation_regression_absolute_error=regression,model_and_code_unchanged=True,input_windows_sha256=sha(args.windows),record_sha256=inputs,artifact_sha256=hashes,elapsed_s=time.time()-started,training='environment-held-out existing frozen fits; fresh raw replay',evaluation='validation and test previously viewed; descriptive reproduction',guaranteed=False,protocol=cfg['protocol'])
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
 print(summary.to_string(index=False));print(json.dumps({k:v for k,v in audit.items() if k not in ['record_sha256','artifact_sha256']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
