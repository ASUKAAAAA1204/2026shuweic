from pathlib import Path
import sys,json,gzip
import numpy as np
import pandas as pd
from rotation_inference import predict_model
from replay_engine import MotionState,FixedPolicy,replay
from rotation_state import factory
BASE=Path(__file__).resolve().parents[1];Q2=BASE.parent;SPACE=Q2.parent;WORK=SPACE.parent

def aggregate(d,keys,metrics):
 for level in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:
  cols=list(dict.fromkeys(keys+level));d=d.groupby(cols,as_index=False)[metrics].mean() if cols else d[metrics].mean().to_frame().T
 return d

def radius(model,age,speed,scale):
 a=np.clip(np.ceil(age*30),1,30)/30;v=np.log1p(speed/scale);x=np.array([a,a*a,v,a*v]);return max(0.,float(np.r_[1.,(x-model['mu'])/model['sd']]@model['coefficient']))*scale

def main():
 out=BASE/'results';out.mkdir(exist_ok=True);cfg=json.loads((BASE/'model/model.json').read_text());fold=json.loads((BASE/'model/fold_scales.json').read_text());prior=json.loads((BASE/'model/prior_models.json').read_text());op=json.loads((BASE/'model/oof_prior_models.json').read_text());ret=json.loads((BASE/'model/oof_return.json').read_text());windows=pd.read_csv(BASE/'data/windows.csv');windows=windows[windows.split.isin(['train','validation'])];records={};models={};rows=[];logs=[];maximum=0.
 with gzip.open(out/'points.jsonl.gz','wt',encoding='utf8') as stream:
  for row in windows.itertuples():
   uid=row.episode_uuid
   if uid not in records:
    with np.load(WORK/f'处理结果/v1/records/{uid}.npz',allow_pickle=False) as z:records[uid]={k:z[k].copy() for k in z.files}
   z=records[uid];t=z[row.side+'__t_s'];x=z[row.side+'__position'].mean(1);i=row.start_index;j=row.end_index_exclusive;anchor=float(t[i-1]);times=t[i:j];past={}
   for k,v in z.items():
    ct=z.get(k.split('__')[0]+'__t_s');past[k]=v[ct<=anchor].copy() if ct is not None and v.ndim and len(v)==len(ct) else v.copy()
   if row.split=='train':
    path=BASE/f'model/oof_rotation/{row.environment_id}_parameters.npz';scale=fold[row.environment_id][row.dataset];pm=next(v for v in op if v['held']==row.environment_id and v['dataset']==row.dataset);rm=next(v for v in ret if v['held']==row.environment_id and v['dataset']==row.dataset)
   else:path=BASE/'model/rotation_full.npz';scale=cfg['scale'][row.dataset];pm=prior[row.dataset];rm=cfg['return_models'][row.dataset]
   if str(path) not in models:
    with np.load(path,allow_pickle=False) as zz:models[str(path)]={k:zz[k].copy() for k in zz.files}
   initial=predict_model(models[str(path)],past,row.side,anchor,times-anchor,row.dataset)+x[i-1];Parent=factory(times,initial,MotionState);instances=[]
   class Joint(Parent):
    def __init__(self,*args):super().__init__(*args);self.ranges=[];instances.append(self)
    def predict(self,now):
     p=super().predict(now);self.ranges.append(radius(rm if self.returns else pm,now-self.last_time,float(np.linalg.norm(self.velocity)),scale));return p
   run=replay(times,x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),state_factory=Joint);meta={k:getattr(row,k) for k in ['window_id','episode_uuid','dataset','environment_id','split','side']};errs=[];cover=[];width=[];loss=[]
   for p,rad in zip(run['points'],instances[0].ranges,strict=True):
    err=p['error']/scale;rr=rad/scale;errs.append(err);cover.append(float(err<=rr));width.append(2*rr);loss.append(max(.9*(err-rr),-.1*(err-rr)));stream.write(json.dumps(dict(**meta,scale=scale,radius=rad,**p))+'\n')
   logs.extend(dict(**meta,**v) for v in run['actions']);rows.append(dict(**meta,nade=np.mean(errs),nfde=errs[-1],coverage=np.mean(cover),width=np.mean(width),pinball=np.mean(loss),requests=run['metrics']['requests'],waiting_s=run['metrics']['waiting_s']))
 d=pd.DataFrame(rows);d.to_csv(out/'windows.csv',index=False);pd.DataFrame(logs).to_csv(out/'actions.csv.gz',index=False);metrics=['nade','nfde','coverage','width','pinball','requests','waiting_s'];summary=aggregate(d,['split'],metrics);summary.to_csv(out/'summary.csv',index=False)
 for split,folder in [('train','formal_training'),('validation','development')]:
  reference=pd.read_csv(BASE/f'reference/{split}_windows.csv');reference=reference[reference.method=='rotation_switch'];a=d[d.split==split].merge(reference,on='window_id',suffixes=('_new','_old'));maximum=max(maximum,float(abs(a.nade_new-a.nade_old).max()),float(abs(a.nfde_new-a.nfde_old).max()))
 for split,folder in [('train','range_nested_training'),('validation','range_development')]:
  reference=pd.read_csv(BASE/f'reference/{split}_range_summary.csv').iloc[0];actual=summary.set_index('split').loc[split]
  maximum=max(maximum,max(abs(actual[k]-reference[k]) for k in ['coverage','width','pinball']))
 assert len(d)==1739 and maximum<1e-9
 (out/'audit.json').write_text(json.dumps(dict(passed=True,windows=len(d),maximum_regression_difference=maximum,raw_input_replayed=True,pure_inference=True,test_reads=0),indent=2),encoding='utf8');print(summary.to_string(index=False));print('max regression difference',maximum)
if __name__=='__main__':main()
