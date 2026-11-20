from common import *
import time,collections

def logistic_fit(X,y,lam=1.):
 mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1;Z=np.c_[np.ones(len(X)),(X-mu)/sd];b=np.zeros(Z.shape[1]);reg=np.eye(len(b))*lam;reg[0,0]=0
 for _ in range(35):
  p=1/(1+np.exp(-np.clip(Z@b,-30,30)));w=p*(1-p);step=np.linalg.solve(Z.T@(w[:,None]*Z)+reg+np.eye(len(b))*1e-8,Z.T@(p-y)+reg@b);b-=step
  if np.linalg.norm(step)<1e-8:break
 return dict(mu=mu.tolist(),sd=sd.tolist(),coef=b.tolist(),lambda_=lam)
def logistic_predict(m,X):return 1/(1+np.exp(-np.clip(np.c_[np.ones(len(X)),(X-np.array(m['mu']))/np.array(m['sd'])]@m['coef'],-30,30)))
def auc(y,p):
 pos=p[y==1];neg=p[y==0]
 if not len(pos) or not len(neg):return None
 vals=np.unique(p);rank={float(v):i for i,v in enumerate(vals)};counts=np.bincount([rank[float(v)] for v in neg],minlength=len(vals));before=np.cumsum(counts)-counts
 return float(sum(before[rank[float(v)]]+.5*counts[rank[float(v)]] for v in pos)/(len(pos)*len(neg)))
def replay_window(w,z,protocol,radii,method,delay,frac,return_trace=False):
 side=w['side'];t=z[side+'__t_s'];x=state(z,side);i=int(w['start_index']);j=int(w['end_index_exclusive']);anchor=i-1
 start=time.perf_counter();obs=[(float(t[anchor-1]),x[anchor-1].copy()),(float(t[anchor]),x[anchor].copy())];pending=None;requests=0;lastrequest=-1e30;fixed=set();errors=[];covered=[];radius=[];wait=0;actions=[];trace=[]
 for k in range(i,j):
  now=float(t[k]);rel=now-float(t[i]);typ='hold' if method=='hold' else 'constant_velocity'
  if pending is not None and now+1e-12>=pending[0]:
   obs.append((pending[1],pending[2]));pending=None
  def predict():
   age=now-obs[-1][0];v=(obs[-1][1]-obs[-2][1])/(obs[-1][0]-obs[-2][0]) if typ!='hold' and obs[-1][0]>obs[-2][0] else np.zeros(3);b=min(30,max(1,int(np.ceil(age*30))));return obs[-1][1]+v*age,radii[f'{w["dataset"]}|{typ}|{b}']
  pred,rad=predict();request=False
  if method=='fixed_review':
   due=[v for v in [.3,.6] if rel+1e-12>=v and v not in fixed]
   if due:fixed.add(due[0]);request=True
  if method=='adaptive_review':request=rad>=protocol['train_scale'][w['dataset']]*frac and now-lastrequest>=.2-1e-12
  if request and pending is None and requests<2 and now+delay<=float(t[i])+1+1e-12:
   pending=(now+delay,now,x[k].copy());requests+=1;lastrequest=now;wait+=delay;actions.append({'request_s':rel,'return_s':rel+delay})
   if delay==0:obs.append((now,x[k].copy()));pending=None;pred,rad=predict()
  trace.append(dict(time=now,prediction=pred.tolist(),requests=requests,pending_return=pending[0] if pending else None));err=float(np.linalg.norm(pred-x[k]));errors.append(err);covered.append(err<=rad);radius.append(rad)
 result={**{k:w[k] for k in ['window_id','episode_uuid','dataset','environment_id','split','side']},'method':method,'delay_s':delay,'tolerance_fraction':frac,'ade':float(np.mean(errors)),'fde':errors[-1],'p95':float(np.quantile(errors,.95)),'coverage90':float(np.mean(covered)),'mean_radius90':float(np.mean(radius)),'within_tolerance':float(np.mean(np.array(errors)<=protocol['train_scale'][w['dataset']]*frac)),'requests':requests,'cost':requests,'waiting_s':wait,'runtime_s':time.perf_counter()-start,'actions':json.dumps(actions,separators=(',',':'))}
 return (result,trace) if return_trace else result

def run(out):
 out=Path(out);d=out/'data';q=out/'q12';q.mkdir(exist_ok=True);protocol=load(d/'protocol.json');windows=readcsv(d/'windows.csv');records=readcsv(d/'records.csv');cache={};cal=collections.defaultdict(list);openrows=[];reliability=[]
 for r in records:
  if r['split']=='test':continue
  with np.load(DATA/'records'/(r['episode_uuid']+'.npz')) as z:cache[r['episode_uuid']]={k:z[k].copy() for k in z.files}
 for w in windows:
  if w['split']=='test':continue
  z=cache[w['episode_uuid']];side=w['side'];t=z[side+'__t_s'];x=state(z,side);i=int(w['start_index']);j=int(w['end_index_exclusive']);anchor=i-1;vel=(x[anchor]-x[anchor-1])/(t[anchor]-t[anchor-1])
  for model in ['hold','constant_velocity']:
   for h in [.2,.6,1.]:
    ii=np.arange(i,j)[t[i:j]<=t[i]+h+1e-12];age=t[ii]-t[anchor];pred=x[anchor]+(age[:,None]*vel if model!='hold' else 0);err=np.linalg.norm(pred-x[ii],axis=1)
    openrows.append({**{k:w[k] for k in ['window_id','episode_uuid','dataset','environment_id','split','side']},'model':model,'horizon_s':h,'ade':float(err.mean()),'fde':float(err[-1]),'p95':float(np.quantile(err,.95)),'samples':len(ii)})
    if w['split']=='train' and h==1.:
     for a,e in zip(age,err):cal[(w['dataset'],model,min(30,max(1,int(np.ceil(a*30)))))].append(float(e))
 radii={}
 for src in protocol['train_scale']:
  for model in ['hold','constant_velocity']:
   for b in range(1,31):
    values=cal.get((src,model,b),[])
    if not values:values=[v for (s,m,_),vv in cal.items() if s==src and m==model for v in vv]
    radii[f'{src}|{model}|{b}']=float(np.quantile(values,.9,method='higher')) if values else None
 dump(q/'prediction_radius_train.json',radii)
 # Quality forecasting: historical observable features only, labels from next 0.2 seconds.
 feats=[];labels=[];ids=[]
 for r in records:
  if r['split']=='test':continue
  z=cache[r['episode_uuid']]
  for side in ['left_hand','right_hand']:
   t=z[side+'__t_s'];x=state(z,side);bad=z[side+'__bad'];known=z[side+'__quality_known'];stride=6
   for i in range(6,len(t)-7,stride):
    lo=int(np.searchsorted(t,t[i]-.2));hi=int(np.searchsorted(t,t[i]+.2,side='right'))
    if hi<=i+1 or not known[lo:hi].all() or not np.isfinite(x[lo:i+1]).all():continue
    speed=np.linalg.norm(np.diff(x[lo:i+1],axis=0),axis=1)/np.diff(t[lo:i+1]);feature=[float(np.median(speed)),float(np.std(speed)),float(bad[lo:i+1].mean()),float(bad[i])]
    feats.append(feature);labels.append(int(bad[i+1:hi].any()));ids.append({**{k:r[k] for k in ['episode_uuid','dataset','environment_id','split']},'side':side,'index':i})
 X=np.asarray(feats);y=np.asarray(labels);tr=np.array([x['split']=='train' for x in ids]);model=logistic_fit(X[tr],y[tr]);model['constant_probability']=float(y[tr].mean());dump(q/'reliability_model.json',model)
 for method in ['constant','logistic']:
  p=np.full(len(y),model['constant_probability']) if method=='constant' else logistic_predict(model,X)
  for split in ['train','validation']:
   for src in ['all',*protocol['train_scale']]:
    ix=np.array([r['split']==split and (src=='all' or r['dataset']==src) for r in ids]);yy=y[ix];pp=p[ix]
    if not len(yy):continue
    reliability.append(dict(method=method,split=split,dataset=src,n=len(yy),positives=int(yy.sum()),brier=float(np.mean((pp-yy)**2)),logloss=float(np.mean(-yy*np.log(np.clip(pp,1e-12,1))-(1-yy)*np.log(np.clip(1-pp,1e-12,1)))),auc=auc(yy,pp),mean_probability=float(pp.mean()),observed_rate=float(yy.mean())))
 writecsv(q/'reliability_metrics.csv',reliability)
 pp=logistic_predict(model,X);quality_rows=[dict(**meta,label=int(label),probability=float(prob),constant_probability=model['constant_probability']) for meta,label,prob in zip(ids,y,pp)]
 writecsv(q/'reliability_predictions.csv',quality_rows)
 calibration=[]
 for split in ['train','validation']:
  rr=[r for r in quality_rows if r['split']==split]
  for lo in [0,.01,.05,.1,.2,.5]:
   hi={0:.01,.01:.05,.05:.1,.1:.2,.2:.5,.5:1.0000001}[lo];v=[r for r in rr if lo<=r['probability']<hi]
   if v:calibration.append(dict(split=split,bin_lo=lo,bin_hi=min(1.,hi),n=len(v),mean_probability=float(np.mean([r['probability'] for r in v])),observed_rate=float(np.mean([r['label'] for r in v]))))
 writecsv(q/'reliability_calibration.csv',calibration)
 q2=[]
 for w in windows:
  if w['split']!='validation':continue
  z=cache[w['episode_uuid']];side=w['side'];t=z[side+'__t_s'];x=state(z,side);i=int(w['start_index']);j=int(w['end_index_exclusive']);anchor=i-1
  for method in ['hold','constant_velocity','fixed_review','adaptive_review']:
   for delay in protocol['delays_s']:
    for frac in protocol['tolerance_fractions']:
     q2.append(replay_window(w,z,protocol,radii,method,delay,frac))
 # Independent validation open-loop interval coverage.
 for row in openrows:
  if row['split']!='validation':continue
  w=next(w for w in windows if w['window_id']==row['window_id']);z=cache[w['episode_uuid']];side=w['side'];t=z[side+'__t_s'];x=state(z,side);i=int(w['start_index']);j=int(w['end_index_exclusive']);a=i-1;sel=np.arange(i,j)[t[i:j]<=t[i]+row['horizon_s']+1e-12];age=t[sel]-t[a];v=(x[a]-x[a-1])/(t[a]-t[a-1]);pred=x[a]+(age[:,None]*v if row['model']!='hold' else 0);err=np.linalg.norm(pred-x[sel],axis=1);rad=np.array([radii[f'{w["dataset"]}|{row["model"]}|{min(30,max(1,int(np.ceil(b*30))))}'] for b in age]);row['coverage90']=float(np.mean(err<=rad));row['mean_radius90']=float(rad.mean())
 writecsv(q/'open_loop_windows.csv',openrows);writecsv(q/'replay_windows.csv',q2)
 summaries=[]
 for metricrows,keys in [(openrows,['split','dataset','model','horizon_s']),(q2,['dataset','method','delay_s','tolerance_fraction'])]:
  buckets=collections.defaultdict(list)
  for r in metricrows:buckets[tuple(r[k] for k in keys)].append(r)
  for key,rr in buckets.items():
   # Aggregate windows within record, then records within environment.
   for metric in ['ade','fde','p95','coverage90','mean_radius90','requests','cost','waiting_s','within_tolerance']:
    v=[r for r in rr if metric in r]
    if not v:continue
    rb=collections.defaultdict(list)
    for r in v:rb[(r['episode_uuid'],r['environment_id'])].append(float(r[metric]))
    vals=[float(np.mean(a)) for a in rb.values()];env=[k[1] for k in rb];summaries.append({**dict(zip(keys,key)),'metric':metric,**grouped_boot(vals,env),'windows':len(v),'records':len(rb)})
 writecsv(q/'summary.csv',summaries);dump(q/'checks.json',dict(open_loop_rows=len(openrows),replay_rows=len(q2),quality_examples=len(y),quality_train=int(tr.sum()),test_evaluated=False,request_limit_pass=all(r['requests']<=2 for r in q2),nonfinite_metrics=sum(not np.isfinite(r['ade']) for r in q2),state='21-point centroid'))
 print('q12 complete',len(q2),'replay rows',flush=True)
