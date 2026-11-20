from common import *
import collections

def design(f,family):return np.r_[f,[float(family==x) for x in ['P','A','L']]]
def fit(X,y,lam):
 mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1;Z=np.c_[np.ones(len(X)),(X-mu)/sd];reg=np.eye(Z.shape[1])*lam;reg[0,0]=0;b=np.linalg.solve(Z.T@Z+reg,Z.T@y)
 return dict(mu=mu.tolist(),sd=sd.tolist(),coef=b.tolist())
def predict(m,X):return np.exp(np.clip(np.c_[np.ones(len(X)),(X-np.array(m['mu']))/np.array(m['sd'])]@m['coef'],-10,10))
def run(out):
 out=Path(out);md=out/'models';md.mkdir(exist_ok=True);segs=readcsv(DATA/'segments_private.csv');rows=[]
 for s in segs:
  if s['eligible']!='1' or s['split']=='test':continue
  for length in [.2,.6,1.]:
   with np.load(DATA/'private'/(s['unit_id']+'_'+str(length)+'.npz')) as p:f=prefix_features(p)
   rows.append(dict(unit_id=s['unit_id'],environment_id=s['environment_id'],split=s['split'],family=s['family_id'],length=length,duration=float(s['duration_s']),features=f))
 model={'families':{},'ridge':{},'quantiles':[.5,.8,.9]};metrics=[];cvrows=[]
 for family in ['P','A','L']:
  vals=[r['duration'] for r in rows if r['split']=='train' and r['length']==.2 and r['family']==family]
  model['families'][family]={str(q):float(np.quantile(vals,q)) for q in [.5,.8,.9]}
 for length in [.2,.6,1.]:
  rr=[r for r in rows if r['split']=='train' and r['length']==length];X=np.array([design(r['features'],r['family']) for r in rr]);y=np.log([r['duration'] for r in rr]);env=np.array([r['environment_id'] for r in rr]);best=None
  for lam in [.1,1.,10.,100.]:
   pp=np.zeros(len(y))
   for e in sorted(set(env)):
    keep=env!=e;m=fit(X[keep],y[keep],lam);pp[~keep]=predict(m,X[~keep])
   mae=float(np.mean(abs(pp-np.exp(y))));cvrows.append(dict(length=length,lambda_=lam,environment_oof_mae=mae))
   if best is None or mae<best[0]:best=(mae,lam,pp.copy())
  m=fit(X,y,best[1]);res=np.exp(y)-best[2];m.update(lambda_=best[1],oof_mae=best[0],residual_quantiles={str(q):float(np.quantile(res,q)) for q in [.5,.8,.9]});model['ridge'][str(length)]=m
  for r in [r for r in rows if r['length']==length]:
   pred=float(predict(m,design(r['features'],r['family'])[None])[0])
   for typ in ['family','ridge']:
    p=model['families'][r['family']]['0.5'] if typ=='family' else max(.001,pred+m['residual_quantiles']['0.5'])
    metrics.append({k:r[k] for k in ['unit_id','environment_id','split','family','length','duration']}|dict(model=typ,prediction=p,absolute_error=abs(p-r['duration']),interval80_lo=max(.001,pred+float(np.quantile(res,.1))) if typ=='ridge' else None,interval80_hi=max(.001,pred+float(np.quantile(res,.9))) if typ=='ridge' else None))
 dump(md/'duration.json',model);writecsv(out/'q3_duration_predictions.csv',metrics);writecsv(out/'q3_duration_train_cv.csv',cvrows)
 print('duration model fitted train only',flush=True)
