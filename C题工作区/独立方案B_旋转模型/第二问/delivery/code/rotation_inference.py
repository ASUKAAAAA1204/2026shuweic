"""Frozen rotation inference; no training or filesystem side effects."""
import numpy as np
from scipy.spatial.transform import Rotation
HORIZONS=(.2,.6,1.)

VECTORS=[slice(0,3)]+[slice(i,i+3) for i in range(6,69,3)]+[slice(i,i+3) for i in (70,73,77,80,84,87)]
def frame(x):
 v=np.array([x[s] for s in VECTORS]);norm=np.linalg.norm(v,axis=1);primary=v[0] if norm[0]>1e-8 else v[np.argmax(norm)];scale=max(float(norm.max()),1e-12)
 if np.linalg.norm(primary)<1e-8:return np.eye(3),False,0.
 a=primary/np.linalg.norm(primary);perp=v-(v@a)[:,None]*a;j=int(np.argmax(np.linalg.norm(perp,axis=1)));ratio=float(np.linalg.norm(perp[j])/scale)
 if ratio<1e-3:return np.eye(3),False,ratio
 b=perp[j]/np.linalg.norm(perp[j]);c=np.cross(a,b);return np.stack([a,b,c],axis=1),True,ratio
def rotate(x,R):
 y=x.copy()
 for s in VECTORS:y[s]=x[s]@R
 return y

def rot(q):
 q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
 return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))

def features_at(z,side,anchor_time,ages):
 def channel(s):
  t=z[s+'__t_s'];ix=np.flatnonzero(t<=anchor_time);return t[ix],z[s+'__position'][ix].mean(1),ix
 t,x,ix=channel(side);a=len(t)-1
 if a<4 or anchor_time-t[-1]>.05:raise ValueError('insufficient current history')
 h=np.flatnonzero(t>=t[-1]-.2-1e-8);ht=t[h]-t[-1];hx=x[h]-x[-1];v=(hx[-1]-hx[-2])/(ht[-1]-ht[-2]);vv=np.diff(hx,axis=0)/np.diff(ht)[:,None];speed=np.linalg.norm(vv,axis=1);turn=1-np.dot(vv[-1],vv[0])/max(np.linalg.norm(vv[-1])*np.linalg.norm(vv[0]),1e-12)
 ii=np.flatnonzero(t>=t[-1]-2.-1e-8);p=z[side+'__position'][ix];good=np.isfinite(p).all((1,2))&z[side+'__quality_known'][ix]&~z[side+'__bad'][ix];ok=len(ii)>=30 and t[-1]-t[ii[0]]>=1.9 and good[ii].all() and np.max(np.diff(t[ii]))<=3*np.median(np.diff(t))
 tt=t[ii]-t[-1] if ok else ht;xx=x[ii]-x[-1] if ok else hx;hist=np.stack([np.interp(np.linspace(-2,0,21),tt,xx[:,j]) for j in range(3)],1);extra=[]
 other='right_hand' if side=='left_hand' else 'left_hand';ot,op,oi=channel(other);og=z[other+'__finite'][oi]&z[other+'__quality_known'][oi]&~z[other+'__bad'][oi]
 for span in (.2,.6,2.):
  targets=np.array([anchor_time-span,anchor_time]);jj=np.searchsorted(ot,targets,side='right')-1;available=min(jj)>=0
  if available:
   kk=np.arange(jj[0],jj[1]+1);available=len(kk)>1 and og[kk].all() and np.isfinite(op[kk]).all() and max(targets-ot[jj])<=.05 and np.max(np.diff(ot[kk]))<=3*np.median(np.diff(ot))
  extra.extend(np.r_[op[jj[1]]-op[jj[0]],op[jj[1]]-x[-1],1.] if available else np.zeros(7))
 raw=np.r_[v,np.median(speed),np.std(speed),turn,hist.ravel(),float(ok),extra];R=frame(raw)[0];X=np.r_[rotate(raw,R),R.ravel()]
 headt,headp,_=channel('head');targets=ht+anchor_time;jj=np.searchsorted(headt,targets,side='right')-1;headok=min(jj)>=1
 if headok:headok=max(targets-headt[jj])<=.05 and np.isfinite(headp[jj]).all()
 step=lambda tau:tau*(-np.expm1(-np.asarray(ages)/tau))
 if headok:
  hh=headp[jj]-headp[jj[-1]];hv=(hh[-1]-hh[-2])/(ht[-1]-ht[-2]);rv=((hx-hh)[-1]-(hx-hh)[-2])/(ht[-1]-ht[-2]);base=step(1.)[:,None]*hv+step(.2)[:,None]*rv
 else:base=step(.2)[:,None]*v
 return X,R,base

def rotation_features(t,q,t0):
 keep=t<=t0;t=t[keep];q=q[keep];a=len(t)-1;out=[]
 for span in (.2,.6,2.):
  j=int(np.searchsorted(t,t0-span,side='right')-1);ok=j>=0 and t0-t[a]<=.05
  if ok:ok=bool(t0-span-t[j]<=.05 and np.isfinite(q[[j,a]]).all() and max(abs(np.linalg.norm(q[[j,a]],axis=1)-1))<=1e-3)
  if ok:
   M=rot(q[a])@rot(q[j]).T;v=Rotation.from_matrix(M).as_rotvec()/max(t[a]-t[j],1e-8)
  else:v=np.zeros(3)
  out.extend(np.r_[v,float(ok)])
 return np.array(out)


def predict_model(parameters,z,side,t0,ages,source):
 x,R,base=features_at(z,side,t0,ages);s=rotation_features(z['head__t_s'],z['head__quaternion'][:,0],t0)
 for j in (0,4,8):s[j:j+3]=s[j:j+3]@R
 x=np.r_[x,s];pred=[]
 for seed in (20261120,20261121,20261122):
  model={k:parameters[f'{seed}_{k}'] for k in ('mu','sd','W','phase','feature_mean','coef','scales')};scale=model['scales'][('EgoDemo','EgoPro','EgoStandard').index(source)]
  feature=x.copy();mask=np.ones(112,bool);mask[[5,69,76,83,90]]=False;mask[91:]=False;feature[mask]/=scale;feature=(feature-model['mu'])/model['sd']
  phi=np.sqrt(2)*np.cos(feature@model['W']/2+model['phase'])-model['feature_mean'];local=(np.r_[phi,1.]@model['coef']).reshape(3,3)*scale;world=local@R.T
  pred.append(base+np.stack([np.interp(ages,[0,*HORIZONS],np.r_[0,world[:,j]]) for j in range(3)],axis=1))
 return np.mean(pred,axis=0)
