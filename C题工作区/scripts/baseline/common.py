from pathlib import Path
import sys,json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'运行依赖314'))
import numpy as np
DATA=ROOT/'处理结果/v1';CONFIG=ROOT/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints'
def readcsv(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r)) or ['empty']
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def dump(p,o):
 Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(z,side):return z[side+'__position'].mean(axis=1)
def prefix_features(p):
 v=[]
 for side in ['left_hand','right_hand']:
  t=p[side+'__t_s'];xyz=state(p,side);good=p[side+'__quality_known']&~p[side+'__bad']&np.isfinite(xyz).all(axis=1);dt=np.diff(t);steps=np.linalg.norm(np.diff(xyz,axis=0),axis=1);speed=steps[dt>0]/dt[dt>0]
  v.extend([len(t),float(good.mean()) if len(t) else 0,float(np.linalg.norm(xyz[-1]-xyz[0])) if len(t)>1 else 0,float(steps.sum()),float(np.median(speed)) if len(speed) else 0])
 return np.asarray(v,dtype=float)
def grouped_boot(values,groups,n=2000):
 means=np.asarray([np.mean([v for v,g in zip(values,groups) if g==e]) for e in sorted(set(groups))])
 if not len(means):return dict(mean=None,lo=None,hi=None,clusters=0)
 if len(means)==1:return dict(mean=float(means[0]),lo=None,hi=None,clusters=1)
 rng=np.random.default_rng(20261120);b=means[rng.integers(0,len(means),(n,len(means)))].mean(axis=1)
 return dict(mean=float(means.mean()),lo=float(np.quantile(b,.025)),hi=float(np.quantile(b,.975)),clusters=len(means))
