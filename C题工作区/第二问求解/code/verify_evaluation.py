"""Independent score/action audit from persisted evaluation logs."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
KEY=['window_id','method','length_s','delay_s']
def audit(split):
 out=BASE/'results'/f'S5_{split}';m=pd.read_csv(out/'windows.csv');p=pd.read_csv(out/'points.csv.gz');a=pd.read_csv(out/'actions.csv.gz');f=json.loads((BASE/'frozen_model.json').read_text());old=json.loads((out/'audit.json').read_text())
 assert old['frozen_model_sha256']==hashlib.sha256((BASE/'frozen_model.json').read_bytes()).hexdigest()
 for name,digest in f['code_sha256'].items():assert hashlib.sha256((BASE/'code'/name).read_bytes()).hexdigest()==digest,name
 pred=p[['pred_x','pred_y','pred_z']].to_numpy();ref=p[['ref_x','ref_y','ref_z']].to_numpy();e=np.sqrt(np.sum((pred-ref)**2,axis=1));delta=float(np.max(np.abs(e-p.error)));assert delta<1e-9
 p['recalc']=e;g=p.groupby(KEY,sort=False);z=g.recalc.agg(ade='mean',fde='last',p95=lambda x:x.quantile(.95)).reset_index();join=m.merge(z,on=KEY,validate='one_to_one',suffixes=('','_independent'))
 for field in ['ade','fde','p95']:
  err=float(np.max(np.abs(join[field]-join[field+'_independent'])));delta=max(delta,err);assert err<1e-9
 req=a[a.event=='request'];ret=a[a.event=='return'];counts=req.groupby(KEY).size().rename('request_count').reset_index();j=m.merge(counts,on=KEY,how='left').fillna({'request_count':0});assert np.array_equal(j.requests,j.request_count) and (j.cost==j.requests).all() and (j.requests<=2).all()
 assert (ret.time<=ret.processed_at+1e-9).all();q=req[KEY+['request_id','time','arrival']].merge(ret[KEY+['request_id','time','sample_time','processed_at']],on=KEY+['request_id'],validate='one_to_one',suffixes=('_req','_ret'));assert np.allclose(q.time_req,q.sample_time,atol=1e-9) and np.allclose(q.arrival,q.time_ret,atol=1e-9)
 assert not req.duplicated(KEY+['sample_time']).any()
 spans=req.assign(span=req.arrival-req.time).groupby(KEY).span.sum().reset_index();j=m.merge(spans,on=KEY,how='left').fillna({'span':0});assert np.max(np.abs(j.waiting_s-j.span))<1e-9
 assert len(j)==len(m) and m[KEY].drop_duplicates().shape[0]==len(m)
 check=dict(passed=True,window_results=len(m),point_results=len(p),action_rows=len(a),max_independent_score_error=delta,requests=len(req),returns=len(ret),early_returns=0,budget_violations=0,frozen_code_hashes_valid=True,source_model_unchanged=True)
 (out/'independent_audit.json').write_text(json.dumps(check,indent=2),encoding='utf8');print(json.dumps(check),flush=True)
if __name__=='__main__':audit(sys.argv[1])
