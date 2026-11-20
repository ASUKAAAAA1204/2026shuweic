from common import *
from q12 import replay_window

def run(out):
 out=Path(out);p=load(out/'data/protocol.json');r=load(out/'q12/prediction_radius_train.json');ws=[w for w in readcsv(out/'data/windows.csv') if w['split']=='validation'];tests=[]
 for w in ws[:9]:
  with np.load(DATA/'records'/(w['episode_uuid']+'.npz')) as z:a={k:z[k].copy() for k in z.files}
  b={k:v.copy() for k,v in a.items()};i=int(w['start_index']);b[w['side']+'__position'][i:]+=1000
  for method in ['hold','constant_velocity','fixed_review','adaptive_review']:
   for delay in [0,.1,.2]:
    ra,ta=replay_window(w,a,p,r,method,delay,.05,True);rb,tb=replay_window(w,b,p,r,method,delay,.05,True)
    first_return=min((float(w['start_s'])+x['return_s'] for x in json.loads(ra['actions'])),default=float('inf'))
    n=0
    for x,y in zip(ta,tb):
     if x['time']<first_return-1e-12:
      assert x['prediction']==y['prediction'] and x['requests']==y['requests'];n+=1
    tests.append(dict(window_id=w['window_id'],method=method,delay_s=delay,prefix_decisions_checked=n,passed=True))
 writecsv(out/'q2_future_invariance.csv',tests)
 # Window spacing/quality and train-only protocol scale rechecked independently.
 windows=readcsv(out/'data/windows.csv');prev={}
 for w in windows:
  k=(w['episode_uuid'],w['side']);start=float(w['start_s']);end=float(w['end_s'])
  if k in prev:assert start-prev[k]>=1-1e-12
  prev[k]=end
  with np.load(DATA/'records'/(w['episode_uuid']+'.npz')) as z:assert z[w['side']+'__reference_candidate'][int(w['history_start']):int(w['end_index_exclusive'])].all()
 dump(out/'extra_checks.json',dict(q2_future_tests=len(tests),windows_qualification_checked=len(windows),passed=True))
 print('extra causality/window checks passed',flush=True)
