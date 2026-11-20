from common import *
import collections,time
import cv2

def run(out,video=True):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);records=readcsv(DATA/'record_index.csv');segs=readcsv(DATA/'segments_private.csv');meta=load(DATA/'metadata_private.json');windows=[];scales=collections.defaultdict(list);reasons=[];usage=[]
 for r in records:
  uid=r['episode_uuid'];m=meta[uid]['metadata'];r['scene_id']=m['episode']['scene_id'] if r['dataset']=='EgoDemo' else m.get('task_info.scene_id','')
  with np.load(DATA/'records'/(uid+'.npz'),allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];x=state(z,side);head=z['head__position'][:,0];valid=z[side+'__reference_candidate'];dt=np.diff(t);med=float(np.median(dt[dt>0]))
    if r['split']=='train':scales[r['dataset']].extend(np.linalg.norm(x[valid]-head[valid],axis=1).tolist())
    last_end=-1e30;count=0
    for i in range(len(t)):
     if t[i]<last_end+1 or t[i]-t[0]<.2-1e-6:continue
     h=int(np.searchsorted(t,t[i]-.2,side='left'));j=int(np.searchsorted(t,t[i]+1,side='right'))
     if j>=len(t) or t[j-1]-t[i]<1-med*1.1:continue
     if not valid[h:j].all() or (np.diff(t[h:j])>3*med).any():continue
     windows.append(dict(window_id=f'{uid}_{side}_{i}',episode_uuid=uid,dataset=r['dataset'],environment_id=r['environment_id'],split=r['split'],side=side,history_start=h,start_index=i,end_index_exclusive=j,start_s=float(t[i]),end_s=float(t[i]+1)));count+=1;last_end=t[i]+1
    usage.append({**{k:r[k] for k in ['episode_uuid','dataset','split','environment_id']},'side':side,'total_samples':len(t),'reference_samples':int(valid.sum()),'bad_samples':int(z[side+'__bad'].sum()),'windows':count,'fit_allowed':r['split']=='train' and bool(valid.any()),'evaluation_allowed':bool(count),'excluded_samples':int((~valid).sum())})
  for item in meta[uid]['quality_source_rows']:
   vals=[(side,item.get(side+'_bad_reason')) for side in ['left','right']] if r['dataset']=='EgoDemo' else [('both',item[k]) for k in item if k.startswith('problem_type[') and item[k]]
   if r['dataset']!='EgoDemo' and item.get('is_bad','').lower()=='true' and not vals:vals=[('both','unspecified_bad')]
   for side,reason in vals:
    if reason:reasons.append(dict(episode_uuid=uid,dataset=r['dataset'],split=r['split'],side=side,reason=str(reason),frame_or_ns=item.get('frame',item.get('_log_time_ns'))))
 writecsv(out/'records.csv',records);writecsv(out/'windows.csv',windows);writecsv(out/'sample_usage.csv',usage);writecsv(out/'quality_reasons.csv',reasons)
 overlaps=[]
 for field in ['environment_id','scene_id','operator_id']:
  for key in sorted({r[field] for r in records}):
   sp=sorted({r['split'] for r in records if r[field]==key})
   if len(sp)>1:overlaps.append(dict(field=field,value=key,splits='|'.join(sp)))
 writecsv(out/'split_overlaps.csv',overlaps)
 protocol=dict(version='baseline-20261120-v2',state='mean of 21 keypoints for each hand; source units; no quaternion/camera use',mask_lengths_s=[.2,.6,1.],window_s=1.,inter_window_gap_s=1.,history_s=.2,request_cost=1,request_limit=2,delays_s=[0,.1,.2],primary_delay_s=.1,fixed_request_s=[.3,.6],request_semantics='sample original at request time, available after delay',bootstrap_clusters='environment',bootstrap_repetitions=2000,seed=20261120,tolerance_fractions=[.02,.05,.1],train_scale={k:float(np.median(v)) for k,v in scales.items()},test_use='structure only',q2_protocol_status='participant-defined; official missing_observation_experiments.md unavailable',quality_assumption='Demo sparse absent means unflagged; not independent physical truth')
 dump(out/'protocol.json',protocol);checks=0
 for s in segs:
  if s['eligible']!='1':continue
  with np.load(DATA/'records'/(s['episode_uuid']+'.npz')) as z:
   for length in [.2,.6,1.]:
    with np.load(DATA/'private'/(s['unit_id']+'_'+str(length)+'.npz')) as p:
     assert len(p.files)==19
     for ch in ['left_hand','right_hand','head','head_left_camera','head_right_camera']:
      t=z[ch+'__t_s'];sel=(t>=float(s['start_s']))&(t<=float(s['start_s'])+length);assert np.array_equal(p[ch+'__t_s'],t[sel]-float(s['start_s']))
      for key in ['position','quaternion']:assert np.array_equal(p[ch+'__'+key],z[ch+'__'+key][sel],equal_nan=True)
     checks+=1
 videos=[]
 if video:
  for v in readcsv(DATA/'video_audit.csv'):
   p=ROOT/v['path'];cap=cv2.VideoCapture(str(p));n=0;start=time.perf_counter()
   while True:
    ok,im=cap.read()
    if not ok:break
    n+=1
   cap.release();videos.append(dict(path=str(p.relative_to(ROOT)),decoded_frames=n,expected_frames=int(v['frames_reported']),passed=n==int(v['frames_reported']),seconds=time.perf_counter()-start))
   if len(videos)%12==0:print('full video decode',len(videos),'/184',flush=True)
  writecsv(out/'full_video_decode.csv',videos)
 writecsv(out/'input_hashes.csv',[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in (ROOT/'数据集').rglob('*') if p.is_file() and p.suffix!='.mp4'])
 dump(out/'prepare_checks.json',dict(records=len(records),unique_uuid=len({r['episode_uuid'] for r in records}),prefixes_verified=checks,windows=len(windows),window_counts=dict(collections.Counter(w['split'] for w in windows)),environment_overlap=sum(x['field']=='environment_id' for x in overlaps),video_count=len(videos),video_failures=sum(not v['passed'] for v in videos),state_avoids_unverified_joint_order_and_orientation=True))
 print('prepare complete',len(windows),'windows',flush=True)
