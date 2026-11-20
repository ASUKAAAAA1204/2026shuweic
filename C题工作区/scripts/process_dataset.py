from pathlib import Path
import sys,csv,json,hashlib,collections,platform
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'运行依赖314'))
import numpy as np,pyarrow,pyarrow.parquet as pq,cv2
O=R/'处理结果/v1';O.mkdir(parents=True,exist_ok=True)
for d in ['records','private','contact_sheets']: (O/d).mkdir(exist_ok=True)
C=R/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints'
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(n,a):
 if not a:return
 keys=list(dict.fromkeys(k for x in a for k in x))
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(a)
def js(n,a):(O/n).write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf8')
split={x['episode_uuid']:x for x in rows(C/'record_split.csv')};segs=rows(C/'source_segment_index.csv')
records=[];stats=[];videos=[];intervals=[];prefix=[];hashes=[];errors=[];metas={};checks=[]
channels={'left_hand':('hand_left_world','hand_left_world_rotation','pose_left_hand',21),'right_hand':('hand_right_world','hand_right_world_rotation','pose_right_hand',21),'head':('head_world','head_world_rotation','pose_head',1),'head_left_camera':('head_left_camera_position','head_left_camera_rotation','pose_headcam',1),'head_right_camera':('head_right_camera_position','head_right_camera_rotation','pose_right_eye_cam',1)}
for src in ['EgoDemo','EgoPro','EgoStandard']:
 for folder in sorted((R/'数据集'/src).iterdir()):
  if not folder.is_dir():continue
  a={};meta={};camera={};rawseg=[];quality_reasons=[]
  if src=='EgoDemo':
   ep=pq.read_table(folder/'meta/episodes/chunk-000/file-000.parquet').to_pylist()[0];uid=ep['episode_uuid'];env=ep['environment_id'];operator=ep['operator_id']
   table=pq.read_table(folder/'data/chunk-000/file-000.parquet');t=np.asarray(table['timestamp'].to_pylist(),dtype=np.float64);fi=np.asarray(table['frame_index'].to_pylist(),dtype=np.int64)
   info=json.loads((folder/'meta/info.json').read_text());meta={'episode':ep,'info':info};camera={k:v for k,v in ep.items() if 'intrinsic' in k}
   bad=json.loads((folder/'bad_frame_ratio.json').read_text());quality_reasons=bad.get('hand_bad_frame_details',[])
   qmap={int(x['frame']):x for x in quality_reasons}
   rawseg=[(float(x['start_time']),float(x['end_time'])) for x in json.loads((folder/'annotation.json').read_text(encoding='utf8'))['subtask_annotations']]
   for name,(p,q,_,count) in channels.items():
    a[name+'__t_s']=t.copy();a[name+'__frame_index']=fi.copy()
    a[name+'__position']=np.asarray(table['observation.state.'+p].to_pylist(),dtype=np.float64).reshape(-1,count,3)
    a[name+'__quaternion']=np.asarray(table['observation.state.'+q].to_pylist(),dtype=np.float64).reshape(-1,count,4)
    if name in ['left_hand','right_hand']:
     side=name.split('_')[0];a[name+'__quality_known']=np.ones(len(t),dtype=bool);a[name+'__bad']=np.asarray([bool(qmap.get(int(i),{}).get(side+'_bad_reason')) for i in fi])
   origin='Demo timestamp field (float32 source); no time shift';quat='declared qw,qx,qy,qz in info.json; preserved without permutation'
  else:
   meta={x['field']:x['value'] for x in rows(folder/'session_metadata.csv')};uid=meta['task_info.episode_uuid'];env=meta['task_info.environment_id'];operator=meta.get('operator.operator_id','')
   qr=rows(folder/'tables/annotation_bad_frame_pose_hand.csv');quality_reasons=qr
   allrows={name:rows(folder/'tables'/(v[2]+'.csv')) for name,v in channels.items()}
   # Recover shared record log origin exactly from integer nanoseconds and decimal offset.
   from decimal import Decimal
   first=allrows['left_hand'][0];origin_ns=int(first['_log_time_ns'])-int(Decimal(first['_offset_from_first_log_s'])*Decimal(1000000000))
   qt=np.asarray([int(x['_log_time_ns']) for x in qr],dtype=np.int64);qb=np.asarray([x['is_bad'].lower()=='true' for x in qr]);a['quality__timestamp_ns']=qt;a['quality__bad']=qb
   for name,(_,_,_,count) in channels.items():
    rr=allrows[name];ns=np.asarray([int(x['_log_time_ns']) for x in rr],dtype=np.int64)
    a[name+'__timestamp_ns']=ns;a[name+'__publish_timestamp_ns']=np.asarray([int(x['_publish_time_ns']) for x in rr],dtype=np.int64)
    a[name+'__source_offset_s']=np.asarray([float(x['_offset_from_first_log_s']) for x in rr]);a[name+'__t_s']=(ns-origin_ns)/1e9
    a[name+'__position']=np.asarray([[[float(x[f'transforms[{j}].pos.{k}']) for k in 'xyz'] for j in range(count)] for x in rr])
    a[name+'__quaternion']=np.asarray([[[float(x[f'transforms[{j}].quat.{k}']) for k in 'wxyz'] for j in range(count)] for x in rr])
    if name in ['left_hand','right_hand']:
     lookup={int(n):bool(b) for n,b in zip(qt,qb)};a[name+'__quality_known']=np.asarray([int(n) in lookup for n in ns]);a[name+'__bad']=np.asarray([lookup.get(int(n),False) for n in ns])
   rawseg=[(float(x['segment.start_time']),float(x['segment.end_time'])) for x in rows(folder/'tables/annotation_semantic_segments.csv')]
   camera={p.name:rows(p) for p in (folder/'tables').glob('sensor_camera*.csv')};origin=f'shared log origin {origin_ns} ns, recovered from source offset; integer subtraction';quat='CSV explicit quat.w,x,y,z'
  official=split[uid];assert official['environment_id']==env
  metas[uid]={'dataset':src,'folder':folder.name,'metadata':meta,'camera':camera,'quality_source_rows':quality_reasons,'quaternion_basis':quat,'time_basis':origin}
  for name in channels:
   t=a[name+'__t_s'];p=a[name+'__position'];q=a[name+'__quaternion'];dt=np.diff(t);finite=np.isfinite(p).all(axis=(1,2))&np.isfinite(q).all(axis=(1,2));norm=np.linalg.norm(q,axis=2);qok=np.isfinite(norm).all(axis=1)&(np.abs(norm-1)<=.001).all(axis=1)
   ordered=np.r_[True,dt>0];a[name+'__finite']=finite;a[name+'__quaternion_unit_1e3']=qok;a[name+'__time_increasing']=ordered
   median=float(np.median(dt[dt>0])) if (dt>0).any() else 0.;gap=np.r_[False,dt>3*median] if median else np.zeros(len(t),bool);a[name+'__gap_before']=gap
   stat={'episode_uuid':uid,'dataset':src,'split':official['split'],'channel':name,'samples':len(t),'start_s':float(t[0]),'end_s':float(t[-1]),'median_dt_s':median,'nonpositive_dt':int((dt<=0).sum()),'gaps_gt_3median':int(gap.sum()),'nonfinite_frames':int((~finite).sum()),'nonunit_quaternion_frames':int((~qok).sum()),'max_quaternion_norm_error':float(np.nanmax(abs(norm-1)))}
   if name in ['left_hand','right_hand']:
    known=a[name+'__quality_known'];bad=a[name+'__bad'];valid=finite&qok&known&~bad&ordered
    a[name+'__reference_candidate']=valid
    velocity=np.full((len(t),3),np.nan);ok=(dt>0)&valid[1:]&valid[:-1]&~gap[1:];velocity[1:][ok]=(p[1:,0][ok]-p[:-1,0][ok])/dt[ok,None]
    a[name+'__wrist_backward_velocity']=velocity;a[name+'__wrist_position']=p[:,0,:]
    stat.update(quality_unknown=int((~known).sum()),bad_frames=int(bad.sum()),reference_candidates=int(valid.sum()),velocity_valid=int(np.isfinite(velocity).all(axis=1).sum()))
    flag=bad|~known|~finite|~qok
    edges=np.diff(np.r_[False,flag,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
    for lo,hi in zip(starts,ends):intervals.append({'episode_uuid':uid,'split':official['split'],'channel':name,'start_index':int(lo),'end_index_exclusive':int(hi),'start_s':float(t[lo]),'last_sample_s':float(t[hi-1]),'samples':int(hi-lo),'meaning':'quality_or_numeric_unusable; not physical ground truth'})
   stats.append(stat)
  np.savez_compressed(O/'records'/(uid+'.npz'),**a)
  # Numerical conversion is checked against in-memory originals for all arrays.
  with np.load(O/'records'/(uid+'.npz'),allow_pickle=False) as z:
   assert set(z.files)==set(a)
   for k in a:assert np.array_equal(z[k],a[k],equal_nan=True),k
  rs=[s for s in segs if s['episode_uuid']==uid]
  for s in rs:
   i=int(s['source_segment_index']);assert np.allclose(rawseg[i],[float(s['start_s']),float(s['end_s'])],atol=1e-5,rtol=0)
   start=float(s['start_s'])
   for length in [.2,.6,1.]:
    payload={};counts={}
    for name in channels:
     t=a[name+'__t_s'];sel=(t>=start)&(t<=start+length);counts[name]=int(sel.sum());payload[name+'__t_s']=t[sel]-start
     for key in ['position','quaternion']+(['quality_known','bad'] if name in ['left_hand','right_hand'] else []):payload[name+'__'+key]=a[name+'__'+key][sel]
     assert np.all(payload[name+'__t_s']>=0) and np.all(payload[name+'__t_s']<=length+1e-12)
    # Private cache only: filename identity is forbidden to policy; API strips all metadata.
    if s['eligible']=='1':np.savez_compressed(O/'private'/(s['unit_id']+'_'+str(length)+'.npz'),**payload)
    prefix.append({'unit_id':s['unit_id'],'split':s['split'],'eligible':s['eligible'],'length_s':length,**counts,'all_channels_nonempty':all(counts.values()),'samples_after_segment_end':sum(int(((a[n+'__t_s']>=start)&(a[n+'__t_s']<=start+length)&(a[n+'__t_s']>float(s['end_s']))).sum()) for n in channels)})
  for vp in folder.rglob('*.mp4'):
   cap=cv2.VideoCapture(str(vp));opened=cap.isOpened();n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);ok,im=cap.read();last=False
   if n>0:cap.set(cv2.CAP_PROP_POS_FRAMES,n-1);last,imlast=cap.read()
   if ok and len(videos)%12==0:cv2.imwrite(str(O/'contact_sheets'/(uid+'_'+str(len(videos))+'.jpg')),cv2.resize(im,(480,270)))
   cap.release();framefiles=list(vp.parent.glob(vp.stem+'_frames.csv'));indexn=len(rows(framefiles[0])) if framefiles else None
   videos.append({'episode_uuid':uid,'path':str(vp.relative_to(R)),'opened':opened,'frames_reported':n,'fps_reported':fps,'first_decoded':ok,'last_decoded':last,'frame_index_rows':indexn,'frame_index_count_matches':indexn==n if indexn is not None else 'not supplied separately (Demo)'})
  records.append({'dataset':src,'folder':folder.name,'episode_uuid':uid,'environment_id':env,'operator_id':operator,'split':official['split'],'samples_left':len(a['left_hand__t_s']),'segments':len(rawseg),'time_basis':origin,'quaternion_basis':quat})
  print(f'{len(records)}/69 {src} {folder.name}',flush=True)
# Hash original metadata/numeric files; large videos covered by prior archive CRC verification.
for p in (R/'数据集').rglob('*'):
 if p.is_file() and p.suffix!='.mp4':hashes.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(records)==69 and len({r['episode_uuid'] for r in records})==69 and set(split)=={r['episode_uuid'] for r in records}
assert not any(len({r['split'] for r in records if r['environment_id']==e})>1 for e in {r['environment_id'] for r in records})
save('record_index.csv',records);save('channel_quality.csv',stats);save('video_audit.csv',videos);save('natural_unusable_intervals.csv',intervals);save('prefix_coverage.csv',prefix);save('input_sha256.csv',hashes);save('segments_private.csv',segs);js('metadata_private.json',metas)
summary={'records':len(records),'samples_left':sum(x['samples_left'] for x in records),'channels':len(stats),'segments':len(segs),'eligible':sum(x['eligible']=='1' for x in segs),'private_prefixes':len(list((O/'private').glob('*.npz'))),'videos':len(videos),'video_failures':[v for v in videos if not all([v['opened'],v['first_decoded'],v['last_decoded']])],'video_index_mismatch':[v for v in videos if v['frame_index_rows'] is not None and not v['frame_index_count_matches']],'empty_eligible_prefixes':[p for p in prefix if p['eligible']=='1' and not p['all_channels_nonempty']],'split_counts':dict(collections.Counter(x['split'] for x in records)),'sources':dict(collections.Counter(x['dataset'] for x in records)),'quality_totals':{k:sum(x.get(k,0) for x in stats if x['channel'] in ['left_hand','right_hand']) for k in ['samples','nonfinite_frames','nonunit_quaternion_frames','quality_unknown','bad_frames','reference_candidates','velocity_valid']},'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'pyarrow':pyarrow.__version__,'opencv':cv2.__version__},'checks':{'all_array_roundtrip':True,'all_internal_uuid_and_environment_match':True,'all_segments_match':True,'environment_disjoint':True,'all_prefix_timestamps_causal':True},'limits':['video metadata plus first/last decode only','world unit and coordinate directions unconfirmed','Demo quaternion order follows declared metadata, not independently physically calibrated','CSV quality is global hand flag, conservatively shared by both hands; exact timestamp matching only','no interpolation, denoising, quaternion repair or resampling','missing organizer final test seeds and data dictionary remain unresolved']}
js('summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
