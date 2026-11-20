from common import *
import cv2,collections,html

def run(out):
 out=Path(out);a=out/'audit';a.mkdir(exist_ok=True);records=readcsv(out/'data/records.csv');segs=readcsv(DATA/'segments_private.csv');rr=[];selected=[];rawchecks=[]
 import pyarrow.parquet as pq
 for r in records:
  folder=ROOT/'数据集'/r['dataset']/r['folder'];uid=r['episode_uuid']
  with np.load(DATA/'records'/(uid+'.npz')) as z:
   if r['dataset']=='EgoDemo':
    table=pq.read_table(folder/'data/chunk-000/file-000.parquet')
    mapping={'left_hand':('hand_left_world','hand_left_world_rotation'),'right_hand':('hand_right_world','hand_right_world_rotation'),'head':('head_world','head_world_rotation'),'head_left_camera':('head_left_camera_position','head_left_camera_rotation'),'head_right_camera':('head_right_camera_position','head_right_camera_rotation')}
    for name,(pos,quat) in mapping.items():
     for source,key in [(pos,'position'),(quat,'quaternion')]:
      v=np.array(table['observation.state.'+source].to_pylist(),dtype=float).reshape(z[name+'__'+key].shape);assert np.array_equal(v,z[name+'__'+key],equal_nan=True)
     assert np.array_equal(np.array(table['timestamp'].to_pylist()),z[name+'__t_s'])
   else:
    for name,file in [('left_hand','pose_left_hand'),('right_hand','pose_right_hand'),('head','pose_head'),('head_left_camera','pose_headcam'),('head_right_camera','pose_right_eye_cam')]:
     rows=readcsv(folder/'tables'/(file+'.csv'));count=z[name+'__position'].shape[1]
     for keys,suffix in [('xyz','pos'),('wxyz','quat')]:
      v=np.array([[[float(x[f'transforms[{j}].{suffix}.{k}']) for k in keys] for j in range(count)] for x in rows]);assert np.array_equal(v,z[name+('__position' if suffix=='pos' else '__quaternion')],equal_nan=True)
     assert np.array_equal(np.array([int(x['_log_time_ns']) for x in rows],dtype=np.int64),z[name+'__timestamp_ns'])
   rawchecks.append(dict(episode_uuid=uid,channels=5,raw_numeric_recheck=True))
 for src in ['EgoDemo','EgoPro','EgoStandard']:
  rs=sorted([r for r in records if r['dataset']==src],key=lambda r:r['folder'])
  for r in [rs[0],rs[len(rs)//2],rs[-1]]:
   folder=ROOT/'数据集'/src/r['folder'];v=next(folder.rglob('*head_left*.mp4'),None)
   if v is None:v=next((folder/'videos/observation.images.head_left').rglob('*.mp4'))
   with np.load(DATA/'records'/(r['episode_uuid']+'.npz')) as z:
    t=z['left_hand__t_s'];left=state(z,'left_hand');right=state(z,'right_hand');indices=[0,len(t)//2,len(t)-1];cap=cv2.VideoCapture(str(v));fps=cap.get(cv2.CAP_PROP_FPS);ims=[]
    for i in indices:
     cap.set(cv2.CAP_PROP_POS_FRAMES,min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1,round((t[i]-t[0])*fps)));ok,im=cap.read()
     if not ok:raise RuntimeError('selected frame unavailable')
     im=cv2.resize(im,(320,240));cv2.putText(im,f'{src} t={t[i]:.3f}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1);ims.append(im)
    cap.release();sheet=np.concatenate(ims,axis=1);fname=r['episode_uuid']+'.jpg';cv2.imencode('.jpg',sheet)[1].tofile(str(a/fname));selected.append(dict(dataset=src,folder=r['folder'],episode_uuid=r['episode_uuid'],image=fname))
    # Orthographic XY/XZ raw-coordinate plots; these are not camera projections.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="300"><rect width="960" height="300" fill="white"/>']
    for panel,axes in enumerate([(0,1),(0,2)]):
     combined=np.r_[left[:,axes],right[:,axes]];lo=combined.min(0);span=np.maximum(combined.max(0)-lo,1e-9)
     svg.append(f'<text x="{panel*480+20}" y="22">{src}: '+('XY' if panel==0 else 'XZ')+' source axes</text>')
     for data,color in [(left,'#2563eb'),(right,'#dc2626')]:
      v2=(data[:,axes]-lo)/span;pts=' '.join(f'{panel*480+20+x*440:.2f},{280-y*240:.2f}' for x,y in v2[::3]);svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1" points="{pts}"/>')
    svg.append('</svg>');(a/(r['episode_uuid']+'.svg')).write_text(''.join(svg),encoding='utf8')
 writecsv(a/'raw_numeric_recheck.csv',rawchecks);writecsv(a/'visual_sample_index.csv',selected)
 page=['<meta charset="utf-8"><title>九条记录的时间与轨迹检查</title><h1>九条记录的时间与轨迹检查</h1><p>每源按目录排序选择首、中、末。图像为首中末时刻；轨迹为21点中心在源坐标下的XY/XZ正投影。没有把轨迹投到视频像素，没有据此声称相机标定或轴向已验证。</p>']
 for r in selected:page.append(f'<h2>{r["dataset"]} {r["folder"]}</h2><img width="960" src="{r["image"]}"><br><img src="{r["episode_uuid"]}.svg">')
 (a/'九条记录核对.html').write_text('\n'.join(page),encoding='utf8')
 # Offline semantic stratification only; never policy features.
 windows={w['window_id']:w for w in readcsv(out/'data/windows.csv')};groups=collections.defaultdict(list)
 for row in readcsv(out/'q12/open_loop_windows.csv'):
  if row['split']!='validation':continue
  w=windows[row['window_id']];ss=[s for s in segs if s['episode_uuid']==row['episode_uuid'] and float(s['start_s'])<=float(w['start_s'])<float(s['end_s'])];s=ss[0] if ss else None;phase=min(3,int(4*(float(w['start_s'])-float(s['start_s']))/float(s['duration_s']))) if s else -1;family=s['family_id'] if s else 'unannotated'
  groups[(row['model'],row['horizon_s'],family,phase)].append(row)
 for key,items in groups.items():rr.append(dict(model=key[0],horizon_s=key[1],family=key[2],retrospective_phase_quartile=key[3],windows=len(items),ade=float(np.mean([float(r['ade']) for r in items])),note='offline stratification using annotations; never online feature'))
 writecsv(a/'retrospective_phase_metrics.csv',rr)
 reasons=readcsv(out/'data/quality_reasons.csv');counts=collections.Counter((r['dataset'],r['split'],r['side'],r['reason']) for r in reasons);writecsv(a/'quality_reason_counts.csv',[dict(dataset=k[0],split=k[1],side=k[2],reason=k[3],rows=v) for k,v in sorted(counts.items())])
 # Natural bad runs only, separate from numeric mask failures.
 runs=[]
 for r in records:
  with np.load(DATA/'records'/(r['episode_uuid']+'.npz')) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];bad=z[side+'__bad'];edge=np.diff(np.r_[False,bad,False].astype(int))
    for lo,hi in zip(np.where(edge==1)[0],np.where(edge==-1)[0]):runs.append(dict(episode_uuid=r['episode_uuid'],dataset=r['dataset'],split=r['split'],side=side,start_index=int(lo),end_index_exclusive=int(hi),start_s=float(t[lo]),last_bad_s=float(t[hi-1]),next_sample_s=float(t[hi]) if hi<len(t) else None,span_to_next_s=float(t[hi]-t[lo]) if hi<len(t) else None,recovered_within_record=bool(hi<len(t)),reference_error='not defined'))
 writecsv(a/'natural_bad_runs.csv',runs)
 dump(a/'checks.json',dict(raw_records_verified=len(rawchecks),raw_channels_verified=len(rawchecks)*5,visual_samples=len(selected),natural_bad_runs=len(runs),camera_projection_used=False))
 writecsv(a/'source_hashes.csv',[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'数据集').rglob('*')) if p.is_file()])
 print('raw source recheck and visual evidence complete',flush=True)
