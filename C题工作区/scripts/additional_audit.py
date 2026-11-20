from pathlib import Path
import sys,csv,json,collections,hashlib
R=Path('C题工作区');sys.path.insert(0,str(R/'运行依赖314'));import numpy as np
O=R/'处理结果/v1'
def read(n):
 with (O/n).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def save(n,a):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
rr=read('record_index.csv');ss=read('segments_private.csv');out=[];groups=[];times=[]
for r in rr:
 with np.load(O/'records'/(r['episode_uuid']+'.npz')) as z:
  base=z['left_hand__t_s']
  for ch in ['right_hand','head','head_left_camera','head_right_camera']:
   t=z[ch+'__t_s'];idx=np.clip(np.searchsorted(t,base),0,len(t)-1);prev=np.clip(idx-1,0,len(t)-1);err=np.minimum(abs(t[idx]-base),abs(t[prev]-base))
   times.append({'episode_uuid':r['episode_uuid'],'channel':ch,'nearest_error_max_s':float(err.max()),'nearest_error_p95_s':float(np.quantile(err,.95)),'use':'offline audit only; not feature alignment'})
  own=[s for s in ss if s['episode_uuid']==r['episode_uuid']];endprev=None
  for s in own:
   start=float(s['start_s']);end=float(s['end_s']);out.append({**s,'start_before_pose':start<float(base[0])-1e-5,'end_after_last_pose':end>float(base[-1])+1e-5,'end_minus_last_pose_s':end-float(base[-1]),'nonpositive_duration':end<=start,'overlap_previous_s':max(0,endprev-start) if endprev is not None else 0});endprev=end
for fields in [('dataset',),('split',),('environment_id',),('family_id',)]:
 buckets=collections.defaultdict(list)
 for s in ss:buckets[tuple(s[k] for k in fields)].append(s)
 for key,v in buckets.items():
  d=np.asarray([float(x['duration_s']) for x in v]);groups.append({'group_by':'+'.join(fields),'group':'+'.join(key),'segments':len(v),'eligible':sum(x['eligible']=='1' for x in v),'duration_min':float(d.min()),'duration_median':float(np.median(d)),'duration_max':float(d.max())})
save('segments_audit.csv',out);save('segment_group_stats.csv',groups);save('channel_alignment_audit.csv',times)
q=read('channel_quality.csv');agg=[]
for fields in [('dataset','channel'),('split','channel'),('environment_id','channel')]:
 b=collections.defaultdict(list);lookup={r['episode_uuid']:r for r in rr}
 for x in q:b[tuple(lookup[x['episode_uuid']].get(k,x.get(k,'')) for k in fields)].append(x)
 for key,v in b.items():agg.append({'group_by':'+'.join(fields),'group':'+'.join(key),**{k:sum(int(x.get(k) or 0) for x in v) for k in ['samples','bad_frames','quality_unknown','reference_candidates','nonfinite_frames','nonunit_quaternion_frames','gaps_gt_3median','nonpositive_dt']}})
save('quality_group_stats.csv',agg)
manifest=[]
for p in list((R/'scripts').glob('*.py'))+list((R/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints').glob('*')):
 if p.is_file():manifest.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
save('pipeline_config_sha256.csv',manifest)
overlap=[]
for key in ['operator_id']:
 for v in {r[key] for r in rr}:
  splits=sorted({r['split'] for r in rr if r[key]==v})
  if len(splits)>1:overlap.append({'field':key,'value':v,'splits':splits})
(O/'additional_audit.json').write_text(json.dumps({'cross_split_operator':overlap,'segment_end_after_last_sample':sum(x['end_after_last_pose'] for x in out),'segment_start_before_first_sample':sum(x['start_before_pose'] for x in out),'nonpositive_segments':sum(x['nonpositive_duration'] for x in out),'overlapping_segments':sum(x['overlap_previous_s']>0 for x in out),'maximum_channel_nearest_error_s':max(x['nearest_error_max_s'] for x in times)},ensure_ascii=False,indent=2),encoding='utf8')
