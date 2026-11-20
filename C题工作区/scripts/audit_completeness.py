from pathlib import Path
import sys,csv,json,collections
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'工具依赖'))
pq=None
out=root/'数据审计';out.mkdir(exist_ok=True)
config=root/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints'
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,rows):
 if not rows:return
 with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
split=readcsv(config/'record_split.csv');byid={x['episode_uuid']:x for x in split};segments=readcsv(config/'source_segment_index.csv')
records=[];issues=[];schemas=[];rawsegments={}
for source in ['EgoDemo','EgoPro','EgoStandard']:
 for folder in sorted((root/'数据集'/source).iterdir()):
  if not folder.is_dir():continue
  row={'dataset':source,'folder':folder.name,'episode_uuid':'','environment_id':'','split':'','timestamps':0,'segments':0,'videos':len(list(folder.rglob('*.mp4'))),'missing_files':'','parquet_decode_status':'not applicable'}
  missing=[]
  if source=='EgoDemo':
   for rel in ['annotation.json','bad_frame_ratio.json','meta/info.json','meta/tasks.parquet','meta/episodes/chunk-000/file-000.parquet','data/chunk-000/file-000.parquet']:
    if not (folder/rel).is_file():missing.append(rel)
   info=json.loads((folder/'meta/info.json').read_text());row['timestamps']=info['total_frames']
   row['parquet_decode_status']='not decoded; footer magic and metadata declaration only'
   for f in folder.rglob('*.parquet'):
    with f.open('rb') as q:
     first=q.read(4);q.seek(-4,2);last=q.read(4)
    if first!=b'PAR1' or last!=b'PAR1':missing.append('invalid parquet magic '+str(f))
   a=json.loads((folder/'annotation.json').read_text(encoding='utf8'));ss=a.get('subtask_annotations',[]);row['segments']=len(ss)
   # official source index maps E### to UUID; identity provenance explicitly recorded
   ids={s['episode_uuid'] for s in segments if s['dataset']==source and s['sample']==folder.name}
   row['episode_uuid']=next(iter(ids)) if len(ids)==1 else ''
   row['identity_basis']='source_segment_index sample mapping; internal metadata checked separately'
   rawsegments[row['episode_uuid']]=[(float(s['start_time']),float(s['end_time'])) for s in ss]
   features=json.loads((folder/'meta/info.json').read_text())['features']
   for key in ['observation.state.hand_left_world','observation.state.hand_right_world','observation.state.hand_left_world_rotation','observation.state.hand_right_world_rotation','observation.state.head_world','observation.state.head_left_camera_position','observation.state.head_right_camera_position']:
    if key not in features:missing.append(key)
   for v in ['head_left','head_right']:
    if not list((folder/'videos'/('observation.images.'+v)).rglob('*.mp4')):missing.append(v+' video')
  else:
   meta={x['field']:x['value'] for x in readcsv(folder/'session_metadata.csv')}
   row['episode_uuid']=meta.get('task_info.episode_uuid','');row['environment_id']=meta.get('task_info.environment_id','');row['identity_basis']='internal session_metadata.csv'
   needed=['pose_left_hand','pose_right_hand','pose_head','pose_headcam','pose_right_eye_cam','annotation_bad_frame_pose_hand','annotation_semantic_segments','sensor_camera_head_left_intrinsic','sensor_camera_head_right_intrinsic','sensor_camera_head_left_extrinsic','sensor_camera_head_right_extrinsic']
   for name in needed:
    f=folder/'tables'/(name+'.csv')
    if not f.is_file():missing.append(str(f.relative_to(folder)));continue
    vals=readcsv(f);schemas.append({'dataset':source,'folder':folder.name,'file':str(f.relative_to(folder)),'rows':len(vals),'columns':json.dumps(list(vals[0]) if vals else [])})
    if name=='pose_left_hand':row['timestamps']=len(vals)
    if name=='annotation_semantic_segments':
     row['segments']=len(vals);rawsegments[row['episode_uuid']]=[(float(x['segment.start_time']),float(x['segment.end_time'])) for x in vals]
   for v in ['head_left','head_right']+(['left_wrist','right_wrist'] if source=='EgoPro' else []):
    for suffix in ['.mp4','_frames.csv']:
     if not (folder/'videos'/(v+suffix)).is_file():missing.append(v+suffix)
  official=byid.get(row['episode_uuid'])
  if official:
   row['split']=official['split']
   if row['environment_id'] and row['environment_id']!=official['environment_id']:issues.append('environment mismatch '+folder.name)
   row['environment_id']=official['environment_id']
  else:issues.append('unmapped UUID '+folder.name)
  row['missing_files']=';'.join(missing)
  if missing:issues.append('missing channels '+folder.name+': '+str(missing))
  records.append(row)
# keep metadata separate as not all rows share it
metadata=[{'folder':r['folder'],'metadata':r.pop('_metadata','')} for r in records if r.get('_metadata')]
save('demo_episode_metadata.csv',metadata);save('records_completeness.csv',records);save('channel_schemas.csv',schemas)
checks=[]
for s in segments:
 uuid=s['episode_uuid'];i=int(s['source_segment_index']);pairs=rawsegments.get(uuid,[]);ok=i<len(pairs) and abs(pairs[i][0]-float(s['start_s']))<1e-5 and abs(pairs[i][1]-float(s['end_s']))<1e-5
 checks.append({'unit_id':s['unit_id'],'episode_uuid':uuid,'index':i,'match_raw_annotation':ok,'eligible':s['eligible']})
save('segment_matches.csv',checks)
ids=[r['episode_uuid'] for r in records];envsets={k:{r['environment_id'] for r in records if r['split']==k} for k in ['train','validation','test']}
# determine actual split spellings directly
splitcounts=collections.Counter(x['split'] for x in split)
summary={'records':len(records),'unique_uuid':len(set(ids)),'missing_official_uuid':sorted(set(byid)-set(ids)),'duplicates':[x for x,c in collections.Counter(ids).items() if c>1],'official_split_record_counts':dict(splitcounts),'official_environment_counts':{k:len({r['environment_id'] for r in split if r['split']==k}) for k in splitcounts},'source_counts':dict(collections.Counter(r['dataset'] for r in records)),'timestamps_by_source':{k:sum(r['timestamps'] for r in records if r['dataset']==k) for k in ['EgoDemo','EgoPro','EgoStandard']},'segments_raw':sum(r['segments'] for r in records),'segments_index':len(segments),'eligible':sum(s['eligible']=='1' for s in segments),'segment_mismatches':sum(not c['match_raw_annotation'] for c in checks),'issues':issues}
summary['environment_cross_split']=[e for e in {x['environment_id'] for x in split} if len({x['split'] for x in split if x['environment_id']==e})>1]
(root/'数据审计/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False,indent=2))
