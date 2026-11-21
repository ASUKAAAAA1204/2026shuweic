"""Read-only reconciliation of original Q3 rules and processed source units."""
from pathlib import Path
import hashlib,json
import pandas as pd
import pymupdf
BASE=Path(__file__).resolve().parents[1];WORK=BASE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=BASE/'audit';out.mkdir(parents=True,exist_ok=True);cfg=WORK/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints';pdf=cfg/'Problem_3_Requirements.pdf'
 doc=pymupdf.open(pdf);text='\n'.join(line.rstrip() for page in doc for line in page.get_text().splitlines());(out/'official_requirements_extracted.txt').write_text(text,encoding='utf8')
 source=pd.read_csv(cfg/'source_segment_index.csv');processed=pd.read_csv(WORK/'处理结果/v1/segments_private.csv');tasks=pd.read_csv(cfg/'tasks.csv');split=pd.read_csv(cfg/'record_split.csv');conditions=json.loads((cfg/'common_conditions.json').read_text())
 assert len(source)==349 and len(processed)==349 and len(tasks)==36
 assert int(processed.eligible.sum())==343;assert processed.unit_id.is_unique
 # Authoritative original row order must match processed order used by historical replay.
 keys=['episode_uuid','source_segment_index'];assert list(source[keys].itertuples(index=False,name=None))==list(processed[keys].itertuples(index=False,name=None))
 assert ((processed.end_s-processed.start_s-processed.duration_s).abs()<1e-9).all()
 assert (source.eligible.astype(int).to_numpy()==processed.eligible.astype(int).to_numpy()).all()
 for case,g in tasks.groupby('case_id'):
  ids=set(g.job_id)
  for r in g.itertuples():
   parents=[] if pd.isna(r.predecessors) else str(r.predecessors).split('|');assert set(parents)<=ids and r.job_id not in parents
  # Acyclicity checked without assuming IDs are topologically ordered.
  done=set()
  while len(done)<len(ids):
   ready=[r.job_id for r in g.itertuples() if r.job_id not in done and (pd.isna(r.predecessors) or set(str(r.predecessors).split('|'))<=done)]
   assert ready,'cycle';done.update(ready)
 pools=processed[processed.eligible==1].groupby(['split','family_id']).agg(units=('unit_id','size'),records=('episode_uuid','nunique'),environments=('environment_id','nunique'));pools.to_csv(out/'source_pool_support.csv')
 record_split=processed.groupby('environment_id').split.nunique();assert record_split.max()==1
 operators={name:set(g.operator_id.dropna()) for name,g in split.groupby('split')};overlap={a+'__'+b:len(operators[a]&operators[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
 (out/'operator_overlap.json').write_text(json.dumps(dict(global_environment_disjoint=True,operator_overlap_counts=overlap,unseen_operator_generalization_claim=False),indent=2),encoding='utf8')
 missing=[]
 for held in sorted(processed[processed.split=='train'].environment_id.unique()):
  available=set(processed[(processed.split=='train')&(processed.environment_id==held)&(processed.eligible==1)].family_id)
  for case,g in tasks.groupby('case_id'):
   required=set(g.family_id)
   if not required<=available:missing.append(dict(environment_id=held,case_id=case,missing_families=sorted(required-available)))
 (out/'ineligible_environment_cases.json').write_text(json.dumps(missing,indent=2),encoding='utf8')
 result=dict(passed=True,official_pdf_available=True,official_pdf_sha256=sha(pdf),source_units=349,eligible_units=343,original_pool_order_matches_processed=True,split_global_environments=processed.groupby('split').environment_id.nunique().to_dict(),case_jobs=tasks.groupby('case_id').size().to_dict(),ineligible_training_environment_cases=len(missing),validation_seeds=conditions['validation_seeds'],official_test_seeds_available=False,files={p.name:sha(p) for p in cfg.iterdir() if p.is_file()})
 (out/'protocol_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ['files','validation_seeds']},ensure_ascii=False));print(pools.to_string())
if __name__=='__main__':main()
