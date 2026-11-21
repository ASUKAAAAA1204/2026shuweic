from pathlib import Path
import json
import pandas as pd
from engine import Engine,independent_metrics
from baseline import action
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

def load():
    cfg=WORK/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints'
    cond=json.loads((cfg/'common_conditions.json').read_text())
    tasks=pd.read_csv(cfg/'tasks.csv').fillna('');z=pd.read_csv(cfg/'service_zones.csv')
    zones={r.zone_id:[r.x,r.y] for r in z.itertuples()}
    training=pd.DataFrame(json.loads((ROOT/'model/training_family_samples.json').read_text()))
    return cond,tasks,zones,training

def jobs_for(tasks,case):
    return [dict(job_id=r.job_id,family_id=r.family_id,release_s=float(r.release_s),due_s=float(r.due_s),priority_weight=float(r.priority_weight),zone_id=r.zone_id,predecessors=r.predecessors.split('|') if r.predecessors else []) for r in tasks[tasks.case_id==case].sort_values('job_id').itertuples()]
