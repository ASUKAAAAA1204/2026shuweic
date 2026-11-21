"""Finalize S0 protocol and natural quality descriptive evidence, training only."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parents[1];OUT=BASE/'results/S0'

def main():
 p=dict(version='q2-20261121-v1',q1_commit='bf380c4',state='per-hand centroid of 21 points',motion='tau=0.2 damped; continuous last-two initial velocity',durations_s=[.2,.6,1.],delays_s=[0,.1,.2],primary_duration_s=1.,primary_delay_s=.1,request_limit=2,request_cost=1,concurrent_requests=1,request_payload='single timestamp centroid and validity; failed payload charged and disclosed at due time',request_sample='original scoring timestamp',known_clock='original timestamp grid and fixed mask endpoint known equally to policies',arrival='exact request time + delay; process before first scoring timestamp at/after due; zero delay before scoring',terminal='allow return on last scoring point; report terminal benefit separately',missing_information='hide trajectories, derivatives and quality flags except returned payload validity',main_reference='qualified continuous windows; ideal qualified returns; natural bad intervals descriptive only',scale='source median valid hand-centroid to head distance, both hands and training timestamps pooled; exclude global held env for OOF',metrics='frame-mean ADE primary, FDE, within-window P95 (not global P95), tolerance rate, pointwise coverage, diameter, cost, pending and no-request waiting, compute',hierarchy=['window','hand','record','global_environment','source'],bootstrap_repeats=2000,seed=20261121,test_use='index only until freeze; final evaluation separate',official_conventions='missing_observation_experiments.md not found locally; numeric resource choices participant defined')
 (BASE/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
 runs=pd.read_csv(OUT/'train_natural_bad_runs.csv');ret=pd.read_csv(OUT/'retention.csv').query("split=='train'")
 summary=[]
 for source,g in runs.groupby('dataset'):
  complete=g[~g.right_censored & ~g.left_censored]
  rr=ret[ret.dataset==source]
  summary.append(dict(dataset=source,hand_bad_runs=len(g),records=g.episode_uuid.nunique(),environments=g.environment_id.nunique(),left_censored=int(g.left_censored.sum()),right_censored=int(g.right_censored.sum()),complete_runs=len(complete),complete_median_s=complete.duration_to_recovery_s.median(),complete_p90_s=complete.duration_to_recovery_s.quantile(.9),complete_max_s=complete.duration_to_recovery_s.max(),bad_frames=int(rr.bad_frames.sum()),hand_frames=int(rr.frames.sum()),bad_frame_fraction=rr.bad_frames.sum()/rr.frames.sum()))
 pd.DataFrame(summary).to_csv(OUT/'natural_duration_summary.csv',index=False)
 # Raw quality reasons are descriptive and filtered before aggregation; no trajectory test read.
 reasons=pd.read_csv(ROOT/'C题工作区/阶段交付/baseline_v2/data/quality_reasons.csv').query("split=='train'")
 reasons.groupby(['dataset','side','reason']).agg(mark_rows=('episode_uuid','size'),records=('episode_uuid','nunique')).to_csv(OUT/'natural_reason_summary.csv')
 runs['duration_bin']=pd.cut(runs.duration_to_recovery_s,[0,.2,.6,1.,float('inf')],include_lowest=True).astype(str)
 runs.groupby(['dataset','duration_bin']).size().to_csv(OUT/'natural_duration_bins.csv')
 pd.DataFrame(summary).to_markdown if False else None
 print(pd.DataFrame(summary).to_string(index=False))
 print('protocol_sha256',hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest())
if __name__=='__main__':main()
