from pathlib import Path
import sys,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
import s0_audit_replay as s0

def summarize(frame,factor,metrics):
 g=frame.copy()
 for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:g=g.groupby([factor,'method']+lev,as_index=False,dropna=False)[metrics].mean()
 return g

def main():
 out=BASE/'results/stage_diagnostic';out.mkdir(exist_ok=True)
 windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'")
 seg=pd.read_csv(s0.DATA/'segments_private.csv').query("split=='train'")
 index=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'");cache={};labels=[];time_rows=[]
 for w in windows.itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=z[w.side+'__t_s']
   t=cache[key];s=seg[seg.episode_uuid==w.episode_uuid];time_rows.append(dict(episode_uuid=w.episode_uuid,side=w.side,t_min=float(t[0]),t_max=float(t[-1]),semantic_min=float(s.start_s.min()),semantic_max=float(s.end_s.max()),time_basis=index.set_index('episode_uuid').loc[w.episode_uuid,'time_basis']))
  t=cache[key];start=float(t[w.start_index]);end=float(t[w.end_index_exclusive-1]);assert abs(start-w.start_s)<1e-6
  s=seg[(seg.episode_uuid==w.episode_uuid)&(seg.start_s<=start)&(start<seg.end_s)]
  label=dict(window_id=w.window_id,dataset=w.dataset,environment_id=w.environment_id,episode_uuid=w.episode_uuid,side=w.side,start_s=start,last_score_s=end,matches=len(s),family='unknown',progress_third='unknown',boundary='unknown',source_segment_index=-1)
  if len(s)==1:
   r=s.iloc[0];progress=(start-r.start_s)/(r.end_s-r.start_s);assert 0<=progress<1
   label.update(family=r.family_id,task=r.task,source_segment_index=int(r.source_segment_index),semantic_start=r.start_s,semantic_end=r.end_s,progress=progress,progress_third=['early','middle','late'][min(2,int(progress*3))],boundary='crosses_end' if end>=r.end_s else 'internal',third_question_eligible=int(r.eligible))
  labels.append(label)
 lab=pd.DataFrame(labels);lab.to_csv(out/'labels.csv',index=False);pd.DataFrame(time_rows).to_csv(out/'time_alignment.csv',index=False)
 m=pd.read_csv(OLD/'results/S3/update_windows.csv');m=m[m.method.isin(['open','secant'])]
 merged=m.merge(lab[['window_id','family','progress_third','boundary','matches']],on='window_id',validate='many_to_one');assert len(merged)==len(windows)*2
 metrics=['nade','nfde','exceed_0.05','requests','waiting_s']
 for factor in ['family','progress_third','boundary']:
  counts=lab.groupby([factor,'dataset'],dropna=False).agg(windows=('window_id','size'),records=('episode_uuid','nunique'),environments=('environment_id','nunique')).reset_index();counts.to_csv(out/(factor+'_coverage.csv'),index=False)
  summarize(merged,factor,metrics).to_csv(out/(factor+'_summary.csv'),index=False)
  crossed=lab.groupby([factor,'family','dataset','environment_id'],dropna=False).size() if factor!='family' else lab.groupby([factor,'dataset','environment_id'],dropna=False).size()
  crossed.to_csv(out/(factor+'_composition.csv'))
  # paired per-window change, then same hierarchy; never compare unmatched populations
  pivot=merged.pivot(index='window_id',columns='method',values='nade');pair=lab.copy();pair['method']='paired';pair['open_nade']=pair.window_id.map(pivot.open);pair['secant_nade']=pair.window_id.map(pivot.secant);pair['absolute_reduction']=pair.open_nade-pair.secant_nade
  res=summarize(pair,factor,['open_nade','secant_nade','absolute_reduction']);res['gain_pct']=100*res.absolute_reduction/res.open_nade;res.to_csv(out/(factor+'_benefit.csv'),index=False)
 audit=dict(passed=True,train_windows=len(windows),matched_unique=int((lab.matches==1).sum()),unmatched=int((lab.matches==0).sum()),ambiguous=int((lab.matches>1).sum()),train_segments=len(seg),train_records=len(index),semantic_interpretation='offline annotated interval progression, not physical reach/grasp/place labels',development_reads=0,test_reads=0,policy_changed=False)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
 for factor in ['family','progress_third','boundary']:print(factor);print(pd.read_csv(out/(factor+'_summary.csv')).to_string(index=False))
if __name__=='__main__':main()
