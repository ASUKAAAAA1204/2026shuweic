"""Q2 S0 audit and initial fixed-policy replay. Never loads test trajectories."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
Q1=ROOT/'分支测试/q1-ding/q1'
DATA=Q1/'C_Problem_Data/v1/v1'
OUT=Path(__file__).resolve().parents[1]/'results/S0'

def replay(t,x,i,j,method,delay,length):
    last_t=float(t[i-1]);last=x[i-1].copy();v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2]);pending=None;count=0;used=set();rows=[]
    for k in range(i,j):
        rel=float(t[k]-t[i]);now=float(t[k])
        if rel>length+1e-9:break
        if pending is not None and now+1e-12>=pending[0]:
            _,sample_t,value=pending
            v=(value-last)/(sample_t-last_t);last=value;last_t=sample_t;pending=None
        if method=='fixed' and pending is None and count<2:
            due=[a for a in (.3,.6) if a not in used and rel+1e-12>=a]
            if due and rel+delay<=length+1e-12:
                used.add(due[0]);count+=1;pending=(now+delay,now,x[k].copy())
                if delay==0:
                    v=(pending[2]-last)/(now-last_t);last=pending[2];last_t=now;pending=None
        age=now-last_t
        factor=0 if method=='hold' else age if method=='cv' else -.2*np.expm1(-age/.2)
        pred=last+factor*v
        rows.append((rel,float(np.linalg.norm(pred-x[k])),count))
    a=np.asarray(rows);assert len(a)>0 and np.isfinite(a).all() and count<=2
    return dict(ade=float(a[:,1].mean()),fde=float(a[-1,1]),p95=float(np.quantile(a[:,1],.95)),requests=count,waiting_s=count*delay,points=len(a))

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    records=pd.read_csv(DATA/'record_index.csv');official=pd.read_csv(Q1/'C_Problem_Data/record_split.csv')
    assert len(records)==69 and records.episode_uuid.is_unique
    check=records.merge(official,on='episode_uuid',suffixes=('','_official'),validate='one_to_one')
    for col in ['split','environment_id']:assert (check[col]==check[col+'_official']).all()
    assert records.groupby('environment_id').split.nunique().max()==1
    windows=pd.read_csv(Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv')
    windows=windows[windows.split.isin(['train','validation'])].copy()
    assert windows.window_id.is_unique
    hashes=[];bad_runs=[];scales={};cache={};retention=[]
    for r in records[records.split!='test'].itertuples():
        path=DATA/'records'/f'{r.episode_uuid}.npz'
        hashes.append(dict(episode_uuid=r.episode_uuid,split=r.split,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with np.load(path,allow_pickle=False) as z:
            head=z['head__position'][:,0]
            for side in ['left_hand','right_hand']:
                t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);valid=z[side+'__reference_candidate'];bad=z[side+'__bad'].astype(bool)
                cache[(r.episode_uuid,side)]=(t,x,valid)
                if r.split=='train':
                    scales[(r.episode_uuid,side)]=(r.dataset,r.environment_id,np.linalg.norm(x[valid]-head[valid],axis=1))
                    edges=np.diff(np.r_[False,bad,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
                    for a,b in zip(starts,ends):
                        bad_runs.append(dict(episode_uuid=r.episode_uuid,dataset=r.dataset,environment_id=r.environment_id,side=side,start_s=t[a],last_bad_s=t[b-1],duration_to_recovery_s=float(t[b]-t[a]) if b<len(t) else float(t[b-1]-t[a]),left_censored=bool(a==0),right_censored=bool(b==len(t)),bad_frames=b-a))
                retention.append(dict(episode_uuid=r.episode_uuid,dataset=r.dataset,environment_id=r.environment_id,split=r.split,side=side,frames=len(t),reference_frames=int(valid.sum()),bad_frames=int(bad.sum())))
    sources=sorted(records.dataset.unique());envs=sorted(records.loc[records.split=='train','environment_id'].unique())
    scale_table={}
    for held in ['development',*envs]:
        scale_table[held]={s:float(np.median(np.concatenate([v[2] for v in scales.values() if v[0]==s and v[1]!=held]))) for s in sources}
    results=[]
    for w in windows.itertuples():
        t,x,valid=cache[(w.episode_uuid,w.side)];i,j=w.start_index,w.end_index_exclusive
        assert i-w.history_start>=2 and valid[w.history_start:j].all() and np.all(np.diff(t[w.history_start:j])>0)
        if w.split!='train':continue  # development execution awaits frozen selection
        for length in [.2,.6,1.]:
            for method,delays in [('hold',[0]),('cv',[0]),('damped',[0]),('fixed',[0,.1,.2])]:
                for delay in delays:
                    result=replay(t,x,i,j,method,delay,length)
                    result.update(window_id=w.window_id,episode_uuid=w.episode_uuid,environment_id=w.environment_id,dataset=w.dataset,side=w.side,length_s=length,method=method,delay_s=delay)
                    result['nade']=result['ade']/scale_table[w.environment_id][w.dataset];results.append(result)
    frame=pd.DataFrame(results);frame.to_csv(OUT/'train_initial_replay.csv',index=False)
    group=['length_s','method','delay_s'];g=frame.groupby(group+['dataset','environment_id','episode_uuid','side'])[['nade','requests','waiting_s']].mean()
    for keys in [group+['dataset','environment_id','episode_uuid'],group+['dataset','environment_id'],group+['dataset'],group]:g=g.groupby(level=keys).mean()
    g.reset_index().to_csv(OUT/'train_initial_summary.csv',index=False)
    pd.DataFrame(hashes).to_csv(OUT/'input_hashes.csv',index=False);pd.DataFrame(bad_runs).to_csv(OUT/'train_natural_bad_runs.csv',index=False);pd.DataFrame(retention).to_csv(OUT/'retention.csv',index=False)
    windows.groupby(['split','dataset']).agg(windows=('window_id','size'),records=('episode_uuid','nunique'),environments=('environment_id','nunique')).to_csv(OUT/'window_counts.csv')
    (OUT/'fold_scales.json').write_text(json.dumps(scale_table,indent=2),encoding='utf8')
    audit=dict(record_index=69,numerical_records_read=len(hashes),test_trajectory_reads=0,development_replay_rows=0,train_windows=int((windows.split=='train').sum()),development_windows=int((windows.split=='validation').sum()),train_environments=len(envs),qualified_windows_checked=len(windows),initial_replay_rows=len(frame),budget_violations=int((frame.requests>2).sum()),natural_bad_runs=len(bad_runs),elapsed_s=time.perf_counter()-start,status='S0 initial diagnostics; engine not yet independently certified',scale_formula='median distance of valid hand centroid to head position; pool both hands and training timestamps per source excluding global held environment')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit),flush=True);print(g.to_string())
if __name__=='__main__':main()
