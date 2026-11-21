"""Re-evaluate approved Q1 models on the provided neutral window and query manifest."""
import argparse
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd
from q1 import ROOT, DATA, KEYS, aggregate
from q1_approved import geometric_median, damping, TAUS
from approved_filters import filtered_velocities

OUT = ROOT/'results/q1_neutral_preprocessing'
HORIZONS = (.2,.6,1.)
GRID = {'A':[None], 'B':[None], 'C':[(f,0) for f in (.5,1,2,4)],
        'D':[(f,k) for f in (.5,1,2,4) for k in (.5,1,2)],
        'H1':list(TAUS), 'H2':[(s,t) for s in TAUS for t in TAUS]}


def candidates(w,m,scale):
    t,x,v,b,head,age = w
    a = damping(age,.2)
    if m=='A': return [a*v]
    if m=='B': return [a*b]
    if m in ('C','D'):
        velocities = filtered_velocities(t,x,scale)
        return [a*velocities[k] for k in GRID[m]]
    if m=='H1': return [damping(age,s)*v for s in TAUS]
    return [damping(age,s)*head+damping(age,r)*(v-head) for s,r in GRID[m]]


def read(z,name):
    with z.open('第一问中立同窗/'+name) as f:
        return pd.read_csv(f,float_precision='round_trip')


def run(path):
    with ZipFile(path) as z:
        manifest=read(z,'data/windows.csv')
        queries=read(z,'queries.csv')
        protocol=json.loads(z.read('第一问中立同窗/summary.json'))
    assert manifest.window_id.is_unique and not queries.duplicated(['window_id','frame_index']).any()
    official=pd.read_csv(next(DATA.parents[1].rglob('record_split.csv'))).set_index('episode_uuid')
    assert all(np.array_equal(official.loc[manifest.episode_uuid,k],manifest[k]) for k in ['environment_id','split'])
    assert manifest.groupby('environment_id').split.nunique().max()==1
    query={k:g for k,g in queries.groupby('window_id')}
    assert set(query)==set(manifest.window_id)
    metadata=json.loads((DATA/'q1_feature_names.json').read_text(encoding='utf8'))
    obs,targets={},{}
    for (uid,side),group in manifest.groupby(['episode_uuid','side']):
        features=metadata[uid]['metadata'].get('info',{}).get('features',{})
        common=all(features.get('observation.state.'+k+'_world',{}).get('names',[[]])[-1]==['x','y','z'] for k in ('hand_left','hand_right','head'))
        with np.load(DATA/'records'/f'{uid}.npz',allow_pickle=False) as z:
            time=z[side+'__t_s']; joints=z[side+'__position']; head=z['head__position'][:,0]
            ht=z['head__t_s']; hg=z['head__finite'] & z['head__quaternion_unit_1e3'] & z['head__time_increasing']
            gap=z['head__gap_before']
            for w in group.itertuples():
                i,j,lo=w.start_index,w.end_index_exclusive,w.history_start
                t=time[lo:i]; x=joints[lo:i].mean(axis=1); dt=t[-1]-t[-2]
                q=query[w.window_id]
                np.testing.assert_array_equal(q.frame_index,np.arange(i,j))
                np.testing.assert_allclose(q.age_from_anchor_s,time[i:j]-t[-1],atol=1e-12,rtol=0)
                np.testing.assert_allclose(q.timestamp_s,time[i:j],atol=1e-12,rtol=0)
                v=(x[-1]-x[-2])/dt
                joint_v=(joints[i-1]-joints[i-2])/dt
                np.testing.assert_allclose(v,joint_v.mean(0),atol=1e-10)
                vh=None
                if common and np.array_equal(ht[i-2:i],t[-2:]) and hg[i-2:i].all() and not gap[i-1]:
                    vh=(head[i-1]-head[i-2])/dt
                age=q.age_from_anchor_s.to_numpy()
                obs[w.window_id]=(t,x,v,geometric_median(joint_v),vh,age)
                if w.split=='train':
                    targets[w.window_id]=joints[i:j].mean(axis=1)-x[-1]
    OUT.mkdir(exist_ok=True,parents=True)
    pred=[]; params=[]
    for src,g in manifest.groupby('dataset'):
        train=g[g.split=='train']
        test=g[g.split=='test']
        speed=np.array([np.linalg.norm(obs[k][2]) for k in train.window_id])
        scale=np.median(speed[speed>0])
        excursion=np.median([np.linalg.norm(targets[k][-1]) for k in train.window_id])
        np.testing.assert_allclose(excursion,protocol['scale_by_source_training_only'][src],rtol=1e-10)
        for m,grid in GRID.items():
            ids=train if not m.startswith('H') else train[train.window_id.map(lambda k:obs[k][4] is not None)]
            if ids.empty: continue
            weights=1/(ids.groupby('episode_uuid').window_id.transform('size')*ids.groupby('environment_id').episode_uuid.transform('nunique'))
            loss=[]
            for k in ids.window_id:
                w=obs[k]
                error=np.linalg.norm(np.array(candidates(w,m,scale))-targets[k],axis=-1)
                loss.append(np.mean([error[:,w[-1]<=h+1e-9].mean(axis=1) for h in HORIZONS],axis=0))
            pick=np.argmin(np.average(loss,axis=0,weights=weights))
            params.append(dict(dataset=src,model=m,parameter=str(grid[pick]),speed_scale=scale))
            for k in test.window_id:
                w=obs[k]
                if m.startswith('H') and w[4] is None: continue
                pos=candidates(w,m,scale)[pick]+w[1][-1]
                if m=='H2':
                    np.testing.assert_allclose(candidates(w,m,scale)[grid.index((.2,.2))],candidates(w,'A',scale)[0],atol=1e-12)
                pred.append(pd.DataFrame(dict(window_id=k,model=m,frame_index=query[k].frame_index.to_numpy(),x=pos[:,0],y=pos[:,1],z=pos[:,2])))
    pred=pd.concat(pred,ignore_index=True)
    pred.to_csv(OUT/'predictions.csv.gz',index=False)
    pd.DataFrame(params).to_csv(OUT/'parameters.csv',index=False)
    with ZipFile(path) as z:
        ref=read(z,'test_reference.csv')
        archived=read(z,'local_test_ade.csv')
    test=manifest.query("split=='test'")
    keys=['window_id','frame_index']
    assert len(ref)==protocol['test_frames'] and not ref.duplicated(keys).any()
    testqueries=queries[queries.window_id.isin(test.window_id)]
    ages=testqueries.merge(ref,on=keys,validate='one_to_one')
    assert len(ages)==len(ref)==len(testqueries)
    np.testing.assert_allclose(ages.age_from_anchor_s_x,ages.age_from_anchor_s_y,atol=1e-12)
    scored=pred.merge(ref,on=keys,validate='many_to_one',suffixes=('','_true'))
    assert len(scored)==len(pred)
    for m,g in scored.groupby('model'):
        wanted=testqueries
        if m.startswith('H'): wanted=wanted[wanted.window_id.map(lambda k:obs[k][4] is not None)]
        assert len(g)==len(wanted) and not g.duplicated(keys).any()
    scored['error']=np.linalg.norm(scored[['x','y','z']].values-scored[['x_true','y_true','z_true']].values,axis=1)
    rows=[]
    for h in HORIZONS:
        subset=scored[scored.age_from_anchor_s<=h+1e-9].sort_values('frame_index')
        rows.append(subset.groupby(['window_id','model']).error.agg(ade='mean',fde='last').reset_index().assign(horizon_s=h))
    parts=[pd.concat(rows)]
    for m,col in [('MLP_gate','mlp_gate_ade'),('residual_MLP','conditional_residual_mlp_ade')]:
        parts.append(archived[['window_id','horizon_s',col]].rename(columns={col:'ade'}).assign(model=m))
    frame=pd.concat(parts).merge(test[KEYS],on='window_id',validate='many_to_one')
    frame['norm_ade']=frame.ade/frame.dataset.map(protocol['scale_by_source_training_only'])
    frame.to_csv(OUT/'windows.csv',index=False)
    summary,env=aggregate(frame,['dataset','horizon_s','model'],['ade','fde','norm_ade'])
    summary.to_csv(OUT/'summary.csv',index=False)
    env.to_csv(OUT/'environments.csv',index=False)
    for scope,p in [('all',summary[~summary.model.isin(['H1','H2'])]),('EgoDemo',summary.query("dataset=='EgoDemo'"))]:
        table=p.groupby('model').norm_ade.mean().to_frame()
        table['change_vs_A_pct']=100*(table.norm_ade/table.loc['A','norm_ade']-1)
        table.to_csv(OUT/(scope+'.csv'));print(scope,table)
    wide=env.pivot(index=['dataset','environment_id','horizon_s'],columns='model',values='norm_ade')
    for m,b in [('B','A'),('C','A'),('D','A'),('D','C'),('H2','H1')]:
        wide[m+'_minus_'+b]=wide[m]-wide[b]
    wide.to_csv(OUT/'paired_environments.csv')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('archive')
    run(parser.parse_args().archive)
