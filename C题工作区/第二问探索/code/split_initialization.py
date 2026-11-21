from pathlib import Path
import sys,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'));import s0_audit_replay as s0

def main():
 out=BASE/'results/initialization_split';out.mkdir(exist_ok=True);p=pd.read_csv(BASE/'results/causal_calibration/raw_residuals.csv.gz');rec=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'");rows=[]
 for r in rec.itertuples():
  with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    good=z[side+'__reference_candidate'];received_bad=False
    for k,valid in enumerate(good):
     # Capture history before observing current validity; future change cannot affect label.
     rows.append(dict(episode_uuid=r.episode_uuid,side=side,index=k,past_invalid_seen=received_bad))
     if not valid:received_bad=True
 labels=pd.DataFrame(rows);a=p.merge(labels,on=['episode_uuid','side','index'],validate='one_to_one');a['origin']=np.where(a.past_invalid_seen,'after_observed_invalid','before_any_invalid');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());a['nade']=[e/sc[env][s] for e,env,s in zip(a.raw_error,a.environment_id,a.dataset)];a['exceed05']=(a.nade>.05).astype(float);a.to_csv(out/'labeled_residuals.csv.gz',index=False)
 counts=a.groupby(['mode','origin']).agg(points=('index','size'),records=('episode_uuid','nunique'),environments=('environment_id','nunique')).reset_index();counts.to_csv(out/'counts.csv',index=False)
 g=a.groupby(['mode','origin','dataset','environment_id','episode_uuid','side'],as_index=False)[['nade','exceed05']].mean()
 for keys in [['mode','origin','dataset','environment_id','episode_uuid'],['mode','origin','dataset','environment_id'],['mode','origin','dataset'],['mode','origin']]:g=g.groupby(keys,as_index=False)[['nade','exceed05']].mean()
 g.to_csv(out/'summary.csv',index=False)
 # Saved causal calibration coverage broken out by prior-history origin; no new fit/selection.
 old=pd.read_csv(BASE/'results/causal_calibration/oof_points.csv.gz');b=old.merge(labels,on=['episode_uuid','side','index'],validate='many_to_one');b['origin']=np.where(b.past_invalid_seen,'after_observed_invalid','before_any_invalid')
 v=b.groupby(['method','mode','origin','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball']].mean()
 for keys in [['method','mode','origin','dataset','environment_id','episode_uuid'],['method','mode','origin','dataset','environment_id'],['method','mode','origin','dataset'],['method','mode','origin']]:v=v.groupby(keys,as_index=False)[['coverage','width','pinball']].mean()
 v.to_csv(out/'existing_calibration_split.csv',index=False)
 audit=dict(passed=True,train_records=len(rec),label_rows=len(labels),residual_rows=len(a),current_validity_excluded_from_label=True,future_validity_excluded_from_label=True,model_refitted=False,test_reads=0,development_reads=0,warning='past-invalid marker separates early initialization but later tracking may be long after any invalid; do not call all post-invalid frames immediate recovery')
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(counts.to_string(index=False));print(g.to_string(index=False));print(v.query("mode=='recovered_single'").to_string(index=False))
if __name__=='__main__':main()
