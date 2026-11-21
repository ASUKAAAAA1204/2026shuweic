from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from recovery_interface import RecoveryTracker
from recovery_origin_interface import OriginRecoveryTracker
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE.parent/'第二问求解/code'))
import s0_audit_replay as s0

def main():
    checks=[]
    s=OriginRecoveryTracker();assert s.origin=='before_any_invalid'
    s.receive(0,[0,0,0],True,0);s.receive(.03,None,False,.03)
    assert s.origin=='after_observed_invalid';checks.append('origin changes only after invalid receipt')
    for args in [(.03,None,False,.04),(.02,None,False,.04),(.04,[float('nan'),0,0],True,.04),(.04,None,False,float('nan'))]:
        before=(s.last_arrival,s.last_received_sample_time,s.origin,s.predict(.04))
        try:s.receive(*args)
        except ValueError:pass
        else:raise AssertionError('illegal input accepted')
        assert before==(s.last_arrival,s.last_received_sample_time,s.origin,s.predict(.04))
    checks.append('four rejected inputs leave state unchanged')
    s.receive(.04,[1,0,0],True,.05)
    assert s.origin=='after_observed_invalid' and s.predict(.05).mode=='recovered_single'
    try:s.predict(.045)
    except ValueError:pass
    else:raise AssertionError('historical prediction used later information')
    checks.append('arrival boundary enforced')
    labels=pd.read_csv(BASE/'results/initialization_split/labeled_residuals.csv.gz')
    expected={(r.episode_uuid,r.side,r.index):r.origin for r in labels.itertuples()}
    rec=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'")
    n=matched=0;maxdiff=0.
    for r in rec.itertuples():
        with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
            for side in ['left_hand','right_hand']:
                t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);good=z[side+'__reference_candidate']
                gap=3*float(np.median(np.diff(t)));old=RecoveryTracker(gap);new=OriginRecoveryTracker(gap)
                for k,now in enumerate(t):
                    a=old.predict(float(now));b=new.predict(float(now));assert a.mode==b.mode
                    if a.position is not None:
                        d=float(np.max(np.abs(np.asarray(a.position)-np.asarray(b.position))));maxdiff=max(maxdiff,d);assert d<1e-12
                    key=(r.episode_uuid,side,k)
                    if key in expected:assert new.origin==expected[key];matched+=1
                    payload=x[k] if good[k] else None
                    for tracker in [old,new]:tracker.receive(float(now),payload,bool(good[k]),float(now))
                    n+=1
    assert matched==len(labels)
    out=BASE/'results/origin_interface';out.mkdir(exist_ok=True)
    audit=dict(passed=True,checks=checks,train_records=len(rec),hand_timestamps=n,matched_origin_labels=matched,max_prediction_difference=maxdiff,development_reads=0,test_reads=0)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
