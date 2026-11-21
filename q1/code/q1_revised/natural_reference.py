"""Independent recovery-event key/geometry check using only preceding measured samples."""
import numpy as np
import pandas as pd
from audit_position import DATA,OUT,save

def main():
    seg=pd.read_csv(DATA/"segments_private.csv")
    idx=pd.read_csv(DATA/"record_index.csv").query("split!='test'").set_index("episode_uuid")
    intervals=pd.read_csv(DATA/"natural_unusable_intervals.csv").query("split!='test'")
    out=[]
    for (uid,side),b in intervals.groupby(["episode_uuid","channel"]):
        rec=idx.loc[uid];ann=seg[seg.episode_uuid==uid]
        with np.load(DATA/"records"/f"{uid}.npz") as z:
            t,x,gap,good=[z[side+"__"+k] for k in ("t_s","position","gap_before","reference_candidate")]
        x=x.mean(axis=1);ids=np.flatnonzero(good)
        vel=np.zeros_like(x)
        for old,new in zip(ids[:-1],ids[1:]):
            dt=t[new]-t[old]
            vel[new]=(x[new]-x[old])/dt if new==old+1 and not gap[new] else vel[old]*np.exp(-dt/.2)
        for r in b.itertuples():
            start,end=r.start_index,r.end_index_exclusive
            assert not good[start:end].any() and (end==len(good) or good[end])
            for phase,i in [("pre_first",end),("pre_second",end+1)]:
                valid=i<len(t) and good[i] and (phase=="pre_first" or not gap[i])
                prior=ids[ids<i];age=np.nan;decay=np.nan;zero=np.nan
                if valid and len(prior):
                    j=prior[-1];age=t[i]-t[j]
                    d=x[i]-x[j]
                    decay=np.linalg.norm(d+.2*np.expm1(-age/.2)*vel[j])
                    zero=np.linalg.norm(d)
                ss=ann[(ann.start_s<=t[start])&(ann.end_s>t[start])]
                family,stage="unannotated","unannotated"
                if len(ss)==1:
                    s=ss.iloc[0];family=s.family_id
                    stage=["early","middle","late"][min(2,int(3*(t[start]-s.start_s)/s.duration_s))]
                out.append(dict(record=uid,side=side,source=rec.dataset,env=rec.environment_id,split=rec.split,
                           start=start,stop=end,phase=phase,frame=i,valid=valid,age=age,decay_error=decay,
                           zero_displacement_error=zero,family=family,stage=stage))
    save(pd.DataFrame(out),"natural_reference.csv")
if __name__=="__main__":main()
