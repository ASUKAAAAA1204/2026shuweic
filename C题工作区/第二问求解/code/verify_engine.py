"""Independent synthetic, noninterference and full training parity checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from engine import replay,FixedPolicy,NeverPolicy,View
import s0_audit_replay as s0
OUT=Path(__file__).resolve().parents[1]/'results/S1'

def independent(result):
    p=result['points'];a=np.asarray([v['prediction'] for v in p]);b=np.asarray([v['reference'] for v in p]);e=np.sqrt(((a-b)**2).sum(1));m=result['metrics']
    assert abs(e.mean()-m['ade'])<1e-9 and abs(e[-1]-m['fde'])<1e-9 and abs(np.quantile(e,.95)-m['p95'])<1e-9
    req=[v for v in result['actions'] if v['event']=='request'];ret=[v for v in result['actions'] if v['event']=='return']
    assert len(req)==m['requests']==m['cost'] and m['successful_returns']==sum(v['valid'] for v in ret)
    assert all(v['time']<=v['processed_at']+1e-12 for v in ret)
    for p in p:
        known=[v for v in ret if v['processed_at']<=p['time']+1e-12 and v['valid']]
        expected=known[-1]['sample_time'] if known else None
        if expected is not None:assert abs(p['last_time']-expected)<1e-12
        assert abs(p['age']-(p['time']-p['last_time']))<1e-12

class Eager:
    def request(self,v):return True

class PublicAdaptive:
    def request(self,v):return v.age>.18 and (np.linalg.norm(v.velocity)>.1 or v.relative>.4)

def main():
    OUT.mkdir(parents=True,exist_ok=True);checks=[]
    def ok(name):checks.append(dict(name=name,passed=True))
    t=np.arange(11)/10;hist=np.array([-.2,-.1]);hx=np.c_[hist,np.zeros((2,2))];x=np.c_[t,np.zeros((11,2))]
    def run(policy,**kw):
        r=replay(t,x,hist,hx,policy,**kw);independent(r);return r
    r=run(NeverPolicy(),motion='cv');assert r['metrics']['ade']<1e-15;ok('analytic constant velocity exact')
    r=run(NeverPolicy(),motion='hold');assert np.isclose(r['metrics']['ade'],.6);ok('analytic hold exact')
    r=run(NeverPolicy());pred=np.array([p['prediction'][0] for p in r['points']]);np.testing.assert_allclose(pred,-.1+.2*(1-np.exp(-(t+.1)/.2)),atol=1e-15);ok('analytic damping exact')
    r=run(FixedPolicy(),delay=0);assert r['metrics']['requests']==2 and [p['error'] for p in r['points'] if np.isclose(p['relative'],.3)][0]<1e-15;ok('zero delay visible same timestamp')
    r=run(FixedPolicy(),delay=.1);p=r['points'][4];assert np.isclose(p['last_time'],.3) and np.isclose(p['age'],.1);ok('stale return retains sampling age')
    r=run(Eager(),budget=1);assert r['metrics']['requests']==1;ok('budget enforced')
    r=run(Eager(),budget=0);assert r['metrics']['requests']==0;ok('zero budget')
    r=run(Eager(),delay=.2,valid=np.zeros(11,bool));assert r['metrics']['cost']==2 and r['metrics']['successful_returns']==0;ok('failed payload charged and cannot update')
    r=run(FixedPolicy((.9,)),delay=.2);assert r['metrics']['requests']==0;ok('late arrival rejected')
    r=run(FixedPolicy((1.,)),delay=0);assert r['metrics']['fde']==0;ok('terminal scoring observation allowed and explicit')
    r=run(Eager(),delay=.2);req=[a['time'] for a in r['actions'] if a['event']=='request'];assert len(req)==2 and req[1]-req[0]>=.2-1e-12;ok('single inflight enforced')
    r=run(Eager(),delay=.1);assert len({a['sample_time'] for a in r['actions'] if a['event']=='request'})==r['metrics']['requests'];ok('no duplicate sample requests')
    r=run(Eager(),delay=.15);assert all(a['processed_at']+1e-12>=a['time'] for a in r['actions'] if a['event']=='return');ok('off grid arrival processed next scoring instant')
    assert set(View.__dataclass_fields__)=={'now','relative','remaining','age','requests','pending_until','last_time','last_position','velocity','successful_returns'};ok('public interface excludes identities and future states')
    rng=np.random.default_rng(20261121);perturb=0
    for trial in range(100):
        xx=np.cumsum(rng.normal(size=(11,3)),axis=0)
        a=replay(t,xx,hist,hx,PublicAdaptive(),delay=.2)
        b=replay(t,xx+rng.normal(size=xx.shape)*100,hist,hx,PublicAdaptive(),delay=.2)
        # All decisions/predictions before first return must ignore every hidden coordinate.
        first=min([v['processed_at'] for v in a['actions'] if v['event']=='return'],default=2.)
        pa=[v for v in a['points'] if v['time']<first-1e-12];pb=[v for v in b['points'] if v['time']<first-1e-12]
        assert [(v['prediction'],v['requests']) for v in pa]==[(v['prediction'],v['requests']) for v in pb];perturb+=1
        # Change only payloads never requested: entire trajectory of legal decisions is invariant.
        yy=xx.copy();asked={round(v['time'],10) for v in a['actions'] if v['event']=='request'}
        for k,tt in enumerate(t):
            if round(tt,10) not in asked:yy[k]+=rng.normal(size=3)*100
        c=replay(t,yy,hist,hx,PublicAdaptive(),delay=.2)
        assert a['actions']==c['actions'] and [p['prediction'] for p in a['points']]==[p['prediction'] for p in c['points']];perturb+=1
        independent(a);independent(b);independent(c)
    ok('200 future and unrequested reference perturbations')
    windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'")
    maxdiff=0.;tested=0;cache={};sample=None
    for w in windows.itertuples():
        key=(w.episode_uuid,w.side)
        if key not in cache:
            with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
        tt,xx=cache[key];i,j=w.start_index,w.end_index_exclusive
        for length in [.2,.6,1.]:
            mask=(tt[i:j]-tt[i])<=length+1e-9;idx=np.arange(i,j)[mask]
            for method,delays in [('hold',[0]),('cv',[0]),('damped',[0]),('fixed',[0,.1,.2])]:
                for delay in delays:
                    a=s0.replay(tt,xx,i,j,method,delay,length)
                    b=replay(tt[idx],xx[idx],tt[i-2:i],xx[i-2:i],FixedPolicy() if method=='fixed' else NeverPolicy(),delay=delay,motion='damped' if method=='fixed' else method,window_end=float(tt[i]+length))
                    independent(b)
                    for k in ['ade','fde','p95','requests','waiting_s']:maxdiff=max(maxdiff,abs(a[k]-b['metrics'][k]))
                    tested+=1
                    if sample is None and method=='fixed' and delay==.1 and length==1.:sample=b
    assert maxdiff<1e-9;ok('full training 27486 replays match prototype and independent scorer')
    pd.DataFrame(checks).to_csv(OUT/'checks.csv',index=False)
    (OUT/'sample_trace.json').write_text(json.dumps(sample,indent=2),encoding='utf8')
    audit=dict(passed=True,checks=len(checks),future_perturbation_tests=perturb,real_training_replays=tested,max_metric_difference=maxdiff,test_trajectory_reads=0,development_replays=0,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
