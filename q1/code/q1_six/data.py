"""Training-only raw-window reconstruction. Prediction inputs and reference arrays are separate."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'C_Problem_Data/v1/v1'
OUT = ROOT / 'results/q1_six_train'
HORIZONS = (.2, .6, 1.)
EPS = 1e-9


@dataclass
class Obs:
    t: np.ndarray
    x: np.ndarray


@dataclass
class Window:
    key: dict
    obs: Obs
    ages: np.ndarray
    reference: np.ndarray


def load_windows():
    official = pd.read_csv(next((ROOT/'C_Problem_Data').rglob('record_split.csv')))
    train = official.loc[official.split.eq('train')].sort_values('episode_uuid')
    assert len(train)==train.episode_uuid.nunique()==55 and train.environment_id.nunique()==16
    assert not set(train.environment_id) & set(official.loc[official.split.ne('train'),'environment_id'])
    windows = []
    for record in train.itertuples():
        with np.load(DATA/'records'/f'{record.episode_uuid}.npz',allow_pickle=False) as z:
            for side in ('left_hand','right_hand'):
                t=z[side+'__t_s']
                x=z[side+'__position'].mean(axis=1)
                valid=z[side+'__reference_candidate']
                dt=np.diff(t)
                assert np.all(dt>0)
                step=np.median(dt)
                next_time=t[0]+.2
                for i,now in enumerate(t):
                    if now+EPS<next_time:
                        continue
                    lo=np.searchsorted(t,now-.2-EPS)
                    end=np.searchsorted(t,now+1+EPS,side='right')
                    if i-lo<1 or now-t[lo]<.2-step-EPS or t[end-1]-now<1-step-EPS:
                        continue
                    if not valid[lo:end].all() or (dt[lo:end-1]>3*step).any():
                        continue
                    key=dict(window_id=f'{record.episode_uuid}_{side}_{i}',source=record.dataset,
                             env=record.environment_id,record=record.episode_uuid,side=side,
                             start_index=i,start_s=now,history_start=lo,end_exclusive=end)
                    windows.append(Window(key,Obs(t[lo:i+1],x[lo:i+1]),t[i:end]-now,x[i:end]))
                    next_time=now+2
    return windows


def features(obs):
    """Columns used in order: speed at anchor, early/late secant speed, turn in radians."""
    t,x=obs.t,obs.x
    v=(x[-1]-x[-2])/(t[-1]-t[-2])
    speeds=np.full(4,np.nan)
    speeds[0]=np.linalg.norm(v)
    mid=np.argmin(abs(t-(t[-1]-.1)))
    if mid==0 or mid==len(t)-1 or t[-1]-t[0]<.15:
        return v,speeds,np.zeros(3),0.
    early=(x[mid]-x[0])/(t[mid]-t[0])
    late=(x[-1]-x[mid])/(t[-1]-t[mid])
    speeds[1:3]=np.linalg.norm(early),np.linalg.norm(late)
    if min(speeds[:3])<=0:
        return v,speeds,np.zeros(3),0.
    cross=np.cross(early,late)
    norm=np.linalg.norm(cross)
    speeds[3]=np.arctan2(norm,early@late)
    axis=cross/norm if norm else np.zeros(3)
    return v,speeds,axis,2*speeds[3]/(t[-1]-t[0])


def basic(v,h,model='B',c=0.):
    tau=.2
    if model=='A':
        a=tau*h/(tau+h)
    else:
        q=-np.expm1(-h/tau)
        a=tau*(q+c*q*q)
    return a[:,None]*v


def turn(v,h,axis,omega):
    """Exact integral via Rodrigues' formula and the complex exponential."""
    if omega==0:
        return basic(v,h)
    limit=min(.1,np.pi/6/omega)
    active=np.minimum(h,limit)
    rate=-5+1j*omega
    rot=np.expm1(rate*active)/rate
    tail=.2*(np.exp(-5*active)-np.exp(-5*h))
    rot+=tail*np.exp(1j*omega*active)
    parallel=(v@axis)*axis
    return basic(parallel,h)+rot.real[:,None]*(v-parallel)+rot.imag[:,None]*np.cross(axis,v)


def average(h,values):
    return np.trapezoid(values,h,axis=0)/h[-1]


def hierarchy(frame,metrics,extra=()):
    levels=list(extra)+['horizon','source','env','record','side']
    tab=frame.groupby(levels)[metrics].mean()
    for _ in range(4):
        levels=levels[:-1]
        tab=tab.groupby(level=levels).mean()
    return tab
