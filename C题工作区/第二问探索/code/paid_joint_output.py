"""Research-only paid isolated-return output. No reference trajectory input."""
import math
from types import MappingProxyType
import numpy as np

class PaidRange:
 def __init__(self,table,*,protocol='paid_fixed_03_06_delay01',model='distance_return'):
  if protocol!='paid_fixed_03_06_delay01' or model!='distance_return':raise ValueError('unmatched calibration protocol/model')
  snapshot={}
  for key,value in table.items():
   if not np.isfinite(value['radius']) or value['radius']<0:raise ValueError('invalid radius')
   if type(value['points']) is not int or value['points']<=0 or type(value['environments']) is not int or not 0<value['environments']<=value['points']:raise ValueError('invalid support')
   if type(value['age_fallback']) is not bool:raise ValueError('invalid fallback flag')
   snapshot[key]=MappingProxyType(dict(value))
  self.table=MappingProxyType(snapshot)
 def output(self,position,*,now,last_sample_time,returns,source):
  position=np.asarray(position,float)
  if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('invalid position')
  if not math.isfinite(now) or not math.isfinite(last_sample_time) or now<last_sample_time:raise ValueError('invalid time')
  if type(returns) is not int or returns not in (0,1,2):raise ValueError('invalid return count')
  age=now-last_sample_time;age_bin=max(1,min(30,math.ceil(age*30)));key=f'{source}|{int(returns>0)}|{age_bin}'
  if age>1.0+1e-12:return dict(position=position.tolist(),age=age,age_bin=None,radius=None,status='outside_calibration_domain',guaranteed=False)
  if key not in self.table:return dict(position=position.tolist(),age=age,age_bin=age_bin,radius=None,status='no_matching_calibration',guaranteed=False)
  entry=self.table[key]
  return dict(position=position.tolist(),age=age,age_bin=age_bin,radius=float(entry['radius']),status='empirical_range',guaranteed=False,calibration_points=entry['points'],calibration_environments=entry['environments'],age_fallback=entry['age_fallback'])
