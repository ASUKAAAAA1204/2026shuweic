"""Policies use public observation age and fold-fitted radius only."""
import numpy as np
class RadiusPolicy:
 def __init__(self,radii,scale,fraction,delay,anticipate=False):
  self.radii=np.asarray(radii,float);self.scale=scale;self.fraction=fraction;self.delay=delay;self.anticipate=anticipate
 def radius(self,age):return float(self.radii[min(29,max(0,int(np.ceil(age*30))-1))])
 def request(self,v):return self.radius(v.age+(self.delay if self.anticipate else 0))>=self.scale*self.fraction

class BudgetPolicy:
 """Compare no request, now, or later; proxy integrated normalized radius.
 At most one hypothetical second query, halfway from first return to mask end.
 Later candidate at remaining/(remaining_budget+1); future curve evaluated on 12 public grid points.
 No claim of expected actual error reduction or optimal control.
 """
 def __init__(self,radii,scale,cost,delay):self.r=np.asarray(radii,float)/scale;self.cost=cost;self.delay=delay
 def request(self,v):
  left=2-v.requests
  if left<=0 or v.remaining<self.delay:return False
  q=np.linspace(0,v.remaining,12);base=np.full(len(q),v.age)+q
  def loss(schedule):
   ages=base.copy()
   for sample in schedule:
    use=q>=sample+self.delay-1e-12;ages[use]=q[use]-sample
   ix=np.clip(np.ceil(ages*30).astype(int)-1,0,29)
   return float(self.r[ix].mean()*v.remaining+self.cost*len(schedule))
  candidates=[(loss([]),False)]
  for first,isnow in [(0.,True),(v.remaining/(left+1),False)]:
   if first+self.delay>v.remaining+1e-12:continue
   schedules=[[first]]
   second=first+self.delay+(v.remaining-first-self.delay)/2
   if left>1 and second+self.delay<=v.remaining+1e-12:schedules.append([first,second])
   for schedule in schedules:candidates.append((loss(schedule),isnow))
  # Equal proxy losses prefer saving current request.
  return min(candidates,key=lambda a:(a[0],a[1]))[1]
