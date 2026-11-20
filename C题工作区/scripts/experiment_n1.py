from pathlib import Path
import sys,collections,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'阶段交付/stage1_review/corrected_code'));from common import *
O=R/'阶段交付/next_round/N1';O.mkdir(parents=True,exist_ok=True);ws=readcsv(R/'阶段交付/stage1_review/corrected_results/data/windows.csv')
def run(split):
 out=[];windows=[w for w in ws if w['split']==split]
 for uid in sorted({w['episode_uuid'] for w in windows}):
  with np.load(DATA/'records'/(uid+'.npz')) as z:
   for w in [w for w in windows if w['episode_uuid']==uid]:
    t=z[w['side']+'__t_s'];x=state(z,w['side']);i=int(w['start_index']);j=int(w['end_index_exclusive'])
    for method,schedule in [('fixed_reference',[.3,.6]),('spread_budget',[.2,.65])]:
     for delay in [0,.1,.2]:
      obs=[(t[i-2],x[i-2]),(t[i-1],x[i-1])];pending=None;requested=set();errors=[];actions=[]
      for k in range(i,j):
       now=t[k];rel=now-t[i]
       if pending is not None and now>=pending[0]-1e-12:obs.append((pending[1],pending[2]));pending=None
       due=[s for s in schedule if s not in requested and rel>=s-1e-12]
       if due and pending is None and now+delay<=t[i]+1+1e-12:
        requested.add(due[0]);pending=(now+delay,now,x[k]);actions.append(dict(request_s=float(rel),return_s=float(rel+delay)))
        if delay==0:obs.append((now,x[k]));pending=None
       age=now-obs[-1][0];v=(obs[-1][1]-obs[-2][1])/(obs[-1][0]-obs[-2][0]);pred=obs[-1][1]-.2*np.expm1(-age/.2)*v;errors.append(float(np.linalg.norm(pred-x[k])))
      out.append({k:w[k] for k in ['window_id','episode_uuid','environment_id','dataset','split']}|dict(method=method,delay=delay,ade=float(np.mean(errors)),fde=errors[-1],requests=len(actions),cost=len(actions),waiting_s=len(actions)*delay,actions=json.dumps(actions)))
 return out
def score(rows):
 rb=collections.defaultdict(list)
 for r in rows:
  if r['delay']==.1:rb[(r['dataset'],r['environment_id'],r['episode_uuid'],r['method'])].append(r['ade'])
 eb=collections.defaultdict(list)
 for (s,e,u,m),v in rb.items():eb[(s,e,m)].append(float(np.mean(v)))
 sb=collections.defaultdict(list)
 for (s,e,m),v in eb.items():sb[(s,m)].append(float(np.mean(v)))
 return {k:float(np.mean(v)) for k,v in sb.items()}
train=run('train');writecsv(O/'train_windows.csv',train);m=score(train);comp=[dict(dataset=s,relative_improvement=1-m[(s,'spread_budget')]/m[(s,'fixed_reference')]) for s in sorted({r['dataset'] for r in train})];improvement=float(np.mean([r['relative_improvement'] for r in comp]));admit=improvement>=.03 and all(r['relative_improvement']>=-.05 for r in comp);dump(O/'selection_before_validation.json',dict(admitted=admit,improvement=improvement,by_source=comp))
if admit:writecsv(O/'validation_windows.csv',run('validation'))
dump(O/'result.json',dict(admitted=admit,train_rows=len(train),improvement=improvement,by_source=comp,test_evaluated=False,checks=dict(requests_le_two=all(r['requests']<=2 for r in train),finite=all(np.isfinite(r['ade']) for r in train))))
(O/'实验报告.md').write_text(f'# N1 预算分配实验\n\n固定候选(.2,.65)对比(.3,.6)，tau=.2秒。训练记录/环境/来源等权主延迟.1秒，平均相对改善{improvement:.2%}。分源{comp}。\n\n准入{admit}，'+('已运行冻结验证。' if admit else '未达到门槛，不运行验证，不调整时刻。')+'\n\n本轮只比较固定时序，不能据此推断优于所有自适应策略。请求返回前只用历史可见点；测试未评价。',encoding='utf8');print(load(O/'result.json'))
