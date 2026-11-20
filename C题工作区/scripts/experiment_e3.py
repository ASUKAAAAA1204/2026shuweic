"""E3 fixed two-step ordering; pure observation policy, paired environment holdout."""
from pathlib import Path
import sys,math,random,time,collections,copy
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'阶段交付/stage1_review/corrected_code'))
from common import *
from q3 import Engine,independent_metrics
O=R/'阶段交付/stage2_optimization/E3';O.mkdir(parents=True,exist_ok=True)

def act(o,medians,method):
 jobs=o['jobs'];done={j['job_id'] for j in jobs if j['completed']}
 def option(j,now,zone,cost,completed):
  if j['attempted'] or j['release_s']>now+1e-12 or any(p not in completed for p in j['predecessors']):return None
  p=o['alpha']*medians[j['family']]+o['beta'];b=max(.2,p);D=math.dist(zone,j['zone_xy']);s=now+D+.5;fee=2+.1*b+.2*D
  if s+b>o['H']+1e-12 or cost+fee>o['B']+1e-12:return None
  return dict(job=j,p=p,b=b,C=s+p,next=s+max(p,b),fee=fee,D=D,ontime=j['weight'] if s+p<=j['due_s']+1e-12 else 0,completed=j['weight'] if s+p<=o['H']+1e-12 else 0)
 first=[v for j in jobs if (v:=option(j,o['time'],o['zone_xy'],o['cost'],done)) is not None]
 if not first:
  future=[j['release_s'] for j in jobs if not j['attempted'] and o['time']<j['release_s']<=o['H']]
  return dict(type='wait',duration=min(future)-o['time']) if future else dict(type='stop')
 if method=='family_density':
  a=min(first,key=lambda a:(-a['job']['weight']/(a['D']+.5+a['b']),a['job']['due_s'],a['job']['job_id']))
 else:
  ranked=[]
  for a in first:
   j=a['job'];now=a['next'];completed=done|{j['job_id']};remaining=[k for k in jobs if k['job_id']!=j['job_id'] and not k['attempted']];times=[now];future=[k['release_s'] for k in remaining if k['release_s']>now and k['release_s']<=o['H']]
   if future:times.append(min(future))
   second=[v for at in times for k in remaining if (v:=option(k,at,j['zone_xy'],o['cost']+a['fee'],completed)) is not None]
   choices=[(-a['ontime'],-a['completed'],a['C'],a['fee'],j['job_id'],'')]
   for b in second:choices.append((-(a['ontime']+b['ontime']),-(a['completed']+b['completed']),b['C'],a['fee']+b['fee'],j['job_id'],b['job']['job_id']))
   ranked.append((min(choices),a))
  a=min(ranked,key=lambda r:r[0])[1]
 return dict(type='execute',job_id=a['job']['job_id'],reserve=a['b'])

def run_trials(split,fold,training,pool,seeds):
 med={f:float(np.median([float(s['duration_s']) for s in training if s['family_id']==f and s['eligible']=='1'])) for f in ['P','A','L']};rows=[];skips=[];logs=[];diffmax=0;checks=0
 for case in conditions['cases']:
  jobs=alljobs[case['case_id']];pools={f:[s for s in pool if s['family_id']==f and s['eligible']=='1'] for f in ['P','A','L']};missing=sorted({j['family_id'] for j in jobs if not pools[j['family_id']]})
  if missing:skips.append(dict(split=split,fold=fold,case_id=case['case_id'],missing_families='|'.join(missing)));continue
  for alpha,beta in mappings:
   for seed in seeds:
    rng=random.Random(seed);hidden={}
    for j in jobs:
     s=rng.choice(pools[j['family_id']]);hidden[j['job_id']]=dict(duration=float(s['duration_s']),unit_id=s['unit_id'],features={str(l):[] for l in [.2,.6,1.]})
    for method in ['family_density','two_step']:
     e=Engine(jobs,hidden,zones,case['horizon_s'],case['cost_budget'],case['observation_limit'],alpha,beta);tag=dict(split=split,fold=fold,case_id=case['case_id'],alpha=alpha,beta=beta,seed=seed,method=method);start=time.perf_counter()
     while not e.terminal:
      observation=e.observe();a=act(observation,med,method)
      other=copy.deepcopy(observation)
      for j in other['jobs']:j['features']=[999]*10
      assert act(other,med,method)==a;checks+=1
      event=e.step(a);logs.append(dict(**tag,event=event))
     result=e.metrics();ind=independent_metrics(jobs,e.events,e.H);diffmax=max(diffmax,max(abs(float(result[k])-float(ind[k])) for k in result));rows.append(dict(**tag,**result,runtime_s=time.perf_counter()-start))
 return rows,skips,logs,diffmax,checks,med

start=time.perf_counter();segs=readcsv(DATA/'segments_private.csv');conditions=load(CONFIG/'common_conditions.json');zones={r['zone_id']:[float(r['x']),float(r['y'])] for r in readcsv(CONFIG/'service_zones.csv')};alljobs={}
for case in conditions['cases']:
 alljobs[case['case_id']]=[{**r,**{k:float(r[k]) for k in ['release_s','due_s','priority_weight']},'predecessors':r['predecessors'].split('|') if r['predecessors'] else []} for r in sorted(readcsv(CONFIG/'tasks.csv'),key=lambda r:r['job_id']) if r['case_id']==case['case_id']]
mappings=[(1,.5),(.75,.5),(1.5,.5),(1,0),(1,1)];train=[s for s in segs if s['split']=='train'];envs=sorted({s['environment_id'] for s in train});rows=[];skips=[];folds=[];maxdiff=0;count=0
# Freeze exact mechanics before observing outputs.
dump(O/'protocol.json',dict(registry_sha256=sha(R/'阶段交付/stage1_review/执行计划与实验注册表.md'),methods=['family_density','two_step'],second_step_wait='evaluate now and nearest future release; no arbitrary later wait',optional_second='single-step alternative included',source_pool_order='source CSV preserved',training_seeds=[2026112000,2026112029],test_evaluated=False))
with (O/'training_actions.jsonl').open('w',encoding='utf8') as logfile:
 for e in envs:
  fitting=[s for s in train if s['environment_id']!=e];pool=[s for s in train if s['environment_id']==e];rr,ss,ll,dd,cc,med=run_trials('train_oof',e,fitting,pool,range(2026112000,2026112030));rows.extend(rr);skips.extend(ss);maxdiff=max(maxdiff,dd);count+=cc;folds.append(dict(held_environment=e,fitting_environments=sorted({s['environment_id'] for s in fitting}),medians=med,trials=len(rr)))
  for line in ll:logfile.write(json.dumps(line,ensure_ascii=False)+'\n')
  print('E3 fold',e,'trials',len(rr),'skipped cases',len(ss),flush=True)
writecsv(O/'train_trials.csv',rows);writecsv(O/'ineligible_folds.csv',skips);dump(O/'fold_manifest.json',folds)
def paired(rr):
 lookup={(r['fold'],r['case_id'],r['alpha'],r['beta'],r['seed'],r['method']):r for r in rr};out=[]
 for r in rr:
  if r['method']!='two_step':continue
  b=lookup[(r['fold'],r['case_id'],r['alpha'],r['beta'],r['seed'],'family_density')];out.append({k:r[k] for k in ['fold','case_id','alpha','beta','seed']}|dict(on_time_difference=r['weighted_on_time_rate']-b['weighted_on_time_rate'],cost_difference=r['cost']-b['cost']))
 return out
pairs=paired(rows);writecsv(O/'train_paired.csv',pairs);group=collections.defaultdict(list)
for p in pairs:group[(p['fold'],p['case_id'],p['alpha'],p['beta'])].append(p['on_time_difference'])
summary=[dict(fold=k[0],case_id=k[1],alpha=k[2],beta=k[3],difference=float(np.mean(v))) for k,v in group.items()];writecsv(O/'train_fold_summary.csv',summary)
nom=[r for r in summary if r['alpha']==1 and r['beta']==.5];case_means={c:float(np.mean([r['difference'] for r in nom if r['case_id']==c])) if any(r['case_id']==c for r in nom) else None for c in alljobs};mean=float(np.mean([r['difference'] for r in nom])) if nom else None;overall=float(np.mean([r['difference'] for r in summary])) if summary else None;admitted=mean is not None and mean>=.01 and all(v is not None and v>=-.02 for v in case_means.values()) and overall>=0
dump(O/'selection_before_validation.json',dict(admitted=admitted,nominal_difference=mean,case_differences=case_means,five_mapping_difference=overall,evaluable_environment_case_pairs=len({(r['fold'],r['case_id']) for r in summary}),skipped_environment_cases=skips))
if admitted:
 rr,ss,ll,dd,cc,med=run_trials('validation','frozen_train',train,[s for s in segs if s['split']=='validation'],conditions['validation_seeds']);writecsv(O/'validation_trials.csv',rr);vp=paired(rr);writecsv(O/'validation_paired.csv',vp);maxdiff=max(maxdiff,dd);count+=cc
 with (O/'validation_actions.jsonl').open('w',encoding='utf8') as f:
  for line in ll:f.write(json.dumps(line,ensure_ascii=False)+'\n')
 result=[]
 for case in alljobs:
  for alpha,beta in mappings:
   v=[p['on_time_difference'] for p in vp if p['case_id']==case and p['alpha']==alpha and p['beta']==beta];result.append(dict(case_id=case,alpha=alpha,beta=beta,**grouped_boot(v,list(range(len(v))))))
 writecsv(O/'validation_paired_summary.csv',result)
invalid=sum(r['invalid_actions'] for r in rows);checks=dict(folds=len(folds),fold_separation=all(f['held_environment'] not in f['fitting_environments'] for f in folds),invalid_actions=invalid,independent_metric_max_error=maxdiff,observation_feature_invariance_checks=count,test_evaluated=False)
dump(O/'result.json',dict(status='PASS' if checks['fold_separation'] and invalid==0 and maxdiff<=1e-9 else 'FAIL',admitted=admitted,training_trials=len(rows),skipped_environment_cases=len(skips),nominal_difference=mean,case_differences=case_means,five_mapping_difference=overall,checks=checks,runtime_s=time.perf_counter()-start))
text=f'''# E3 两步调度前瞻报告

固定族中位数、50%预留、无复查，仅改变排序。16训练环境留一，逐折族统计来自其余环境；同case同seed配对，5官方映射。持出环境缺少某任务族时整case不可评价，不用训练数据替代。

训练试验数：{len(rows)}；不可评价环境/case：{len(skips)}，详见ineligible_folds.csv。每环境/case/映射先平均30种子差，再等权汇总。

名义按时率差：{mean}；分案例：{case_means}；五映射平均差：{overall}。

准入：{admitted}。{'已运行单一冻结候选验证，详见validation_paired_summary.csv。' if admitted else '未达到冻结准入标准，不运行验证，不修改评分或增加搜索深度追逐结果。保留family_density。'}

注意：缺族导致训练留一只能评价部分环境/case，结果仅适用于这些可评价子集，不代表全部16环境完整泛化。评分仅看两步，可能牺牲更远未来高价值任务，不保证整体改进。

非法动作{invalid}；独立指标误差最大{maxdiff}；{count}次输入特征扰动不改变动作。策略是纯观察函数，没有接收源标识、真实时长或seed；完整hidden仅评估器持有。日志含事后源引用，不能作为策略训练资料。

下一步：汇总E1/E2/E3，冻结实际保留方法，不强行合并未通过候选。测试未评价。本报告不声称全局最优。
'''
(O/'实验报告.md').write_text(text,encoding='utf8');dump(O/'code_hashes.json',{Path(__file__).name:sha(Path(__file__))});print(json.dumps(load(O/'result.json'),ensure_ascii=False,indent=2))
