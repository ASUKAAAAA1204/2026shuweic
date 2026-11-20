from pathlib import Path
import sys,ast,math,random,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'阶段交付/stage1_review/corrected_code'));from common import *
from q3 import Engine
O=R/'阶段交付/mechanism_diagnosis';O.mkdir(parents=True,exist_ok=True);S=R/'阶段交付/stage2_optimization'
# Load only pure policy function, never execute E3 experiment on import.
tree=ast.parse((R/'scripts/experiment_e3.py').read_text(encoding='utf-8-sig'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='act');ns={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_e3_act','exec'),ns);act=ns['act']
rr=[r for r in readcsv(S/'E2/replay_windows.csv') if float(r['delay_s'])==.1 and float(r['tolerance_fraction'])==.05];timing=[]
for r in rr:
 actions=json.loads(r['actions']);timing.append({k:r[k] for k in ['window_id','episode_uuid','environment_id','dataset','method']}|dict(first_request_s=actions[0]['request_s'] if actions else None,second_request_s=actions[1]['request_s'] if len(actions)>1 else None,exhausted_by_03=len(actions)==2 and actions[1]['request_s']<=.3+1e-12))
writecsv(O/'request_timing_windows.csv',timing);summ=[]
for src in ['EgoDemo','EgoPro','EgoStandard']:
 for method in ['age_only','conditional']:
  items=[r for r in timing if r['dataset']==src and r['method']==method];summ.append(dict(dataset=src,method=method,windows=len(items),median_first_s=float(np.median([r['first_request_s'] for r in items if r['first_request_s'] is not None])),median_second_s=float(np.median([r['second_request_s'] for r in items if r['second_request_s'] is not None])),exhausted_by_03_fraction=float(np.mean([r['exhausted_by_03'] for r in items]))))
writecsv(O/'request_timing_summary.csv',summ)
segs=readcsv(DATA/'segments_private.csv');raw=readcsv(CONFIG/'tasks.csv');zones={r['zone_id']:[float(r['x']),float(r['y'])] for r in readcsv(CONFIG/'service_zones.csv')};folds={r['held_environment']:r for r in load(S/'E3/fold_manifest.json')};old=readcsv(S/'E3/train_trials.csv');nom=[r for r in readcsv(S/'E3/train_paired.csv') if float(r['alpha'])==1 and float(r['beta'])==.5];cases={r['case_id']:r for r in load(CONFIG/'common_conditions.json')['cases']};results=[];maxerr=0;invalid=0
for row in nom:
 fold=row['fold'];case=row['case_id'];seed=int(row['seed']);med=folds[fold]['medians'];pool=[s for s in segs if s['split']=='train' and s['environment_id']==fold and s['eligible']=='1'];jobs=[{**j,**{k:float(j[k]) for k in ['release_s','due_s','priority_weight']},'predecessors':j['predecessors'].split('|') if j['predecessors'] else []} for j in sorted(raw,key=lambda r:r['job_id']) if j['case_id']==case];rng=random.Random(seed);hidden={}
 for j in jobs:
  s=rng.choice([s for s in pool if s['family_id']==j['family_id']]);hidden[j['job_id']]=dict(duration=float(s['duration_s']),unit_id=s['unit_id'],features={str(l):[] for l in [.2,.6,1.]})
 c=cases[case]
 def engine():return Engine(jobs,hidden,zones,c['horizon_s'],c['cost_budget'],c['observation_limit'],1,.5)
 values={};divergence=None
 for method in ['family_density','two_step','first_action_swap']:
  e=engine();swapped=False;index=0
  while not e.terminal:
   o=e.observe();a=act(o,med,'family_density' if method=='family_density' else 'two_step')
   if method=='first_action_swap' and not swapped:
    b=act(o,med,'family_density')
    if a!=b:divergence=dict(action_index=index,decision_time=e.time,original=a,replacement=b);a=b;swapped=True
   e.step(a);index+=1
  metrics=e.metrics();values[method]=metrics['weighted_on_time_rate'];invalid+=metrics['invalid_actions']
  if method!='first_action_swap':
   prior=next(r for r in old if r['fold']==fold and r['case_id']==case and int(r['seed'])==seed and r['method']==method and float(r['alpha'])==1 and float(r['beta'])==.5);maxerr=max(maxerr,abs(values[method]-float(prior['weighted_on_time_rate'])))
 results.append(dict(fold=fold,case_id=case,seed=seed,**values,swap_minus_lookahead=values['first_action_swap']-values['two_step'],swap_minus_baseline=values['first_action_swap']-values['family_density'],divergence=json.dumps(divergence)))
writecsv(O/'first_action_intervention.csv',results);groups=collections.defaultdict(list)
for r in results:groups[(r['fold'],r['case_id'])].append(r)
group=[dict(fold=k[0],case_id=k[1],**{m:float(np.mean([r[m] for r in v])) for m in ['swap_minus_lookahead','swap_minus_baseline']}) for k,v in groups.items()];writecsv(O/'intervention_group_summary.csv',group)
mean=float(np.mean([r['swap_minus_lookahead'] for r in group]));counts=collections.Counter('improve' if r['swap_minus_lookahead']>1e-12 else 'worse' if r['swap_minus_lookahead']< -1e-12 else 'tie' for r in results);dump(O/'checks.json',dict(status='PASS' if maxerr<=1e-9 and invalid==0 else 'FAIL',baseline_reproduction_max_error=maxerr,invalid_actions=invalid,nominal_pairs=len(results),test_evaluated=False,parameter_changes=False))
text=['# 决策机制诊断结果','','## 请求时间','','以下为主条件逐窗口描述统计，窗口非独立样本，不用于显著性声明。','','|来源|校准|首请求中位秒|次请求中位秒|0.3秒前耗尽比例|','|---|---|---:|---:|---:|']
for r in summ:text.append(f'|{r["dataset"]}|{r["method"]}|{r["median_first_s"]:.3f}|{r["median_second_s"]:.3f}|{r["exhausted_by_03_fraction"]:.1%}|')
text += ['', '请求预算饱和本身不证明规则错误，但说明需要比较信息到达时刻和后半窗口误差，而不是只统计总请求数。下一轮应优先研究预算分配和触发尺度，不继续用细分分桶替代决策设计。', '', '## 首次分歧动作干预',f'覆盖全部{len(results)}组名义训练配对。只在首次分歧采用当时合法基线动作，随后恢复前瞻，不用未来选择干预。先环境/case平均再等权，干预相对原前瞻按时率变化={mean:.4%}；逐组改善/持平/变差={dict(counts)}。', '', '结果只能说明这种单步替换的总效应，不足以把差异全部归于短视或时长误差。两步策略后续还会重新决策，因此替换后轨迹是新的策略，不是先知纠错。', '',f'原始策略复现最大误差{maxerr}，所有回放非法动作{invalid}。完整逐组结果见first_action_intervention.csv。只覆盖2个完整族训练环境，不能外推到其余环境。','','## 下一轮建议','','保持现有冻结模型；先为请求策略注册“保留后半程信息预算”的固定对照，覆盖全窗口误差与成本；调度先用与完整任务集机会损失相关的评分对照，缺族评价方案另行事先确定。不以本轮分析数据直接训练或继续试探验证集。test保持封存。']
(O/'诊断与下一步建议.md').write_text('\n'.join(text),encoding='utf8');print(json.dumps(dict(mean_swap_effect=mean,counts=dict(counts),checks=load(O/'checks.json')),ensure_ascii=False,indent=2))
