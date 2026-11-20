"""Post-hoc diagnosis only; no candidate fitting/selection on validation/test."""
from pathlib import Path
import sys,collections,math,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'阶段交付/stage1_review/corrected_code'));from common import *
S=R/'阶段交付/stage2_optimization';O=R/'阶段交付/stage2_diagnosis';O.mkdir(parents=True,exist_ok=True)
# Bucket occupancy directly from frozen model, not guessed from result quality.
m=load(S/'E2/train_fitted_calibration.json');occup=[]
for src,v in m.items():
 c=list(v['counts'].values());occup.append(dict(dataset=src,conditional_buckets=len(c),empty=sum(x==0 for x in c),below100=sum(x<100 for x in c),median_points=float(np.median(c)),min_points=min(c),max_points=max(c)))
writecsv(O/'E2_bucket_occupancy.csv',occup)
# Independent train-only historical-feature explanatory check, target mean 1s damped error.
ws=[w for w in readcsv(R/'阶段交付/stage1_review/corrected_results/data/windows.csv') if w['split']=='train'];examples=[]
for uid in sorted({w['episode_uuid'] for w in ws}):
 with np.load(DATA/'records'/(uid+'.npz')) as z:
  for w in [w for w in ws if w['episode_uuid']==uid]:
   t=z[w['side']+'__t_s'];x=state(z,w['side']);i=int(w['start_index']);j=int(w['end_index_exclusive']);h=int(w['history_start']);a=i-1;dx=np.diff(x[h:i],axis=0);norm=np.linalg.norm(dx,axis=1);dt=np.diff(t[h:i]);speed=float(np.median(norm/dt));turn=[]
   for k in range(1,len(dx)):
    if norm[k]>1e-12 and norm[k-1]>1e-12:turn.append(float(np.arccos(np.clip(dx[k]@dx[k-1]/norm[k]/norm[k-1],-1,1))))
   angle=float(np.median(turn)) if turn else 0;age=t[i:j]-t[a];v=(x[a]-x[a-1])/(t[a]-t[a-1]);err=np.linalg.norm(x[a]-.2*np.expm1(-age[:,None]/.2)*v-x[i:j],axis=1)
   examples.append(dict(episode_uuid=uid,environment_id=w['environment_id'],dataset=w['dataset'],speed=speed,turn=angle,mean_error=float(err.mean())))
writecsv(O/'E2_train_features.csv',examples);expl=[]
for src in sorted(m):
 rr=[r for r in examples if r['dataset']==src];y=np.log1p([r['mean_error'] for r in rr]);X=np.array([[np.log1p(r['speed']),r['turn']] for r in rr]);env=np.array([r['environment_id'] for r in rr]);pred=np.zeros(len(y));base=np.zeros(len(y))
 for e in sorted(set(env)):
  tr=env!=e;te=~tr;mu=X[tr].mean(0);sd=X[tr].std(0);sd[sd<1e-10]=1;A=np.c_[np.ones(tr.sum()),(X[tr]-mu)/sd];reg=np.diag([0,1,1]);b=np.linalg.solve(A.T@A+reg,A.T@y[tr]);pred[te]=np.c_[np.ones(te.sum()),(X[te]-mu)/sd]@b;base[te]=y[tr].mean()
 # Environment equal MSE; diagnostic ridge is fixed, not searched and not deployed.
 mse=float(np.mean([np.mean((y[env==e]-pred[env==e])**2) for e in set(env)]));bm=float(np.mean([np.mean((y[env==e]-base[env==e])**2) for e in set(env)]));expl.append(dict(dataset=src,windows=len(rr),environments=len(set(env)),target='log1p mean 1s error',baseline_mse=bm,feature_mse=mse,relative_mse_improvement=1-mse/bm))
writecsv(O/'E2_feature_explanation_oof.csv',expl)
# Replay results diagnose observed effect, not speculative physical causes.
rs=readcsv(S/'E2/replay_summary.csv');effect=[]
for src in sorted(m):
 for metric in ['ade','coverage','radius','requests','waiting_s','fallback_fraction']:
  rr=[r for r in rs if r['dataset']==src and r['metric']==metric and float(r['delay_s'])==.1 and float(r['tolerance_fraction'])==.05];v={r['method']:float(r['mean']) for r in rr};effect.append(dict(dataset=src,metric=metric,age_only=v['age_only'],conditional=v['conditional'],difference=v['conditional']-v['age_only']))
writecsv(O/'E2_replay_main_effect.csv',effect)
# E3 task-family coverage before any proposed new split.
segs=readcsv(DATA/'segments_private.csv');envs=sorted({s['environment_id'] for s in segs if s['split']=='train'});coverage=[]
for e in envs:
 counts=collections.Counter(s['family_id'] for s in segs if s['split']=='train' and s['environment_id']==e and s['eligible']=='1');coverage.append(dict(environment_id=e,**{f:counts[f] for f in ['P','A','L']},complete=all(counts[f] for f in ['P','A','L'])))
writecsv(O/'E3_family_coverage.csv',coverage)
tr=readcsv(S/'E3/train_trials.csv');pairs=readcsv(S/'E3/train_paired.csv');nom=[r for r in pairs if float(r['alpha'])==1 and float(r['beta'])==.5];counts=collections.Counter('win' if float(r['on_time_difference'])>1e-12 else 'loss' if float(r['on_time_difference'])< -1e-12 else 'tie' for r in nom)
logs=collections.defaultdict(list)
with (S/'E3/training_actions.jsonl').open(encoding='utf8') as f:
 for line in f:
  r=json.loads(line)
  if r['alpha']==1 and r['beta']==.5:logs[(r['fold'],r['case_id'],r['seed'],r['method'])].append(r['event'])
jobs=readcsv(CONFIG/'tasks.csv');worst=sorted(nom,key=lambda r:float(r['on_time_difference']))[:10];cases=[];tasks=[]
for ix,r in enumerate(worst):
 key=(r['fold'],r['case_id'],int(r['seed']));a=logs[key+('family_density',)];b=logs[key+('two_step',)];seq=lambda es:[e['action'].get('job_id',e['action']['type']) for e in es];sa=seq(a);sb=seq(b);first=next((i for i,(x,y) in enumerate(zip(sa,sb)) if x!=y),min(len(sa),len(sb)));cases.append(dict(case_number=ix+1,fold=key[0],case_id=key[1],seed=key[2],on_time_difference=float(r['on_time_difference']),first_divergent_action_index=first,baseline_sequence='|'.join(sa),lookahead_sequence='|'.join(sb),note='retrospective divergence; not causal attribution by itself'))
 for j in [j for j in jobs if j['case_id']==key[1]]:
  def outcome(events):
   e=next((e for e in events if e['action'].get('job_id')==j['job_id'] and e['action']['type']=='execute'),None);C=e.get('completion') if e else None;return C,bool(C is not None and C<=float(j['due_s'])+1e-12)
  ca,oa=outcome(a);cb,ob=outcome(b)
  if oa!=ob:tasks.append(dict(case_number=ix+1,job_id=j['job_id'],weight=j['priority_weight'],due_s=j['due_s'],baseline_completion=ca,lookahead_completion=cb,baseline_on_time=oa,lookahead_on_time=ob))
writecsv(O/'E3_worst_divergences.csv',cases);writecsv(O/'E3_changed_task_outcomes.csv',tasks)
# Fixed small-case exact hindsight bound: same frozen reserve, actual durations known ONLY to oracle.
# Enumerate all feasible subsets/orders (S1 eight jobs), stop allowed; not a deployable method.
import random
zones={r['zone_id']:(float(r['x']),float(r['y'])) for r in readcsv(CONFIG/'service_zones.csv')};folds={f['held_environment']:f for f in load(S/'E3/fold_manifest.json')};oracle=[]
small=sorted([r for r in nom if r['case_id']=='S1'],key=lambda r:float(r['on_time_difference']))[:3]
for r in small:
 jj=sorted([j for j in jobs if j['case_id']=='S1'],key=lambda j:j['job_id']);rng=random.Random(int(r['seed']));pool=[s for s in segs if s['split']=='train' and s['environment_id']==r['fold'] and s['eligible']=='1'];actual={j['job_id']:float(rng.choice([s for s in pool if s['family_id']==j['family_id']])['duration_s'])+.5 for j in jj};med=folds[r['fold']]['medians'];best=[0,[]];nodes=[0]
 def dfs(now,zone,cost,done,score,path):
  nodes[0]+=1
  if score>best[0]:best[:]=[score,path]
  if len(done)==len(jj):return
  times=[now]+sorted({float(j['release_s']) for j in jj if j['job_id'] not in done and float(j['release_s'])>now})
  for at in times:
   for j in jj:
    k=j['job_id'];pred=j['predecessors'].split('|') if j['predecessors'] else [];b=med[j['family_id']]+.5;D=math.dist(zones[zone],zones[j['zone_id']]);s=at+D+.5;fee=2+.1*b+.2*D;C=s+actual[k]
    if k in done or float(j['release_s'])>at or any(p not in done for p in pred) or cost+fee>32+1e-12 or s+b>100+1e-12 or C>100+1e-12:continue
    dfs(s+max(b,actual[k]),j['zone_id'],cost+fee,done|{k},score+(float(j['priority_weight']) if C<=float(j['due_s']) else 0),path+[k])
 dfs(0,'Z0',0,set(),0,[]);W=sum(float(j['priority_weight']) for j in jj);key=(r['fold'],'S1',int(r['seed']));outcomes={method:float(next(t['weighted_on_time_rate'] for t in tr if t['fold']==key[0] and t['case_id']=='S1' and int(t['seed'])==key[2] and t['method']==method and float(t['alpha'])==1 and float(t['beta'])==.5)) for method in ['family_density','two_step']};oracle.append(dict(fold=key[0],seed=key[2],oracle_rate=best[0]/W,oracle_path='|'.join(best[1]),nodes=nodes[0],**outcomes,bound='clairvoyant order/subset optimum with frozen reservations; not policy-achievable benchmark'))
writecsv(O/'E3_small_exact_hindsight.csv',oracle)
dump(O/'diagnosis.json',dict(E2_features=expl,E3_nominal_counts=dict(counts),E3_complete_environments=sum(r['complete'] for r in coverage),E3_total_environments=len(envs),exact_cases=len(oracle),test_evaluated=False,no_parameter_changes=True))
lines=['# E2/E3 失败诊断报告','','本轮是事后诊断，不改变候选、阈值、数据划分，不评价测试，不将新诊断回归当作部署模型。','','## E2：已证实与未证实','','|来源|条件桶数|空桶|不足100点|中位样本数|','|---|---:|---:|---:|---:|']
for r in occup:lines.append(f'|{r["dataset"]}|{r["conditional_buckets"]}|{r["empty"]}|{r["below100"]}|{r["median_points"]:.1f}|')
lines+=['','完整训练模型的桶占用见上表；真实留一回退率见E2/admission_by_source.csv，不能用完整模型桶数代替留一折覆盖。EgoPro留一约86.7%点回退，条件模型在该源大部分时间实际上等于原模型。','','固定诊断回归用历史速度/转向预测log1p的一秒平均误差，按来源训练环境留一，与折内常数预测比较；结果如下。它衡量这两个特征的一种简单预测关系，不是因果解释，也不能排除非线性条件关系。','','|来源|环境数|相对MSE改善|','|---|---:|---:|']
for r in expl:lines.append(f'|{r["dataset"]}|{r["environments"]}|{r["relative_mse_improvement"]:.2%}|')
lines+=['','重观测主条件的误差、覆盖、半径、请求及等待变化见E2_replay_main_effect.csv。分桶改善幅度不能单独解释策略收益。观测后可见历史稀疏与训练连续历史不同是代码确认的分布差异，但本轮尚未以受控实验量化它导致多少性能变化，不能定为唯一原因。','','## E3：覆盖不足与决策损失',f'训练名义配对试验胜/平/负：{dict(counts)}。16环境仅2环境具备P/A/L完整池，42环境/case组合不可评价，见E3_family_coverage.csv。这个可行性检查本应在方案冻结前完成。','','最差10组试验的首次动作分歧与完整顺序见E3_worst_divergences.csv；具体从按时变迟到或未执行的任务见E3_changed_task_outcomes.csv。观察到动作分歧和结果差不能单独证明“短视”因果；还可能来自中位时长误差、移动、预留和前驱耦合。','','对S1最差3组作同预留下的全子集/顺序穷举，允许等待已知释放；使用真实时长为先知参考，仅供诊断，不属于合法策略比较，不拟合任何模型。','','|种子|基线按时率|两步按时率|同预留先知界|','|---|---:|---:|---:|']
for r in oracle:lines.append(f'|{r["seed"]}|{r["family_density"]:.4f}|{r["two_step"]:.4f}|{r["oracle_rate"]:.4f}|')
lines+=['','先知差距表明在这些实例中顺序/任务选择尚有空间；不能把全部差距归咎于两步评分，也不能认为可见信息策略能达到先知界。穷举受限固定预留与8任务，不证明原随机决策问题全局最优。','','## 下一步决定','','1. 保留E1，不因E2/E3负结果撤销已有点预测证据。','2. 不继续增加条件分桶；先报告稀疏与回退，若研究校准应另注册训练环境稳健/收缩校准，不借本次验证挑参数。','3. 不直接加大搜索深度；先用已给出的任务结果日志区分丢任务、迟到和预留占用，再注册一个与最终服务目标一致的候选。','4. 如要改E3环境评价，先明确“跨环境任务池组合”和“单环境完整案例”是不同实验，必须新协议，不偷偷补池。','5. 本轮不生成新的最优方法结论。诊断表、先知界和源代码完整保留，测试仍封存。']
(O/'诊断报告.md').write_text('\n'.join(lines),encoding='utf8');dump(O/'code_hashes.json',{Path(__file__).name:sha(Path(__file__))});print(json.dumps(load(O/'diagnosis.json'),ensure_ascii=False,indent=2))
