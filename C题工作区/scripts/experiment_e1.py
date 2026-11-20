"""E1 preregistered damped motion experiment; no test evaluation."""
from pathlib import Path
import sys,time,collections
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'阶段交付/stage1_review/corrected_code'))
from common import *
OUT=ROOT/'阶段交付/stage2_optimization/E1';OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'阶段交付/stage1_review/corrected_results'
MODELS=['hold','constant_velocity',.1,.2,.5,1.]
def displacement(age,v,model):
 if model=='hold':return np.zeros((len(age),3))
 if model=='constant_velocity':return age[:,None]*v
 return (-model*np.expm1(-age/model))[:,None]*v

def evaluate(split,models):
 rows=[];ws=[w for w in readcsv(BASE/'data/windows.csv') if w['split']==split];by=collections.defaultdict(list)
 for w in ws:by[w['episode_uuid']].append(w)
 for uid,windows in by.items():
  with np.load(DATA/'records'/(uid+'.npz')) as z:
   for w in windows:
    side=w['side'];t=z[side+'__t_s'];x=state(z,side);i=int(w['start_index']);j=int(w['end_index_exclusive']);a=i-1;v=(x[a]-x[a-1])/(t[a]-t[a-1])
    for m in models:
     for h in [.2,.6,1.]:
      ix=np.arange(i,j)[t[i:j]<=t[i]+h+1e-12];age=t[ix]-t[a];err=np.linalg.norm(x[a]+displacement(age,v,m)-x[ix],axis=1)
      rows.append({k:w[k] for k in ['window_id','episode_uuid','dataset','environment_id','side','split']}|dict(model=str(m),horizon_s=h,ade=float(err.mean()),fde=float(err[-1]),window_p95=float(np.quantile(err,.95)),fold_held_environment=w['environment_id']))
 return rows

def aggregate(rows):
 rb=collections.defaultdict(list)
 for r in rows:rb[(r['dataset'],r['model'],r['horizon_s'],r['environment_id'],r['episode_uuid'])].append(r['ade'])
 eb=collections.defaultdict(list)
 for (s,m,h,e,uid),v in rb.items():eb[(s,m,h,e)].append(float(np.mean(v)))
 env=[dict(dataset=s,model=m,horizon_s=h,environment_id=e,ade=float(np.mean(v))) for (s,m,h,e),v in eb.items()]
 groups=collections.defaultdict(list)
 for r in env:groups[(r['dataset'],r['model'],r['horizon_s'])].append(r['ade'])
 means={k:float(np.mean(v)) for k,v in groups.items()};return env,means
start=time.perf_counter();train=evaluate('train',MODELS);env,means=aggregate(train);sources=sorted({r['dataset'] for r in train});scores=[]
for tau in [.1,.2,.5,1.]:
 ratios={(s,h):means[(s,str(tau),h)]/min(means[(s,'hold',h)],means[(s,'constant_velocity',h)]) for s in sources for h in [.2,.6,1.]}
 score=float(np.mean([ratios[(s,h)] for s in sources for h in [.6,1.]]));worstlong=max(ratios[(s,h)] for s in sources for h in [.6,1.]);worstshort=max(ratios[(s,.2)] for s in sources)
 scores.append(dict(tau=tau,relative_long_ade=score,improvement=1-score,worst_long_ratio=worstlong,worst_short_ratio=worstshort,admitted=(score<=.97 and worstlong<=1.05 and worstshort<=1.1)))
best=min(scores,key=lambda r:(r['relative_long_ade'],r['tau']));passed=best['admitted']
# Predictions have no fitted parameters. Each training environment supplies held-out errors;
# tau selection uses these errors once. No fictitious per-fold refit is claimed.
dump(OUT/'selection_before_validation.json',dict(scores=scores,selected_tau=best['tau'],admitted=passed,registry_sha256=sha(ROOT/'阶段交付/stage1_review/执行计划与实验注册表.md'),method='parameter-free predictors, held-environment scoring; pooled OOF selection, no claim of nested-CV estimate',test_evaluated=False))
writecsv(OUT/'train_windows.csv',train);writecsv(OUT/'train_environments.csv',env);writecsv(OUT/'train_summary.csv',[dict(dataset=k[0],model=k[1],horizon_s=k[2],ade=v) for k,v in means.items()]);writecsv(OUT/'candidate_scores.csv',scores)
paired=[];validation=[]
if passed:
 validation=evaluate('validation',['hold','constant_velocity',best['tau']]);ve,vm=aggregate(validation);writecsv(OUT/'validation_windows.csv',validation);writecsv(OUT/'validation_environments.csv',ve)
 for s in sources:
  for h in [.2,.6,1.]:
   baseline=min(['hold','constant_velocity'],key=lambda m:(means[(s,m,h)],m));b={r['environment_id']:r['ade'] for r in ve if r['dataset']==s and r['model']==baseline and r['horizon_s']==h};c={r['environment_id']:r['ade'] for r in ve if r['dataset']==s and r['model']==str(best['tau']) and r['horizon_s']==h};values=[b[e]-c[e] for e in b];paired.append(dict(dataset=s,horizon_s=h,baseline_train_selected=baseline,**grouped_boot(values,list(b))))
 writecsv(OUT/'validation_paired.csv',paired)
# Meaningful numerical/causal tests including limits and hidden-future independence.
v=np.array([1.,2.,3.]);age=np.array([0.,.2,1.]);checks=[]
checks.append(dict(name='zero age exact',passed=bool(np.array_equal(displacement(age,v,.2)[0],np.zeros(3)))))
checks.append(dict(name='zero velocity stationary',passed=bool(np.array_equal(displacement(age,np.zeros(3),.2),np.zeros((3,3))))))
checks.append(dict(name='large tau tends constant velocity',passed=bool(np.max(abs(displacement(age,v,1e8)-age[:,None]*v))<1e-7)))
checks.append(dict(name='bounded long displacement',passed=bool(np.linalg.norm(displacement(np.array([100.]),v,.2)[0])<=.2*np.linalg.norm(v)+1e-12)))
checks.append(dict(name='test not evaluated',passed=not any(r['split']=='test' for r in train+validation)))
# All train windows use same original .2/.6/1 cut and initial two history points as baseline.
old=readcsv(BASE/'q12/open_loop_windows.csv');lookup={(r['window_id'],r['model'],float(r['horizon_s'])):float(r['ade']) for r in old if r['split']=='train'}
maxdiff=max(abs(r['ade']-lookup[(r['window_id'],r['model'],r['horizon_s'])]) for r in train if r['model'] in ['hold','constant_velocity'])
checks.append(dict(name='simple baselines source-result agreement',passed=maxdiff<=1e-12,max_difference=maxdiff))
writecsv(OUT/'checks.csv',checks);dump(OUT/'result.json',dict(status='PASS' if all(r['passed'] for r in checks) else 'FAIL',training_selected=best,validation_run=passed,validation_rows=len(validation),runtime_s=time.perf_counter()-start,test_evaluated=False,checks=checks))
lines=['# E1 阻尼恒速实验报告','',f'训练筛选最佳tau={best["tau"]}秒；长时相对ADE={best["relative_long_ade"]:.5f}，平均改善{best["improvement"]:.2%}。',f'最差来源/长时比值={best["worst_long_ratio"]:.5f}；最差来源短时比值={best["worst_short_ratio"]:.5f}。',f'准入结果：{"通过，已按冻结规则运行单一候选验证" if passed else "未通过，不运行候选验证，保留简单基线"}。','','|tau秒|长时改善|最差长时比值|最差短时比值|准入|','|---|---:|---:|---:|---|']
for r in scores:lines.append(f'|{r["tau"]}|{r["improvement"]:.2%}|{r["worst_long_ratio"]:.4f}|{r["worst_short_ratio"]:.4f}|{r["admitted"]}|')
lines+=['','保持状态、恒速、固定tau阻尼模型都无拟合参数；按训练环境持出统计误差，不伪称每折重新拟合。tau用持出误差汇总选择，选择后的训练收益存在选择乐观性，不称最终泛化证据。先记录平均，再环境平均，最后来源/时域等权。测试数据没有评价。','',f'数值/一致性检查{sum(r["passed"] for r in checks)}/{len(checks)}通过；简单基线复核最大差异{maxdiff:.3g}。','', '下一步E2使用本轮准入者；若无准入者，按训练环境等权长时平均ADE选择每来源简单基线，冻结后再校准，不借用验证结果调整。']
(OUT/'实验报告.md').write_text('\n'.join(lines),encoding='utf8');dump(OUT/'code_hashes.json',{Path(__file__).name:sha(Path(__file__))});print(json.dumps(load(OUT/'result.json'),ensure_ascii=False,indent=2))
