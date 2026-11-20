"""E2 preregistered conditional calibration, train-environment holdout."""
from pathlib import Path
import sys,time,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'阶段交付/stage1_review/corrected_code'))
from common import *
O=R/'阶段交付/stage2_optimization/E2';O.mkdir(parents=True,exist_ok=True)
BASE=R/'阶段交付/stage1_review/corrected_results';TAU=load(R/'阶段交付/stage2_optimization/E1/handoff_E2.json')['tau_s'];start=time.perf_counter()
# Freeze implementation decisions before outcomes: median historical speed and turn,
# source-specific fitting thresholds; fallback source-age bucket, >=100 conditional points.
dump(O/'protocol.json',dict(tau=TAU,history_s=.2,age_bins=30,min_conditional_points=100,quantile=.9,quantile_method='higher',thresholds='within-fold source median of window historical median speed/turn',zero_motion_turn=0,calibration='each window point once; no multi-horizon duplicates',threshold_equality='high if >= median',validation_rule='only admitted candidate; no threshold refit using validation',test_evaluated=False))
def samples(split):
 result=[];ws=[w for w in readcsv(BASE/'data/windows.csv') if w['split']==split];by=collections.defaultdict(list)
 for w in ws:by[w['episode_uuid']].append(w)
 for uid,ww in by.items():
  with np.load(DATA/'records'/(uid+'.npz')) as z:
   for w in ww:
    t=z[w['side']+'__t_s'];x=state(z,w['side']);i=int(w['start_index']);j=int(w['end_index_exclusive']);a=i-1;lo=int(w['history_start']);step=np.diff(x[lo:i],axis=0);dt=np.diff(t[lo:i]);speeds=np.linalg.norm(step,axis=1)/dt;norm=np.linalg.norm(step,axis=1);turn=[]
    for k in range(1,len(step)):
     if norm[k]>1e-12 and norm[k-1]>1e-12:turn.append(float(np.arccos(np.clip(step[k]@step[k-1]/norm[k]/norm[k-1],-1,1))))
    speed=float(np.median(speeds)) if len(speeds) else 0;angle=float(np.median(turn)) if turn else 0;v=(x[a]-x[a-1])/(t[a]-t[a-1]);age=t[i:j]-t[a];pred=x[a]-TAU*np.expm1(-age[:,None]/TAU)*v;err=np.linalg.norm(pred-x[i:j],axis=1)
    result.append({**{k:w[k] for k in ['window_id','episode_uuid','environment_id','dataset','split','side']},'speed':speed,'turn':angle,'bins':np.minimum(30,np.maximum(1,np.ceil(age*30).astype(int))),'errors':err})
 return result

def fit(rows):
 model={}
 for src in sorted({r['dataset'] for r in rows}):
  rr=[r for r in rows if r['dataset']==src];sm=float(np.median([r['speed'] for r in rr]));tm=float(np.median([r['turn'] for r in rr]));buckets=collections.defaultdict(list);conditional=collections.defaultdict(list)
  for r in rr:
   for b,e in zip(r['bins'],r['errors']):buckets[int(b)].append(float(e));conditional[(int(b),int(r['speed']>=sm),int(r['turn']>=tm))].append(float(e))
  base={str(b):float(np.quantile(v,.9,method='higher')) for b,v in buckets.items()};cond={};counts={}
  for b in range(1,31):
   if str(b) not in base:
    near=min(buckets,key=lambda k:abs(k-b));base[str(b)]=base[str(near)]
   for s in [0,1]:
    for t in [0,1]:
     key=f'{b}|{s}|{t}';v=conditional.get((b,s,t),[]);counts[key]=len(v);cond[key]=float(np.quantile(v,.9,method='higher')) if len(v)>=100 else base[str(b)]
  model[src]=dict(speed_median=sm,turn_median=tm,base=base,conditional=cond,counts=counts)
 return model

def evaluate(rows,model,fold):
 out=[]
 for r in rows:
  if r['dataset'] not in model:continue
  m=model[r['dataset']];s=int(r['speed']>=m['speed_median']);t=int(r['turn']>=m['turn_median'])
  for method in ['age_only','conditional']:
   radii=np.array([m['base'][str(b)] if method=='age_only' else m['conditional'][f'{b}|{s}|{t}'] for b in r['bins']])
   fallback=float(np.mean([m['counts'][f'{b}|{s}|{t}']<100 for b in r['bins']])) if method=='conditional' else 0.
   out.append({k:r[k] for k in ['window_id','episode_uuid','environment_id','dataset','split','side']}|dict(fold=fold,method=method,coverage=float(np.mean(r['errors']<=radii)),radius=float(radii.mean()),ade=float(r['errors'].mean()),fallback_fraction=fallback))
 return out

def aggregate(rows):
 rb=collections.defaultdict(list)
 for r in rows:rb[(r['dataset'],r['method'],r['environment_id'],r['episode_uuid'])].append(r)
 eb=collections.defaultdict(list)
 for (src,m,e,u),rr in rb.items():eb[(src,m,e)].append({k:float(np.mean([r[k] for r in rr])) for k in ['coverage','radius','ade','fallback_fraction']})
 env=[dict(dataset=s,method=m,environment_id=e,**{k:float(np.mean([r[k] for r in rr])) for k in rr[0]}) for (s,m,e),rr in eb.items()];summary=[]
 for src in sorted({r['dataset'] for r in env}):
  for m in ['age_only','conditional']:
   rr=[r for r in env if r['dataset']==src and r['method']==m];summary.append(dict(dataset=src,method=m,environments=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['coverage','radius','ade','fallback_fraction']}))
 return env,summary
train=samples('train');envs=sorted({r['environment_id'] for r in train});oof=[];folds=[]
for e in envs:
 fitting=[r for r in train if r['environment_id']!=e];held=[r for r in train if r['environment_id']==e];model=fit(fitting);assert all(r['environment_id']!=e for r in fitting);oof.extend(evaluate(held,model,e));folds.append(dict(held_environment=e,fitting_environments=sorted({r['environment_id'] for r in fitting}),fitting_windows=len(fitting),held_windows=len(held)));print('E2 fold',e,flush=True)
env,summary=aggregate(oof);writecsv(O/'train_oof_windows.csv',oof);writecsv(O/'train_oof_environments.csv',env);writecsv(O/'train_summary.csv',summary);dump(O/'fold_manifest.json',folds)
comparisons=[]
for src in sorted({r['dataset'] for r in summary}):
 a=next(r for r in summary if r['dataset']==src and r['method']=='age_only');b=next(r for r in summary if r['dataset']==src and r['method']=='conditional');comparisons.append(dict(dataset=src,baseline_coverage=a['coverage'],conditional_coverage=b['coverage'],coverage_error_reduction=abs(a['coverage']-.9)-abs(b['coverage']-.9),radius_ratio=b['radius']/a['radius'],coverage_drop=a['coverage']-b['coverage'],fallback_fraction=b['fallback_fraction']))
improvement=float(np.mean([r['coverage_error_reduction'] for r in comparisons]));admitted=improvement>0 and all(r['radius_ratio']<=1.2 and r['coverage_drop']<=.02 for r in comparisons);writecsv(O/'admission_by_source.csv',comparisons)
dump(O/'selection_before_validation.json',dict(admitted=admitted,mean_coverage_error_reduction=improvement,comparisons=comparisons,test_evaluated=False))
finalmodel=fit(train);dump(O/'train_fitted_calibration.json',finalmodel)
if admitted:
 val=evaluate(samples('validation'),finalmodel,'frozen_train');ve,vs=aggregate(val);writecsv(O/'validation_windows.csv',val);writecsv(O/'validation_environments.csv',ve);writecsv(O/'validation_summary.csv',vs)
checks=[dict(name='all 16 train environment folds',passed=len(envs)==16),dict(name='each fold excludes held environment',passed=all(f['held_environment'] not in f['fitting_environments'] for f in folds)),dict(name='one point prediction for both calibrators',passed=all(abs(next(x['ade'] for x in oof if x['window_id']==r['window_id'] and x['method']=='age_only')-r['ade'])<1e-12 for r in oof if r['method']=='conditional')),dict(name='test not evaluated',passed=all(r['split']=='train' for r in oof)),dict(name='finite radius nonnegative',passed=all(np.isfinite(r['radius']) and r['radius']>=0 for r in oof))]
writecsv(O/'checks.csv',checks);dump(O/'result.json',dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',admitted=admitted,train_windows=len(train),oof_rows=len(oof),folds=len(envs),validation_run=admitted,mean_coverage_error_reduction=improvement,runtime_s=time.perf_counter()-start,checks=checks,test_evaluated=False))
lines=['# E2 条件误差范围校准报告','',f'固定E1阻尼预测器tau={TAU}秒；16训练环境留一，{len(train)}个训练窗口。每折速度/转向分界及90%分位数仅拟合其余训练环境。','', '|来源|原覆盖|条件覆盖|覆盖偏差减少|半径比|回退比例|','|---|---:|---:|---:|---:|---:|']
for r in comparisons:lines.append(f'|{r["dataset"]}|{r["baseline_coverage"]:.4f}|{r["conditional_coverage"]:.4f}|{r["coverage_error_reduction"]:.4f}|{r["radius_ratio"]:.4f}|{r["fallback_fraction"]:.3f}|')
lines += ['',f'训练准入：{admitted}；来源等权覆盖偏差减少={improvement:.6f}。','若未准入，停止候选验证和重观测替换，保留age-only校准；不得改阈值或减少100点要求以追求通过。未准入也构成已完成的负结果，不称阶段二整体完成。','若准入，validation_summary为冻结后单次验证；重观测收益尚需单独回放，不能仅凭覆盖结果宣布E2全部完成。','', '95%环境区间不能靠单环境估计；不将窗口当独立环境。训练样本桶分位数使用点级经验分位数，评价使用记录→环境等权；稀疏条件桶回退，需结合回退比例解释。','',f'检查{sum(c["passed"] for c in checks)}/{len(checks)}通过。测试集未评价；未改动基线及E1。']
(O/'实验报告.md').write_text('\n'.join(lines),encoding='utf8');dump(O/'code_hashes.json',{Path(__file__).name:sha(Path(__file__))});print(json.dumps(load(O/'result.json'),ensure_ascii=False,indent=2))
