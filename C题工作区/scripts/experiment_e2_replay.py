from pathlib import Path
import sys,collections,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'阶段交付/stage1_review/corrected_code'));from common import *
O=R/'阶段交付/stage2_optimization/E2';model=load(O/'train_fitted_calibration.json');P=load(R/'阶段交付/stage1_review/corrected_results/data/protocol.json');ws=[w for w in readcsv(R/'阶段交付/stage1_review/corrected_results/data/windows.csv') if w['split']=='validation'];rows=[]
for uid in sorted({w['episode_uuid'] for w in ws}):
 with np.load(DATA/'records'/(uid+'.npz')) as z:
  for w in [w for w in ws if w['episode_uuid']==uid]:
   t=z[w['side']+'__t_s'];x=state(z,w['side']);i=int(w['start_index']);j=int(w['end_index_exclusive']);h=int(w['history_start']);m=model[w['dataset']]
   for method in ['age_only','conditional']:
    for delay in [0,.1,.2]:
     for frac in [.02,.05,.1]:
      start=time.perf_counter();visible=[(float(t[k]),x[k].copy()) for k in range(h,i)];pending=None;lastrequest=-1e30;n=0;errors=[];covers=[];radii=[];events=[];fallbacks=[]
      for k in range(i,j):
       now=float(t[k])
       if pending is not None and now>=pending[0]-1e-12:visible.append((pending[1],pending[2]));pending=None
       def predict():
        recent=[v for v in visible if v[0]>=visible[-1][0]-.2-1e-12];steps=np.diff(np.array([v[1] for v in recent]),axis=0);dt=np.diff([v[0] for v in recent]);norm=np.linalg.norm(steps,axis=1);speed=float(np.median(norm/dt)) if len(dt) else 0.;turn=[]
        for a in range(1,len(steps)):
         if norm[a]>1e-12 and norm[a-1]>1e-12:turn.append(float(np.arccos(np.clip(steps[a]@steps[a-1]/norm[a]/norm[a-1],-1,1))))
        angle=float(np.median(turn)) if turn else 0.;b=min(30,max(1,int(np.ceil((now-visible[-1][0])*30))));key=f'{b}|{int(speed>=m["speed_median"])}|{int(angle>=m["turn_median"])}';radius=m['base'][str(b)] if method=='age_only' else m['conditional'][key];v=(visible[-1][1]-visible[-2][1])/(visible[-1][0]-visible[-2][0]);pred=visible[-1][1]-.2*np.expm1(-(now-visible[-1][0])/.2)*v;return pred,radius,method=='conditional' and m['counts'][key]<100
       pred,rad,fb=predict()
       if rad>=P['train_scale'][w['dataset']]*frac and pending is None and n<2 and now-lastrequest>=.2-1e-12 and now+delay<=t[i]+1+1e-12:
        pending=(now+delay,now,x[k].copy());n+=1;lastrequest=now;events.append(dict(request_s=float(now-t[i]),return_s=float(now+delay-t[i])))
        if delay==0:visible.append((now,x[k].copy()));pending=None;pred,rad,fb=predict()
       err=float(np.linalg.norm(pred-x[k]));errors.append(err);covers.append(err<=rad);radii.append(rad);fallbacks.append(fb)
      rows.append({k:w[k] for k in ['window_id','episode_uuid','dataset','environment_id','side']}|dict(method=method,delay_s=delay,tolerance_fraction=frac,ade=float(np.mean(errors)),fde=errors[-1],coverage=float(np.mean(covers)),radius=float(np.mean(radii)),requests=n,cost=n,waiting_s=n*delay,fallback_fraction=float(np.mean(fallbacks)),runtime_s=time.perf_counter()-start,actions=json.dumps(events)))
writecsv(O/'replay_windows.csv',rows);summaries=[]
for key in sorted({(r['dataset'],r['method'],r['delay_s'],r['tolerance_fraction']) for r in rows}):
 rr=[r for r in rows if (r['dataset'],r['method'],r['delay_s'],r['tolerance_fraction'])==key]
 for metric in ['ade','fde','coverage','radius','requests','cost','waiting_s','fallback_fraction']:
  rb=collections.defaultdict(list)
  for r in rr:rb[(r['episode_uuid'],r['environment_id'])].append(r[metric])
  summaries.append(dict(zip(['dataset','method','delay_s','tolerance_fraction'],key))|dict(metric=metric,**grouped_boot([float(np.mean(v)) for v in rb.values()],[k[1] for k in rb])))
writecsv(O/'replay_summary.csv',summaries)
val=readcsv(O/'validation_summary.csv');comp=[]
for src in sorted(model):
 a=next(r for r in val if r['dataset']==src and r['method']=='age_only');b=next(r for r in val if r['dataset']==src and r['method']=='conditional');comp.append(dict(dataset=src,baseline_coverage=float(a['coverage']),conditional_coverage=float(b['coverage']),coverage_error_reduction=abs(float(a['coverage'])-.9)-abs(float(b['coverage'])-.9),radius_ratio=float(b['radius'])/float(a['radius'])))
writecsv(O/'validation_comparison.csv',comp)
# Paired environment differences, equal records first. No cross-environment CI for one env.
ve=readcsv(O/'validation_environments.csv');pairs=[]
for src in sorted(model):
 a={r['environment_id']:r for r in ve if r['dataset']==src and r['method']=='age_only'};b={r['environment_id']:r for r in ve if r['dataset']==src and r['method']=='conditional'};v=[abs(float(a[e]['coverage'])-.9)-abs(float(b[e]['coverage'])-.9) for e in a];pairs.append(dict(dataset=src,**grouped_boot(v,list(a))))
writecsv(O/'validation_paired_coverage.csv',pairs)
retain=all(r['coverage_error_reduction']>0 and r['radius_ratio']<=1.2 for r in comp) and all(r['lo'] is not None and r['lo']>0 for r in pairs)
checks=dict(rows=len(rows),expected=212*2*3*3,request_limit=all(r['requests']<=2 for r in rows),finite=all(np.isfinite(r['ade']) for r in rows),test_evaluated=False)
dump(O/'replay_checks.json',checks);dump(O/'handoff.json',dict(retain_conditional=retain,decision='keep age-only unless consistent supported gain; conditional remains exploratory',point_predictor='damped tau=.2',test_evaluated=False))
with (O/'实验报告.md').open('a',encoding='utf8') as f:
 f.write('\n\n## 验证及重观测\n\n|来源|原覆盖|条件覆盖|覆盖偏差减少|半径比|\n|---|---:|---:|---:|---:|\n')
 for r in comp:f.write(f'|{r["dataset"]}|{r["baseline_coverage"]:.4f}|{r["conditional_coverage"]:.4f}|{r["coverage_error_reduction"]:.5f}|{r["radius_ratio"]:.4f}|\n')
 f.write(f'\n完成{len(rows)}条重观测回放，3延迟×3容差，两方法共享阻尼点预测与预算。主条件.1秒/5%容差详见replay_summary.csv，逐窗口请求与返回见replay_windows.csv。验证配对覆盖误差区间见validation_paired_coverage.csv；单环境不可估计跨环境区间。\n\n条件校准作为最终保留方法：{retain}。当前保留age-only为正式对照，条件校准按探索结果报告，不继续调整分桶追逐验证。该保守结论不等于条件校准在任何场景都无用。测试未评价。\n\n注意：回放观测更新后历史可能稀疏，速度/转向特征仅用已返回观测，和连续可见历史校准分布不同，回放结果单独报告，不能用开环覆盖代替。\n')
print(json.dumps(dict(checks=checks,comparisons=comp,retain=retain),ensure_ascii=False,indent=2))
