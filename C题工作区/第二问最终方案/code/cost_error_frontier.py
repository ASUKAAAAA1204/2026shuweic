"""Predeclared fixed-policy resource levels, no adaptive policy or parameter search."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from candidate_replay_engine import replay,FixedPolicy
from run_complete import aggregate
BASE=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--windows',type=Path,required=True);a=ap.parse_args();cfg=json.loads((BASE/'model/model.json').read_text(encoding='utf8'));folds=json.loads((BASE/'model/fold_scales.json').read_text(encoding='utf8'));w=pd.read_csv(a.windows);cache={};rows=[]
 for r in w.itertuples():
  if r.episode_uuid not in cache:
   with np.load(a.data_root/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[r.episode_uuid]={s:(z[s+'__t_s'],z[s+'__position'].mean(axis=1)) for s in ['left_hand','right_hand']}
  t,x=cache[r.episode_uuid][r.side];i=int(r.start_index);j=int(r.end_index_exclusive);scale=folds[r.environment_id][r.dataset] if r.split=='train' else cfg['scale'][r.dataset]
  for budget in [0,1,2]:
   run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),budget=budget,delay=.1,window_end=float(t[j-1]));m=run['metrics'];rows.append({k:getattr(r,k) for k in ['window_id','split','dataset','environment_id','episode_uuid','side']}|dict(budget=budget,nade=m['ade']/scale,nfde=m['fde']/scale,exceed_005=np.mean([p['error']/scale>.05 for p in run['points']]),**m))
 d=pd.DataFrame(rows);out=BASE/'results';d.to_csv(out/'cost_error_windows.csv.gz',index=False);metrics=['nade','nfde','exceed_005','requests','cost','waiting_s','strategy_compute_s'];s=aggregate(d,['split','budget'],metrics);s.to_csv(out/'cost_error_frontier.csv',index=False);aggregate(d,['split','budget','dataset'],metrics).to_csv(out/'cost_error_sources.csv',index=False)
 old=pd.read_csv(out/'window_metrics.csv').set_index('window_id');new=d[d.budget==2].set_index('window_id');diff=float(abs(old.nade-new.nade).max());assert diff<1e-12
 lines=['# 固定策略的成本—误差取舍','','保持同一状态、tau=0.2阻尼、割线更新、1秒窗口和0.1秒延迟。预算为0、1、2时，分别无请求、只在0.3秒请求、在0.3/0.6秒请求。单次成本1。三档是预先定义的资源对照，不优化单次请求时刻，不证明全部策略的Pareto最优。','','|划分|请求预算|实际平均成本|NADE|NFDE|超过5%比例|累计等待/秒|','|---|---:|---:|---:|---:|---:|---:|']
 for r in s.itertuples():lines.append(f'|{r.split}|{r.budget}|{r.cost:.2f}|{r.nade:.6f}|{r.nfde:.6f}|{r.exceed_005:.2%}|{r.waiting_s:.2f}|')
 lines+=['','## 解释限制','','固定预算下比较误差与误差约束下最小成本是不同目标。若主要目标为现有两次预算内精度，当前保留两次固定请求；若追求节省观测，必须明确允许增加多少误差。不能只因一次观测成本减半便认定其更优。','','例如可用表中数据判断0.05或0.1归一化平均误差约束下哪档可行，但这只是所比较三档中的最小成本；平均误差约束不等于95%时刻达标，更不等于物理安全要求。不得按已查看测试结果重新设置偏好后宣称独立最优。','','范围模型仍仅用于当前两次固定观测协议；没有向0/1次协议转移覆盖声明。所有划分此前已查看，结果为描述性复现。完整来源与逐窗口共同指标见 results/cost_error_sources.csv 与 cost_error_windows.csv.gz。']
 (BASE/'第二问成本与误差取舍.md').write_text('\n'.join(lines)+'\n',encoding='utf8');(out/'cost_error_audit.json').write_text(json.dumps(dict(windows=len(w),replays=len(d),budget_grid=[0,1,2],delay=.1,request_schedule=[.3,.6],maximum_current_scheme_difference=diff,model_selection=False,global_optimum_claim=False),indent=2),encoding='utf8');print(s.to_string(index=False))
if __name__=='__main__':main()
