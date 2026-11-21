"""Descriptive tolerance horizons of the adopted fixed policy, no safety guarantee."""
from pathlib import Path
import gzip,json
import numpy as np
import pandas as pd
from run_complete import aggregate
BASE=Path(__file__).resolve().parents[1]
def main():
 out=BASE/'results';groups={}
 with gzip.open(out/'joint_points.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   p=json.loads(line);groups.setdefault(p['window_id'],[]).append(p)
 rows=[]
 for wid,points in groups.items():
  meta={k:points[0][k] for k in ['window_id','split','dataset','environment_id','episode_uuid','side']}
  for h in [.0,.05,.1,.2,.3,.4,.6,.8,1.]:
   selected=[p for p in points if p['relative']<=h+1e-9];e=np.array([p['error']/p['scale'] for p in selected])
   for tol in [.02,.05,.1]:rows.append(meta|dict(horizon=h,tolerance=tol,nade=float(e.mean()),point_success=float(np.mean(e<=tol)),whole_prefix_success=float(np.all(e<=tol)),mean_error_success=float(e.mean()<=tol),observed_last_relative=selected[-1]['relative'],points=len(e)))
 d=pd.DataFrame(rows);metrics=['nade','point_success','whole_prefix_success','mean_error_success','observed_last_relative'];keys=['split','horizon','tolerance']
 s=aggregate(d,keys,metrics);s.to_csv(out/'tolerance_horizons.csv',index=False)
 src=aggregate(d,keys+['dataset'],metrics);src.to_csv(out/'tolerance_horizon_sources.csv',index=False)
 assert (d.whole_prefix_success<=d.point_success+1e-12).all()
 for _,g in s.groupby(['split','tolerance']):assert (np.diff(g.sort_values('horizon').whole_prefix_success)<=1e-12).all()
 bounds=[]
 for (split,tol),g in s.groupby(['split','tolerance']):
  g=g.sort_values('horizon')
  for criterion,col in [('mean_error','nade'),('90pct_points','point_success'),('90pct_whole_prefix','whole_prefix_success')]:
   ok=(g[col]<=tol) if col=='nade' else (g[col]>=.9)
   # Require every earlier evaluated horizon to pass, not a later recovery after returns.
   valid=g[np.logical_and.accumulate(ok.to_numpy())]
   bounds.append(dict(split=split,tolerance=tol,criterion=criterion,last_passing_grid_horizon=float(valid.horizon.max()) if len(valid) else None,passes_at_first_scoring_frame=bool(ok.iloc[0]),right_censored=bool(ok.all())))
 b=pd.DataFrame(bounds);b.to_csv(out/'tolerance_supported_grid.csv',index=False)
 baseline=pd.read_csv(out/'required_baseline_summary.csv');baseline=baseline[baseline.method=='adopted_fixed']
 cross=s[s.tolerance==.05].merge(baseline,left_on=['split','horizon'],right_on=['split','duration'],suffixes=('_h','_b'))
 error=float(abs(cross.nade_h-cross.nade_b).max());complement=float(abs(cross.point_success-(1-cross.exceed_005)).max())
 assert len(cross)==9 and error<1e-12 and complement<1e-12
 (out/'tolerance_crosscheck.json').write_text(json.dumps(dict(passed=True,paired_split_horizon_checks=9,maximum_nade_difference=error,point_success_complement_max_difference=complement),indent=2),encoding='utf8')
 lines=['# 固定方案在自设容差内的经验持续时长','','容差为来源训练尺度的2%、5%、10%，不代表物理安全阈值。下表是检查网格上从评分首帧连续通过的最晚时长，不是连续时间精确交点；晚期返回带来的重新达标不会抹去此前超差。0秒表示评分首帧，历史锚点仍严格早于该帧。空值表示首帧已未通过。','','|划分|容差|标准|最后通过的检查时长/秒|全网格通过|','|---|---:|---|---:|---|']
 for r in b.itertuples():lines.append(f'|{r.split}|{r.tolerance:.0%}|{r.criterion}|{r.last_passing_grid_horizon if pd.notna(r.last_passing_grid_horizon) else "首帧未通过"}|{r.right_censored}|')
 lines+=['','mean_error：窗口内平均误差经层级平均后不超容差；90pct_points：平均逐点达标比例至少90%；90pct_whole_prefix：整段从未超容差的窗口层级比例至少90%。三者是不同描述目标，不等价。不同前缀的点数不同，保留实际末时间均值在原始表中。','','所有比例仍按窗口、手、记录、环境和来源逐层等权。该结果来自已经查看的记录，未给出统计安全保证，不应用于认证机器人持续操作。没有将经验范围覆盖率误当成容差达标概率。完整来源与时长曲线见 results/tolerance_horizon_sources.csv。','','本项给出当前采用方案的经验可持续边界，没有证明资源最优，也未解决新独立数据缺失。']
 (BASE/'第二问容差与持续时长.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(b[b.tolerance==.05].to_string(index=False))
if __name__=='__main__':main()
