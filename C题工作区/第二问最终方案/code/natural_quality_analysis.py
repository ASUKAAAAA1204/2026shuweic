"""Observed quality-flag episodes, not verified physical occlusion or true error."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);args=ap.parse_args();index=pd.read_csv(args.data_root/'record_index.csv');records=[];episodes=[]
 for r in index.itertuples():
  with np.load(args.data_root/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];known=z[side+'__quality_known'];bad=z[side+'__bad'];ref=z[side+'__reference_candidate'];finite=z[side+'__finite'];dt=np.diff(t);assert np.all(dt>0)
    gap=3*float(np.median(dt));flag=known&bad;meta=dict(split=r.split,dataset=r.dataset,environment_id=r.environment_id,episode_uuid=r.episode_uuid,side=side)
    records.append(meta|dict(points=len(t),quality_known=int(known.sum()),bad_points=int(flag.sum()),unknown_points=int((~known).sum()),reference_points=int(ref.sum()),nonreference_points=int((~ref).sum()),known_bad_fraction=float(flag.sum()/known.sum()) if known.any() else None,gap_threshold_s=gap))
    stream_start=len(episodes);k=0
    while k<len(t):
     if not flag[k]:k+=1;continue
     start=k;k+=1
     while k<len(t) and flag[k] and t[k]-t[k-1]<=gap:k+=1
     stop=k;observed_right=stop<len(t) and t[stop]-t[stop-1]<=gap and known[stop] and not bad[stop]
     observed_left=start>0 and t[start]-t[start-1]<=gap and known[start-1] and not bad[start-1]
     # Run ends by known good, unknown quality, large timestamp gap, or record edge.
     reason='known_good' if observed_right else ('record_end' if stop==len(t) else 'time_gap' if t[stop]-t[stop-1]>gap else 'unknown_quality')
     recovery_pair=bool(observed_right and stop+1<len(t) and t[stop+1]-t[stop]<=gap and ref[stop] and ref[stop+1])
     episodes.append(meta|dict(first_bad_time=float(t[start]),last_bad_time=float(t[stop-1]),bad_samples=stop-start,observed_bad_span_s=float(t[stop-1]-t[start]),first_bad_to_first_good_s=float(t[stop]-t[start]) if observed_right else None,left_transition_observed=bool(observed_left),right_transition_observed=bool(observed_right),end_reason=reason,first_good_reference=bool(ref[stop]) if observed_right else False,two_reference_recovery_samples=recovery_pair))
    starts=flag & np.r_[True,(~flag[:-1])|(np.diff(t)>gap)]
    assert int(starts.sum())==len(episodes)-stream_start
 frame=pd.DataFrame(records);runs=pd.DataFrame(episodes);out=BASE/'results';frame.to_csv(out/'natural_quality_records.csv',index=False);runs.to_csv(out/'natural_quality_episodes.csv',index=False)
 sums=frame.groupby(['split','dataset'],as_index=False)[['points','quality_known','bad_points','unknown_points','reference_points','nonreference_points']].sum();sums['known_bad_fraction']=sums.bad_points/sums.quality_known;sums.to_csv(out/'natural_quality_sources.csv',index=False)
 summary=[]
 for (split,src),g in runs.groupby(['split','dataset']):
  complete=g[g.left_transition_observed&g.right_transition_observed]
  summary.append(dict(split=split,dataset=src,runs=len(g),both_transitions_observed=len(complete),observed_span_median_s=float(g.observed_bad_span_s.median()),observed_span_p90_s=float(g.observed_bad_span_s.quantile(.9)),complete_transition_duration_median_s=float(complete.first_bad_to_first_good_s.median()) if len(complete) else None,recovered_two_reference_samples=int(g.two_reference_recovery_samples.sum())))
 summary=pd.DataFrame(summary);summary.to_csv(out/'natural_quality_duration_summary.csv',index=False)
 assert frame.points.sum()==frame.quality_known.sum()+frame.unknown_points.sum();assert int(runs.bad_samples.sum())==int(frame.bad_points.sum())
 audit=dict(records=len(index),hand_streams=len(frame),points=int(frame.points.sum()),bad_points=int(frame.bad_points.sum()),runs=len(runs),all_bad_samples_accounted=True,position_error_computed_in_bad_intervals=False,quality_label_is_not_physical_occlusion_truth=True)
 (out/'natural_quality_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
 lines=['# 自然质量下降与恢复：全量数据分析','','分析69条记录的双手流，仅以quality_known且bad为已知低质量。未知质量单列；reference_candidate不成立还可能包含数值、时间、质量等条件，不能全部解释为遮挡。下表为原始采样点比例，不是时间占比，也不是环境等权风险估计。','','|划分|来源|已知质量点|标坏点|标坏比例|未知质量点|','|---|---|---:|---:|---:|---:|']
 for r in sums.itertuples():lines.append(f'|{r.split}|{r.dataset}|{r.quality_known}|{r.bad_points}|{r.known_bad_fraction:.2%}|{r.unknown_points}|')
 lines+=['','## 持续时间与恢复','','以连续已知标坏点构成片段；相邻时间差超过该手流中位采样间隔的3倍时拆段，未知质量也结束已知标坏片段。observed_bad_span_s为最后坏点减最初坏点，单点段为0，不能理解为真实遮挡持续0秒。只有两侧均观测到连续已知好点时，才汇总first_bad_to_first_good_s；它仍是离散采样定义，不是精确遮挡起止。边界、时间缺口和未知质量截断单独保留，不将其当作完整事件。','','恢复统计要求首次好点及下一连续点均满足reference_candidate，表示有两个合格观测可恢复速度估计，不表示恢复误差已达标。缺失内部不计算位置真误差；原始质量标记并非独立物理遮挡真值。','','人工遮罩从合格参考窗口隐藏数据，能够计算与记录的预测差异；自然标坏段的坐标自身不再是可靠参照。两者的保留机制与样本难度不同，不能把人工遮罩覆盖率直接迁移到自然段。','','完整片段、截断类型、持续时间、恢复支持保存在 results/natural_quality_episodes.csv 和 natural_quality_duration_summary.csv；逐记录分母与排除量在 natural_quality_records.csv。此次未修改预测器，也未证明自然恢复定位精度。']
 (BASE/'第二问自然缺失与恢复分析.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(audit));print(summary.to_string(index=False))
if __name__=='__main__':main()
