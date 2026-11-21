"""Audit the adopted model's phase/source failures without refitting or selection."""
from pathlib import Path
import gzip,json
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
def main():
 out=BASE/'results';rows=[]
 with gzip.open(out/'joint_points.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   p=json.loads(line);q=p['range'];err=p['error']/p['scale'];radius=q['radius']/p['scale'];delta=err-radius
   rows.append({k:p[k] for k in ['split','dataset','environment_id','episode_uuid','side','window_id']}|dict(phase=q['phase'],coverage=float(delta<=0),width=2*radius,pinball=max(.9*delta,-.1*delta),nade=err))
 d=pd.DataFrame(rows);metrics=['coverage','width','pinball','nade'];keys=['split','dataset','phase']
 window=d.groupby(keys+['environment_id','episode_uuid','side','window_id'],as_index=False)[metrics].mean()
 g=window
 for lev in [['environment_id','episode_uuid','side'],['environment_id','episode_uuid'],['environment_id']]:g=g.groupby(keys+lev,as_index=False)[metrics].mean()
 g.to_csv(out/'phase_environments.csv',index=False)
 summary=g.groupby(keys,as_index=False)[metrics].mean()
 support=d.groupby(keys).agg(points=('coverage','size'),windows=('window_id','nunique'),records=('episode_uuid','nunique'),environments=('environment_id','nunique')).reset_index()
 summary=summary.merge(support,on=keys,validate='one_to_one');summary['below_85_percent']=summary.coverage<.85
 summary.to_csv(out/'phase_sources.csv',index=False)
 lines=['# 当前方案分阶段失败边界审查','','本审查复用全量运行保存的逐点输出，不训练、不选模型、不修改冻结参数。覆盖率按窗口、手、记录、环境逐层平均；来源分别报告。85%是已有来源工程门槛在子组中的描述性参考线，不能冒充新注册验收标准。','','|划分|来源|阶段|覆盖率|环境数|记录数|窗口数|评分点|','|---|---|---|---:|---:|---:|---:|---:|']
 for r in summary.itertuples():lines.append(f'|{r.split}|{r.dataset}|{r.phase}|{r.coverage:.2%}|{r.environments}|{r.records}|{r.windows}|{r.points}|')
 lines+=['','## 可确认的问题与处理边界','','1. 总体覆盖通过不能推导首次返回前、返回后均可靠；低覆盖子组必须单列。','2. 环境数少的来源不能给出可信的跨环境保证；评分点多不能补足独立环境数量。','3. 不能仅根据这些已查看结果重调参数并宣布独立改善。若未来改模型，必须重新匹配其范围模型。','4. 当前付费观测范围不适用于自然坏帧段；该段没有独立定位真值，不能用被判坏的位置自身证明修复精度。','5. 当前继续保留已采用方案；本次审查没有产生新候选，也没有解决新增独立数据与官方协议缺失。','','详表见 results/phase_sources.csv 与 results/phase_environments.csv。']
 (BASE/'第二问分阶段失败边界.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(summary.to_string(index=False))
if __name__=='__main__':main()
