import gzip
from pathlib import Path
import json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
def table(d,cols):
 lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
 for r in d[cols].itertuples(index=False,name=None):lines.append('|'+'|'.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in r)+'|')
 return '\n'.join(lines)
def main():
 out=BASE/'results/validation';tr=pd.read_csv(out/'trials.csv');summary=pd.read_csv(out/'summary.csv');selected=summary[summary.method=='density_minimum'];nom=selected[(selected.alpha==1)&(selected.beta==.5)];pairs=[];rng=np.random.default_rng(20261121)
 for mapping,g in tr.groupby(['alpha','beta']):
  for case,z in g.groupby('case_id'):
   wide=z.pivot(index='seed',columns='method',values='weighted_on_time_rate')
   for baseline in ['edf_minimum','density_median','edf_median']:
    delta=(wide.density_minimum-wide[baseline]).to_numpy();draw=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(axis=1);lo,hi=np.quantile(draw,[.025,.975]);pairs.append(dict(alpha=mapping[0],beta=mapping[1],case_id=case,baseline=baseline,mean_delta=delta.mean(),lo95=lo,hi95=hi))
 paired=pd.DataFrame(pairs);paired.to_csv(out/'paired_differences.csv',index=False)
 stats=tr.groupby(['case_id','alpha','beta','method']).weighted_on_time_rate.agg(['mean','std','min','max']).reset_index();stats.to_csv(out/'variation.csv',index=False)
 failures=[]
 with gzip.open(out/'actions.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);e=r['event']
   if r['method']=='density_minimum' and r['alpha']==1 and r['beta']==.5 and e.get('censored',0):failures.append({k:r[k] for k in ['case_id','seed','alpha','beta']}|dict(job_id=e['action']['job_id'],decision_time=e['decision_time'],service_start=e['service_start'],realized_duration=e['realized_duration'],reserve=e['reserve'],source_reference=e['source_reference']))
 pd.DataFrame(failures).to_csv(out/'censored_examples.csv',index=False)
 duration=pd.read_csv(BASE/'results/duration_information/summary.csv');audit=json.loads((BASE/'audit/independent_delivery_checks.json').read_text());info=json.loads((BASE/'results/duration_information/decision.json').read_text());model=json.loads((BASE/'model/family_duration.json').read_text())
 text='''# 第三问：不确定作业时长下的补货调度完整解答

## 结论与研究目标

以附件固定任务为对象，主目标最大化加权按时完成率。训练证据选择：任务族经验时长模型、权重/预计占用排序、最小合法预留0.2秒、无付费复核、无可执行任务时等到最近释放时刻，否则停止。方案是有约束的可复现启发式，不宣称全局最优。

第一问提供同一手部21点质心、时间与质量语义；第二问提供“信息有延迟、风险估计必须匹配决策条件”的分析原则。第三问重新建模的是语义作业区间的时长，而非空间位置误差。任务族不是商品类别，一个源区间不是一个商品；不报告实体机器人成功率。

## 数据、协议与信息边界

官方依据为Problem_3_Requirements.pdf及同目录CSV/JSON。349区间中343个长度至少1.2秒进入抽样；训练246、验证35、测试62。环境划分16/4/4，不按相邻帧随机拆分。合格抽样不保证观测质量。源索引行序与处理后索引已逐项核对。

每案例按job_id排序，random.Random(seed)逐项从同族/同split源池choice有放回抽样；每案例重置生成器，策略之间复用相同分配。验证使用官方2026111700—2026111729，三个案例五映射，每方法450回放。训练探索使用提前固定的2026112000—2026112029。策略不接收seed、源标识、总时长、区间终点或未披露前缀。当前方案没有从执行反馈跨试验学习。

S1/S2/S3任务8/12/16，H=100/100/125，B=32/34/40，复核上限4/6/8。释放后免费提供0.2源秒前缀；两次复核依次扩展至0.6/1.0，每次1情景秒和1成本。等待不增加披露。执行、复核、等待单设备串行。第三问不能使用第二问0.1秒返回延迟。

## 数学模型与预留结构

任务j权重w_j，释放r_j，截止d_j，前驱集合P_j。当前位置z、时间t，移动距离D_j，服务开始s_j=t+D_j+0.5。实际服务p_j=alpha*d_source+beta；执行前未知。选择预留b_j>=0.2，费用f_j=2+0.1b_j+0.2D_j，要求累计费用不超B且s_j+b_j<=H；不能以隐藏p_j做执行前检查。

完成C_j=s_j+p_j，设备释放s_j+max(p_j,b_j)。跨H未完即截尾并终止，不退款。主指标J=sum(w_j*1{完成且C_j<=d_j})/sum(w_j)。所有未完成任务保留分母；拖期将未完C_j置H，再求sum(w_j*max(C_j-d_j,0))，不除总权重。

**最小预留的结构依据。**固定一个任务的开始和实际时长，缩小b不改变C，不增加执行费用、设备占用或计划可行性限制；较早可用的资源可通过合法等待复现原后续安排。因此，当主目标不惩罚超预留时，小预留存在弱优势。并非某个时间依赖排序必然逐试验占优，也不是预留能改变真实p。代价是观测超预留显著增加，必须共同报告。

本题没有规定超预留额外罚款，不暗加规则。若以容量风险为主目标，则需另行定义偏好并遵守每案例名义平均按时率至少0.50要求；本解答主目标是按时率，不能将最小预留称为低容量风险方案。预计实际时长与预留严格分离。

## 时长模型、区间与额外前缀的信息价值

采用训练族经验分布F_f。点预测为族中位数，区间采用10%/90%经验分位数，服务映射为alpha乘以源时长分位数再加beta。它是经验分布，不承诺80%跨环境覆盖。最终全训练分位数见model/family_duration.json。冻结模型在35个验证源单元上按环境等权MAE为4.0333源秒，名义80%区间覆盖77.68%、平均宽度11.4525源秒、90%分位损失1.1940；不是450次抽样任务的独立预测样本数。逐族、环境及区间记录见validation_units与validation_families。

用固定ridge=100的对数时长模型作为信息对照，不搜索新模型。每个前缀提取两手采样数、有效比例、合格点端点位移、仅相邻合格点路径长度及中位速度，再加族指示。坏坐标不进入几何量；无未来填充。外层整环境留一，前缀区间的加性残差由外层训练中的内层环境留一产生，避免外层泄漏。此模型用于诊断，不部署。

训练环境等权评价如下（单位源秒，coverage80为名义80%区间的实测覆盖）：

'''+table(duration,['method','length','mae','coverage80','width80','pinball90'])+'''

1秒前缀比0.2秒改善约0.102源秒，13/16环境改善，但未胜族统计基线。部分区间损失改善伴随更宽范围；不能以一项损失证明复核值得。缺少稳定的基线优势与决策增量证据，因此不使用付费复核，不把简单固定复核负结果扩展成“任何信息都无价值”。

## 排序算法、等待与停止

在所有已释放、未尝试且前驱完成的任务中，先计算预计服务p_hat=max(0.2,alpha*族中位数+beta)。给定b=0.2，预计占用c_j=D_j+0.5+max(p_hat,b)。剔除执行费用或计划预留不可行的任务，选择w_j/c_j最大者；同分依次按截止和job_id。只提交当前执行，实际反馈后重新决策。

若当前无可执行任务但有H前尚未释放任务，等待至最近释放；否则停止。该规则可能等待一个后续仍不可行的任务，合法但不声称最优。晚于截止仍允许执行，能影响完成率和拖期。没有先验承诺每项都完成。

```text
初始化时间0、区域Z0、费用0；逐试验清空状态
循环：读取当前公开任务条件、已返回信息、执行反馈
  枚举已释放且前驱完成的未尝试任务
  用训练族中位数估时，检查费用与最小预留时域合法性
  若有候选，执行最大权重/预计占用者，预留0.2
  否则若有未来释放，等待到最近释放；否则停止
  H到达、非法动作或停止时结束
```

## 有限实验与冻结取舍

四个固定对照为EDF/density×最小/中位预留，均无复核。训练完整案例环境留一只有两个环境可评价：16×3个环境/案例中42个缺任务族，不能用训练源填补。训练共3600回放；这是支持限制，不是16环境完整调度泛化。

唯一新增排序候选利用直接后继价值：日志中S2/S3各4/22个未完成任务受未完前驱阻塞。候选用可立即解锁且预计按时、预算可行的直接后继收益修正当前密度，不扩大搜索深度。1800训练回放平均提升0.139个百分点，条件种子重抽样区间约[-0.046,0.417]个百分点，未晋级。第二个付费复核候选因信息筛选失败而不启动，不用剩余预算另造路线。旧两步前瞻失败不重复。

在本轮验证前已冻结density_minimum。验证历史上已被查看，本轮没有以其调参；不能称完全独立确认。没有按验证中S1偏好EDF改成分案例策略。

## 名义验证结果

'''+table(nom,['case_id','weighted_completion_rate','weighted_on_time_rate','weighted_tardiness_s','cost','observations','elapsed_s','observed_overrun_s','reserved_idle_s','censored_units','invalid_actions'])+'''

三个案例平均按时率约81.57%，各自均高于0.50，但此条件不能替代主目标表现。零闲置来自最小预留结构，不代表实际没有容量风险；超预留及截尾明显，S3平均接近H。EDF在S1优于最终方案，density在S2/S3优于EDF，偏好存在冲突。

同种子比较最终方案与三个基线，名义加权按时率差及条件95%区间：

'''+table(paired[(paired.alpha==1)&(paired.beta==.5)],['case_id','baseline','mean_delta','lo95','hi95'])+'''

区间基于固定验证源池30个抽样重复，条件于现有少量环境；不是对新环境的泛化置信保证，也未校正项目历史选择。完整逐种子差、标准差/最小/最大值在paired_differences.csv与variation.csv。

## 五映射敏感性

'''+table(selected,['case_id','alpha','beta','weighted_on_time_rate','weighted_completion_rate','cost','observed_overrun_s','censored_units'])+'''

按规定只改变alpha或beta，不构造全因子网格。服务变长时截止和H风险增加；最小预留不修复该风险。所有对照和方案的完整11项指标保存在results/validation/summary.csv与trials.csv，不能只引用本表有利项。

## 失败与可靠性边界

执行前只检查s+b而非隐藏s+p，故合法指令也可能截尾。具体截尾任务、源引用、开始与实际时长保存在censored_examples.csv及逐动作日志，不能删除失败样本。长尾任务族时长和样本不足造成真实容量风险；族区间仅约66.6%训练外层覆盖，并不达到名义80%。不使用加宽区间包装可靠性成功。

S1的EDF优势说明单一密度偏好可能牺牲紧截止任务；本轮不再通过验证特设规则。前驱解锁候选未有稳定收益，不能宣称其更复杂所以更优。

自然手部坏标记、位置误差和时长风险是不同变量。原始前缀更长未稳定胜族基线，不表示原始数据毫无信息，更不表示所有模型不可提升。本解答不依赖不存在的定位真值或视频额外输入。

## 复现、验收与测试缺口

结果包括四方法1800验证回放，最终方案450次；源分配按官方行序和随机序列独立重建。独立检查从任务条件和实际源时长重算每个动作的几何、费用、前驱、H、完成和全部核心计分，最大差'''+str(audit['maximum_score_difference'])+'''。所有试验非法动作0；隐藏未来扰动检查20763次，另有数值边界与质量坐标扰动检查。

正式最终测试需要组织方额外冻结的30种子，本地没有，故未编造或提交“官方最终测试成绩”。可复现的规定验证、五映射和完整论证已完成；该外部步骤必须明确留待种子获得后执行，不能将目前成绩写成官方测试。源数据、Python/numpy/pandas/PyMuPDF环境、模型与代码哈希见交付入口和清单。

## 最终选择

采用简单而证据透明的族统计、密度排序、最小预留、无复核。它利用题目允许的目标与费用结构，减少无收益信息开销；仍有截止冲突、长尾和截尾风险，不是所有指标都优于EDF。保留全部共同指标和失败证据，不宣称全局最优或实体机器人性能。
'''
 (BASE/'第三问完整解答.md').write_text(text,encoding='utf8');print('Report written; selected nominal mean',nom.weighted_on_time_rate.mean())
if __name__=='__main__':main()
