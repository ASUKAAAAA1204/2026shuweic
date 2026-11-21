"""Build final report from frozen results; no fitting or policy changes."""
from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
from update_calibration_experiment import paired_ci
BASE=Path(__file__).resolve().parents[1]
def table(df):
 cols=list(df.columns);lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
 for row in df.itertuples(index=False,name=None):lines.append('|'+'|'.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+'|')
 return '\n'.join(lines)
def main():
 f=json.loads((BASE/'frozen_model.json').read_text());train=pd.read_csv(BASE/'results/S2/summary.csv');tables={};comparisons=[];sources={};runtime=[]
 for split in ['validation','test']:
  out=BASE/'results'/f'S5_{split}';a=pd.read_csv(out/'summary.csv');primary=a[(a.length_s==1)&((a.delay_s==.1)|a.method.isin(['hold','cv','damped']))]
  tables[split]=table(primary[['method','nade','nfde','exceed_0.05','coverage','path_covered','requests','waiting_s']])
  env=pd.read_csv(out/'environments.csv');g=env[(env.length_s==1)&((env.delay_s==.1)|env.method.isin(['hold','cv','damped']))]
  for ref,cand in [('damped','fixed'),('fixed','budget'),('fixed','threshold'),('fixed','delay_aware')]:
   item=paired_ci(g,ref,cand);item['split']=split;comparisons.append(item)
  ss=pd.read_csv(out/'sources.csv').query("method=='fixed' and length_s==1 and delay_s==0.1");sources[split]=table(ss[['dataset','nade','coverage','coverage_separate_exploratory','exceed_0.05']])
  m=pd.read_csv(out/'windows.csv').query("method=='fixed' and length_s==1 and delay_s==0.1")
  runtime.append(dict(split=split,windows=len(m),median_window_compute_ms=m.strategy_compute_s.median()*1000,p95_window_compute_ms=m.strategy_compute_s.quantile(.95)*1000,max_window_compute_ms=m.strategy_compute_s.max()*1000))
  m.sort_values('nade',ascending=False).head(10).to_csv(out/'worst_fixed_windows.csv',index=False)
  # Identify paid queries whose returns were not processed by last scoring point.
  act=pd.read_csv(out/'actions.csv.gz');nr=len(act[act.event=='request']);nt=len(act[act.event=='return'])
  (out/'end_boundary_diagnostic.json').write_text(json.dumps(dict(requests=nr,returns_processed_before_last_score=nt,paid_requests_without_processed_return_before_last_score=nr-nt,explanation='nominal mask endpoint may be after last original scoring timestamp; queries due in that gap are paid but cannot improve scored frames; waiting includes their full in-flight delay'),indent=2),encoding='utf8')
 pd.DataFrame(comparisons).to_csv(BASE/'paired_final_comparisons.csv',index=False);pd.DataFrame(runtime).to_csv(BASE/'runtime_summary.csv',index=False)
 support=[]
 for source,scale in f['scale'].items():
  for frac in [.02,.05,.1]:
   prefix=0
   for age in range(1,31):
    if f['radii']['damped'][f'{source}|{age}']>scale*frac:break
    prefix=age
   support.append(dict(dataset=source,tolerance_fraction=frac,continuous_accepted_age_bins=prefix,empirical_age_upper_s=prefix/30,physical_guarantee=False))
 pd.DataFrame(support).to_csv(BASE/'empirical_prediction_duration.csv',index=False)
 tr=train[(train.length_s==1)&((train.delay_s==.1)|train.method.isin(['hold','cv','damped']))]
 nat=pd.read_csv(BASE/'results/S0/natural_duration_summary.csv');scal=pd.read_csv(BASE/'results/S3/calibration_summary.csv');fixedrank=pd.read_csv(BASE/'results/S4_fixed/full_train_ranking.csv');adaptive=json.loads((BASE/'results/S4_adaptive/selection.json').read_text());auditdev=json.loads((BASE/'results/S5_validation/independent_audit.json').read_text());audittest=json.loads((BASE/'results/S5_test/independent_audit.json').read_text())
 text=f'''# 第二问完整解答与实验报告

## 结论与正式方案

第一问采用bf380c4的单手21点质心和tau=0.2秒固定阻尼。第二问正式保留：最后两次已有效接收观测的割线速度更新，0.3/0.6秒固定请求，每窗最多2次、每次成本1、单挂起；主返回延迟0.1秒。年龄误差球按来源、观测年龄的训练残差90%分位数给出；正式对照仍用开环年龄球，必须同时报告其欠覆盖。回传前后分开校准保留为探索结果，未通过预先规定的跨环境置信区间门槛，不能冒充正式90%保证。

研究结果支持“固定请求能显著降低给定回放条件下的平均预测误差”，不支持“已经实现5%容差可靠运行”或“固定请求全局最优”。同成本自适应ADE改善≥3%的争取目标未达成，成本节省≥15%且ADE退化≤3%的争取目标也未达成。工程与复现检查通过不等于性能目标达成。

## 状态、更新与动作模型

状态x是单手21个三维关键点的几何中心，保持来源坐标单位。设最近两个有效接收的采样时刻为s1<s2，位置为x1,x2，则v=(x2-x1)/(s2-s1)。任意当前时刻t≥s2，以h=t-s2计算x_hat(t)=x2+0.2[1-exp(-h/0.2)]v。

请求在固定时刻到期后的首个原始采样点发出；采样于s的信息到达时间为s+delay，只能到达后使用。即使在t>s+delay才处理，其观测年龄仍为t-s，不把陈旧位置直接当当前状态。有效返回更新最后两次观测；无效返回收费但不更新。人工遮罩主实验全部为连续合格参考，因此坏返回主要通过合成边界测试检验，不能据此声称真实自然遮挡时总能恢复。

策略只接收公开View：当前时刻、剩余时间、观测年龄、剩余预算所需计数、挂起返回时间、已知位置/速度和已收到的有效返回数；不接收源UUID、未来状态、真实结束阶段或隐藏质量。回放评估器私有持有参考，统一给各方法使用已知原始时间表。第一问三状态风险是源标坏事件概率，本轮未把它未经验证地转成请求成功概率，也未宣称它改善了请求策略。

## 实验条件与指标

主条件：1秒缺失、0.2秒可见历史、返回延迟0.1秒、预算2。敏感性：缺失0.2/0.6/1秒，延迟0/0.1/0.2秒，容差为训练来源尺度2%/5%/10%。原题第二问补充约定文件本地未找到，以上资源数值均为参赛者统一实验设定，不是官方强制数值。完整规则见protocol.json。

来源尺度为合格手质心到头部的距离中位数，池化训练双手和时间点；训练留一排除全局持出环境，嵌套选择同时排除外层与内层环境。误差先按该尺度归一化，再按窗口→手别→记录→环境→来源等权。主ADE是原始隐藏帧平均距离；NP95为每窗P95的分层均值，不是全体帧的总体P95。coverage为逐点评价，path_covered为整窗所有点均落入对应球的窗口比例，两者不同。

等待分为累计挂起时间与没有新请求的时间。单挂起下累计挂起时间等于请求延迟之和；固定延迟时等待与成本高度相关，不能当成两个独立收益重复计分。终点规则允许零延迟最后计分帧观测改善FDE。名义窗口终点可能在最后原始计分点之后；在这段空隙才返回的付费请求无计分收益，已单列end_boundary_diagnostic.json，没有把它们隐藏或删除。

## 数据、缺失与天然质量

索引69记录、24全局环境；训练55记录/16环境，开发7/4，最终测试7/4。合格人工窗口1527/212/390。开发曾被历史研究使用，不能当成完全未探索的独立检验。所有参数与代码在本轮开发前冻结；最终测试在开发独立核验通过后一次开启，没有据测试结果重新拟合或调参。

训练自然坏段描述如下。手别共享全局质量标记可能重复事件，不能把段数当独立遮挡数；原始原因还含near edge。Standard有2段左删失，不用于完整持续时间分位数。

{table(nat[['dataset','hand_bad_runs','records','environments','complete_median_s','complete_p90_s','bad_frame_fraction']])}

自然低质量段没有独立真实位置参考，不计算其内部真实定位误差。天然缺失多较短，1秒遮罩是压力场景；人工合格段容易选择较易观测时期，不能直接推断真实遮挡效果。原因分布和恢复信息见results/S0。

## 训练基线与消融

{table(tr[['method','nade','nfde','exceed_0.05','coverage','requests','waiting_s']])}

全训练27486条共同指标回放涵盖三档时长/延迟；S2使用严格环境排除的半径。首轮主条件数字与外部汇总四舍五入后相同，只验证基线，不把外部后续实验当作已验证。

更新消融：保留衰减旧速度相对稀疏割线ADE恶化17.86%，95%改善区间[-23.30%,-11.73%]，三来源都退化，故淘汰。

误差球消融如下，点预测和请求相同。回传后分开校准的分位损失降低10.31%，范围增长24.01%，但全局环境配对95%改善区间[-1.37%,18.03%]跨零，按预注册标准未替换。1950个留一分位数经独立重算，最大差约9.71e-17。

{table(scal)}

六组固定时刻公平嵌套选择，每个外层均选0.3/0.6秒。全训练候选排序：

{table(fixedrank)}

## 自适应请求结果与选型

阈值3参数、延迟感知3参数、预算前瞻4参数。严格16外层环境留一，每个外层在其他15环境做内层校准/尺度隔离并选择；共229050条内层结果、15270条外层候选结果。预算前瞻是训练半径积分的代理，不是已知真实VoI，完整定义见S4前瞻定义补充.md。

内层筛出的自适应组合外层NADE=0.092189，固定请求=0.082197；改善为{adaptive['primary_comparison']['gain_pct']:.2f}%，95%区间[{adaptive['primary_comparison']['lo95']:.2f}%,{adaptive['primary_comparison']['hi95']:.2f}%]。请求均为2，不能节省成本；故保持固定请求。该结果只淘汰本次有限候选，不能排除其他自适应方案。

## 开发确认（未再调参）

{tables['validation']}

分来源正式固定请求：

{sources['validation']}

## 冻结后的最终测试

{tables['test']}

分来源正式固定请求：

{sources['test']}

固定请求相对不请求阻尼的改善包含额外购买信息的收益，不是同成本算法提升。固定请求与三种自适应方法在主条件均花费2次，可作同成本比较；副条件必须读取实际成本，不能只看相同预算上限。

最终测试固定请求NADE约0.09780，5%尺度超差负担约61.25%；原球逐点覆盖81.82%，整路径覆盖约37.09%。因此不能解释为满足90%可靠性。分开校准探索球逐点覆盖约87.74%，改善了描述但仍不能证明路径可靠，且不得在看到测试后把它升级为正式方案。

## 可持续预测时长与可行性

按全训练冻结开环经验球，从第1个年龄桶起连续满足“半径≤容差”的最长前缀如下。桶宽1/30秒；它只是经验范围与容差的交点，不是物理安全时长，也不是具有90%整路径保证的时间。可能返回0；非单调曲线首次超限后不重新宣告安全。

{table(pd.DataFrame(support))}

请求延迟可能已超过上述经验支持时长，此时必须报告原容差条件不可满足，不能用“继续预测”代替“满足要求”。本方案主目标是给定资源改善平均误差，不是强制5%可靠可行性。

## 计算与审计

{table(pd.DataFrame(runtime))}

上表为单窗策略状态更新/请求计算计时，含运行时Python开销，不含训练拟合、CSV写入与参考误差计分；不是实际机器人设备延迟。正式固定方案P95目标10毫秒的是否达成可直接由表核对。

引擎16项检查、200组未来/未请求参考扰动通过；三种自适应额外150组扰动通过。全部27486条引擎对照回放独立计分最大差2.28e-14。开发{auditdev['window_results']}条/测试{audittest['window_results']}条结果与原始逐点预测、参考、动作日志独立复算，最大差分别{auditdev['max_independent_score_error']:.2e}/{audittest['max_independent_score_error']:.2e}；预算违规和提前使用返回信息均0。

bootstrap使用全局环境ID整组抽样，来源等权。训练2000次中1933次包含所有来源，其余缺源重采样不进入区间。最终仅4个独立环境，分源更少；所有配对区间附paired_final_comparisons.csv，不能以窗口数冒充独立环境数。

冻结模型SHA-256：{hashlib.sha256((BASE/'frozen_model.json').read_bytes()).hexdigest()}。实现版本：Python {platform.python_version()}，NumPy {np.__version__}，pandas {pd.__version__}。完整审计、输入哈希和模型参数均已保存。

## 为什么提升有限，哪些目标没有达成

1. 固定请求两次后，信息预算和平均误差已形成强基线；纯年龄阈值容易提前用完额度。各方法同样花满2次，不能靠覆盖变化创造成本节省。
2. 旧速度衰减丢掉了返回点之间的位移信息；本次数据支持割线更新，但不证明它是理论最优。
3. 尾部校准受环境异质、帧相关和策略选择影响，覆盖变好仍需代价；改善区间跨零，不能只看点估计。
4. 预算前瞻用半径作为未来误差收益代理，其预测不精确；“考虑未来”本身不保证决策更好。
5. 真正自然遮挡的参考缺失，人工遮罩结论不能替代真实系统验证。

明确未达成：自适应相对固定ADE再改善3%、同精度节省15%成本、所有来源90%覆盖及5%容差可靠运行。没有以修改门槛或测试后调参包装成功。已交付的是完整可复现的第二问有限候选研究和诚实的最优候选选择，而非全局最优或实际机器人安全证明。

## 交付索引

- 执行计划与量化目标.md、protocol.json：范围和规则。
- frozen_model.json：正式选择、训练参数、校准球、代码与协议哈希。
- results/S0—S4_*：输入、自然质量、基线、消融、嵌套选择。
- results/S5_validation、results/S5_test：逐窗/逐点/逐动作结果、来源环境汇总、最差窗口及独立审计。
- paired_final_comparisons.csv、runtime_summary.csv、empirical_prediction_duration.csv：统计及解释。
- README.md：运行与复核方式；manifest.csv：交付文件哈希。
'''
 supplement=BASE/'阶段历史补充.md'
 if supplement.exists():text+='\n\n'+supplement.read_text(encoding='utf8')
 (BASE/'第二问完整解答与实验报告.md').write_text(text,encoding='utf8')
 (BASE/'requirements.txt').write_text(f'numpy=={np.__version__}\npandas=={pd.__version__}\n',encoding='utf8')
 print('report saved',len(text),flush=True)
if __name__=='__main__':main()
