# 终稿模型敏感性分析复现说明

[论文小节](../../reports/三问终稿模型敏感性分析.md)使用标签 `v1.0.0-final`，对应提交 `9f8bc05fae9d8ce32da7fa7c5e28a82960085819`。需要运行模型时，按下文创建冻结快照。脚本依赖 Python、NumPy、SciPy、pandas 和 Matplotlib。

在完整克隆的仓库根目录运行以下命令，可准备模型回放所用的独立快照。轨迹回放还需从赛事附件恢复 v1 数值记录，将各条记录的 `<episode_uuid>.npz` 放在本地 `C_Problem_Data/v1/v1/records/`。第三问调度回放使用冻结提交中的任务与工期池，可直接运行。

```powershell
git worktree add --detach tmp/final_v1_20261122 v1.0.0-final
git -C tmp/final_v1_20261122 rev-parse HEAD
```

下列命令重建三问敏感性结果。

```powershell
python -B -X utf8 results/final_v1_sensitivity/run_q1_q3.py
python -B -X utf8 results/final_v1_sensitivity/q2/run_q2.py
python -B -X utf8 results/final_v1_sensitivity/plot_intervals.py
```

| 论文结果 | 数据与口径 |
|---|---|
| 第一问的历史截短实验（新增） | [汇总](q1_history.csv)和[逐窗记录](q1_history_windows.csv.gz)。同一开发窗同步截短三个通道，未来真值与冻结半径相同。每个预测时长先取窗内逐点均值，再等权聚合至记录、环境和来源。完整历史下的 212 × 3 个窗口和时长组合复核 NADE 与末帧误差，三个覆盖率复核原汇总。 |
| 第一问测试集的预测时长（已有） | [终稿时长表](frozen_tables/q1_test_horizons.csv)，取 test、B。开发集截短实验和测试集原有完整历史实验分别报告。 |
| 第二问的遮挡时长和预算（已有） | [窗长与延迟](frozen_tables/q2_validation_horizon_delay.csv)，取 validation、rotation_switch。[预算表](frozen_tables/q2_test_budget.csv)，取 test、B。 |
| 第二问冻结误差球随延迟变化（新增） | [延迟汇总](q2/summary.csv)、[逐窗记录](q2/per_window.csv.gz)和[核验](q2/audit.json)。212 个开发窗各回放 3 次。误差球参数未重标定，按终稿依次对窗、手、记录、环境及来源聚合。名义条件复核逐窗位置指标及覆盖、直径汇总，三档延迟复核已归档的位置误差。覆盖率均按逐点命中计算。 |
| 第三问的工期映射（已有）及配对区间（新增统计） | [终稿逐实例记录](../../C题工作区/正式方案/results/q3/b_k64/trials.csv.gz)与[本节汇总](q3_mapping.csv)。固定 B_K64，在 3 × 30 × 5 个案例、种子和映射组合上汇总。逐种子先计算三案例平均按时率相对名义映射的差，再用固定随机种子 20261122 做 10 000 次有放回重抽，取 2.5% 和 97.5% 分位数。区间逐项计算。 |

新增轨迹实验均使用 212 个开发窗口。论文中的测试数值仅摘自发布时的汇总。调度分析复用原有 450 次官方验证，重抽样次数不计入独立验证实例数。

## 五维报告附件与复现顺序

[五维度实验报告](../../reports/终稿模型五维度稳定性实验报告.md)中的相对链接对应各次实验归档。CSV 保存汇总或明细，较大的逐窗及逐实例文件使用 gzip 压缩，可用 pandas.read_csv 直接读取。冻结表摘录保留原数值文本，分别从终稿的 q1_horizons.csv、q2_cost.csv 和第二问 position_sensitivity/summary.csv 选取上表所列划分及模型。三行两列结果图可使用已归档的统计表直接重画，默认中文字体为 Microsoft YaHei。[矢量 PDF](../../figures/final_v1_sensitivity/三问终稿模型敏感性分析_三行两列审核稿.pdf) · [高清 PNG](../../figures/final_v1_sensitivity/三问终稿模型敏感性分析_三行两列审核稿.png) · [绘图脚本](plot_sensitivity_grid.py)

从仓库根目录按下列顺序运行，可以重建上表以外的报告结果。若只复算扩展批次的配对区间、尾部或种子统计，直接运行 schedule_statistics.py 即可，它使用已上传的回放记录。S2 专项复验读取此前批次的种子及输入指纹，确保新批次独立。

```powershell
python -B -X utf8 results/final_v1_sensitivity/plot_sensitivity_grid.py
python -B -X utf8 results/final_v1_stability/tail/tail.py
python -B -X utf8 results/final_v1_stability/seed_replay.py
python -B -X utf8 results/final_v1_stability/seed_summary.py
python -B -X utf8 results/final_v1_remaining/trajectory/replay.py
python -B -X utf8 results/final_v1_remaining/environment_pairs.py
python -B -X utf8 results/final_v1_remaining/schedule_sources.py
python -B -X utf8 results/final_v1_remaining/capacity_bound.py
python -B -X utf8 results/final_v1_evidence_extension/history_horizons.py
python -B -X utf8 results/final_v1_evidence_extension/trajectory/factorial.py
python -B -X utf8 results/final_v1_evidence_extension/spaced_windows.py
python -B -X utf8 results/final_v1_evidence_extension/tail_transfer.py
python -B -X utf8 results/final_v1_evidence_extension/run_schedule.py
python -B -X utf8 results/final_v1_evidence_extension/schedule_statistics.py
python -B -X utf8 results/final_v1_s2_tail/replay_s2.py
python -B -X utf8 results/final_v1_external_site_audit/check_inventory.py
```

地点清单的核查入口将冻结版记录划分与已发布的逐窗实验交叉核对。提供赛事原包 Raw_Data 目录时，可加参数 `--raw-dir <目录>` 核对三个 ZIP 的记录目录。[地点核查脚本](../final_v1_external_site_audit/check_inventory.py) · [逐地点结果](../final_v1_external_site_audit/site_inventory.csv)
