# 2026 数维杯 C 题最终版

当前组合：第一问冻结旋转残差模型；第二问旋转起步、首次有效返回后阻尼割线更新；第三问B_K64联合场景前瞻。

- [最终路线](C题工作区/最终采用路线确认.md)
- [第一问](C题工作区/独立方案B_旋转模型/第一问/README.md)
- [第二问](C题工作区/独立方案B_旋转模型/第二问/delivery/README.md)
- [第三问B最终方案](C题工作区/第三问B最终方案/README.md)
- [终稿补验、逐次回放与论文图表](results/终稿补验索引.md)

## 论文图与统一回放

最终版本为 `v1.0.0-final`。总技术路线图说明三问的数据来源与联系，求解流程图展示再观测更新、可行动作筛选及完工反馈。

| 论文图件 | 矢量 PDF | 可编辑源图 | 高清预览 |
|---|---|---|---|
| 总技术路线图 | [PDF](figures/fig_roadmap.pdf) | [DrawIO](figures/fig_roadmap.drawio) | [PNG](figures/fig_roadmap.png) |
| 三问求解流程图 | [PDF](figures/fig_three_problem_flow.pdf) | [DrawIO](figures/fig_three_problem_flow.drawio) | [PNG](figures/fig_three_problem_flow.png) |

第一、二问在 390 个测试窗的 NADE 分别为 0.112023 和 0.093538。第三问在 30 个官方验证种子上的名义加权按时率为 90.9606%，加权完成率为 95.2778%。[三问解答与版本对照](C题工作区/正式方案/三问解答与版本对照.md) · [统一入口](C题工作区/正式方案/code/run_all.py) · [逐指标比较](C题工作区/正式方案/results/comparison.csv) · [回放核验](C题工作区/正式方案/results/q3/b_k64/verification.json)

在仓库根目录执行以下命令可复现第三问的 450 个官方实例。全量重跑时去掉 `--q3-only`，用 `--processed` 指定已恢复的赛事 v1 预处理数据目录。

```powershell
python -B -X utf8 "C题工作区/正式方案/code/run_all.py" --q3-only --output tmp/final_replay
```

历史E验收推荐不代表当前主方法。以下为早期基线说明，按历史资料阅读。

---

# 2026 数维杯 C 题

本仓库保存数据审计、处理程序、三问基线、冻结实验协议和验证结果。

## 成果入口

- [阶段交付说明](C题工作区/阶段交付/baseline_v2/交付说明.md)
- [三问基线报告](C题工作区/阶段交付/baseline_v2/三问基线完整报告.md)
- [验收结果](C题工作区/阶段交付/baseline_v2/验收结果.json)
- [执行计划](C题工作区/本阶段执行计划与交付目标.md)
- [程序入口](C题工作区/scripts/baseline/run_pipeline.py)

目前完成三问基线闭环，并非最终测试或比赛最终方案。方法比较使用验证集，测试集仅做结构审计。协议假设与限制见报告。

## 数据与复现

原始比赛资料、往年论文、原视频与轨迹、压缩包、运行依赖、完整处理数组和逐动作JSONL日志未上传，保留于本地。仓库保留汇总、逐试验CSV、代码和校验清单；因此仅克隆仓库不能直接执行全量复现。

使用合法取得的比赛附件恢复 `C题工作区/数据集/{EgoDemo,EgoPro,EgoStandard}` 及 `C题工作区/Restocking_Problem_and_Data_EN`，使用Python 3.14并按阶段交付目录内requirements-lock.txt安装依赖到 `C题工作区/运行依赖314`。先执行scripts中的process_dataset.py、finalize_processing.py、additional_audit.py生成v1数值产物，再运行：

```powershell
python C题工作区/scripts/baseline/run_pipeline.py --reproduce
```

程序会全量解码视频、生成模型和结果，并在独立目录复现。日志与数组在本地生成。源文件和代码哈希用于检查输入版本。
