# 2026 数维杯 C 题

## 当前正式方案

第一问采用旋转位置模型与训练留一残差误差球，源质量风险采用三状态模型。第二问采用旋转起步、首次有效返回后切换阻尼。第三问采用截止机会损失调度器。

[三问解答与版本对照](C题工作区/正式方案/三问解答与版本对照.md) · [统一复现入口](C题工作区/正式方案/code/run_all.py) · [正式模型调用](C题工作区/正式方案/code/adopted.py) · [三问结果](C题工作区/正式方案/results/comparison.csv)


## 历史基线与预处理入口

- [阶段交付说明](C题工作区/阶段交付/baseline_v2/交付说明.md)
- [三问基线报告](C题工作区/阶段交付/baseline_v2/三问基线完整报告.md)
- [验收结果](C题工作区/阶段交付/baseline_v2/验收结果.json)
- [执行计划](C题工作区/本阶段执行计划与交付目标.md)
- [程序入口](C题工作区/scripts/baseline/run_pipeline.py)

上述链接记录早期基线阶段及其当时的测试访问范围。

## 数据与复现

原始比赛资料、往年论文、原视频与轨迹、压缩包、运行依赖、完整处理数组和逐动作JSONL日志未上传，保留于本地。仓库保留汇总、逐试验CSV、代码和校验清单；因此仅克隆仓库不能直接执行全量复现。

使用合法取得的比赛附件恢复 `C题工作区/数据集/{EgoDemo,EgoPro,EgoStandard}` 及 `C题工作区/Restocking_Problem_and_Data_EN`，使用Python 3.14并按阶段交付目录内requirements-lock.txt安装依赖到 `C题工作区/运行依赖314`。先执行scripts中的process_dataset.py、finalize_processing.py、additional_audit.py生成v1数值产物，再运行：

```powershell
python C题工作区/scripts/baseline/run_pipeline.py --reproduce
```

程序会全量解码视频、生成模型和结果，并在独立目录复现。日志与数组在本地生成。源文件和代码哈希用于检查输入版本。
