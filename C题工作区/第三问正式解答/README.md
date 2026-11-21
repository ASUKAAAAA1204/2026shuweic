# 第三问正式解答与复现入口

阅读 [第三问完整解答](第三问完整解答.md)。采用：训练任务族中位时长、权重/预计占用排序、最小合法预留0.2秒、无付费复核。代码、数据审计、冻结模型、全部共同指标、敏感性、逐动作记录和独立核验集中于本目录。

## 一键复现

在原项目目录中执行：

```powershell
python "C题工作区/第三问正式解答/code/run_all.py"
```

依赖Python、numpy、pandas、PyMuPDF。脚本以自身位置定位项目，不依赖调用时工作目录。数据依赖同级 `处理结果/v1/segments_private.csv`、`处理结果/v1/private/*.npz` 及 `Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints/` 的原始PDF/CSV/JSON；原始数据不随本次Git提交重复上传。首次数据准备过程沿用项目既有预处理，本入口不声称从加密压缩包重新解压。

## 文件说明

- `协议与目标冻结.md`、`audit/selection_before_validation.json`：目标、规则和验证前选择记录。
- `model/family_duration.json`：训练族10%、50%、90%时长分位数。
- `results/baseline_train`：3600次训练固定对照，包含缺族支持限制。
- `results/duration_information`：嵌套环境留一前缀信息检验和冻结族模型验证。
- `results/validation`：四方法共1800次验证；采用方法450次，覆盖三个案例、30种子、五映射。
- `actions.jsonl.gz`：无损压缩逐动作日志，可用Python gzip逐行读取。源引用是执行后评价日志，不进入策略。
- `audit/independent_delivery_checks.json`：独立重建1800条试验的抽样、费用、约束和评分。
- `audit/chain_candidate_decision.json`：唯一新排序候选未通过的决策摘要；未采用实现和中间结果保留于本地研究归档，不放入采用方案。
- `audit/delivery_manifest.json`：运行步骤、环境、冻结模型与文件哈希；证明复现一致，不证明全局最优。

## 关键结果与限制

名义验证加权按时率S1/S2/S3为92.71%/75.56%/76.46%。最小预留减少费用/闲置，但超预留明显，不是低容量风险方案。S1的EDF更好，当前没有按已查看验证结果临时切换方法。模型经验区间不保证跨环境覆盖。

规定验证与五映射已完成；组织方额外冻结的最终测试30种子尚未提供，不编造官方测试结果。验证数据历史已被查看、训练调度只有两个全族环境、操作者可能跨环境，均限制泛化结论。完整答案据此报告证据边界，不宣称实体机器人性能。
