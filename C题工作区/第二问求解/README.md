# 第二问复现与交付入口

正式方案及结论见[第二问完整解答与实验报告](第二问完整解答与实验报告.md)。性能争取目标未达成的项目在报告中明确披露。最终测试已按冻结方案完成一次，不能继续把它称为封存数据，也不能据此调参。

## 环境与输入

本机Python：C:/Users/33671/AppData/Local/Python/pythoncore-3.14-64/python.exe。仅需requirements.txt列出的NumPy、pandas。输入沿用同仓库 `分支测试/q1-ding/q1/C_Problem_Data` 和该检出目录的第一问冻结窗口。保持当前仓库目录结构运行；脚本用自身位置定位仓库，不依赖当前工作目录。输入哈希在results/S0与S5_*保存。

## 首先复核现有交付

从本目录运行（终端需将python映射到上述解释器或等效环境）：

```powershell
python -B code/verify_engine.py
python -B code/verify_policies.py
python -B code/verify_training_artifacts.py
python -B code/verify_evaluation.py validation
python -B code/verify_evaluation.py test
python -B code/verify_delivery.py
```

前五条只核验既有参数与结果，训练回放校验不重新选模型；verify_engine会更新S1审计日志中的当前代码列表，若希望交付文件逐字节不变，应先仅运行verify_delivery检查现有manifest，再在复制的完整仓库中运行其他核验。

## 从头重建研究的顺序

在隔离的完整仓库副本中保留输入、code、计划和注册文件，使用新的第二问结果目录。当前实现输出目录固定到本求解目录，因此不能直接在原交付目录删除或覆盖冻结文件来重新选型。需要复现者在副本中把原results/frozen_model.json另行保留后运行：

```powershell
python -B code/s0_audit_replay.py
python -B code/finalize_s0.py
python -B code/verify_engine.py
python -B code/update_calibration_experiment.py
python -B code/fixed_schedule_experiment.py
python -B code/adaptive_policy_experiment.py
python -B code/complete_s2.py
python -B code/verify_policies.py
python -B code/verify_training_artifacts.py
python -B code/freeze_and_evaluate.py freeze
python -B code/freeze_and_evaluate.py validation
python -B code/verify_evaluation.py validation
python -B code/freeze_and_evaluate.py test
python -B code/verify_evaluation.py test
python -B code/context_diagnostics.py
python -B code/build_report.py
```

freeze和evaluate默认拒绝覆盖已完成的冻结/评价文件；这是防止重复查看开发/测试结果后偷偷替换选型。新脚本的只读统计和独立验证不改变冻结推理代码。重建耗时、运行时、CSV压缩字节和附加统计文件时间戳可能不同；核心数值用1e-9容差比较，不要求全部文件字节相同。

## 文件口径

- frozen_model.json记录开发前冻结的选型、参数、代码和协议哈希。
- results/S0：数据审计、训练自然质量、尺度及初始回放。
- results/S1：因果引擎与策略非干扰检查。
- results/S2：全部训练基线的共同指标。
- results/S3：更新/校准配对、留一半径、独立分位数核对。
- results/S4_fixed与S4_adaptive：固定调优与严格嵌套候选比较。
- results/S5_validation与S5_test：一次冻结评价，含原始逐点/逐动作日志及独立审计。
- results/diagnostics：仅评价用的任务、阶段和历史分层；不可作策略输入。
- manifest.csv：除manifest本身和交付验收.json之外的全部交付文件字节数与SHA-256。
- 交付验收.json：检查结果；若性能目标未达，不会用工程通过冒充性能达标。
