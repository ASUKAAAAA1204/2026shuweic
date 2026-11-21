# 第一问模型、实验与复现

[第一问论文图 v5（正文 3 张、附录 2 张）](figures/q1_redblue_refined/第一问论文图_红蓝精修审核稿_v5.pdf) · [绘图入口](figures/q1_redblue_refined/build_review.py)

在 `q1` 目录下安装 `requirements.txt` 中的依赖，使用装有 Microsoft YaHei 字体的环境运行 `python -B figures/q1_redblue_refined/build_review.py`。脚本从仓内实验结果及关键点记录重绘，输出同名 v5 PDF。时间窗图和逐窗散点图复用已归档的矢量底稿，可通过 `python -B figures/q1_redblue_review/build_review.py` 重新生成底稿。

[第一问主报告](reports/问题一优化结果.md) · [Word 稿](reports/问题一_优化稿.docx) · [实验总表](reports/数维杯C题对比试验汇总.md) · [实验设计与优化路线](reports/第一问实验方案与优化路线_待审核.md)

| 任务 | 最终选型 | 基线与扩展证据 |
|---|---|---|
| 手部位置预测 | 固定阻尼，时间常数 0.2 秒。以单手 21 点质心为位置，用最后两帧估计速度。 | 低频沿向残差 MLP 为扩展对照。在两套训练环境留一窗口中误差分别下降 2.92% 和 2.83%，均低于预设 3% 门槛。[本地比较](reports/RESULTS_REPORT.md) · [中立窗口](reports/第一问_中立同窗复评.md) |
| 观测可靠性 | 三状态查表，估计未来 0.2 秒内出现源质量坏标记的概率。持续未标坏、刚恢复、当前标坏的训练平滑概率分别为 0.60%、19.13%、88.46%。 | 二状态为基线，三状态加运动提升树为扩展对照。树的训练留一 Brier 再降低 0.083%，开发集报警精确率为 3.02%。[状态参数，取 held=development](results/q1_reliability/state3_rates.csv) · [候选结果](reports/第一问_可靠性最终候选_实验结果.md) |
| 位置不确定性 | 按距最后观测的时间，用训练残差校准 90% 误差球。 | [阻尼参数与半径](results/q1/motion_model.json) · [覆盖率](results/q1/motion_summary.csv) |

## 正式窗口 A–F 实验

恒速外推为主基线，末值保持为补充基线。在 1,527 个训练窗和 212 个开发窗上，固定阻尼相对恒速的归一化 ADE 分别降低 21.55% 和 26.34%。低频残差 MLP 相对阻尼的训练改善为 0.73%，开发改善为 3.17%，保留为扩展对照。

[本轮结果报告](reports/第一问_统一预处理与缺帧恢复_实验结果.md) · [批准的实验设计](reports/第一问_统一预处理与缺帧恢复_实验设计.md) · [预处理依据](references/当前推荐的数据预处理主方案.md) · [逐窗结果与校验](results/q1_revised/) · [代码](code/q1_revised/)

在 q1 目录执行 `python -B code/q1_revised/verify_delivery.py` 可用仓内数据核对已发布结果。全量重跑 A–F 时，先将环境变量 `Q1_FULL_V1` 指向原完整 v1 数组目录（含 records、原始四元数和五通道时钟），`Q1_ARCHIVES` 指向含三份官方源 ZIP 的目录，再执行 `python -B code/q1_revised/run_all.py`。原实验输入哈希见 [inputs.csv](results/q1_revised/inputs.csv)，本次发布的对应文件哈希见 [inputs_checkout.csv](results/q1_revised/inputs_checkout.csv)。重跑会更新结果和本机计时。

## 实验索引

| 内容 | 归档 |
|---|---|
| 原基线复核、固定阻尼、误差球、早期质量模型 | [q1](results/q1/) |
| 标记掩码、速度估计、坐标与预处理配对 | [q1_ablation](results/q1_ablation/) · [q1_approved](results/q1_approved/) · [q1_neutral_preprocessing](results/q1_neutral_preprocessing/) |
| 阻尼优化、近邻模型、沿向回归 | [q1_six_train](results/q1_six_train/) · [q1_along_ridge](results/q1_along_ridge/) |
| kNN / MLP 同窗比较、函数型回归与低频残差 | [q1_knn_mlp](results/q1_knn_mlp/) · [q1_curves](results/q1_curves/) |
| 中立窗口复评、阶段阻尼消融 | [q1_neutral_models](results/q1_neutral_models/) · [q1_phase](results/q1_phase/) |
| 可靠性、首次变坏、误报及运动和视觉对照 | [q1_reliability](results/q1_reliability/) · [q1_reliability_candidates](results/q1_reliability_candidates/) |

[reports](reports/) 收录各轮设计、审批稿、结果和调研。[code](code/) 收录模型与评分程序。[figures](figures/) 收录现有图表。[references](references/) 收录外部来稿、早期中文稿和原版中立同窗评测包。来稿中的结果采用各自的窗口口径。运动提升树的 joblib 是扩展对照的拟合文件。

## 复现

从仓库根目录进入 q1 后运行下列命令。归档计时环境使用 Python 3.14，实际计时、数值库和线程设置见各轮 run.json。训练模型或生成汇总时，脚本会写入本目录的 results。

```powershell
cd q1
python -m pip install -r requirements.txt
python -B code/q1_six/audit/verify.py results/q1_six_train
python -B code/q1_reliability_candidates/verify.py
```

[主模型与分桶误差球](code/q1.py)、[三状态拟合](code/q1_reliability/run.py)、[低频残差模型](code/q1_curves/)及[运动提升树](code/q1_reliability_candidates/)均含训练或回放入口。中立窗口的复评脚本需要位置参数，例如 `python -B code/q1_neutral_models/replay.py references/第一问中立同窗.zip`。位置训练、已探索过的中立测试复评、可靠性开发评价分别按对应报告解释。

[C_Problem_Data](C_Problem_Data/) 包含官方划分、索引和 69 条记录中第一问代码使用的数值字段。NPZ 按原数组无损抽取手部位置、质量掩码、时间与头部参照等字段。轴名文件只收录预处理实验读取的头、手坐标定义。早期基线直接使用仓库已有的[阶段交付](../C题工作区/阶段交付/)。视频比较可从归档的逐帧特征缓存和逐锚点特征表复算。若需从像素重新提取特征，将官方 EgoDemo.zip、EgoPro.zip、EgoStandard.zip 放入本目录的 C_Problem_Data 下，并给 video_inputs.py 的 --decoder-python 指定安装了 OpenCV 的解释器。原始视频沿用赛方分发渠道。

[文件校验清单](manifest.csv) 给出本目录各文件的字节数与 SHA-256，清单自身除外。
