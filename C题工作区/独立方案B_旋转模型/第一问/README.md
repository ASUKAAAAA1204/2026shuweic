# 方案B第一问：冻结旋转运动模型

采用头手运动分解基线、112维因果历史和三种子高斯随机傅里叶残差预测。参数位于model/model_parameters.npz，纯推理位于code/rotation_inference.py，调用predict_model(parameters,z,side,t0,ages,source)。依赖numpy、scipy；无需运行历史训练脚本。

参数SHA256：8f34fa90c6484f53a53deed01374d1e373aacee8c1328a9e7ad7c2bdbd4d6237。

原214开发窗逐点预测已复现，与归档差0。原队列训练来源等权归一化ADE0.347717314，开发0.359515625；与第二问的归一化尺度及队列不同。模型输出0.2/0.6/1秒位移，中间结点插值。

本目录交付第一问采用的运动预测模型。可靠性三状态模型与误差范围沿用既有第一问资料，不把此参数解释为包含所有第一问子任务。evidence保存原选型与复现依据。
