# C题前置工作记录

## 已完成

- 从C题数据集说明确认密码，并成功解开外层ZIP。原下载文件未修改。
- 外层包含10个文件：三个原始子ZIP、六份JSON/CSV配置和Problem_3_Requirements.pdf。
- 三个原始子ZIP本身未加密，已全部解压至“数据集”。
- 全部989个文件逐项完成大小与CRC32核对，全部通过，总大小6,096,375,863字节。详见“数据解压校验.csv”。解压软件亦均返回Everything is Ok。
- 记录目录数量：EgoDemo 31、EgoPro 14、EgoStandard 24，共69。
- 已生成中文题面PDF、可编辑DOCX与文字MD；保留5幅原始图、原公式及参考文献。图内英文保留，图题已译；中文版重排6页，不声称复刻原7页版式。

## 目录

- 数据集/EgoDemo、EgoPro、EgoStandard：已展开数据。
- Restocking_Problem_and_Data_EN/Raw_Data：解密得到的原始子ZIP。
- Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints：任务表、环境划分、配置和第三问完整要求PDF。
- 中文题面：中文PDF、DOCX、MD，原英文PDF与提取图片。
- scripts：中文文档生成及PDF导出脚本。

## 实际格式与题面差异

实测EgoDemo含124个JSON、93个Parquet、80个MP4；EgoPro含252个CSV、56个MP4；EgoStandard含336个CSV、48个MP4。实际发布的后两者已经是CSV表及视频，不是题面所述MCAP。本次未发现MCAP文件。后续应按实测schema处理，不再预设必须解析MCAP；未对载荷逐字段语义作认证。

外层及三个子包均未发现题面提到的Data_Documentation目录、attachment_guide.md、record_index.csv、missing_observation_experiments.md、tasks_and_evaluation.md、file_manifest.csv或Problem_Statement中文参考PDF。不得将自己生成文件冒充官方缺失文件。

已存在的Problem_3_Requirements.pdf为3页，包含抽样、允许信息、动作、核算和评价细则，可作为第三问主要本地协议来源。已导出其原文文本供后续检索。该PDF明确：前缀不得包含语义描述、隐藏源标识或总时长；不能用完整集合、前缀匹配或种子反查真值；全视频不是第三问策略输入。每次试验重置策略，不能跨验证/测试试验积累真值训练。

## 尚未完成

本次仅执行解包、完整性核验和题面翻译；未完成轨迹质量审计、官方split统计核验、字段对齐、模型训练或仿真结果。中文译本供阅读，争议以原英文与正式通知为准。原图内部文字保留英文，避免修改图像改变题意。
