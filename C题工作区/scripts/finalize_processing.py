from pathlib import Path
import sys,json,csv,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'运行依赖314'));sys.path.insert(0,str(R/'scripts'))
import numpy as np
from prefix_api import load_prefix
O=R/'处理结果/v1'
def read(n):
 with (O/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
s=json.loads((O/'summary.json').read_text(encoding='utf8'));rec=read('record_index.csv');st=read('channel_quality.csv');sg=read('segments_private.csv');pc=read('prefix_coverage.csv')
mask=[];tests=0
for r in rec:
 with np.load(O/'records'/(r['episode_uuid']+'.npz')) as z:
  for side in ['left_hand','right_hand']:
   t=z[side+'__t_s'];valid=z[side+'__reference_candidate'];dt=np.diff(t);med=float(np.median(dt[dt>0]));stride=max(1,round(2/med))
   for length in [.2,.6,1.]:
    for i in range(1,len(t),stride):
     j=int(np.searchsorted(t,t[i]+length,side='left'))
     if j>=len(t) or not valid[i-1:j+1].all() or (np.diff(t[i-1:j+1])>3*med).any():continue
     mask.append({'episode_uuid':r['episode_uuid'],'split':r['split'],'channel':side,'start_index':i,'end_index_exclusive':j,'nominal_length_s':length,'start_s':t[i],'next_visible_s':t[j],'purpose':'candidate only; adjacent candidates overlap; select disjoint episodes/windows before experiments'})
for row in sg:
 if row['eligible']!='1':continue
 start=float(row['start_s'])
 with np.load(O/'records'/(row['episode_uuid']+'.npz')) as z:
  for length in [.2,.6,1.]:
   p=load_prefix(row['unit_id'],length)
   assert len(p)==19 and all(any(k.endswith('__'+v) for v in ['t_s','position','quaternion','quality_known','bad']) for k in p)
   for name in ['left_hand','right_hand','head','head_left_camera','head_right_camera']:
    t=z[name+'__t_s'];sel=(t>=start)&(t<=start+length)
    for v in ['position','quaternion']:
     assert np.array_equal(p[name+'__'+v],z[name+'__'+v][sel],equal_nan=True)
    assert np.array_equal(p[name+'__t_s'],t[sel]-start)
   tests+=1
with (O/'artificial_mask_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(mask[0]));w.writeheader();w.writerows(mask)
s['checks']['all_1029_prefixes_source_rechecked']=tests==1029;s['artificial_mask_candidates']=len(mask);s['prefix_source_rechecks']=tests
(O/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# C题数据处理报告（v1）','','## 1. 处理结论','',f"已实际读取并转换 {s['records']} 条记录、{s['samples_left']} 个左手时间点、{s['channels']} 条位姿通道。69条内部UUID和环境全部与官方划分一致；全部349个语义区间起止与官方索引一致。343个合格区间已生成1029份原始数值前缀缓存，逐份回查源数组一致。",'',f"来源记录数：{s['sources']}；官方划分记录数：{s['split_counts']}。环境互斥已验证，不据此声称操作者互斥。",'','## 2. 已执行流程','','1. 兼容环境读取全部Parquet、CSV和JSON，确认内部身份，避免把目录名误当episode UUID。','2. 五种位姿通道统一为位置[N,J,3]、四元数[N,J,4]，手J=21、其他J=1；保存为69份压缩NPZ。','3. Demo保留源timestamp；CSV保存整数纳秒、publish纳秒和原offset，通过整数减法生成相对秒。没有全源强制重采样。','4. 全量检查非有限、非递增时间、超过本通道3倍正时间差中位数的间隙、四元数范数偏差及质量标注。异常标记，不修正数值。','5. 生成双手transform[0]位置和后向速度；速度仅使用当前与前一帧，质量不满足或跨间隙时为NaN。transform[0]在Demo对应wrist，CSV缺少官方关键点字典，仍需确认跨源语义。','6. 审计全部视频元信息、首尾可解码性及可用帧索引；仅少量首帧缩略图，不是全视频逐帧验证。','7. 对合格语义区间按闭区间[start,start+L]提取0.2/0.6/1.0秒原始数值；不插补、不利用后续点修正。策略时间归零，不暴露绝对时间。','8. 为参考候选窗口生成固定0.2/0.6/1.0秒人工遮罩候选，仅供后续实验抽取；候选可能重叠，不能直接当独立样本。','9. 所有数组保存后逐元素回读核对；1029前缀全部回查原始转换轨迹；记录输入非视频文件SHA256，视频等原文件另有前序全量CRC校验。','','## 3. 量化结果','','|项目|结果|','|---|---|']
for k,v in s['quality_totals'].items():lines.append(f'|双手通道合计 {k}|{v}|')
lines += [f"|视频数|{s['videos']}|",f"|视频打开/首尾解码失败|{len(s['video_failures'])}|",f"|视频帧索引数量不一致|{len(s['video_index_mismatch'])}|",f"|合格区间存在空通道前缀|{len(s['empty_eligible_prefixes'])}|",f"|人工遮罩候选|{len(mask)}|",'','双手合计分母为左右手分别计数，不能与单路66070帧直接混用。下面的异常数量是结构审计，不用于在测试集上选择模型阈值。','','|来源|通道|帧数|非递增|采样间隙|非有限|非单位四元数|低质量|质量未知|参考候选|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for src in s['sources']:
 for ch in ['left_hand','right_hand','head','head_left_camera','head_right_camera']:
  rr=[x for x in st if x['dataset']==src and x['channel']==ch]
  totals=[sum(int(x.get(k) or 0) for x in rr) for k in ['samples','nonpositive_dt','gaps_gt_3median','nonfinite_frames','nonunit_quaternion_frames','bad_frames','quality_unknown','reference_candidates']]
  lines.append('|'+src+'|'+ch+'|'+'|'.join(map(str,totals))+'|')
lines+=['','## 4. 质量标记的准确含义','','`finite`：位置和四元数均为有限数；`quaternion_unit_1e3`：本帧全部关键点范数距1不超过0.001，仅为数值诊断，没有归一化。`gap_before`：相邻时间差大于该通道正间隔中位数的3倍，是间隙提示，不是异常运动识别器。','','Demo按稀疏bad_frame_ratio手别原因映射到frame_index，未列出的帧按该文件稀疏约定视为未标坏；该假设需官方字段文档确认。CSV使用is_bad的全局手部质量标记，保守同时作用双手；只在整数纳秒完全一致时认定质量已知，否则为unknown，不使用未来最近邻填标签。原始原因全部保存在metadata_private.json。','','reference_candidate要求数值有限、四元数通过上述检查、质量已知且未标坏、时间递增；只是可评估参考候选，不能称独立物理真值。姿态数值连续不代表追踪准确；本阶段没有用运动速度阈值删除快动作。','','## 5. 产物与使用方法','','- `records/*.npz`：完整数值轨迹，含所有split，仅离线可信训练/评估使用。','- `record_index.csv`：内部身份、环境、operator、split、时间/四元数依据。','- `channel_quality.csv`：每记录每通道质量与时间统计。','- `natural_unusable_intervals.csv`：自然质量/数值不可用连续区间，右端为exclusive索引，时间字段为末样本时间而非区间外推终点。','- `artificial_mask_candidates.csv`：全参考窗口候选；train训练、validation选方法、test只做冻结后的评估。','- `prefix_coverage.csv`：含不合格区间在内的逐长度覆盖审计。','- `private/*.npz`、`segments_private.csv`、`metadata_private.json`：可信评估端私有文件，禁止策略访问目录、文件名、隐藏映射和完整轨迹。','- `video_audit.csv`、`contact_sheets`：视频数量、帧率、首尾及抽样缩略图。','- `input_sha256.csv`、`summary.json`：来源校验和汇总。','','运行命令：','','```powershell',"& 'C:\\Users\\33671\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe' 'C题工作区\\scripts\\process_dataset.py'", "& 'C:\\Users\\33671\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe' 'C题工作区\\scripts\\finalize_processing.py'",'```','','策略前缀入口：可信评估器调用`scripts/prefix_api.py`的`load_prefix(unit_id, length)`后只把返回的19个数值数组交给策略。不得把调用参数、文件路径或源记录信息传入策略。此接口是数据分离约定，不是操作系统安全沙箱；正式实验需限制策略文件系统访问，并由评估器控制何时允许查询。前缀本身仍可能存在轨迹指纹，禁止与隐藏完整轨迹建立反查库。','','## 6. 未消除的限制与下一步','','原始坐标单位、坐标方向、CSV关键点顺序以及Demo四元数声明顺序尚缺官方字段字典作独立验证。统一数组保留源值，没有凭数值猜测和交换四元数；跨源姿态融合、物理速度解释与相机投影前须补齐/验证这些定义。相机参数原样保存在私有元数据，Demo8维内参排列未擅自转换为K；CSV内外参保留明确字段。','','原附件缺少Data_Documentation、正式缺失观测实验协议和组织方冻结的最终测试种子等，处理数据不能补出这些官方材料。问题三现有公开验证种子不能冒充最终测试。','','下一步可立即开展：在train上建立逐源基线和自然缺失分布；确认坐标/关键点后实现因果缺失恢复；按环境隔离和不重叠遮罩冻结实验；在validation比较方法并选择阈值；最终只运行一次冻结test评估。问题三先建立仅接收数值前缀的策略和可信评估器，再进行模型比较。数据资格、方法优越性、结论最优性是三个不同命题，本报告只完成前者及建模数据准备。','','## 7. 验收证据','',json.dumps(s['checks'],ensure_ascii=False,indent=2),'',f"依赖版本：{s['dependencies']}。",'','视频失败详表、空前缀与帧索引差异见summary.json；即使条目为0，也不代表中间所有视频帧都经过解码验证。']
(O/'数据处理完整报告.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(s,ensure_ascii=False,indent=2))

