from common import *
import argparse,subprocess,collections,platform,time

def compute(out,video):
 import prepare,q12,duration,q3,checks,checks_extra,audit,summarize
 prepare.run(out/'data',video);q12.run(out);duration.run(out);q3.run(out);checks.run(out);checks_extra.run(out);audit.run(out);summarize.run(out)
def reproduce(out):
 other=out.parent/(out.name+'_reproduced')
 subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(other),'--stage','compute','--skip-video'],check=True)
 return compare_reproduction(out)

def compare_reproduction(out):
 other=out.parent/(out.name+'_reproduced');comparisons=[]
 files=['data/protocol.json','data/windows.csv','data/records.csv','data/sample_usage.csv','data/input_hashes.csv','audit/source_hashes.csv','models/duration.json','models/frozen_selection.json','q12/prediction_radius_train.json','q12/reliability_model.json','q12/reliability_metrics.csv','q12/reliability_predictions.csv','q12/reliability_calibration.csv','q12/open_loop_windows.csv','q12/summary.csv','q3/trials.csv','q3/summary.csv','q3/paired_differences.csv','q3/paired_summary.csv']
 for f in files:comparisons.append(dict(path=f,mode='sha256 identical',passed=sha(out/f)==sha(other/f)))
 a=readcsv(out/'q12/replay_windows.csv');b=readcsv(other/'q12/replay_windows.csv');assert len(a)==len(b)
 for rows in [a,b]:
  for r in rows:r.pop('runtime_s',None)
 comparisons.append(dict(path='q12/replay_windows.csv',mode='all fields identical except runtime',passed=a==b))
 for p in sorted((out/'q3').glob('*.jsonl')):comparisons.append(dict(path='q3/'+p.name,mode='all actions and observations identical',passed=sha(p)==sha(other/'q3'/p.name)))
 writecsv(out/'reproduction.csv',comparisons);dump(out/'reproduction.json',dict(checks=len(comparisons),passed=sum(r['passed'] for r in comparisons),all_passed=all(r['passed'] for r in comparisons),directory=str(other)))
 assert all(r['passed'] for r in comparisons),'reproduction mismatch'

def report(out):
 (out/'规则与证据对照.md').write_text(Path(__file__).with_name('rules_template.md').read_text(encoding='utf8'),encoding='utf8')
 q=load(out/'q3/checks.json');q12=load(out/'q12/checks.json');ac=load(out/'audit/checks.json');tests=load(out/'test_summary.json');pc=load(out/'data/prepare_checks.json');selection=load(out/'models/frozen_selection.json');proto=load(out/'data/protocol.json');rows=readcsv(out/'q12/summary.csv');tr=readcsv(out/'q3/trials.csv');reasons=readcsv(out/'audit/quality_reason_counts.csv')
 text=['# C题三问基线闭环报告','','## 本阶段结论','','本报告交付数据准备、固定研究协议、三问可复现基线及可信评估器。最终验收状态以《验收结果.json》为准。所有模型拟合使用train，所有方法比较使用validation，test只作结构检查。不是最终测试结果，也不证明全局最优。','','## 数据与协议',f"共{pc['records']}条记录，{ac['raw_channels_verified']}条位姿通道独立回查原文件；{pc['prefixes_verified']}份前缀回查一致。固定窗口{pc['windows']}个，划分分布{pc['window_counts']}。五通道同一记录时间对齐已在v1审计；2个语义终点比最后轨迹点晚不足0.01秒，保留原端点不补帧。",'',f"状态是各手21关键点几何中心；坐标单位只称源单位，不假设米；不使用四元数、相机投影、商品状态或机器人性能。训练尺度为{proto['train_scale']}，主要容差为该尺度5%，另报2%/10%。",'','窗口1秒，前置0.2秒历史，窗口之间至少1秒间隔；不同遮罩长度是同一窗口上的配对实验，不能视为独立样本。按记录汇总后环境等权，环境聚类bootstrap 2000次。验证环境只有4个，分源可能只有1个；单环境组不估计环境置信区间，其他小样本区间也须谨慎解释。训练残差90%分位数仅提供经验预测球，不保证条件覆盖或分布无关覆盖。','','问题二为参与者研究协议：单次请求成本1，最多2次，主延迟0.1秒并检查0/0.2秒；请求时刻原始样本延迟返回，不能把返回时刻未来样本提前提供。该约定不冒充缺失的官方实验文件。','','## 问题一：状态变化和观测可靠性','','保持末值与恒速模型使用相同代表点。可靠性模型用当前及过去0.2秒的速度和质量预测未来0.2秒是否出现标坏，与训练标坏比例常数预测器比较。模型参数和标准化在q12/reliability_model.json；常数/线性状态模型无待拟合运动参数；不确定性半径由训练集残差产生。','','|来源|模型|遮罩秒|验证ADE|环境95%区间|90%球实际覆盖|','|---|---|---:|---:|---|---:|']
 for src in ['EgoDemo','EgoPro','EgoStandard']:
  for model in ['hold','constant_velocity']:
   for h in [.2,.6,1.]:
    rr=[r for r in rows if r.get('split')=='validation' and r['dataset']==src and r.get('model')==model and float(r.get('horizon_s') or 0)==h];m=next(r for r in rr if r['metric']=='ade');c=next(r for r in rr if r['metric']=='coverage90');ci=f"[{float(m['lo']):.5g}, {float(m['hi']):.5g}]" if m['lo'] and m['hi'] else '不可估计（仅1环境）';text.append(f"|{src}|{model}|{h}|{float(m['mean']):.5g}|{ci}|{float(c['mean']):.3f}|")
 text+=['','图见q12/validation_ade.svg，数据见figure_source.csv。不同来源原坐标单位未独立确认，因此不跨源混合解释绝对位置误差。按语义任务族和区间相对四分位阶段的结果见audit/retrospective_phase_metrics.csv；阶段使用区间终点，只用于事后解释，绝不输入在线策略。','','|质量预测方法|验证Brier|验证AUC|实际标坏率|预测均值|','|---|---:|---:|---:|---:|']
 for r in readcsv(out/'q12/reliability_metrics.csv'):
  if r['split']=='validation' and r['dataset']=='all':text.append(f"|{r['method']}|{float(r['brier']):.5g}|{r['auc'] or '不可定义'}|{float(r['observed_rate']):.4f}|{float(r['mean_probability']):.4f}|")
 text+=['','质量概率分箱校准见q12/reliability_calibration.csv，逐样本预测见reliability_predictions.csv。CSV源global质量标记保守用于双手，原始problem_type含手别原因另行完整保留；这会扩大保守排除范围，该模型预测的是相应源质量标记而非独立手部追踪真值。质量标签是源追踪标记，不能解释为任务失败标签。AUC在单类组不可定义。预测优劣以验证结果为准，不以模型复杂程度为准。','','## 问题二：请求、延迟与恢复','','|来源|方法|ADE|90%覆盖|平均半径|请求数|等待秒|','|---|---|---:|---:|---:|---:|---:|']
 for src in ['EgoDemo','EgoPro','EgoStandard']:
  for method in ['hold','constant_velocity','fixed_review','adaptive_review']:
   rr=[r for r in rows if r['dataset']==src and r.get('method')==method and float(r.get('delay_s') or -1)==.1 and float(r.get('tolerance_fraction') or 0)==.05];vals={r['metric']:float(r['mean']) for r in rr};text.append('|'+src+'|'+method+'|'+'|'.join(f'{vals[k]:.5g}' for k in ['ade','coverage90','mean_radius90','requests','waiting_s'])+'|')
 text+=['','全部7632行回放包括三档延迟和三档容差，见q12/replay_windows.csv，逐窗口请求/返回时间在actions字段。runtime_s记录每窗口计算时间，不含训练和磁盘读取，复现时允许不同。固定请求在0.3/0.6秒；自适应按预测球达到容差触发，并限制在途请求和至少0.2秒请求间隔。','','自然质量区间见audit/natural_bad_runs.csv，原因分布如下。只有恢复到未标坏表示观测标记恢复，没有独立真值时不计算自然遮挡内真实位置误差。人工遮罩从高质量连续区间抽样，不能直接外推为自然遮挡性能。','','|来源|划分|手别|原因|标记数|','|---|---|---|---|---:|']
 for r in reasons:text.append('|'+ '|'.join(r[k] for k in ['dataset','split','side','reason','rows'])+'|')
 text+=['','## 问题三：调度和时长','','严格使用官方抽样、3案例、30种子与5映射，不做全因子扩展。四种策略各三档预留分位数，共5400次；同案例和种子的源任务配对。线性时长模型在训练环境留一验证中选择正则参数，50/80/90%预留残差也来自训练环境留一预测。族分位数仅拟合训练区间。','','主目标为加权按时完成率，完成率和成本作为次级排序；本阶段没有以低服务换低成本作为主目标。各情景下所有共同指标和方差/区间见q3/summary.csv，配对差异区间见q3/paired_summary.csv。','','|策略|预留分位数|名义按时率|完成率|成本|','|---|---:|---:|---:|---:|']
 for r in selection['ranking']:text.append(f"|{r['method']}|{r['quantile']}|{r['on_time']:.4f}|{r['completion']:.4f}|{r['cost']:.3f}|")
 text+=[f"\n验证集选择：{selection['selected']}。这是同一验证集上的选择结果，不能用作无偏最终性能估计。",'','固定复查可能增加前缀信息，也会消耗时间和预算；须看paired_summary.csv判断收益及波动，不能因使用复查就宣称更优。完整时长预测误差和残差区间见q3_duration_predictions.csv，训练CV见q3_duration_train_cv.csv。', '',f"独立计分最大差异{q['independent_metric_max_error']:.3g}；{q['hidden_future_invariance_checks']}次隐藏未来扰动检查；非法动作{q['invalid_actions']}。策略独立进程只读冻结模型，私有源文件读取被运行时访问守卫拒绝。此守卫用于可复现代码的访问审计，不宣称抵御任意恶意原生代码的操作系统安全沙箱。",'','## 失败模式与结论边界','','1. 恒速在转向、加减速时可能迅速失准，保持末值可能胜过恒速；逐窗口误差排序见failure_cases.csv。','2. 训练残差球可能在验证环境欠覆盖或过宽，需要后续改善校准，不能只报告名义90%。','3. 自适应请求可能耗尽预算却不改善误差，固定复查可能拖延高价值任务。','4. 源质量标签本身不是独立真值；Demo稀疏未列出视为未标坏属于已声明约定。','5. 未解决的物理单位/朝向/相机标定已从本阶段模型依赖中移除；后续若使用这些信息必须重新核实。','6. 官方问题二详细协议和最终测试种子仍缺失，最终测试不在本阶段交付范围。','7. 三名操作者跨train/test，结论只讨论官方跨环境划分，不声称未见操作者泛化。','','## 复现与交付','','从本工作区根目录运行：','','```powershell',"& 'C:\\Users\\33671\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe' 'C题工作区\\scripts\\baseline\\run_pipeline.py' --reproduce",'```','','入口默认执行全部步骤及全视频解码；首次验证既有v1转换与原始数值一致，保留原始数据和v1。输出采用独立目录；依赖位于运行依赖314，版本锁见environment.json。若从新机器开始，需要先按requirements-lock.txt安装对应Python3.14兼容依赖，并恢复原始数据与v1准备产物。若需重建v1，按其报告运行process_dataset.py、finalize_processing.py和additional_audit.py，再运行本入口。','','每个输出表均有逐记录或逐试验来源；结果日志只在可信评估端含事后源引用，策略输入不含seed、unit_id、语义、终点和完整时长。报告、协议和规则矩阵共同构成本阶段说明。']
 (out/'三问基线完整报告.md').write_text('\n'.join(text),encoding='utf8')
 failure=[]
 candidates=readcsv(out/'q12/replay_windows.csv')
 for src in ['EgoDemo','EgoPro','EgoStandard']:
  for method in ['hold','constant_velocity','fixed_review','adaptive_review']:
   rr=[r for r in candidates if r['dataset']==src and r['method']==method and float(r['delay_s'])==.1 and float(r['tolerance_fraction'])==.05]
   failure.extend(sorted(rr,key=lambda r:float(r['ade']),reverse=True)[:5])
 writecsv(out/'failure_cases.csv',failure)
 durationrows=readcsv(out/'q3_duration_predictions.csv');ds=[]
 for method in ['family','ridge']:
  for length in [.2,.6,1.]:
   rr=[r for r in durationrows if r['split']=='validation' and r['model']==method and float(r['length'])==length];ds.append(dict(model=method,length=length,**grouped_boot([float(r['absolute_error']) for r in rr],[r['environment_id'] for r in rr])))
 writecsv(out/'q3_duration_summary.csv',ds)
 import cv2,pyarrow
 dump(out/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,pyarrow=pyarrow.__version__,platform=platform.platform(),executable=sys.executable))
 (out/'requirements-lock.txt').write_text(f'numpy=={np.__version__}\npyarrow=={pyarrow.__version__}\nopencv-python-headless==5.1.0.95\n',encoding='utf8')
 writecsv(out/'code_config_hashes.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [*Path(__file__).parent.glob('*.py'),*CONFIG.glob('*')] if p.is_file()])

def accept(out):
 checks=[]
 def c(name,passed,evidence):checks.append(dict(name=name,passed=bool(passed),evidence=evidence))
 p=load(out/'data/prepare_checks.json');q=load(out/'q3/checks.json');t=load(out/'test_summary.json');a=load(out/'audit/checks.json');e=load(out/'extra_checks.json');v=readcsv(out/'data/full_video_decode.csv');r=load(out/'reproduction.json');q12=load(out/'q12/checks.json')
 c('all records unique',p['records']==p['unique_uuid']==69,'data/prepare_checks.json');c('all raw numerical channels checked',a['raw_channels_verified']==345,'audit/raw_numeric_recheck.csv');c('all 1029 prefixes source exact',p['prefixes_verified']==1029,'data/prepare_checks.json');c('environment disjoint',p['environment_overlap']==0,'data/split_overlaps.csv');c('all windows qualified and separated',e['windows_qualification_checked']==p['windows'] and e['passed'],'extra_checks.json');c('all videos fully decoded',len(v)==184 and all(x['passed']=='True' for x in v),'data/full_video_decode.csv');c('nine visual records saved',a['visual_samples']==9 and len(list((out/'audit').glob('*.jpg')))>=9,'audit/九条记录核对.html');c('no orientation or unit dependent model',p['state_avoids_unverified_joint_order_and_orientation'],'data/protocol.json');c('boundary and privacy tests',t['passed']==t['total'] and t['total']>=20,'tests.csv');c('all 5400 paired trials complete',q['trials']==5400,'q3/trials.csv');c('no invalid policy actions',q['invalid_actions']==0,'q3/checks.json');c('independent metrics agree',q['independent_metric_max_error']<=1e-9,'q3/checks.json');c('q3 future noninterference',q['hidden_future_invariance_checks']>=5400,'q3/checks.json');c('q2 future noninterference',e['q2_future_tests']>=100,'q2_future_invariance.csv');c('q2 budgets and finite results',q12['request_limit_pass'] and q12['nonfinite_metrics']==0,'q12/checks.json');c('no test performance used',not q['test_evaluated'] and not q12['test_evaluated'],'q12/checks.json, q3/checks.json');c('independent reproduction',r['all_passed'],'reproduction.csv');c('report and frozen model exist',all((out/x).is_file() for x in ['三问基线完整报告.md','models/frozen_selection.json','code_config_hashes.csv','规则与证据对照.md']),'delivery files')
 result=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',required=len(checks),passed=sum(x['passed'] for x in checks),blockers=[x for x in checks if not x['passed']],checks=checks,scope='data closure and reproducible three-problem baseline; not final competition solution');dump(out/'验收结果.json',result);writecsv(out/'验收清单.csv',checks);print(json.dumps(result,ensure_ascii=False,indent=2));return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'阶段交付/baseline_v2');ap.add_argument('--stage',choices=['all','compute','report','accept'],default='all');ap.add_argument('--skip-video',action='store_true');ap.add_argument('--reproduce',action='store_true');args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if args.stage in ['all','compute']:compute(out,not args.skip_video)
 if args.reproduce or args.stage=='all':reproduce(out)
 if args.stage in ['all','report']:report(out)
 if args.stage in ['all','accept']:assert accept(out)['status']=='PASS','delivery gates failed'
if __name__=='__main__':main()
