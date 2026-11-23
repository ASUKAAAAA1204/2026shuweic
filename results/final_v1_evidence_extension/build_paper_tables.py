"""Render the frozen five-dimension results as Markdown and A4 three-line tables."""
from pathlib import Path
import textwrap

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pandas as pd

EXT = Path(__file__).resolve().parent
R = EXT.parent
OUT = R.parent/"figures/five_dimension_tables"
S = R/"final_v1_sensitivity"
REM = R/"final_v1_remaining"
TAIL = R/"final_v1_stability/tail"
FORMAL = R.parent/"tmp/final_v1_20261122/C题工作区/正式方案/results"
pages = []


def page(name):
    pages.append((name, []))


def table(title, headers, rows, note="", widths=None):
    h = headers.split("|")
    rows = [list(map(str,r)) for r in rows]
    assert rows and all(len(r)==len(h) for r in rows)
    pages[-1][1].append((title,h,rows,note,widths))


def read(folder, name):
    return pd.read_csv(folder/name)


def fmt(x,n=2):
    return f"{x:.{n}f}".replace("-", "−")


def pct(x):
    return fmt(100*x)


def span(lo,hi):
    return f"[{fmt(lo)}, {fmt(hi)}]"


def build():
    page("五维稳定性实验结果")
    table("五个维度的主要对照结果","维度|固定条件与比较|主要结果",[
        ["参数敏感性","第三问　β=0.5 秒，α 从 1 增至 1.5","平均按时率 89.92% → 71.81%"],
        ["跨环境稳定性","第二问　短历史，回传延迟翻倍","已有 8/8 地误差上升 20.95%–35.72%"],
        ["配对不确定性","第三问　300 个工期种子，长工期减名义","−18.10 个百分点，95% 区间 [−18.79, −17.41]"],
        ["尾部风险","第三问　S2 名义工况，两批各 300 次","最差一成均值 70.42% / 70.69%"],
        ["内部种子稳定性","第三问　长工期复验，baseline / B_K64","单例标准差均值 0 / 5.85 个百分点"]],
        "轨迹主指标为归一化平均位移误差（NADE），越低越好。第三问使用加权按时率，越高越好。以下各表的精度与复验判据均按舍入前数值计算。",[.17,.41,.42])
    grid=read(EXT,"grid_means.csv")
    q1=read(S,"q1_history.csv")
    table("第一问：可见历史与预测时长","可见历史|0.2 秒 NADE|0.6 秒 NADE|1.0 秒 NADE|1.0 秒覆盖率（%）",
          [[label,*[fmt(x,5) for x in g.nade],pct(g.coverage.iloc[-1])] for key,label in
           [("full","完整"),("0.75s","0.75 秒"),("0.30s","0.30 秒")] for g in [q1[q1.history==key].sort_values("horizon")]],
          "同一批 212 个开发窗，各预测时长均取整个前缀的平均位移误差。")
    h=read(EXT,"history_interaction.csv").query("environment=='all'").set_index("split")
    table("第一问：截短至 0.30 秒历史的绝对误差增量","划分|0.2 秒|0.6 秒|1.0 秒|1.0 秒减 0.2 秒",
          [[label,*[fmt(v,5) for v in h.loc[k,["diff_02","diff_06","diff_10","interaction"]]]] for k,label in
           [("validation","开发"),("test","既有测试")]],"差值为短历史 NADE 减完整历史 NADE。开发集 3/4 地、既有测试集 4/4 地的一秒增量高于 0.2 秒增量。")
    page("敏感性｜回传延迟与实际工期")
    d=read(S/"q2","summary.csv")
    table("第二问：固定两次请求，改变单次延迟","延迟（秒）|NADE|末帧误差|球直径|覆盖率（%）|等待（秒）",
          [[fmt(x.delay),fmt(x.nade,5),fmt(x.nfde,5),fmt(x.norm_diameter,5),pct(x.coverage),fmt(x.waiting_s)] for x in d.itertuples()],
          "开发集 212 窗，遮挡一秒。遮挡缩短至 0.2 秒后，三档 NADE 均为 0.03958。")
    d=read(EXT/"trajectory","summary.csv").query("selection=='all' and scope=='split' and metric=='nade'")
    table("第二问：历史长度与回传延迟的交互","划分|完整／0.10 秒|完整／0.20 秒|短／0.10 秒|短／0.20 秒|交互差",
          [[{"test":"既有测试","validation":"开发"}[x.split],*[fmt(v,5) for v in [x.full_d01,x.full_d02,x.short_d01,x.short_d02,x.did]]] for x in d.itertuples()],
          "各格为 NADE。交互差为短历史下的延迟效应减完整历史下的延迟效应。短历史为 0.30 秒。")
    d=read(FORMAL,"q2_cost.csv").query("model=='B' and split=='test'")
    table("第二问：固定 0.10 秒延迟，改变请求预算","预算（次）|NADE|末帧误差|等待（秒）",
          [[x.budget,fmt(x.nade,5),fmt(x.nfde,5),fmt(x.waiting_s)] for x in d.itertuples()],"同一批 390 个既有测试窗。")
    official=read(S,"q3_mapping.csv")
    table("第三问：官方五种工期映射下的调度结果（%）","α|β（秒）|S1 按时率|S2 按时率|S3 按时率|平均按时率|平均完成率",
          [[fmt(x.alpha),fmt(x.beta),*[pct(v) for v in [x.S1,x.S2,x.S3,x.on_time,x.completion]]] for x in official.itertuples()],
          "每档三个案例、每例 30 个官方工期种子。按时率、完成率按任务权重计算，三个案例等权。")
    table("第三问：扩展九档的平均加权按时率（%）","α|β = 0 秒|β = 0.5 秒|β = 1 秒",
          [[fmt(k),*[pct(v) for v in g.sort_values("beta").overall]] for k,g in grid.groupby("alpha")],
          "每格 300 个新工期种子 × 3 个案例，内部种子固定。")
    page("跨环境稳定性｜已有地点与工期来源")
    env=read(REM,"environment_results.csv")
    rows=[]
    for (q,s),g in env.groupby(["question","split"]):
        a=g[g.scope=="all"].iloc[0]; e=g[g.scope=="environment"]; omit=g[g.scope=="omit_environment"].nade_change_pct
        rows.append([q.upper(),{"test":"既有测试","validation":"开发"}[s],fmt(a.nade_change_pct),f"{sum(e.nade_change_pct>0)}/4",span(omit.min(),omit.max()),pct(a.coverage_nominal),pct(a.coverage_stress)])
    table("单因素扰动与逐一排除地点的结果","问题|划分|增幅（%）|同向地点|排除一地范围（%）|名义覆盖率（%）|扰动覆盖率（%）",rows,
          "第一问截短历史，第二问在完整历史下将延迟翻倍。增幅指 NADE 相对增幅。")
    d=read(EXT,"spaced_environments.csv").sort_values(["split","environment_id"],ascending=[False,True])
    table("第二问：间隔选窗后各地点的配对误差","地点|划分|窗数|延迟 0.10 秒|延迟 0.20 秒|NADE 增幅（%）",
          [[("超市" if "supermarket" in x.environment_id else "餐厅" if "restaurant" in x.environment_id else "快递站")+x.environment_id[-4:],
            "开发" if x.split=="validation" else "既有测试",x.n,fmt(x.nade_01,5),fmt(x.nade_02,5),fmt(x.change_pct)] for x in d.itertuples()],
          "短历史 0.30 秒，同记录同手的保留窗至少间隔 2 秒，共 308 窗。仓库 24 个地点中，16 地用于训练，4 地用于开发，4 地用于既有测试。符合外部盲验条件的新地点为 0，14 地方案尚未运行。")
    d=read(REM,"source_summary.csv")
    table("第三问：更换 A、P 任务族工期来源后的加权按时率（%）","A 族来源|P 族来源|S1|S2|S3|三例均值",
          [[["快递站 0022","超市 0147"][x.a],["餐厅 0004","超市 0074"][x.p],*[pct(v) for v in [x.S1,x.S2,x.S3,x.overall]]] for x in d.itertuples()],
          "L 族来源固定，四组共 360 次回放。餐厅、超市的 P 族可用工期分别为 3 条、12 条。")
    page("配对不确定性｜效应量与区间")
    d=read(REM,"paired_uncertainty.csv").query("metric=='nade'")
    table("地点等权重抽下的 NADE 相对变化","问题|划分|平均差（%）|中间 95% 范围（%）|双侧符号检验 p",
          [[x.question.upper(),"开发" if x.split=="validation" else "既有测试",fmt(x.env_equal_delta),span(x.resample_p025,x.resample_p975),fmt(x.exact_twosided_sign_p,3)] for x in d.itertuples()],
          "每个划分四个地点，枚举 256 种有放回重抽组合。区间描述已观测地点构成的影响。")
    extra=read(EXT,"paired_grid.csv"); first=official.query("alpha==1.5 and beta==0.5").iloc[0]
    rows=[["长工期减名义","官方 30",fmt(first.delta_pp),span(first.low_pp,first.high_pp),fmt((first.high_pp-first.low_pp)/2)]]
    for x in extra.itertuples():
        rows.append(["长工期减名义" if x.contrast=="long-minus-nominal" else "α 与 β 的交互","扩展 "+str(x.n),fmt(x.delta_pp),span(x.lower_pp,x.upper_pp),fmt(x.halfwidth_pp)])
    table("第三问：工期效应及配对区间（百分点）","对比|工期种子|平均差|配对 95% 区间|半宽",rows,
          "以外部种子为单位成组重抽 10,000 次。扩展前 30 个包含在全部 300 个中。主对比固定 β=0.5 秒，交互比较 β 从 0.5 增至 1 时，α=1.5 相对 α=1 的额外变化。主对比半宽 0.69，达到 ≤2 个百分点的预设要求。")
    d=read(REM,"source_effects.csv")
    names=["换 A，P 取餐厅","换 A，P 取超市","换 P，A 取快递站","换 P，A 取超市"]
    labels=dict(zip(["A1-A0_P0","A1-A0_P1","P1-P0_A0","P1-P0_A1"],names))
    table("名义工况下更换工期来源的配对差（百分点）","来源对比|平均按时率差|配对 95% 区间",
          [[labels[x.contrast],fmt(x.delta_pp),span(x.low_pp,x.high_pp)] for x in d.itertuples()],
          "A 从快递站改为超市，P 从餐厅改为超市。每项固定三个案例和 30 个外部种子。")
    d=read(REM,"capacity_summary.csv").groupby("alpha").mean(numeric_only=True)
    table("调度可行性对长工期损失的约束（%）","工况|实际按时率|按时率上界|不可避免损失下界",
          [[name,pct(x.observed),pct(x.ceiling),pct(x.unavoidable_loss_floor)] for name,(_,x) in zip(["名义","长工期"],d.iterrows())],
          "按必要工时、前驱、移动和预算求放宽约束后的上界。上界与实际按时率之差只给出可改善空间的上限。")
    page("尾部风险｜困难窗口与最差一成实例")
    d=read(EXT,"tail_threshold_transfer.csv").query("environment=='all'")
    out=[]
    for x in d.itertuples():
        name="第一问" if x.question=="q1" else "第二问"
        out.append([name,fmt(x.threshold,5),f"{round(x.n*x.exceed_nominal)} / {x.n}",pct(x.exceed_nominal),f"{round(x.n*x.exceed_stress)} / {x.n}",pct(x.exceed_stress),f"{x.new} / {x.recovered}"])
    table("开发名义 P90 阈值在既有测试窗上的超阈情况","问题|阈值|名义超阈|占比（%）|扰动超阈|占比（%）|新增／恢复",out,
          "严格大于阈值计为超阈。第一问缩短历史，第二问延迟翻倍，测试窗等权。")
    d=read(TAIL,"paired_tail.csv")
    table("锁定开发集原有最差 22 窗后的误差","问题|名义 NADE|扰动 NADE|相对增幅（%）|仍属最差组",
          [[x.comparison[:2].upper(),fmt(x.nominal_tail_mean,5),fmt(x.stress_tail_mean,5),fmt(100*x.tail_change_mean/x.nominal_tail_mean),f"{x.overlap_count}/22"] for x in d.itertuples()],
          "212 个开发窗，先锁定名义条件下最差 22 窗，再对同窗评分。")
    d=read(EXT,"tail_replication.csv");old=read(TAIL,"q3_summary.csv").query("beta==0.5").set_index(["case_id","alpha"])
    table("六组工况最差一成的平均加权按时率（%）","案例／工况|官方最低 3 次|扩展最低 30 次|前半最低 15 次|后半最低 15 次|差（百分点）",
          [[x.case_id+(" 名义" if x.alpha==1 else " 长工期"),pct(old.at[(x.case_id,x.alpha),"worst3_mean_on_time"]),pct(x.all),pct(x.first150),pct(x.last150),fmt(x.gap_pp)] for x in d.itertuples()],
          "官方每组 30 例，扩展每组 300 例。两半绝对差 ≤5 个百分点为通过，六组中五组通过，S2 名义工况相差 8.89 个百分点。")
    d=read(R/"final_v1_s2_tail","batches.csv")
    table("S2 名义工况专项复验","新批次|回放次数|尾部例数|全批均值（%）|尾部均值（%）",
          [[x.batch,x.n,x.worst_n,fmt(x.overall_pct),fmt(x.tail_pct)] for x in d.itertuples()],
          "两批外部种子独立，计算使用未舍入值。两批尾部均值绝对差为 0.28 个百分点，通过同一 5 个百分点判据。")
    page("内部抽样种子稳定性｜baseline 与终稿配对")
    d=read(R/"q3_baseline_seed_control","summary.csv")
    for batch,name in [("original","首批"),("replication","独立复验批")]:
        g=d[d.batch==batch].sort_values(["alpha","beta","method"],ascending=[True,True,False])
        table(name+"：固定任务和实际工期后更换内部种子",
              "α／β|策略|按时率均值（%）|单例 SD（百分点）|首步一致（%）|全程一致（%）|得分变化例数",
              [[f"{x.alpha:g}／{x.beta:g}",x.method,fmt(x.mean_rate_pct),fmt(x.mean_within_instance_sd_pp,3),
                fmt(x.first_action_pair_agreement_pct),fmt(x.path_pair_agreement_pct),f"{x.n_instances_rate_varies}/{x.n_instances}"] for x in g.itertuples()],
              "每行 90 个实例，每例 20 个种子。先在单例内求样本标准差（SD）及两两一致率（190 对），再对实例等权平均。")
    return pages


def render():
    OUT.mkdir(parents=True,exist_ok=True)
    mpl.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"pdf.fonttype":42})
    md=["# 冻结模型五维稳定性实验结果表","","模型版本为 `v1.0.0-final`。baseline 单列于第三问内部种子对照。",""]
    num=0
    with PdfPages(OUT/"五维结果三线表.pdf") as pdf:
        for pg,(section,tables) in enumerate(pages,1):
            fig=plt.figure(figsize=(210/25.4,297/25.4))
            ax=fig.add_axes([0,0,1,1]); ax.set_axis_off()
            ax.set(xlim=(0,210),ylim=(297,0))
            def text(x,y,s,size=9,**kw):
                ax.text(x,y,s,fontsize=size,va="center",**kw)
            text(17,14,section,12,weight="bold")
            text(193,14,f"{pg} / {len(pages)}",8,ha="right",color="#555555")
            y=27
            md.extend(["## "+section,""])
            for title,headers,rows,note,widths in tables:
                num+=1
                label=f"表 {num}　{title}"
                text(105,y,label,10,ha="center",weight="bold")
                md.extend([label,"","| "+" | ".join(headers)+" |","|"+"|".join(["---"]+["---:"]*(len(headers)-1))+"|"])
                left,right=15,195
                if widths is None:
                    def weight(column):
                        return max(sum(1 if ord(c)>255 else .57 for c in s) for s in column)+1.3
                    weights=[weight(col) for col in zip(headers,*rows)]
                    widths=[v/sum(weights) for v in weights]
                edges=[left]
                for w in widths: edges.append(edges[-1]+w*(right-left))
                y+=4
                def line(pos,pt): ax.plot([left,right],[pos,pos],color="#263441",linewidth=pt,solid_capstyle="butt")
                line(y,.9)
                for i,row in enumerate([headers,*rows]):
                    height=6.1 if i else 7.1
                    for j,cell in enumerate(row):
                        fontsize=8.4 if pg!=1 or num!=1 else 8
                        anchor=(edges[j]+edges[j+1])/2
                        item=ax.text(anchor,y+height/2,cell,fontsize=fontsize,ha="center",va="center",
                                     color="#20272d",weight="semibold" if i==0 else "normal")
                        renderer=fig.canvas.get_renderer()
                        available=(edges[j+1]-edges[j]-1.2)*fig.bbox.width/210
                        width=item.get_window_extent(renderer).width
                        if width>available: item.set_fontsize(fontsize*available/width)
                    y+=height
                    if not i: line(y,.5)
                    else: md.append("| "+" | ".join(row)+" |")
                line(y,.9)
                md.extend(["","注："+note,""])
                y+=4
                for part in textwrap.wrap("注："+note,width=53,break_long_words=True):
                    text(left,y,part,8.1,color="#53575a");y+=4.1
                y+=6
            assert y<294,(pg,y)
            pdf.savefig(fig)
            plt.close(fig)
    (OUT/"五维结果表.md").write_text("\n".join(md),encoding="utf8")


if __name__=="__main__":
    build()
    render()
