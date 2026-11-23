"""Build the five review plates from the frozen comparison and archived seed repeats."""
from zipfile import ZIP_DEFLATED, ZipFile

import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_input.seed_plot_data import load_seed_points
from plot_baseline_stability_lines import CASES, STYLE, VERSIONS, lines, panels, q3_results
from plot_final_baseline import RESULTS
from plot_five_dimensions import BLUSH, BLUE, INK, NAVY, RED, ROOT, RULE, SENS, SUB, canvas, note, style
from schedule_statistics import paired

OUT = ROOT / "figures/final_v1_stability_review"


def key(fig, baseline, line=True):
    handles = []
    for model, label in zip(VERSIONS, [baseline, "终稿 v1.0.0-final"]):
        settings = STYLE[model].copy()
        if not line:
            settings["ls"] = "none"
        handles.append(Line2D([], [], **settings, ms=6, lw=1.7, label=label))
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(.958,.918), ncol=2)


def sensitivity(q):
    fig = canvas("参数敏感性", "前两问使用相同的 390 个测试窗。第三问固定 β = 0.5 秒，三个案例各使用相同的 30 个官方工期种子。")
    key(fig,"对照基线")
    axes = panels(fig, ["第一问 · 预测时长","第二问 · 观测预算","第三问 · 工期倍率"])
    for ax, file, index, xlabel in zip(axes, ["q1_horizons.csv","q2_cost.csv"],
                                       ["horizon","budget"], ["预测时长（秒）","允许请求次数"]):
        data = pd.read_csv(RESULTS/file).query("split=='test'").pivot(index=index,columns="model",values="nade")
        lines(ax,data.rename(columns={"original":"基线","B":"终稿"}))
        ax.set(xlabel=xlabel,ylabel="归一化位移误差（NADE）")
    lines(axes[2],q.query("beta==.5").groupby("alpha")[VERSIONS].mean())
    axes[2].set(xlabel="工期倍率 α",ylabel="加权按时率（%）")
    note(fig,"前两问的基线为固定阻尼，误差越低越好。第三问的基线为上一正式版，按时率越高越好。")
    return fig


def sites():
    raw = pd.read_csv(RESULTS/"q2_windows.csv.gz").query("split in ['test','validation']")
    assert raw.requests.eq(2).all() and np.allclose(raw.waiting_s,.2)
    levels = ["split","environment_id","episode_uuid","side","window_id"]
    pairs = raw.pivot(index=levels,columns="model",values="nade")[["original","B"]]
    assert pairs.notna().all().all() and len(pairs)==602
    counts = pairs.groupby(level=levels[:2]).size()
    assert counts.groupby(level="split").sum().to_dict()=={"test":390,"validation":212}
    data = pairs
    for group in [levels[:-1],levels[:-2],levels[:2]]:
        data = data.groupby(level=group).mean()
    fig = canvas("跨环境稳定性", "第二问按地点比较两版的归一化位移误差。每对点使用同一批窗口、两次观测及相同的返回延迟。")
    key(fig,"固定阻尼",False)
    axes = panels(fig,["开发集 · 212 窗","既有测试集 · 390 窗"])
    names = {"restaurant":"餐厅","supermarket":"超市","station":"快递站"}
    for ax,split,x in zip(axes,["validation","test"],[.164,.641]):
        ax.set_position([x,.245,.295,.439])
        ax.grid(False)
        style(ax,"x")
        labels = []
        for y,(site,row) in enumerate(data.loc[split].iterrows()):
            fields = site.split("_")
            labels.append(f"{names[fields[-2]]} {fields[-1]}\n{counts.loc[split,site]} 窗")
            ax.plot([row.original,row.B],[y,y],lw=2.1,color="#94A9BD",zorder=2)
            ax.plot(row.original,y,"o",mfc="white",mec=BLUE,ms=8,mew=1.5)
            ax.plot(row.B,y,"D",color=RED,ms=4.7)
        ax.set(xlim=(.052,.126),xticks=[.06,.08,.10,.12],ylim=(3.55,-.55),
               yticks=range(4),yticklabels=labels,xlabel="归一化位移误差（越低越好）")
        ax.tick_params(axis="y",length=0,labelsize=9)
        ax.spines["left"].set_visible(False)
    note(fig,"地点内依次对窗口、手侧和记录取平均。八个地点均已参与此前审核，图中结果仅覆盖这些已观测地点。")
    return fig


def forest(q):
    indexed = q.set_index(["alpha","beta","case_id","seed"])[VERSIONS].sort_index()
    diff = indexed.loc[1.5,.5] - indexed.loc[1.,.5]
    assert diff.shape==(90,2) and diff.notna().all().all()
    overall = diff.groupby(level="seed").mean()
    fig = canvas("配对不确定性", "第三问固定 β = 0.5 秒，比较工期倍率从 1 增至 1.5 后的加权按时率变化。负值表示按时率下降。")
    key(fig,"上一正式版",False)
    ax = fig.add_axes([.175,.215,.495,.493])
    style(ax,"x")
    ax.axvline(0,color=SUB,lw=1,ls=":")
    ax.set(yticks=[3,2,1,0],yticklabels=CASES+["三案例等权"],ylim=(-.50,3.50),
           xlabel="长工期减名义工期的按时率差（百分点）")
    ax.tick_params(axis="y",length=0,pad=9)
    ax.spines["left"].set_visible(False)
    ax.axhline(.5,color=RULE,lw=.85)
    ax.text(1.09,1.04,"平均差  [配对 95% 区间]",transform=ax.transAxes,color=SUB,fontsize=9)
    low = 0.
    for y,case in zip([3,2,1,0],CASES+["三案例等权"]):
        part = overall if case=="三案例等权" else diff.xs(case)
        assert len(part)==30
        for model,dy in zip(VERSIONS,[.15,-.15]):
            s = paired("long",part.sort_index()[model]/100)
            center,left,right = [s[k] for k in ["delta_pp","lower_pp","upper_pp"]]
            if case=="三案例等权" and model=="终稿":
                ref = pd.read_csv(SENS/"q3_mapping.csv").query("alpha==1.5 and beta==.5").iloc[0]
                np.testing.assert_allclose([center,left,right],ref[["delta_pp","low_pp","high_pp"]].to_numpy(float),atol=.02)
            color = STYLE[model]["color"]
            ax.errorbar(center,y+dy,xerr=[[center-left],[right-center]],fmt=STYLE[model]["marker"],
                        color=color,mfc="white" if model=="基线" else RED,ms=5.8,lw=1.6,capsize=3)
            ax.text(1.09,y+dy,f"{center:.2f}   [{left:.2f}, {right:.2f}]".replace("-","−"),
                    transform=ax.get_yaxis_transform(),va="center",fontsize=9.5,color=color)
            low = min(low,left)
    ax.set_xlim(5*np.floor(low/5)-1,1.5)
    note(fig,"每版按同一组 30 个工期种子配对重抽 10,000 次。三案例等权行先在每个种子内取案例平均，再按种子重抽。")
    return fig


def tail(q):
    fig = canvas("尾部风险", "第三问名义工况。每个案例使用 30 个共同的官方工期种子，分别按两版的加权按时率从低到高排序。")
    key(fig,"上一正式版")
    axes = panels(fig,CASES)
    rank = np.arange(1,31)
    for ax,case in zip(axes,CASES):
        part = q.query("case_id==@case and alpha==1 and beta==.5")
        frame = pd.DataFrame({m:np.cumsum(np.sort(part[m]))/rank for m in VERSIONS},index=rank)
        np.testing.assert_allclose(frame.loc[3], [part[m].nsmallest(3).mean() for m in VERSIONS])
        lines(ax,frame)
        for l in ax.lines:
            l.set_markevery([0,2,9,19,29])
        ax.axvline(3,color=SUB,lw=.85,ls=":",zorder=0)
        ax.set(xticks=[1,3,10,20,30],ylim=(58,102),xlabel="纳入最差的 k 次",ylabel="平均加权按时率（%）")
        assert frame.min().min()>58
        ax.text(.04,.93,f"最差 3 次    {frame.loc[3,'基线']:.2f} → {frame.loc[3,'终稿']:.2f}%",
                transform=ax.transAxes,color=INK,fontsize=8.3,bbox=dict(fc="white",ec="none",alpha=.9,pad=1))
    note(fig,"每条曲线取本版本最差 k 次的平均值。竖虚线 k = 3 对应最差一成，k = 30 对应整组。标注顺序为基线 → 终稿。")
    return fig


def seed_boxes(q):
    points = load_seed_points()
    fig = canvas("内部抽样种子稳定性", "第三问固定任务、实际工期与 K = 64。每个红点代表一个内部种子在同一组 90 个实例上的平均按时率。")
    axes = panels(fig,["名义工期 · α = 1","长工期 · α = 1.5"])
    fig.text(.946,.917,"基线内部种子复验：缺测",ha="right",va="center",color=BLUE,fontsize=11,weight="bold")
    for ax,alpha in zip(axes,[1,1.5]):
        official = q.query("alpha==@alpha and beta==.5")[VERSIONS].mean()
        np.testing.assert_allclose(official["终稿"],points.query("alpha==@alpha and internal_seed==20261122").rate_pct.item())
        ax.axhline(official["基线"],ls="--",color=BLUE,lw=1.55,zorder=1)
        for x,batch in enumerate(["original","replication"],1):
            values = points.query("alpha==@alpha and batch==@batch").sort_values("internal_seed").rate_pct.to_numpy()
            assert len(values)==20
            ax.boxplot(values,positions=[x-.12],widths=.25,patch_artist=True,showfliers=False,
                       boxprops=dict(facecolor=BLUSH,edgecolor=RED,linewidth=1.35),
                       whiskerprops=dict(color=RED),capprops=dict(color=RED),medianprops=dict(color=RED,lw=2.2))
            # Offsets only separate dots with tied or nearby values.
            dx = .08+((np.arange(len(values))*7)%20)/19*.16
            ax.scatter(x+dx,values,s=18,facecolors=RED,edgecolors="white",linewidths=.45,zorder=4)
        bottom = 87.8 if alpha==1 else 68.8
        assert points.query("alpha==@alpha").rate_pct.between(bottom,bottom+5).all()
        ax.set(xticks=[1,2],xticklabels=["终稿首批 · 20 个","终稿复验批 · 20 个"],
               xlim=(.51,2.50),ylim=(bottom,bottom+5),ylabel="种子平均加权按时率（%）")
        ax.text(.5,-.20,f"蓝虚线   基线正式值 {official['基线']:.2f}%",transform=ax.transAxes,
                ha="center",va="top",fontsize=9,color=BLUE)
    note(fig,"箱体为中间 50%，箱内横线为中位数，须线延伸至 1.5 倍四分位距内的最远点。蓝虚线给出上一正式版的单次配置参照。")
    return fig


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    q = q3_results()
    plates = [("01_参数敏感性_三联折线图",sensitivity(q)),("02_跨环境稳定性_地点哑铃图",sites()),
              ("03_配对不确定性_分组森林图",forest(q)),("04_尾部风险_累积尾均值图",tail(q)),
              ("05_内部种子稳定性_箱线散点图",seed_boxes(q))]
    with fitz.open() as combined, ZipFile(OUT/"五张高清PNG.zip","w",compression=ZIP_DEFLATED) as archive:
        for name,fig in plates:
            pdf,png = OUT/(name+".pdf"), OUT/(name+".png")
            fig.savefig(pdf,metadata={"Title":name,"Subject":"v1.0.0-final"})
            fig.savefig(png,dpi=450)
            plt.close(fig)
            with fitz.open(pdf) as doc:
                assert len(doc)==1
                combined.insert_pdf(doc)
            archive.write(png,arcname=png.name)
        assert len(combined)==5
        combined.save(OUT/"终稿与基线_五维审核图册.pdf",garbage=4,deflate=True)


if __name__=="__main__":
    main()
