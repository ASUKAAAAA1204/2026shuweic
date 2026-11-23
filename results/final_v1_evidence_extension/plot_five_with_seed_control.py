"""Five matched review plates, with the paired baseline experiment on plate 5."""
from pathlib import Path
from zipfile import ZipFile

import fitz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from PIL import Image

from plot_five_dimensions import (EXT, ROOT, NAVY, BLUE, INK, RED, SUB, RULE, BLUSH, PALE,
                                  canvas, environments, heading, note, style, tail, uncertainty)

OUT = ROOT / "figures/five_dimensions_seed_control"
HEIGHT = 6.2
FILES = []


def save(fig, name):
    pdf = OUT / (name + ".pdf")
    fig.savefig(pdf, metadata={"Title": name, "Subject": "v1.0.0-final"})
    fig.savefig(pdf.with_suffix(".png"), dpi=450)
    FILES.append(pdf)
    plt.close(fig)


def cells(fig, rect, data, col_labels, row_labels, digits, worse_high):
    ax = fig.add_axes(rect)
    scaled = (data - data.min()) / np.ptp(data)
    risk = scaled if worse_high else 1 - scaled
    for (row, col), val in np.ndenumerate(data):
        # Label all values; the endpoint colours distinguish good and poor performance.
        t = risk[row, col]
        start, end, f = (PALE, NAVY, 1 - 2*t) if t < .5 else (BLUSH, RED, 2*t-1)
        rgb = np.array(to_rgb(start))*(1-f) + np.array(to_rgb(end))*f
        ax.add_patch(plt.Rectangle((col-.5, row-.5), 1, 1, color=rgb, ec="white", lw=2))
        ax.text(col, row, f"{val:.{digits}f}", ha="center", va="center", fontsize=12,
                color="white" if f>.67 else INK, weight="bold")
    ax.set(xlim=(-.5,len(col_labels)-.5),ylim=(len(row_labels)-.5,-.5),
           xticks=range(len(col_labels)),xticklabels=col_labels,yticks=range(len(row_labels)),yticklabels=row_labels)
    ax.tick_params(length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def sensitivity():
    fig = canvas("三问模型对输入扰动的响应",
                 "第一问缩短可见历史，第二问改变历史和回传延迟，第三问调整实际工期。", HEIGHT)
    for x, title, sub in [(.070,"第一问  ·  历史长度","短历史减完整历史的 NADE"),
                          (.398,"第二问  ·  回传延迟","既有测试集的 NADE"),
                          (.707,"第三问  ·  工期映射","三案例平均加权按时率（%）")]:
        heading(fig,x,.754,title)
        fig.text(x,.695,sub,fontsize=9.6,color=SUB)
    diff = pd.read_csv(EXT/"history_interaction.csv").query("environment=='all'").set_index("split")
    ax = fig.add_axes([.088,.32,.227,.29])
    style(ax)
    for split,label,marker,color in [("validation","开发","o",BLUE),("test","既有测试","D",RED)]:
        ax.plot([.2,.6,1],diff.loc[split,["diff_02","diff_06","diff_10"]].to_numpy(float),
                label=label,marker=marker,ms=5,color=color,lw=1.8)
    ax.set(xticks=[.2,.6,1],xlabel="预测时长（秒）",ylim=(0,.006),yticks=[0,.002,.004,.006])
    ax.legend(loc="upper left",fontsize=8)
    test = pd.read_csv(EXT/"trajectory/summary.csv").query(
        "selection=='all' and scope=='split' and split=='test' and metric=='nade'").iloc[0]
    cells(fig,[.425,.32,.202,.29],np.array([[test.full_d01,test.full_d02],[test.short_d01,test.short_d02]]),
          ["0.10","0.20"],["完整","0.30 秒"],5,True).set_xlabel("单次回传延迟（秒）")
    schedule = pd.read_csv(EXT/"grid_means.csv").pivot(index="alpha",columns="beta",values="overall")*100
    cells(fig,[.744,.32,.19,.29],schedule.to_numpy(),["0","0.5","1"],["α=0.75","α=1","α=1.5"],
          2,False).set_xlabel("固定增量 β（秒）")
    for x,msg in [(.070,"0.30 秒可见历史\n开发 212 窗，既有测试 390 窗"),(.398,"一秒遮挡，固定两次请求\n390 个相同的既有测试窗"),
                  (.707,"每格 300 个外部种子 × 3 个案例\n同一个任务在九档共享来源工期")]:
        fig.text(x,.157,msg,fontsize=9.2,color=SUB,linespacing=1.55)
    note(fig,"NADE 越低越好，加权按时率越高越好。热图按各自面板内的实测范围着色，蓝色表示表现较好，红色表示较差。")
    save(fig,"01_敏感性")


def seeds():
    data = pd.read_csv(EXT.parent/"q3_baseline_seed_control/summary.csv").query("alpha in [1,1.5] and beta==0.5")
    idx = data.set_index(["alpha","batch","method"])
    assert len(idx)==8 and idx.index.is_unique and data.n_instances.eq(90).all()
    fig = canvas("内部抽样种子对按时率与首步决策的影响",
                 "第三问 · 每种工况固定 90 个实例，首批和复验批各 20 个种子 · β 固定为 0.5 秒", HEIGHT)
    fig.legend(handles=[Line2D([],[],marker=m,color=c,ls="",label=l,ms=7) for m,c,l in
                        [("o",BLUE,"baseline"),("D",RED,"终稿 B_K64")]],
               loc="upper right",bbox_to_anchor=(.95,.955),ncol=2,fontsize=10,handletextpad=.16)
    labels=["名义 · 首批","名义 · 复验","长工期 · 首批","长工期 · 复验"]
    rows=[(1,"original"),(1,"replication"),(1.5,"original"),(1.5,"replication")]
    panels=[("mean_rate_pct","平均加权按时率","按时率（%）",62,98,[65,75,85,95]),
            ("mean_within_instance_sd_pp","单实例得分波动","标准差均值（百分点）",-1,7.2,[0,2,4,6]),
            ("first_action_pair_agreement_pct","首步动作一致率","一致率（%）",30,112,[40,60,80,100])]
    for j,(metric,title,unit,lo,hi,ticks) in enumerate(panels):
        x=.158+j*.273
        heading(fig,x,.75,title)
        ax=fig.add_axes([x,.244,.228,.43])
        style(ax,"x")
        ax.set(xlim=(lo,hi),ylim=(-.55,4.05),yticks=[3.5,2.5,1,0],yticklabels=labels if j==0 else [],
               xticks=ticks,xlabel=unit)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y",length=0,pad=9)
        ax.axhline(1.75,color=RULE,lw=.8)
        for (alpha,batch),y in zip(rows,[3.5,2.5,1,0]):
            values=[idx.at[(alpha,batch,model),metric] for model in ["baseline","B_K64"]]
            ax.plot(values,[y,y],color="#ABBCCD",lw=1.5)
            for v,c,m in zip(values,[BLUE,RED],["o","D"]):
                ax.scatter(v,y,s=40,marker=m,color=c,zorder=3)
                label=(f"{v:.3f}" if metric.startswith("mean_within") and 0<v<.1 else
                       f"{v:.2f}".rstrip("0").rstrip("."))
                above = m == "D"
                ax.annotate(label,(v,y),xytext=(0,6 if above else -6),textcoords="offset points",
                            ha="center",va="bottom" if above else "top",color=c,fontsize=8.7)
    note(fig,"每个实例在 20 个内部种子间计算样本标准差及首步一致率（190 对），再对实例等权平均。点间连线连接同一工况和批次。")
    save(fig,"05_内部种子稳定性")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sensitivity()
    environments(HEIGHT,save)
    uncertainty(HEIGHT,save)
    tail(HEIGHT,save)
    seeds()
    with fitz.open() as combined, ZipFile(OUT/"五维结果图_高清PNG.zip","w") as archive:
        for pdf in FILES:
            with fitz.open(pdf) as src:
                assert len(src)==1 and src[0].get_text().strip()
                combined.insert_pdf(src)
            png=pdf.with_suffix(".png")
            with Image.open(png) as img:
                assert img.size==(5040,2790)
            archive.write(png,png.name)
        assert len(combined)==5
        combined.save(OUT/"五维结果图_审核稿.pdf",garbage=4,deflate=True)


if __name__=="__main__":
    main()
