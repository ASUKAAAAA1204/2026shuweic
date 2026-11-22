"""Paper-width review figure for the three frozen-model sensitivity experiments."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent
OUT = DATA.parents[1] / "figures/final_v1_sensitivity"
RED, PINK = "#CE4844", "#F7A9A8"
BLUE, NAVY = "#2A73BA", "#1F4E89"
LIGHT, GRID = "#E8EEF8", "#B6CAED"
INK, SUB = "#202936", "#475568"
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"],
                     "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": SUB, "ytick.color": SUB, "axes.edgecolor": SUB,
                     "axes.linewidth": .6, "pdf.fonttype": 42, "axes.unicode_minus": False,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.facecolor": "white"})


def header(fig, x, y, letter, title, sub):
    fig.text(x, y, letter, fontsize=15, weight="bold", color=NAVY, va="baseline")
    fig.text(x+.033, y+.001, title, fontsize=12.5, weight="bold", va="baseline")
    fig.text(x+.033, y-.032, sub, fontsize=8.4, color=SUB)


def style(ax, grid_axis="y"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.grid(axis=grid_axis, color=LIGHT, lw=.7)
    ax.set_axisbelow(True)


def panel_a(fig):
    data = pd.read_csv(DATA/"q1_history.csv").pivot(index="horizon", columns="history", values="nade")
    relative = 100 * (data.div(data["full"], axis=0)-1)
    header(fig, .038, .946, "A", "历史截短与质心预测", "212 个开发窗口 · 完整历史为参照")
    ax = fig.add_axes([.104, .588, .325, .237])
    style(ax)
    ax.axhline(0, color=SUB, lw=.85)
    centers = np.arange(len(relative))
    for col, offset, marker, color, face, label in [
        ("0.75s", -.085, "o", BLUE, "white", "保留 0.75 秒"),
        ("0.30s", .085, "D", RED, PINK, "保留 0.30 秒")]:
        ax.plot(centers+offset, relative[col], ls="none", marker=marker, mew=1.4,
                mec=color, mfc=face, ms=6, label=label)
    ax.legend(loc="lower left", bbox_to_anchor=(-.01, 1.03), ncol=2, frameon=False,
              borderpad=0, handlelength=1.1, handletextpad=.4, columnspacing=1.2, fontsize=8.4)
    last = relative.loc[1.0, "0.30s"]
    ax.annotate(f"+{last:.2f}%", (centers[-1]+.085,last), xytext=(-4,8),
                textcoords="offset points", ha="center", color=RED, fontsize=9, weight="bold")
    ax.set(xticks=centers, xticklabels=[f"{h:.1f}" for h in relative.index],
           xlim=(-.36,2.38), ylim=(-2.6,3.8), yticks=[-2,-1,0,1,2,3],
           xlabel="预测时长（秒）", ylabel="NADE 相对变化（%）")
    ax.set_yticklabels(["−2","−1","0","+1","+2","+3"])
    fig.text(.104,.507,"正值表示误差高于完整历史设置。",fontsize=8.2,color=SUB)


def panel_b(fig):
    data = pd.read_csv(DATA/"q2/summary.csv").set_index("delay").sort_index()
    header(fig,.505,.946,"B","回传延迟与误差球","212 个开发窗口 · 遮挡 1 秒，每窗请求 2 次")
    upper = fig.add_axes([.584,.722,.372,.107])
    lower = fig.add_axes([.584,.573,.372,.096], sharex=upper)
    for ax in (upper, lower):
        style(ax)
        ax.axvspan(.093,.107, color=LIGHT, zorder=0)
        ax.set_xlim(-.015,.29)
    for key, label, marker, color, line, dy in [
        ("norm_diameter", "球直径", "D", RED, "--", 7),
        ("nade", "NADE", "o", BLUE, "-", -7)]:
        series = 100 * data[key]/data.at[.1,key]
        upper.plot(series.index,series,ls=line,color=color,marker=marker,ms=6 if marker=="D" else 4,
                   mfc="white" if marker=="D" else BLUE, lw=1.3, mew=1.15,
                   zorder=3 if marker=="o" else 2)
        upper.annotate(f"{label} +{series.iloc[-1]-100:.2f}%",(.2,series.iloc[-1]),
                       xytext=(4,dy),textcoords="offset points",va="center",fontsize=8,color=color)
    upper.set(ylim=(60,145), yticks=[70,100,130], ylabel="相对指数")
    upper.tick_params(labelbottom=False)
    upper.text(0,1.11,"名义延迟 0.10 秒时，两项指数均为 100",transform=upper.transAxes,fontsize=8.2,color=SUB)
    lower.axhline(90,ls=(0,(3,2)),lw=.85,color=NAVY)
    lower.text(.287,90.2,"标定目标 90%",ha="right",va="bottom",fontsize=7.8,color=NAVY)
    pct = data.coverage*100
    lower.plot(pct.index,pct,ls="none",color=NAVY,marker="o",ms=4.2)
    for delay, value in pct.items():
        above = delay == 0
        lower.annotate(f"{value:.2f}%",(delay,value),xytext=(0,5 if above else -5),
                       textcoords="offset points",ha="center",va="bottom" if above else "top",fontsize=8,color=NAVY)
    lower.set(ylim=(83,91.7),yticks=[85,90],xticks=data.index,xticklabels=["0","0.10","0.20"],
              xlabel="单次回传延迟（秒）",ylabel="覆盖率（%）")


def panel_c(fig):
    data = pd.read_csv(DATA/"q3_mapping.csv").set_index(["alpha","beta"])
    nominal = data.loc[(1,.5),"on_time"]*100
    header(fig,.038,.438,"C","服务时长映射与调度按时率",
           f"3 个官方案例 × 30 个配对种子 · 固定 K = 64 · 名义工况按时率 {nominal:.2f}%")
    ax = fig.add_axes([.307,.112,.405,.226])
    style(ax,"x")
    ax.axvline(0,color=SUB,lw=.9)
    ax.set(xlim=(-21.2,10),ylim=(-.65,3.75),yticks=[],xticks=[-20,-15,-10,-5,0,5,10],
           xlabel="加权按时率相对名义工况的变化（百分点）")
    ax.set_xticklabels(["−20","−15","−10","−5","0","+5","+10"])
    ax.spines["bottom"].set_visible(False)
    def row_y(y):
        return .112+(y+.65)/4.4*.226
    fig.text(.051,.353,"固定条件",fontsize=8.6,weight="bold",color=NAVY)
    fig.text(.181,.353,"调整后的取值",fontsize=8.6,weight="bold",color=NAVY)
    fig.text(.746,.353,"配对差 [95% 区间]（百分点）",fontsize=8.6,weight="bold",color=NAVY)
    for key, y, label in [((.75,.5),3.05,"α = 0.75"),((1.5,.5),2.15,"α = 1.50"),
                          ((1,0),.95,"β = 0.00 秒"),((1,1),.05,"β = 1.00 秒")]:
        row = data.loc[key]
        ax.plot([row.low_pp,row.high_pp],[y,y],color=NAVY,lw=1.8,solid_capstyle="round")
        ax.plot([row.low_pp,row.high_pp],[y,y],"|",ms=5.5,color=NAVY)
        ax.plot(row.delta_pp,y,"o",color=NAVY,ms=5.5)
        fig.text(.181,row_y(y),label,va="center",fontsize=9)
        interval = f"{row.delta_pp:+.2f}  [{row.low_pp:+.2f}, {row.high_pp:+.2f}]".replace("-","−")
        fig.text(.746,row_y(y),interval,va="center",color=NAVY,fontsize=8.8,
                 weight="bold" if key==(1.5,.5) else "normal")
        if key == (1.5,.5):
            fig.text(.746,row_y(y)-.0205,f"按时率 {nominal:.2f}% → {row.on_time*100:.2f}%",
                     va="center",color=SUB,fontsize=7.7)
    fig.text(.051,row_y(2.60),"附加时长\nβ = 0.50 秒",va="center",fontsize=9,color=SUB)
    fig.text(.051,row_y(.5),"工期系数\nα = 1.00",va="center",fontsize=9,color=SUB)
    fig.add_artist(Line2D([.04,.963],[row_y(1.55)]*2,color=GRID,lw=.5,transform=fig.transFigure))
    fig.text(.05,.032,"点为三案例等权平均的配对差，横线为按种子成组重抽 10 000 次得到的 95% 区间。",fontsize=8.1,color=SUB)


if __name__ == "__main__":
    fig = plt.figure(figsize=(9.5,6.85), facecolor="white")
    panel_a(fig)
    panel_b(fig)
    panel_c(fig)
    OUT.mkdir(exist_ok=True,parents=True)
    for ext in ("pdf","png"):
        fig.savefig(OUT/f"三问终稿模型敏感性分析_审核稿.{ext}",dpi=300)
    plt.close(fig)
