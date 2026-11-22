"""Final-model sensitivity: absolute responses and paired heatmaps on one sheet."""
from decimal import Decimal, ROUND_HALF_UP

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from plot_triptych import BLUE, DATA, GRID, INK, LIGHT, NAVY, OUT, PINK, RED, SUB, style

SIGNED = LinearSegmentedColormap.from_list("performance", [
    NAVY, "#87AEE6", "#FFFFFF", PINK, RED], N=257)
plt.rcParams.update({"legend.frameon": False, "axes.labelsize": 9, "font.size": 9})


def signed(value):
    # Remove binary arithmetic noise before conventional half-up rounding.
    num = Decimal(f"{abs(value):.9f}").quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
    return ("−" if value < 0 else "+") + str(num) if num else "0.00"


def section(fig, y, letter, title, context):
    fig.text(.045, y, letter, color=NAVY, weight="bold", fontsize=17, va="center")
    fig.text(.082, y, title, weight="bold", fontsize=13, va="center")
    fig.text(.961, y, context, color=SUB, fontsize=8.5, ha="right", va="center")
    fig.add_artist(plt.Line2D([.045,.961],[y-.022]*2,transform=fig.transFigure,lw=.75,color=GRID))


def heatmap(fig, box, values, rows, columns, title, bound, unit, *, reversed_sign=False):
    ax = fig.add_axes(box)
    nr, nc = values.shape
    cmap = SIGNED.reversed() if reversed_sign else SIGNED
    norm = Normalize(-bound, bound)
    for i in range(nr):
        for j in range(nc):
            v = values[i,j]
            color = cmap(norm(v))
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=color,edgecolor="#CAD4E0",lw=.55))
            # WCAG relative luminance chooses the more legible of white and dark ink.
            rgb = np.array(color[:3])
            lum = np.dot(np.where(rgb <= .04045,rgb/12.92,((rgb+.055)/1.055)**2.4),[.2126,.7152,.0722])
            ax.text(j,i,signed(v),ha="center",va="center",fontsize=10.3,
                    color="white" if lum<.17 else "#15202F",weight="semibold")
    ax.set(xlim=(-.5,nc-.5),ylim=(nr-.5,-.5),xticks=range(nc),yticks=range(nr),
           xticklabels=columns,yticklabels=rows)
    ax.xaxis.tick_top()
    ax.tick_params(length=0,pad=5,labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title,loc="left",fontsize=9.4,weight="bold",pad=9)
    # Native PDF vector colour swatches; numeric domains are specific to each question.
    x = box[0]+.049
    y = box[1]-.029
    width = box[2]-.097
    for i in range(48):
        fig.add_artist(Rectangle((x+width*i/48,y),width/48+.00008,.005,
                       transform=fig.transFigure,color=cmap(i/47),lw=0))
    for dx, label in [(0,f"−{bound}{unit}"),(.5,"0"),(1,f"+{bound}{unit}")]:
        fig.text(x+width*dx,y-.004,label,ha="center",va="top",fontsize=7.3,color=SUB)


def main():
    q1 = pd.read_csv(DATA/"q1_history.csv").pivot(index="history",columns="horizon",values="nade")
    q2 = pd.read_csv(DATA/"q2/summary.csv").set_index("delay").sort_index()
    archive = pd.read_csv(DATA/"frozen_tables/q2_validation_horizon_delay.csv")
    delay = archive.query("split == 'validation' and method == 'rotation_switch'").pivot(
        index="horizon",columns="delay",values="nade").sort_index()
    q3 = pd.read_csv(DATA/"q3_mapping.csv").set_index(["alpha","beta"])
    configs = [(.75,.5),(1.5,.5),(1.,0.),(1.,1.)]
    cases = ["S1","S2","S3"]
    baseline = q3.loc[(1.,.5)]
    selected = q3.loc[configs]
    hist_change = 100*(q1.loc[["0.75s","0.30s"]]/q1.loc["full"]-1)
    delay_change = 100*(delay.div(delay[.1],axis=0)-1)
    case_change = (selected[cases]-baseline[cases])*100
    np.testing.assert_allclose(delay.loc[1.],q2.nade,atol=1e-12,rtol=0)
    np.testing.assert_allclose(case_change.mean(axis=1),selected.delta_pp,atol=1e-12,rtol=0)
    assert delay_change.loc[.2].eq(0).all() and q2.windows.eq(212).all()

    fig = plt.figure(figsize=(9.3,9.35), facecolor="white")
    section(fig,.967,"A","可见历史与质心预测","212 个开发窗口")
    section(fig,.669,"B","回传延迟与遮挡预测","212 个开发窗口 · 计划请求时刻为 0.3 秒和 0.6 秒")
    section(fig,.344,"C","服务时长与调度按时率","固定 K = 64 · 3 个官方案例 × 30 个配对种子")

    ax = fig.add_axes([.099,.751,.340,.14])
    style(ax)
    for h, label, color, marker, line, size in [
        ("full","完整历史",NAVY,"o","-",8),
        ("0.75s","最近 0.75 秒",BLUE,"s","--",5.5),
        ("0.30s","最近 0.30 秒",RED,"D",":",3.6)]:
        ax.plot(q1.columns,q1.loc[h],label=label,ls=line,lw=1.55,color=color,marker=marker,
                ms=size,mfc="white" if h!="0.30s" else PINK,mew=1.1)
    ax.set(xticks=[.2,.6,1],xticklabels=["0.2","0.6","1.0"],yticks=[0,.05,.1,.15,.2],
           ylim=(0,.208),xlim=(.13,1.07),ylabel="归一化平均位移误差",xlabel="预测时长（秒）")
    ax.legend(ncol=3,loc="lower left",bbox_to_anchor=(-.06,1.025),borderaxespad=0,
              columnspacing=.85,handlelength=1.45,handletextpad=.4,fontsize=8.1)
    heatmap(fig,[.645,.769,.303,.101],hist_change.to_numpy(),["历史 0.75 秒","历史 0.30 秒"],
            ["预测 0.2 秒","0.6 秒","1.0 秒"],"NADE 相对完整历史的变化（%）",3,"%")

    upper = fig.add_axes([.099,.525,.359,.074])
    lower = fig.add_axes([.099,.416,.359,.065],sharex=upper)
    for ax in (upper,lower):
        style(ax)
        ax.axvspan(.093,.107,color=LIGHT,zorder=0)
        ax.set_xlim(-.016,.292)
    for metric,label,color,marker,line,dy in [
        ("norm_diameter","球直径",RED,"D","--",6),("nade","NADE",BLUE,"o","-",-6)]:
        ratio = 100*q2[metric]/q2.at[.1,metric]
        upper.plot(q2.index,ratio,color=color,marker=marker,ls=line,lw=1.4,
                   ms=6 if metric!="nade" else 3.7,mfc="white" if metric!="nade" else color)
        upper.annotate(f"{label} +{ratio.iloc[-1]-100:.2f}%",(.2,ratio.iloc[-1]),xytext=(4,dy),
                       textcoords="offset points",fontsize=7.8,va="center",color=color)
    upper.set(ylim=(60,145),yticks=[70,100,130],ylabel="相对指数")
    upper.tick_params(labelbottom=False)
    upper.text(-.055,1.22,"遮挡 1 秒 · 名义延迟 0.10 秒时指数为 100",transform=upper.transAxes,
               color=INK,fontsize=8.8,weight="semibold")
    lower.axhline(90,color=BLUE,ls="--",lw=.85)
    lower.text(.285,90.18,"目标 90%",ha="right",va="bottom",color=NAVY,fontsize=7.5)
    lower.plot(q2.index,100*q2.coverage,"o",color=NAVY,ms=4)
    for x,val in (100*q2.coverage).items():
        lower.annotate(f"{val:.2f}%",(x,val),xytext=(0,4 if x==0 else -5),
                       textcoords="offset points",ha="center",va="bottom" if x==0 else "top",fontsize=7.8)
    lower.set(ylim=(82.7,92),yticks=[85,90],xticks=[0,.1,.2],xticklabels=["0","0.10","0.20"],
              ylabel="覆盖率（%）",xlabel="单次回传延迟（秒）")
    heatmap(fig,[.645,.457,.303,.125],delay_change.to_numpy(),
            ["遮挡 0.2 秒","遮挡 0.6 秒","遮挡 1.0 秒"],["延迟 0 秒","0.10 秒","0.20 秒"],
            "NADE 相对同窗长名义延迟的变化（%）",35,"%")
    fig.text(.607,.387,"遮挡 0.2 秒时，窗口在首次计划请求前结束。",fontsize=8.1,color=SUB)

    # Match row centres of the intervals and case heatmap exactly in figure coordinates.
    ax = fig.add_axes([.191,.115,.245,.145])
    style(ax,"x")
    labels = ["α = 0.75","α = 1.50","β = 0.00 秒","β = 1.00 秒"]
    ax.axvline(0,color=SUB,lw=.9)
    for i,r in enumerate(selected.itertuples()):
        ax.plot([r.low_pp,r.high_pp],[i,i],color=NAVY,lw=1.8)
        ax.plot([r.low_pp,r.high_pp],[i,i],"|",color=NAVY,ms=5)
        ax.plot(r.delta_pp,i,"o",color=NAVY,ms=5)
        ax.annotate(signed(r.delta_pp),(r.high_pp,i),xytext=(4,0),textcoords="offset points",
                    va="center",fontsize=8.5,color=NAVY)
    ax.set(xlim=(-22,14),xticks=[-20,-10,0,10],ylim=(3.5,-.5),yticks=range(4),yticklabels=labels,
           xlabel="平均按时率变化（百分点）")
    ax.tick_params(axis="y",pad=7)
    fig.text(.066,.279,"三案例平均配对差及 95% 区间",fontsize=9.4,weight="bold")
    heatmap(fig,[.645,.115,.303,.145],case_change.to_numpy(),labels,cases,
            "各案例按时率相对名义工况的变化（百分点）",21,"",reversed_sign=True)
    fig.text(.055,.055,f"α 升至 1.50 时，平均按时率由 {baseline.on_time:.2%} 降至 {selected.iloc[1].on_time:.2%}。",fontsize=8.3,color=INK)
    fig.text(.055,.025,"名义映射为 α = 1.00、β = 0.50 秒，每次仅改变一项。区间按种子配对重抽 10 000 次。",fontsize=8.1,color=SUB)
    fig.text(.635,.025,"热图：红色表示性能变差，蓝色表示改善。",fontsize=8.2,color=INK)
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","png"):
        fig.savefig(OUT/f"三问终稿模型敏感性分析_三行两列审核稿.{ext}",dpi=450)
    plt.close(fig)


if __name__ == "__main__":
    main()
