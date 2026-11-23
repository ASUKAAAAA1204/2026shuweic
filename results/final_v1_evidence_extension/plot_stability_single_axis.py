"""One coordinate system per stability question, using archived observations."""
from zipfile import ZipFile, ZIP_DEFLATED

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_stability_review import (
    BLUSH, BLUE, CASES, INK, RED, RESULTS, ROOT, RULE, STYLE, SUB, VERSIONS,
    canvas, forest, key, load_seed_points, note, q3_results, style,
)

OUT = ROOT / "figures/final_v1_single_axis"


def sensitivity(q):
    fig = canvas("参数敏感性", "三问各自以基线在最难档的损失为 100。数值越低，表示该档损失越小。每段折线只连接同一问的三个档位。")
    key(fig,"对照基线")
    ax = fig.add_axes([.11,.24,.83,.45])
    style(ax)
    tables = []
    for name,x,order in [("q1_horizons.csv","horizon",[.2,.6,1]),("q2_cost.csv","budget",[2,1,0])]:
        data = pd.read_csv(RESULTS/name).query("split=='test'").pivot(index=x,columns="model",values="nade")
        tables.append(data.rename(columns={"original":"基线","B":"终稿"}).loc[order])
    rates = q.query("beta==.5").groupby("alpha")[VERSIONS].mean()
    tables.append(100-rates.loc[[.75,1,1.5]])
    for pos,tab,title in zip([0,4,8],tables,["第一问 · 预测时长","第二问 · 观测预算","第三问 · 工期倍率"]):
        loss = 100*tab/tab["基线"].iloc[-1]
        for model in VERSIONS:
            ax.plot(np.arange(3)+pos,loss[model],**STYLE[model],ms=6,lw=2)
        ax.text(pos+1,1.08,title,ha="center",transform=ax.get_xaxis_transform(),color=INK,fontsize=11)
    ax.set(xticks=[0,1,2,4,5,6,8,9,10],xticklabels=["0.2 秒","0.6 秒","1 秒","2 次","1 次","0 次","0.75","1.0","1.5"],
           ylabel="相对损失指数",ylim=(0,110),xlim=(-.35,10.35))
    note(fig,"前两问取 390 个既有测试窗的 NADE，基线为固定阻尼。第三问取 100% 减加权按时率，基线为上一正式版，β = 0.5 秒。")
    return fig


def sites():
    keys = ["split","environment_id","episode_uuid","side","window_id"]
    data = pd.read_csv(RESULTS/"q2_windows.csv.gz").query("split in ['validation','test']")
    assert data.requests.eq(2).all() and np.allclose(data.waiting_s,.2)
    data = data.pivot(index=keys,columns="model",values="nade")[["original","B"]]
    assert data.shape==(602,2) and data.notna().all().all()
    sizes = data.groupby(level=keys[:2]).size()
    for group in [keys[:-1],keys[:-2],keys[:2]]:
        data = data.groupby(level=group).mean()
    fig = canvas("跨环境稳定性", "第二问在八个已观测地点比较同窗误差。每行一个地点，两个点越接近，两版的误差越接近。")
    key(fig,"固定阻尼",False)
    ax = fig.add_axes([.263,.206,.59,.53])
    style(ax,"x")
    labels,positions = [],[]
    names = {"restaurant":"餐厅","station":"快递站","supermarket":"超市"}
    for split,offset,group in [("validation",0,"开发"),("test",4.8,"测试")]:
        for i,(site,r) in enumerate(data.loc[split].iterrows()):
            y = offset+i
            name = site.split("_")
            labels.append(f"{group}   {names[name[-2]]} {name[-1]}   ·   {sizes.loc[split,site]} 窗")
            positions.append(y)
            ax.plot([r.original,r.B],[y,y],lw=2,color="#9BAEBF")
            ax.plot(r.original,y,"o",ms=8,mec=BLUE,mfc="white",mew=1.4)
            ax.plot(r.B,y,"D",ms=4.7,color=RED)
            ax.text(1.04,y,f"{100*(r.original-r.B)/r.original:+.1f}%",transform=ax.get_yaxis_transform(),va="center",color=INK)
    ax.text(1.04,1.025,"误差降幅",transform=ax.transAxes,fontsize=9,color=SUB)
    ax.axhline(3.9,color=RULE,lw=1)
    ax.set(yticks=positions,yticklabels=labels,ylim=(8.35,-.6),xlabel="归一化位移误差 NADE（越低越好）",xlim=(.053,.126))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y",length=0)
    note(fig,"两版在每个窗口均请求两次观测。地点内依次对窗口、手侧和记录取平均。此图覆盖 212 个开发窗与 390 个既有测试窗。")
    return fig


def tail(q):
    fig = canvas("尾部风险", "第三问名义工况下，分别取两版最差的 k 次计算平均按时率。颜色区分版本，线型区分案例。")
    from matplotlib.lines import Line2D
    handles = [Line2D([],[],color=c,marker=m,ls="none",mfc=fc,label=s)
               for c,m,fc,s in [(BLUE,"o","white","上一正式版"),(RED,"D",RED,"终稿")]]
    fig.legend(handles=handles,loc="center right",bbox_to_anchor=(.947,.916),ncol=2)
    ax = fig.add_axes([.115,.21,.72,.52])
    style(ax)
    k = np.arange(1,31)
    ax.axvline(3,color=SUB,ls=":",lw=.9)
    for case,ls in zip(CASES,["-","--","-."]):
        part = q.query("case_id==@case and alpha==1 and beta==.5")
        ends = []
        for model in VERSIONS:
            means = np.cumsum(np.sort(part[model]))/k
            np.testing.assert_allclose(means[2],part[model].nsmallest(3).mean())
            paint = {**STYLE[model],"ls":ls}
            ax.plot(k,means,**paint,markevery=[0,2,9,19,29],ms=4.8,lw=1.85)
            ends.append(means[-1])
        ax.plot([30.7,31.7],[np.mean(ends)]*2,ls=ls,lw=1.35,color=SUB)
        ax.text(32,np.mean(ends),case,va="center",color=INK)
    ax.set(xlim=(.4,34),ylim=(59,101),xticks=[1,3,5,10,15,20,25,30],
           ylabel="平均加权按时率（%）",xlabel="纳入最差的 k 次")
    note(fig,"每个案例使用相同的 30 个官方工期种子。竖虚线 k = 3 标出最差一成，k = 30 是该案例全部实例的平均水平。")
    return fig


def seeds(q):
    points = load_seed_points()
    fig = canvas("内部抽样种子稳定性", "红点为终稿单个内部种子在 90 个固定实例上的平均按时率，纵轴显示其与同工况基线正式按时率的差。")
    fig.text(.946,.914,"基线内部种子复验缺测",ha="right",color=BLUE,weight="bold")
    ax = fig.add_axes([.12,.23,.815,.505])
    style(ax)
    ax.axhline(0,color=BLUE,ls="--",lw=1.4)
    refs = []
    for alpha,start in [(1,1),(1.5,3.35)]:
        official = q.query("alpha==@alpha and beta==.5")[VERSIONS].mean()
        refs.append(official["基线"])
        np.testing.assert_allclose(points.query("alpha==@alpha and internal_seed==20261122").rate_pct.item(),official["终稿"])
        for i,batch in enumerate(["original","replication"]):
            x = start+i
            vals = points.query("alpha==@alpha and batch==@batch").sort_values("internal_seed").rate_pct.to_numpy()-official["基线"]
            assert len(vals)==20
            ax.boxplot(vals,positions=[x-.12],widths=.25,patch_artist=True,showfliers=False,
                       boxprops=dict(facecolor=BLUSH,edgecolor=RED),medianprops=dict(color=RED,lw=2),
                       whiskerprops=dict(color=RED),capprops=dict(color=RED))
            ax.scatter(x+.07+(np.arange(20)*7%20)*.009,vals,s=24,c=RED,edgecolors="white",linewidths=.5,zorder=3)
    ax.set(xticks=[1,2,3.35,4.35],xticklabels=["名义 · 首批","名义 · 复验批","长工期 · 首批","长工期 · 复验批"],
           ylabel="终稿与同工况基线正式值之差（百分点）",xlim=(.5,4.95))
    note(fig,f"两批各 20 个内部种子，K = 64。蓝色零线对应名义基线 {refs[0]:.2f}%、长工期基线 {refs[1]:.2f}%。每个箱体给出终稿中间 50% 的范围。")
    return fig


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    q = q3_results()
    plots = [("01_参数敏感性",sensitivity(q)),("02_跨环境稳定性",sites()),("03_配对不确定性",forest(q)),
             ("04_尾部风险",tail(q)),("05_内部种子稳定性",seeds(q))]
    with fitz.open() as doc, ZipFile(OUT/"五张高清PNG.zip","w",ZIP_DEFLATED) as archive:
        for name,fig in plots:
            assert len(fig.axes)==1
            pdf,png = OUT/(name+".pdf"),OUT/(name+".png")
            fig.savefig(pdf,metadata={"Title":name})
            fig.savefig(png,dpi=450)
            plt.close(fig)
            with fitz.open(pdf) as single:
                doc.insert_pdf(single)
            archive.write(png,arcname=png.name)
        doc.save(OUT/"五维单图审核稿.pdf",garbage=4,deflate=True)


if __name__=="__main__":
    main()
