"""One full-width manuscript figure from the archived five-dimension experiments."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "figures/five_dimensions_paper"
BLUE, RED, NAVY = "#2A73BA", "#CE4844", "#1F4E89"
INK, GRID = "#242C35", "#DFE4E9"
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"], "font.size":7.2,
    "axes.labelsize":7.4, "xtick.labelsize":6.8, "ytick.labelsize":6.8, "text.color":INK,
    "axes.labelcolor":INK, "axes.edgecolor":"#727D85", "axes.linewidth":.55,
    "xtick.color":INK,"ytick.color":INK,"xtick.major.size":2.5,"ytick.major.size":2.5,
    "xtick.major.width":.55,"ytick.major.width":.55,"axes.spines.top":False,
    "axes.spines.right":False,"axes.unicode_minus":True,"pdf.fonttype":42,"legend.frameon":False})


def panel(fig, x, y, letter, title):
    fig.text(x,y,letter,va="center",fontfamily="DejaVu Sans",weight="bold",fontsize=10.5)
    fig.text(x+.030,y,title,va="center",fontsize=8.8,weight="bold")


def axis(fig,x,y,w,h,grid="y"):
    ax=fig.add_axes((x,y,w,h))
    ax.set_axisbelow(True)
    ax.grid(axis=grid,color=GRID,lw=.45)
    ax.tick_params(pad=2.2)
    return ax


def key(fig,x,y,groups):
    handles=[Line2D([],[],color=c,marker=m,ls="none",ms=4,label=label) for label,c,m in groups]
    fig.legend(handles=handles,loc="center right",bbox_to_anchor=(x,y),ncol=len(groups),
               borderaxespad=0,handlelength=.9,handletextpad=.28,columnspacing=1.0,fontsize=7)


def sensitivity(fig):
    panel(fig,.024,.973,"A","输入扰动敏感性")
    key(fig,.968,.973,[("开发",BLUE,"o"),("既有测试",RED,"D")])
    xs=[.087,.407,.739]
    for x,txt in zip(xs,["第一问  ·  历史截短","第二问  ·  延迟翻倍","第三问  ·  加权按时率（%）"]):
        fig.text(x,.916,txt,fontsize=7.8,weight="semibold")
    ax=axis(fig,xs[0],.747,.214,.14)
    history=pd.read_csv(HERE/"history_interaction.csv").query("environment=='all'").set_index("split")
    q2=pd.read_csv(HERE/"trajectory/summary.csv").query("selection=='all' and scope=='split' and metric=='nade'").set_index("split")
    delay=axis(fig,xs[1],.747,.216,.14)
    for split,color,marker in [("validation",BLUE,"o"),("test",RED,"D")]:
        inc=history.loc[split,["diff_02","diff_06","diff_10"]].to_numpy(float)
        ax.plot([.2,.6,1],1000*inc,color=color,marker=marker,ls="-" if split=="validation" else "--",ms=3.3,lw=1)
        offset=-.095 if split=="validation" else .095
        delay.bar(np.arange(2)+offset,q2.loc[split,["full_delay","short_delay"]].to_numpy(float)*1000,
                  width=.17,color=color)
    ax.set(xticks=[.2,.6,1],xlabel="预测时长（秒）",ylabel="ΔNADE（×10⁻³）",ylim=(0,6.1),yticks=[0,2,4,6])
    delay.set(xticks=[0,1],xticklabels=["完整","0.30 秒"],xlabel="可见历史",ylabel="ΔNADE（×10⁻³）",
              ylim=(0,30),yticks=[0,10,20,30],xlim=(-.5,1.5))
    grid=pd.read_csv(HERE/"grid_means.csv").pivot(index="alpha",columns="beta",values="overall")*100
    a=fig.add_axes([xs[2],.747,.228,.14])
    cmap=LinearSegmentedColormap.from_list("rate",[RED,"#FDE6E6","#E8EEF8","#87AEE6",NAVY])
    edges=np.arange(4)-.5
    a.pcolormesh(edges,edges,grid.to_numpy(),vmin=68,vmax=99,cmap=cmap,shading="flat")
    a.set(xlim=(-.5,2.5),ylim=(2.5,-.5))
    for (i,j),v in np.ndenumerate(grid.to_numpy()):
        a.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6.8,
               color="white" if v<70 or v>94 else INK)
    a.set(xticks=range(3),xticklabels=["0","0.5","1"],yticks=range(3),yticklabels=["0.75","1","1.5"],
          xlabel="固定增量 β（秒）",ylabel="工期倍率 α")
    a.tick_params(length=0,pad=2)
    for s in a.spines.values():s.set_visible(False)


def comparisons(fig):
    panel(fig,.024,.641,"B","跨环境稳定性")
    panel(fig,.348,.641,"C","配对不确定性")
    panel(fig,.685,.641,"D","S2 名义工况尾部")
    sites=pd.read_csv(HERE/"spaced_environments.csv").sort_values(["split","environment_id"],ascending=[False,True])
    assert len(sites)==8 and sites.n.sum()==308
    a=axis(fig,.089,.396,.212,.192,"x")
    for y,r in enumerate(sites.itertuples()):
        dev=r.split=="validation"
        a.hlines(y,0,r.change_pct,color="#CED6DD",lw=.8)
        a.plot(r.change_pct,y,"o" if dev else "D",color=BLUE if dev else RED,ms=3.4)
    a.set(yticks=range(8),yticklabels=[s[-4:] for s in sites.environment_id],ylim=(7.55,-.55),
          xlim=(0,40),xticks=[0,10,20,30,40],xlabel="NADE 增幅（%）",ylabel="地点编号")
    paired=pd.read_csv(HERE/"paired_grid.csv").query("contrast=='long-minus-nominal'").sort_values("n")
    old=pd.read_csv(HERE.parent/"final_v1_sensitivity/q3_mapping.csv").query("alpha==1.5 and beta==0.5").iloc[0]
    rows=[(old.delta_pp,old.low_pp,old.high_pp),*paired[["delta_pp","lower_pp","upper_pp"]].itertuples(index=False,name=None)]
    a=axis(fig,.434,.396,.192,.192,"x")
    for i,(v,lo,hi) in enumerate(rows):
        color=RED if i==2 else NAVY
        a.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="D" if i==2 else "o",color=color,lw=1.25,ms=3.7,capsize=2)
        a.annotate(f"{v:.2f}".replace("-","−"),(v,i),xytext=(0,5),textcoords="offset points",ha="center",fontsize=6.7)
    a.set(yticks=[0,1,2],yticklabels=["官方 30","扩展前 30","扩展 300"],ylim=(2.48,-.60),
          xticks=[-22,-20,-18,-16],xlim=(-22.4,-15),xlabel="按时率差（百分点）")
    prev=pd.read_csv(HERE/"tail_replication.csv").query("case_id=='S2' and alpha==1").iloc[0]
    batches=pd.read_csv(HERE.parent/"final_v1_s2_tail/batches.csv").set_index("batch")
    a=axis(fig,.754,.396,.205,.192)
    for y,color,marker in [([prev.first150*100,prev.last150*100],RED,"D"),
                            (batches.loc[["A","B"],"tail_pct"].to_list(),BLUE,"o")]:
        a.plot([0,1],y,color=color,ms=4,marker=marker,lw=1.2)
        for j,v in enumerate(y):
            dy=-9 if color==BLUE and j==1 else 5
            a.annotate(f"{v:.2f}",(j,v),xytext=(0,dy),textcoords="offset points",ha="center",fontsize=6.8,color=color)
    a.set(xlim=(-.19,1.19),ylim=(62,78),xticks=[0,1],xticklabels=["第一组","第二组"],
          yticks=[64,68,72,76],ylabel="尾部平均按时率（%）")
    a.legend(["首轮分半","独立复验"],loc="lower right",fontsize=6.1,handlelength=1.1,borderpad=.12,labelspacing=.24)


def internal_seeds(fig):
    panel(fig,.024,.294,"E","内部种子稳定性")
    fig.text(.260,.294,"每对点从左至右为首批、复验批",fontsize=6.7,va="center",color="#52606B")
    key(fig,.969,.294,[("baseline",BLUE,"o"),("终稿 B_K64",RED,"D")])
    data=pd.read_csv(HERE.parent/"q3_baseline_seed_control/summary.csv").query("beta==0.5 and alpha in [1,1.5]")
    assert data.n_instances.eq(90).all() and len(data)==8
    data=data.set_index(["method","alpha","batch"])
    cols=[("mean_rate_pct","平均加权按时率（%）",(63,96),[65,75,85,95]),
          ("mean_within_instance_sd_pp","单实例标准差均值（百分点）",(-.32,6.5),[0,2,4,6]),
          ("first_action_pair_agreement_pct","首步动作一致率（%）",(34,106),[40,60,80,100])]
    for x,(metric,title,limits,ticks) in zip([.087,.408,.739],cols):
        fig.text(x,.239,title,fontsize=7.5)
        a=axis(fig,x,.064,.218,.147)
        for alpha,left in [(1.,0),(1.5,2.6)]:
            for model,col,marker in [("baseline",BLUE,"o"),("B_K64",RED,"D")]:
                vals=[data.at[(model,alpha,b),metric] for b in ["original","replication"]]
                a.plot([left,left+1],vals,marker=marker,ms=3.8,color=col,lw=1.25)
        a.axvline(1.8,color=GRID,lw=.6,ls=":")
        a.set(xlim=(-.45,4.05),ylim=limits,yticks=ticks,xticks=[.5,3.1],xticklabels=["名义","长工期"])


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(180/25.4,155/25.4),facecolor="white")
    sensitivity(fig)
    comparisons(fig)
    internal_seeds(fig)
    fig.savefig(OUT/"五维稳定性_正文组合图.pdf",metadata={"Title":"三问模型的敏感性与稳定性"})
    fig.savefig(OUT/"五维稳定性_正文组合图.png",dpi=600)
    plt.close(fig)
