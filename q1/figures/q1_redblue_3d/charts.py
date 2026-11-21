"""Figures on the common development windows and the environment-held-out training results."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from style import read,NAVY,BLUE,RED,SKY,PINK,PALEBLUE

MODELS=["CV","Hold","B","C"]
NAMES=["恒速外推","末值保持","固定阻尼","阻尼＋残差 MLP"]
COLORS=[RED,SKY,NAVY,BLUE]


def waterfall(values,labels,colors,x,xlabels,zlabel,markers=None):
    fig=plt.figure(figsize=(6.1,4.1))
    ax=fig.add_subplot(projection="3d",proj_type="ortho")
    ax.view_init(elev=21,azim=-59)
    for j,(v,color) in enumerate(zip(values,colors)):
        y=np.full(len(x),j)
        verts=[(x[0],j,0),*zip(x,y,v),(x[-1],j,0)]
        ax.add_collection3d(Poly3DCollection([verts],facecolor=color,edgecolor="none",alpha=.14))
        ax.plot(x,y,v,marker=(markers or ["s","^","o","D"])[j],ms=5,color=color,lw=2.1,label=labels[j])
    ax.set(xticks=x,xticklabels=xlabels,yticks=np.arange(len(labels)),yticklabels=[],
           xlim=(min(x),max(x)),ylim=(-.25,len(labels)-.75),zlim=(0,np.max(values)*1.12))
    ax.text2D(1.14,.52,zlabel,transform=ax.transAxes,rotation=90,fontsize=9,va="center",clip_on=False)
    ax.tick_params(labelsize=8,pad=0)
    ax.set_box_aspect((1.25,1.12,.9),zoom=.96)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.set_facecolor("#F4F7FB")
    fig.legend(loc="upper center",bbox_to_anchor=(.5,1.02),ncol=2,fontsize=9)
    fig.subplots_adjust(left=.02,right=.93,bottom=.03,top=.87)
    return fig


def position_waterfall():
    z=read("position_horizons.csv").query("split=='validation'").pivot(index="model",columns="horizon",values="nade")
    return waterfall(z.loc[MODELS].values,NAMES,COLORS,[.2,.6,1.],["0.2 秒","0.6 秒","1.0 秒"],"归一化 ADE")


def recovery_waterfall():
    z=read("recovery_natural_summary.csv").query("split=='validation' and scope=='common'")
    z=z.pivot(index="model",columns="phase",values="norm_error").loc[
        ["hold_two","recent6_hold","zero_velocity","decay"],["pre_first","pre_second"]]
    return waterfall(z.values,["等待两帧","最近六帧回退","首帧＋零速","首帧＋衰减旧速"],
                     [RED,BLUE,SKY,NAVY],[0,1],["首帧前","第二帧前"],"归一化位置误差")


def environment_boxes():
    env=read("position_environments.csv").query("split=='train' and model in ['B','CV']")
    e=env.groupby(["env","horizon","model"]).nade.mean().unstack()
    gains=(100*(1-e.B/e.CV)).unstack()
    assert gains.shape==(16,3)
    fig,ax=plt.subplots(figsize=(5.7,3.15),layout="constrained")
    ax.boxplot(gains.to_numpy(),positions=[0,1,2],widths=.4,showfliers=False,patch_artist=True,
               boxprops=dict(facecolor=PALEBLUE,edgecolor=BLUE),medianprops=dict(color=NAVY,lw=2))
    offsets=np.linspace(-.12,.12,len(gains))
    for j,(_,col) in enumerate(gains.items()):
        ax.scatter(j+offsets,col,color=np.where(col>=0,BLUE,RED),s=20,alpha=.85,edgecolors="white",linewidths=.45,zorder=3)
    ax.axhline(0,color=RED,ls="--",lw=.85)
    ax.set(xticks=[0,1,2],xticklabels=["0.2 秒","0.6 秒","1.0 秒"],ylabel="固定阻尼相对恒速的 ADE 降幅（%）")
    ax.grid(axis="y",alpha=.16)
    return fig


def radar():
    base=read("position_overall.csv").query("split=='validation'").set_index("model")
    windows=read("curve/window_metrics.csv.gz").query("split=='validation' and horizon==1")
    windows=windows[windows.seed==windows.loc[windows.model=="C","seed"].min()]
    assert (windows.groupby("model").window_id.nunique()==212).all()
    timing=read("curve/timing.csv").query("split=='validation'")
    env=read("position_environments.csv").query("split=='validation'")
    worst=env.groupby(["model","env","source"]).nade.mean().groupby(level=["model","env"]).mean().groupby(level="model").max()
    raw=pd.DataFrame({"平均 ADE":base.nade,"末帧 FDE":base.nfde,
        "1 秒 P90":windows.groupby("model").nade.quantile(.9),"最差环境":worst,
        "耗时 P50":timing.groupby("model").us.median()}).loc[MODELS]
    scores=100*raw.min()/raw
    angle=np.arange(5)*2*np.pi/5
    angle=np.r_[angle,angle[0]]
    fig,ax=plt.subplots(figsize=(5.1,4.2),subplot_kw={"projection":"polar"})
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    for i,(_,row) in enumerate(scores.iterrows()):
        v=np.r_[row.values,row.iloc[0]]
        ax.plot(angle,v,marker=["s","^","o","D"][i],mfc="none" if i==3 else COLORS[i],ms=4,
                ls=["--",":","-","-."][i],color=COLORS[i],lw=1.7,label=NAMES[i])
    ax.set(xticks=angle[:-1],xticklabels=raw.columns,ylim=(0,105),yticks=[25,50,75,100])
    ax.tick_params(axis="y",labelsize=7)
    ax.set_rlabel_position(40)
    ax.grid(alpha=.3)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2)
    fig.subplots_adjust(bottom=.23,top=.91)
    return fig,raw


def risks():
    q=read("quality_predictions.csv.gz")
    r=read("quality_rates.csv").query("held=='development'").set_index("quality_state")
    v=q.query("split=='validation'").groupby("state").label.agg(["sum","count","mean"])
    keys=["clean_history","recent_recovery","flagged"]
    fig,ax=plt.subplots(figsize=(6.7,3.1),layout="constrained")
    for i,k in enumerate(keys):
        train=r.loc[k];dev=v.loc[k]
        p1,p2=100*train.p,100*dev["mean"]
        ax.plot([p1,p2],[i-.13,i+.13],color="#A5B3C3",lw=1)
        for x,y,col,mark,count,n,align in [(p1,i-.13,NAVY,"o",train.events,train.samples,"bottom"),
                                           (p2,i+.13,RED,"s",dev["sum"],dev["count"],"top")]:
            ax.scatter(x,y,c=col,marker=mark,s=38,zorder=3)
            right=x>67
            ax.annotate(f"{x:.2f}%   {int(count):,}/{int(n):,}",(x,y),
                        xytext=(-8 if right else 8,0),textcoords="offset points",
                        va="center",ha="right" if right else "left",fontsize=9,color=col)
    for c,m,n in [(NAVY,"o","训练 · 平滑风险"),(RED,"s","开发 · 实际阳性率")]:
        ax.scatter([],[],c=c,marker=m,label=n)
    ax.legend(loc="upper left",bbox_to_anchor=(0,1.23),ncol=2)
    ax.set(yticks=range(3),yticklabels=["持续未标坏","刚恢复","当前标坏"],ylim=(2.45,-.45),
           xlim=(-2,101),xlabel="未来 0.2 秒至少一次源标坏（%）")
    ax.grid(axis="x",alpha=.17)
    assert q.query("split=='validation' and state=='recent_recovery' and label==1").environment_id.nunique()==1
    return fig


def brier_bars():
    scores=read("quality_scores.csv").query("subset=='all'").set_index("split")
    fig,axes=plt.subplots(1,2,figsize=(6.7,3.25),layout="constrained")
    for ax,split,base,name in zip(axes,["train","validation"],[7.85,29.55],["训练环境留一","开发集"]):
        vals=1000*scores.loc[split,["State","State3","Motion"]].astype(float)
        bars=ax.bar(range(3),vals-base,bottom=base,color=[PINK,NAVY,SKY],edgecolor=[RED,NAVY,BLUE],width=.57)
        ax.bar_label(bars,labels=[f"{x:.3f}" for x in vals],padding=3,fontsize=9)
        ax.set(ylim=(base,base+.87),xticks=range(3),xticklabels=["二状态","三状态","运动提升树"],title=name)
        ax.set_ylabel("Brier 分数 × 1,000")
        ax.grid(axis="y",alpha=.18)
        ax.plot([-.016,.016],[-.018,.018],transform=ax.transAxes,color="#384A61",clip_on=False,lw=1.1)
    return fig
