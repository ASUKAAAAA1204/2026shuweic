"""Reliability panels: state risks and enlarged Brier-score bars."""
import matplotlib.pyplot as plt
import numpy as np
from style import read,NAVY,BLUE,RED,SKY,PINK

STATES=["clean_history","recent_recovery","flagged"]
STATE_LABELS=["持续未标坏","刚恢复","当前标坏"]
MODEL_KEYS=["State","State3","Motion"]
MODEL_LABELS=["二状态","三状态","运动提升树"]


def risks():
    """Training smoothed risk versus the observed dev rate, in paired bars."""
    rates=read("quality_rates.csv").query("held=='development'").set_index("quality_state")
    dev=read("quality_predictions.csv.gz").query("split=='validation'").groupby("state").label.agg(["sum","count","mean"])
    fig,ax=plt.subplots(figsize=(6.7,3.3))
    fig.subplots_adjust(left=.145,right=.975,bottom=.175,top=.91)
    height=.30
    for i,key in enumerate(STATES):
        train=rates.loc[key]
        obs=dev.loc[key]
        p_train,p_dev=100*float(train.p),100*float(obs["mean"])
        y_train,y_dev=i-.17,i+.17
        b1=ax.barh(y_train,p_train,height=height,color=NAVY,edgecolor="none",zorder=2,
                   label="训练 · 平滑风险" if i==0 else None)
        b2=ax.barh(y_dev,p_dev,height=height,color=RED,edgecolor="none",zorder=2,
                   label="开发 · 实际阳性率" if i==0 else None)
        ax.text(p_train+2.2,y_train,f"{p_train:.2f}%   {int(train.events):,}/{int(train.samples):,}",
                va="center",ha="left",fontsize=9,color=NAVY)
        ax.text(p_dev+2.2,y_dev,f"{p_dev:.2f}%   {int(obs['sum']):,}/{int(obs['count']):,}",
                va="center",ha="left",fontsize=9,color=RED)
        if i<2:
            ax.axhline(i+.5,color="#E4EAF1",lw=.8,zorder=0)
    ax.legend(handles=[b1[0],b2[0]],labels=["训练 · 平滑风险","开发 · 实际阳性率"],loc="upper right",ncol=2,frameon=False,
              handlelength=1.15,handletextpad=.5,columnspacing=1.8,borderaxespad=.15)
    ax.set(xlim=(0,122),xticks=[0,20,40,60,80,100],
           yticks=range(3),yticklabels=STATE_LABELS,ylim=(2.56,-.56),
           xlabel="未来 0.2 秒至少一次源标坏（%）")
    ax.tick_params(axis="y",length=0,pad=6,labelsize=9.5)
    ax.tick_params(axis="x",labelsize=9)
    ax.grid(axis="x",alpha=.16)
    return fig


def brier_bars():
    """Narrow adjoining bars within each split, with whitespace between splits."""
    scores=read("quality_scores.csv").query("subset=='all'").set_index("split")
    fig,axes=plt.subplots(1,2,figsize=(6.7,3.3),gridspec_kw={"wspace":0})
    fig.subplots_adjust(left=.10,right=.94,bottom=.205,top=.82,wspace=.24)
    span=.87
    for ax,split,base,title in zip(axes,["train","validation"],[7.85,29.55],
                                   ["训练环境留一","开发集"]):
        vals=1000*scores.loc[split,MODEL_KEYS].astype(float).to_numpy()
        width=.76
        x=(np.arange(3)-1)*width
        ax.bar(x,vals-base,bottom=base,width=width,color=[PINK,NAVY,SKY],
               edgecolor="none",linewidth=0,zorder=2)
        for xi,v in zip(x,vals):
            ax.annotate(f"{v:.3f}",(xi,v),xytext=(0,-5),textcoords="offset points",
                        ha="center",va="top",fontsize=9,color="#24374D")
        ax.set_xlim(-1.63,1.63)
        ax.set_ylim(base+span,base)
        ax.set_xticks(x)
        ax.set_xticklabels(["二状态","三状态","运动\n提升树"])
        ticks=base+np.arange(5)*.2
        ax.set_yticks(ticks,[f"{v:.2f}" for v in ticks])
        ax.set_yticks(ticks[:-1]+.1,minor=True)
        ax.tick_params(axis="y",which="minor",length=2)
        ax.set_title(title,fontsize=11,pad=9,color="#24374D")
        ax.tick_params(axis="x",length=0,pad=7,labelsize=9.5)
        ax.tick_params(axis="y",length=3,pad=4,labelsize=9)
    axes[0].yaxis.tick_left()
    axes[1].yaxis.tick_right()
    axes[0].set_ylabel("Brier 分数 ×1,000",labelpad=7)
    axes[0].spines["left"].set_visible(True)
    axes[1].spines["right"].set_visible(True)
    axes[0].spines["right"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    for ax in axes:
        for side in ("left","right"):
            ax.spines[side].set_color("#C9D3DE")
            ax.spines[side].set_linewidth(.8)
    return fig
