import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from charts import radar as base_radar, MODELS, NAMES, COLORS
from style import read, NAVY, RED, BLUE, SKY, PALEBLUE, PINK


def position_waterfall():
    table=read("position_horizons.csv").query("split=='validation'").pivot(index="model",columns="horizon",values="nade")
    table=table.loc[["C","B","CV","Hold"]]
    fig=plt.figure(figsize=(5.8,4.15))
    ax=fig.add_subplot(projection="3d",proj_type="ortho",computed_zorder=False)
    ax.view_init(elev=32,azim=-39)
    x=table.columns.to_numpy(float)
    for j,(model,row) in enumerate(table.iterrows()):
        i=MODELS.index(model)
        v=row.to_numpy()
        vertices=[(x[0],j,0),*zip(x,np.full(len(x),j),v),(x[-1],j,0)]
        ax.add_collection3d(Poly3DCollection([vertices],facecolor=COLORS[i],edgecolor="none",alpha=.13,zorder=1+.1*j))
        ax.plot(x,[j]*len(x),v,marker=["s","^","o","D"][i],color=COLORS[i],ms=5,lw=2.3,
                zorder=10+j,label=NAMES[i])
    ax.set(xticks=x,xticklabels=["0.2 秒","0.6 秒","1.0 秒"],yticks=range(4),yticklabels=[],
           xlim=(.2,1.),ylim=(-.2,3.2),zlim=(0,.32))
    ax.text2D(1.10,.55,"归一化 ADE",transform=ax.transAxes,rotation=90,fontsize=9)
    ax.set_box_aspect((1.45,.94,1.05))
    ax.tick_params(labelsize=8,pad=0)
    fig.legend(loc="upper center",ncol=2,fontsize=9)
    fig.subplots_adjust(top=.89,bottom=.05)
    return fig


def radar():
    fig,raw=base_radar()
    ax=fig.axes[0]
    scores=100*raw.min()/raw
    ang=np.arange(6)*2*np.pi/5
    # Paint larger areas first, then bring all four outlines above every filled area.
    areas={k:np.dot(s,np.roll(s,1)) for k,s in zip(scores.index,scores.to_numpy())}
    fills=["#F4BBBB","#D5E4F5","#AEC3DE","#D8E7F3"]
    for rank,key in enumerate(sorted(areas,key=areas.get,reverse=True)):
        r=scores.loc[key].to_numpy()
        ax.fill(ang,np.r_[r,r[0]],facecolor=COLORS[MODELS.index(key)],edgecolor="none",alpha=.23,zorder=1+rank*.1)
    for line in ax.lines:
        line.set_zorder(5)
    return fig,raw


def environment_boxes():
    env=read("position_environments.csv").query("split=='train' and model in ['B','CV']")
    e=env.groupby(["env","horizon","model"]).nade.mean().unstack()
    gain=(100*(1-e.B/e.CV)).unstack()
    assert gain.shape==(16,3) and gain.notna().all().all()
    fig,ax=plt.subplots(figsize=(6.5,3.15),layout="constrained")
    locations=np.arange(3)
    ax.boxplot(gain.to_numpy(),positions=locations,widths=.42,patch_artist=True,showfliers=False,
               boxprops={"facecolor":PALEBLUE,"edgecolor":NAVY},medianprops={"color":NAVY,"lw":1.5},manage_ticks=False)
    jitter=np.linspace(-.115,.115,len(gain))
    for i,col in enumerate(gain):
        ax.scatter(locations[i]+jitter,gain[col],c=BLUE,s=17,alpha=.85,zorder=3)
    ax.axhline(0,color="#727B89",ls="--",lw=.9)
    ax.set(xticks=range(3),xticklabels=["0.2 秒","0.6 秒","1.0 秒"],ylabel="固定阻尼相对恒速的 ADE 降幅（%）")
    ax.set_xlim(-.48,2.48)
    ax.grid(axis="y",alpha=.15)
    return fig


def recovery_bars():
    data=read("recovery_natural_summary.csv").query("split=='validation' and scope=='common'")
    table=data.pivot(index="model",columns="phase",values="norm_error")
    fig,ax=plt.subplots(figsize=(6.0,3.25),layout="constrained")
    for i,(key,name,col) in enumerate([("hold_two","等待两帧",RED),("recent6_hold","最近六帧回退",BLUE),
                                        ("zero_velocity","首帧＋零速",SKY),("decay","首帧＋衰减旧速",NAVY)]):
        v=table.loc[key,["pre_first","pre_second"]].to_numpy()
        ax.bar(np.arange(2)+(i-1.5)*.19,v,width=.17,color=col,label=name)
    ax.set(xticks=[0,1],xticklabels=["接收首帧前（50 段）","接收第二帧前（44 段）"],
           ylim=(0,.41),ylabel="归一化位置误差")
    ax.grid(axis="y",alpha=.15)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),ncol=2)
    return fig
