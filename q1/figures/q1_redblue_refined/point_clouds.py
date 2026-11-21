import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from style import read,BLUE,NAVY,RED


def wire(ax,center,r):
    u,v=np.meshgrid(np.linspace(0,2*np.pi,25),np.linspace(0,np.pi,15))
    ax.plot_wireframe(center[0]+r*np.cos(u)*np.sin(v),center[1]+r*np.sin(u)*np.sin(v),
                      center[2]+r*np.cos(v),rstride=2,cstride=2,color="#668DC0",lw=.48,alpha=.65)


def space(ax,points):
    lo,hi=points.min(0),points.max(0)
    middle=(lo+hi)/2
    radius=max(hi-lo)*.52
    for dim,axis in enumerate("xyz"):
        getattr(ax,f"set_{axis}lim")(middle[dim]-radius,middle[dim]+radius)
        getattr(ax,f"{axis}axis").set_major_locator(MaxNLocator(3))
        getattr(ax,f"set_{axis}label")(axis.upper(),labelpad=-3,fontsize=7.2)
    ax.set_box_aspect((1,1,1))
    ax.view_init(23,-55)
    ax.tick_params(labelsize=6.7,pad=-2)


def sphere_example(points):
    g=points.query("interface=='continuous'")
    candidates=[]
    for key,window in g.groupby("window_id",sort=True):
        if not window.coverage.eq(1).all():
            continue
        window=window.sort_values("age")
        row=window.iloc[(window.age-.6).abs().argmin()]
        c=window[["cx","cy","cz"]].to_numpy()
        xyz=c+window[["dx","dy","dz"]].to_numpy()
        if np.linalg.norm(xyz-row[["cx","cy","cz"]].to_numpy(float),axis=1).max()<=row.radius:
            span=np.linalg.norm(np.ptp(xyz,axis=0))/row.radius
            candidates.append((span,key))
    if not candidates:
        raise ValueError("No window satisfies both coverage conditions")
    _,key=max(candidates)
    g=g[g.window_id==key].sort_values("age")
    row=g.iloc[(g.age-.6).abs().argmin()]
    center=row[["cx","cy","cz"]].to_numpy(float)
    delta=row[["dx","dy","dz"]].to_numpy(float)
    centers=g[["cx","cy","cz"]].to_numpy()
    refs=centers+g[["dx","dy","dz"]].to_numpy()
    fig=plt.figure(figsize=(4.8,3.8))
    ax=fig.add_subplot(projection="3d",proj_type="ortho")
    wire(ax,center,row.radius)
    ax.plot(*centers.T,c=NAVY,lw=1.5,label="预测球心轨迹")
    ax.plot(*refs.T,c=RED,lw=1.2,ls="--",label="参考质心轨迹")
    ax.scatter(*center,c=NAVY,s=33,marker="D")
    ax.scatter(*(center+delta),c=RED,s=30)
    space(ax,np.vstack([refs,centers,center-row.radius,center+row.radius]))
    ax.legend(loc="upper left",bbox_to_anchor=(-.16,1.01),fontsize=8.5)
    return fig,row


def all_clouds(points):
    fig=plt.figure(figsize=(13.5,3.4))
    sets=[("continuous","连续遮罩"),("pre_first","恢复首帧前"),("pre_second","恢复第二帧前"),
          ("missing","无可用半径")]
    for i,(kind,title) in enumerate(sets):
        ax=fig.add_subplot(1,4,i+1,projection="3d",proj_type="ortho")
        g=points[points.radius.isna()] if kind=="missing" else points[(points.interface==kind)&points.radius.notna()]
        full_n=len(g)
        if kind=="continuous":
            g=g.sort_values(["window_id","frame_index"]).sample(n=full_n//2,random_state=20261121)
            assert len(g)==3163
        v=g[["dx","dy","dz"]].to_numpy()
        if kind=="missing":
            ax.scatter(*v.T,c="#6B7382",marker="x",s=22)
            space(ax,np.vstack([v,np.zeros(3)]))
            info=f"{len(v)} 点 · 原坐标残差"
        else:
            v=v/g.radius.to_numpy()[:,None]
            # A radial transform keeps directions and the unit-sphere boundary exact.
            r=np.linalg.norm(v,axis=1)
            v=v*(np.log2(1+r)/np.where(r==0,1,r))[:,None]
            wire(ax,np.zeros(3),1.)
            inside=g.coverage.to_numpy().astype(bool)
            for m,col in [(inside,BLUE),(~inside,RED)]:
                ax.scatter(*v[m].T,c=col,s=3.2 if i==0 else 15,alpha=.32 if i==0 else .83,
                           edgecolors="none",depthshade=False,rasterized=(i==0))
            space(ax,np.vstack([v,[-1,-1,-1],[1,1,1]]))
            h=1 if kind=="continuous" else 0
            row=read(f"cal_{kind}_summary.csv").query("split=='validation' and source=='ALL' and horizon==@h").iloc[0]
            info=(f"显示 {len(g):,}/{full_n:,}（50%）\n全量加权覆盖 {row.coverage:.2%}"
                  if kind=="continuous" else f"{len(g):,} 点 · 加权覆盖 {row.coverage:.2%}")
        ax.set_title(title+"\n"+info,fontsize=10.5,pad=4)
    fig.subplots_adjust(left=.016,right=.98,top=.81,bottom=.08,wspace=.13)
    return fig
