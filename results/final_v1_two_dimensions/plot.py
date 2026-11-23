"""Publication figures from the frozen window- and trial-level scores."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BLUE = ("#2A73BA", "#B6CAED")
RED = ("#CE4844", "#F7A9A8")
SPLITS = [("validation", "开发窗", BLUE, 212), ("test", "既有测试窗", RED, 390)]
BATCHES = [("original", "第一批", BLUE), ("fresh", "第二批", RED)]
HISTORY = ["full", "0.75s", "0.60s", "0.45s", "0.30s"]
HLAB = ["完整", "0.75", "0.60", "0.45", "0.30"]
DELAYS = [.1, .125, .15, .175, .2]
DLAB = ["100", "125", "150", "175", "200"]
plt.rcParams.update({"font.family":"Microsoft YaHei", "font.size":10, "pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False,
                     "axes.edgecolor":"#697786","axes.labelcolor":"#263341",
                     "xtick.color":"#445360","ytick.color":"#445360"})


def half_violin(ax, x, values, colors, direction, rng):
    """Mirror the two samples around one categorical position."""
    v = np.asarray(values, float)
    assert len(v)>1 and np.isfinite(v).all()
    dark, light = colors
    y = np.linspace(v.min(), v.max(), 200)
    if y[-1] > y[0]:
        bw = v.std(ddof=1) * len(v)**(-.2)
        kde = np.exp(-.5*((y[:,None]-v)/bw)**2).mean(axis=1)
        width = .42*kde/kde.max()
        ax.fill_betweenx(y,x,x+direction*width,color=light,alpha=.64,linewidth=0)
        spread = np.interp(v,y,width)
    else:
        spread = np.full(len(v),.17)
    ax.scatter(x+direction*rng.uniform(.03,.95,len(v))*spread,v,s=8 if len(v)<60 else 4,
               alpha=.64 if len(v)<60 else .37,color=dark,linewidths=0,zorder=2)
    marker_x = x + direction*.09
    low,high = np.quantile(v,[.25,.75])
    ax.vlines(marker_x,low,high,color=dark,linewidth=2.9,zorder=3)
    ax.vlines(marker_x,low,high,color="white",linewidth=1.7,zorder=4)
    ax.scatter(marker_x,v.mean(),s=18,color="white",edgecolors=dark,linewidths=.55,zorder=5)


def split_legend(fig):
    fig.legend(handles=[Patch(facecolor=RED[1],label="左 · 既有测试窗"),
                        Patch(facecolor=BLUE[1],label="右 · 开发窗")],
               loc="outside upper center",ncol=2,frameon=False)


def style(ax, title, labels, xlabel, ylabel, ylim):
    ax.set_title(title,loc="left",fontsize=11,fontweight="bold",pad=6)
    ax.set(xticks=np.arange(len(labels)),xticklabels=labels,xlabel=xlabel,ylabel=ylabel,
           xlim=(-.45,len(labels)-.5),ylim=ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y",color="#E8EEF8",linewidth=.7)
    ax.set_axisbelow(True)


def save(fig,name,dpi=165):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",pad_inches=.08)
    fig.savefig(OUT/f"{name}.png",dpi=dpi,bbox_inches="tight",pad_inches=.08)
    plt.close(fig)


def dist_row(fig, slot, frame, facets, group_col, levels, labels, xlabel, question, rng):
    grid = slot.subgridspec(1,len(facets))
    count=0
    for col,(key,value,label) in enumerate(facets):
        ax=fig.add_subplot(grid[col])
        for split,_,colors,n in SPLITS:
            g=frame[(frame.split==split)&(frame[key]==value)]
            for x,level in enumerate(levels):
                rows=g[g[group_col]==level].sort_values("window_id")
                assert len(rows)==rows.window_id.nunique()==n
                half_violin(ax,x,rows.nade,colors,-1 if split=="test" else 1,rng)
                count+=len(rows)
        style(ax,f"{question} · {label}",labels,xlabel,
              "逐窗 NADE" if col==0 else "",(0,frame[frame[key]==value].nade.max()*1.08))
    assert count==len(frame)


def heatmaps(fig,slot,s):
    grid=slot.subgridspec(1,3,width_ratios=[1,1,.045])
    cmap=LinearSegmentedColormap.from_list("rate",["#CE4844","#F7A9A8","#FDE6E6",
                "#E8EEF8","#B6CAED","#87AEE6","#2A73BA","#1F4E89"])
    cmap.set_bad("#F3F4F5")
    for i,(batch,name,_) in enumerate(BATCHES):
        ax=fig.add_subplot(grid[i])
        g=s.query("batch==@batch and case_id=='overall' and metric=='mean'")
        z=g.pivot(index="beta",columns="alpha",values="rate_pct").sort_index()
        assert z.shape==(3,5) and np.isfinite(z.to_numpy()).all()
        im=ax.imshow(z,cmap=cmap,norm=Normalize(65,100),aspect="auto")
        for (row,col),v in np.ndenumerate(z.to_numpy()):
            rgb=cmap(im.norm(v))[:3]
            color="white" if np.dot(rgb,[.2126,.7152,.0722])<.53 else "#182C42"
            ax.text(col,row,f"{v:.2f}",ha="center",va="center",
                    color=color,fontsize=10)
        ax.set(xticks=range(5),xticklabels=[f"{x:g}" for x in z.columns],yticks=range(3),
               yticklabels=["0","0.5","1"],xlabel="工期倍率 α",ylabel="附加工时 β（秒）")
        ax.set_title(f"第三问 · {name}",loc="left",fontsize=11,fontweight="bold")
        ax.set_xticks(np.arange(5)-.5,minor=True); ax.set_yticks(np.arange(3)-.5,minor=True)
        ax.grid(which="minor",color="white",linewidth=1.2); ax.tick_params(which="minor",length=0)
    bar=fig.colorbar(im,cax=fig.add_subplot(grid[2]),ticks=[65,75,85,95,100])
    bar.set_label("平均加权按时率（%）")


def sensitivity(q1,q2,s,rng,english=False):
    fig=plt.figure(figsize=(13.0,8.6),layout="constrained")
    grid=fig.add_gridspec(3,1,height_ratios=[1,1,.85])
    dist_row(fig,grid[0],q1,[("horizon",h,f"预测 {h:.1f} 秒") for h in [.2,.6,1]],
             "history",HISTORY,HLAB,"可见历史长度（秒）","第一问",rng)
    dist_row(fig,grid[1],q2,[("history","full","完整历史"),("history","short","最近 0.30 秒历史")],
             "delay",DELAYS,DLAB,"单次回传延迟（毫秒）","第二问",rng)
    split_legend(fig)
    heatmaps(fig,grid[2],s)
    if english:
        from matplotlib.text import Text

        labels = {
            "左 · 既有测试窗": "Left: Test windows",
            "右 · 开发窗": "Right: Development windows",
            "可见历史长度（秒）": "Visible history length (s)",
            "逐窗 NADE": "Window-level NADE",
            "第二问 · 完整历史": "Q2 · Full history",
            "第二问 · 最近 0.30 秒历史": "Q2 · Last 0.30 s of history",
            "单次回传延迟（毫秒）": "Observation return delay (ms)",
            "第三问 · 第一批": "Q3 · Batch 1",
            "第三问 · 第二批": "Q3 · Batch 2",
            "附加工时 β（秒）": "Added service time β (s)",
            "工期倍率 α": "Service time multiplier α",
            "平均加权按时率（%）": "Mean weighted on-time rate (%)",
        }
        for h in (.2,.6,1.):
            labels[f"第一问 · 预测 {h:.1f} 秒"] = f"Q1 · {h:.1f} s forecast horizon"
        for ax in fig.axes[:3]:
            ax.set_xticks(range(len(HLAB)), ["Full", *HLAB[1:]])
        for item in fig.findobj(Text):
            if item.get_text() in labels:
                item.set_text(labels[item.get_text()])
        save(fig,"Sensitivity_Analysis_EN",dpi=300)
    else:
        save(fig,"敏感性分析")


def tail(q1, q2, summary):
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(10.6, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    thresholds = pd.read_csv(OUT/"trajectory_thresholds.csv", dtype={"level":str})
    panels = [
        ("q1", q1[q1.horizon==1], "history", ["full","0.30s"],
         ["完整历史","最近 0.30 秒历史"], "A   第一问 · 可见历史缩短"),
        ("q2", q2[q2.history=="full"], "delay", [.1,.2],
         ["回传 100 毫秒","回传 200 毫秒"], "B   第二问 · 回传延迟增加"),
    ]
    for col, (q, frame, key, levels, labels, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0,col])
        test = frame[frame.split=="test"]
        ref = frame[(frame.split=="validation") & (frame[key]==levels[0])]
        assert len(ref)==ref.window_id.nunique()==212
        threshold = ref.nade.quantile(.9)
        limit = test[test[key].isin(levels)].nade.max()*1.03
        ax.axvline(threshold, color="#7A8590", linestyle=":", linewidth=1.1)
        ax.text(threshold, 103, f"P90 = {threshold:.3f}", va="bottom", ha="center",
                color="#48586B", fontsize=9)
        for k, (level, label, color) in enumerate(zip(levels,labels,[BLUE[0],RED[0]],strict=True)):
            g = test[test[key]==level]
            assert len(g)==g.window_id.nunique()==390
            scores, counts = np.unique(g.nade.to_numpy(),return_counts=True)
            remaining = 100*(len(g)-counts.cumsum())/len(g)
            ax.step(np.r_[0,scores,limit],np.r_[100,remaining,0],where="post",label=label,
                    color=color,linewidth=1.6,linestyle="-" if k==0 else "--")
            pct = 100*(g.nade>threshold).mean()
            stored = thresholds[(thresholds.question==q)&(thresholds.split=="test")&
                                (thresholds.level==str(level))].iloc[0]
            np.testing.assert_allclose([threshold,pct],[stored.threshold,stored.exceed_pct])
            ax.scatter(threshold,pct,s=24,color=color,edgecolors="white",linewidths=.6,zorder=4)
            ax.annotate(f"{pct:.2f}%", (threshold,pct), xytext=(10,9 if k else -11),
                        textcoords="offset points",fontsize=9,color=color,fontweight="bold",
                        bbox=dict(facecolor="white",alpha=.9,edgecolor="none",pad=.6))
        ax.set(xlim=(0,limit),ylim=(0,112),yticks=[0,20,40,60,80,100],
               xlabel="误差阈值（NADE）",ylabel="超过阈值的测试窗比例（%）")
        ax.set_title(title,loc="left",fontweight="bold",fontsize=12)
        ax.grid(axis="y",color="#E8EEF8",linewidth=.7)
        ax.legend(loc="upper right",bbox_to_anchor=(1,.91),fontsize=9)

    ax = fig.add_subplot(grid[1,:])
    paths = [OUT.parent/"final_v1_evidence_extension/duration_grid.csv.gz",OUT/"fresh_trials.csv.gz"]
    for idx,(batch,color,marker,path) in enumerate(zip(["original","fresh"],[RED[0],BLUE[0]],
                                                       ["o","s"],paths,strict=True)):
        raw = pd.read_csv(path,usecols=["case_id","alpha","beta","seed","weighted_on_time_rate"])
        for n,case in enumerate(["S1","S2","S3"]):
            ends = []
            for alpha in [1.5,1.]:
                g = raw.query("case_id==@case and alpha==@alpha and beta==.5")
                assert g.seed.nunique()==len(g)==300
                value = 100*g.weighted_on_time_rate.nsmallest(30).mean()
                archived = summary.query("batch==@batch and case_id==@case and alpha==@alpha and beta==.5 and metric=='tail10'")
                np.testing.assert_allclose(value,archived.rate_pct.item())
                ends.append(value)
            y = 2-n+(.13 if idx==0 else -.13)
            ax.plot(ends,[y,y],color=color,lw=1.8)
            for k,value in enumerate(ends):
                ax.scatter(value,y,s=48,marker=marker,facecolor=color if k else "white",
                           edgecolor=color,linewidth=1.4,zorder=3)
                ax.annotate(f"{value:.2f}%",(value,y),xytext=(7 if k else -7,0),
                            textcoords="offset points",ha="left" if k else "right",va="center",
                            color=color,fontsize=9)
    ax.set(xlim=(36,98),ylim=(-.45,2.45),xticks=[40,50,60,70,80,90],
           yticks=[2,1,0],yticklabels=["S1","S2","S3"],
           xlabel="最低 10% 回放的平均加权按时率（%）")
    ax.set_title("C   第三问 · 两批回放的尾部按时率",loc="left",fontweight="bold",fontsize=12)
    handles = [Line2D([],[],ls="-",color=c,marker=m,label=l) for c,m,l in
               [(RED[0],"o","第一批"),(BLUE[0],"s","第二批")]]
    handles += [Line2D([],[],ls="",marker="o",markeredgecolor="#415064",
                       markerfacecolor=fc,label=l) for fc,l in [("white","长工期"),("#415064","名义工期")]]
    ax.legend(handles=handles,loc="lower right",bbox_to_anchor=(1,1.015),ncol=4,
              borderaxespad=0,fontsize=9,columnspacing=1.4,handlelength=1.4)
    ax.grid(axis="x",color="#E8EEF8",linewidth=.8)
    ax.set_axisbelow(True)
    save(fig,"尾部风险")


if __name__=="__main__":
    q1=pd.read_csv(OUT/"q1_windows.csv.gz")
    q2=pd.read_csv(OUT/"q2_windows.csv.gz")
    s=pd.read_csv(OUT/"schedule_cells.csv")
    assert (len(q1),len(q2))==(9030,6020)
    rng=np.random.default_rng(20261122)
    sensitivity(q1,q2,s,rng)
    tail(q1,q2,s)

