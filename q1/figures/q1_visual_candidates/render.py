"""Rebuild three complementary Q1 figures from the frozen experiment outputs."""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.patches import Patch
import fitz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q1_original_redblue"))
from style import read, start, NAVY, BLUE, PALEBLUE, PINK, PALERED, RED


def export(fig, name):
    for fmt in ("pdf","png"):
        fig.savefig(HERE / f"{name}.{fmt}",dpi=300,bbox_inches="tight",pad_inches=.15)
    plt.close(fig)


def heatmap():
    env = read("position_environments.csv").query("split=='train' and model in ['B','CV']")
    assert not env.duplicated(["source","env","horizon","model"]).any()
    avg = env.groupby(["env","horizon","model"]).nade.mean().unstack()
    gain = (100*(1-avg.B/avg.CV)).unstack().sort_values(1.,ascending=False)
    assert gain.shape==(16,3) and gain.notna().all().all()
    fig,ax=plt.subplots(figsize=(5,5.5))
    fig.subplots_adjust(left=.27,right=.93,top=.87,bottom=.17)
    palette = LinearSegmentedColormap.from_list("q1_diverging",[RED,PINK,"#FFFFFF","#96BBE8",NAVY])
    limit=10*np.ceil(np.max(np.abs(gain.to_numpy()))/10)
    scale=Normalize(-limit,limit)
    im=ax.imshow(gain,cmap=palette,norm=scale,aspect="auto")
    types={"retail_x_supermarket":"超市","retail_x_convenience":"便利店","home_x_apartment":"住宅",
           "office_x_corporate_office":"办公室","logistics_x_courier_station":"快递站"}
    labels=[f'{types[s.rsplit("_",1)[0]]}  {s.rsplit("_",1)[1]}' for s in gain.index]
    ax.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(3),xticklabels=["0.2 秒","0.6 秒","1.0 秒"])
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-.5,3),minor=True)
    ax.set_yticks(np.arange(-.5,16),minor=True)
    ax.grid(which="minor",color="white",lw=1.1)
    ax.tick_params(which="both",length=0,pad=5)
    for s in ax.spines.values(): s.set_visible(False)
    for (i,j),v in np.ndenumerate(gain.to_numpy()):
        rgb=np.array(palette(scale(v))[:3])
        luminance=np.dot(np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4),[.2126,.7152,.0722])
        ax.text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=8.6,color="white" if luminance<.24 else "#1A3044")
    bar=fig.colorbar(im,cax=fig.add_axes([.29,.091,.62,.018]),orientation="horizontal",ticks=[-80,-40,0,40,80])
    bar.outline.set_visible(False)
    bar.set_label("归一化 ADE 降幅（%）",labelpad=2)
    export(fig,"heatmap")


def scatter():
    raw=read("curve/window_metrics.csv.gz").query("split=='validation' and horizon==1 and model in ['B','CV']")
    data=raw.pivot(index=["window_id","record","env"],columns="model",values="nade").dropna()
    assert len(data)==212 and len(raw)==2*len(data) and (data>0).all().all()
    fig=plt.figure(figsize=(5.15,5))
    gs=fig.add_gridspec(2,2,width_ratios=[5,1],height_ratios=[1,5],hspace=.045,wspace=.045,
                        left=.15,right=.96,top=.94,bottom=.17)
    ax=fig.add_subplot(gs[1,0])
    top=fig.add_subplot(gs[0,0],sharex=ax)
    side=fig.add_subplot(gs[1,1],sharey=ax)
    lo,hi=.006,1.7
    assert data.min().min()>lo and data.max().max()<hi
    ax.set(xscale="log",yscale="log",xlim=(lo,hi),ylim=(lo,hi),aspect="equal",adjustable="box")
    # Anchor the marginal panels to the final square plot after layout.
    fig.canvas.draw()
    pos=ax.get_position()
    top.set_position([pos.x0,pos.y1+.009,pos.width,.088])
    side.set_position([pos.x1+.01,pos.y0,.075,pos.height])
    bins=np.geomspace(lo,hi,24)
    top.hist(data.CV,bins=bins,color=PINK,edgecolor="white",lw=.5)
    side.hist(data.B,bins=bins,orientation="horizontal",color="#87AEE6",edgecolor="white",lw=.5)
    for marginal in (top,side): marginal.set_axis_off()
    top.text(1,1,"恒速误差分布",transform=top.transAxes,ha="right",va="top",fontsize=8)
    ax.plot([lo,hi],[lo,hi],color="#5F6F80",ls="--",lw=1,zorder=1)
    for wins,c,marker,name in [(True,BLUE,"o","阻尼误差更小"),(False,RED,"^","恒速误差更小")]:
        subset=data[(data.B<data.CV) if wins else (data.B>data.CV)]
        ax.scatter(subset.CV,subset.B,s=19,marker=marker,facecolor=c,edgecolor="white",
                   linewidth=.35,alpha=.77,label=f"{name}（{len(subset)}）",zorder=2)
    assert (data.B!=data.CV).all()
    ticks=[.01,.03,.1,.3,1.]
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_ticks(ticks)
        axis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:g}'))
        axis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="恒速外推 · 归一化 ADE",ylabel="固定阻尼 · 归一化 ADE")
    ax.grid(alpha=.13)
    ax.legend(loc="upper left",fontsize=8,handletextpad=.2)
    export(fig,"scatter")
    return int((data.B<data.CV).sum())


def volcano():
    df=read("position_paired.csv").query("split=='train'").set_index(["model","reference"])
    assert len(df)==3 and (df.holm>0).all()
    fig,ax=plt.subplots(figsize=(5.15,4.35))
    fig.subplots_adjust(left=.14,right=.95,bottom=.18,top=.93)
    threshold=-np.log10(.05)
    ax.axhspan(threshold,5,color=PALEBLUE,alpha=.53,zorder=0)
    ax.axhline(threshold,color="#788797",ls="--",lw=1)
    ax.axvline(0,color="#788797",lw=.8)
    ax.text(-4.4,threshold+.10,"Holm 校正 p = 0.05",fontsize=8.3,color="#4B5968")
    models=[("B","Hold","固定阻尼 vs 末值保持",NAVY,"o",-6,7,"right"),
            ("B","CV","固定阻尼 vs 恒速外推",BLUE,"o",-6,8,"right"),
            ("C","B","残差 MLP vs 固定阻尼",RED,"D",9,4,"left")]
    for model,ref,name,col,marker,dx,dy,align in models:
        row=df.loc[(model,ref)]
        x,y=row.gain_pct,-np.log10(row.holm)
        ax.scatter(x,y,c=col,marker=marker,s=72,edgecolor="white",zorder=4)
        ax.annotate(f"{name}\n{x:+.2f}%",(x,y),xytext=(dx,dy),textcoords="offset points",
                    ha=align,fontsize=9,color=col)
    ax.set(xlim=(-5,32),ylim=(-.1,5),xticks=[0,10,20,30],yticks=range(6),
           xlabel="相对参考模型的归一化 ADE 降幅（%）",ylabel=r"$-\log_{10}$（Holm 校正 p 值）")
    ax.grid(axis="y",alpha=.14)
    export(fig,"volcano")


def review(wins):
    from matplotlib import font_manager
    font=Path(font_manager.findfont("Microsoft YaHei"))
    css=f"@font-face {{font-family:han; src:url('{font.name}');}} * {{font-family:han;color:#213C58;}}"
    archive=fitz.Archive(str(font.parent))
    target=HERE/"第一问_高级图型候选.pdf"
    items=[
        ("A  环境与预测时长","heatmap","16 个训练留出环境。每格比较该环境内来源等权的固定阻尼与恒速误差，按 1 秒降幅排序。蓝色表示改善，红色表示退化。"),
        ("B  同一窗口的预测误差","scatter",f"开发集 1 秒预测，一个点对应一个窗口。虚线为等误差线，下方 {wins} 个窗固定阻尼误差更小。两轴使用相同对数刻度，上、右直方图显示窗口数分布。"),
        ("C  改善幅度与统计证据","volcano","每点对应一项预设比较。训练集按 16 个环境做配对符号随机化检验，三项比较做 Holm 校正。阴影区表示校正 p 值小于 0.05。")]
    with fitz.open() as doc:
        pg=doc.new_page(width=1080,height=536)
        def words(rect,s,size):
            spare,_=pg.insert_htmlbox(fitz.Rect(*rect),f"<div style='font-size:{size}pt'>{s}</div>",css=css,
                                      archive=archive,scale_low=1)
            assert spare>=0
        words((24,15,1040,51),"固定阻尼的误差改善出现在哪里？",21)
        for i,(title,name,caption) in enumerate(items):
            left=24+350*i
            words((left,65,left+331,96),title,13)
            with fitz.open(HERE/f"{name}.pdf") as f:
                pg.show_pdf_page(fitz.Rect(left,102,left+331,419),f,0)
            words((left,437,left+331,524),caption,10)
        doc.subset_fonts()
        doc.save(target,garbage=4,deflate=True)
    with fitz.open(target) as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(1.55,1.55)).save(HERE/"overview.png")


if __name__=="__main__":
    start()
    heatmap()
    wins=scatter()
    volcano()
    review(wins)
