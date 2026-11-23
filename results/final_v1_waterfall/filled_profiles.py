"""Five filled profile waterfalls on the frozen comparison's experimental support."""
from dataclasses import dataclass
from zipfile import ZipFile, ZIP_DEFLATED

import fitz
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from build import HERE, ROOT, RESULTS, VERSIONS, CASES, q3_results
from plot_five_dimensions import RED, BLUE, NAVY, SUB, RULE, canvas, note

OUT = ROOT/"figures/final_v1_filled_waterfalls"
COLORS = {"基线":BLUE,"终稿":RED}


@dataclass
class Plate:
    title: str
    description: str
    xlabel: str
    height: str
    profiles: list
    ticks: list
    ticklabels: list
    overall: np.ndarray
    headline: str
    foot: str
    smooth: bool = False


def rows(group, frame):
    return [(group,model,frame.index.to_numpy(float),frame[model].to_numpy(float)) for model in VERSIONS]


def perturbation(q):
    frames = []
    for file,axis,order in [("q1_horizons.csv","horizon",[.2,.6,1]),("q2_cost.csv","budget",[2,1,0])]:
        raw = pd.read_csv(RESULTS/file).query("split=='test'").pivot(index=axis,columns="model",values="nade")
        frames.append(raw.loc[order].rename(columns={"original":"基线","B":"终稿"})[VERSIONS])
    frames.append(-q.query("beta==.5").groupby("alpha")[VERSIONS].mean().loc[[1.,1.5]])
    profiles, ends = [],[]
    for group,data in zip(["一问","二问","三问"],frames):
        change = data-data.iloc[0]
        scaled = 100*change/change["基线"].iloc[-1]
        scaled.index = np.linspace(0,100,len(scaled))
        profiles.extend(rows(group,scaled))
        ends.append(scaled.iloc[-1])
    return Plate("参数敏感性","各问以本版本的起始档为零点，以同问基线的末档损失为 100。圆点和菱形标记实际测过的档位。",
                 "扰动进度（%）","相对损失指数",profiles,[0,50,100],["0","50","100"],np.mean(ends,axis=0),
                 "末档指数 · 三问等权","一问时长 0.2 / 0.6 / 1 秒，二问请求 2 / 1 / 0 次，三问工期倍率 1 / 1.5。三问仅有两个对照端点。")


def sites():
    keys=["split","environment_id","episode_uuid","side","window_id"]
    raw=pd.read_csv(RESULTS/"q2_windows.csv.gz").query("split in ['validation','test']")
    assert raw.requests.eq(2).all() and np.allclose(raw.waiting_s,.2)
    paired=raw.pivot(index=keys,columns="model",values="nade")[["original","B"]]
    assert paired.shape==(602,2) and paired.notna().all().all()
    for levels in [keys[:-1],keys[:-2],keys[:2]]:
        paired=paired.groupby(level=levels).mean()
    frame=pd.concat([paired.loc["validation"],paired.loc["test"]]).rename(columns={"original":"基线","B":"终稿"})
    ids=[s.split("_")[-1] for s in frame.index]
    frame.index=np.arange(8)
    return Plate("跨环境稳定性","第二问同一地点的两版使用相同窗口。曲线高度是该地的平均误差，地点编号按开发集、既有测试集排列。",
                 "地点编号","归一化位移误差（NADE）",rows("",frame),list(range(8)),ids,frame[VERSIONS].mean().to_numpy(),
                 "八地等权 NADE","前四地共 212 个开发窗，后四地共 390 个既有测试窗。各地点内依次对窗口、手侧和记录取平均，地点间的连线用于辨读。")


def uncertainty(q):
    data=q.set_index(["alpha","beta","case_id","seed"])[VERSIONS].sort_index()
    delta=(data.loc[1.5,.5]-data.loc[1.,.5]).groupby(level="seed").mean().sort_index()
    assert delta.shape==(30,2)
    rng=np.random.default_rng(20261122)
    boot=delta.to_numpy()[rng.integers(0,30,(10000,30))].mean(axis=1)
    width=.5
    edges=np.arange(np.floor(boot.min()/width)*width-width,np.ceil(boot.max()/width)*width+2*width,width)
    frame=pd.DataFrame(index=(edges[:-1]+edges[1:])/2)
    for j,model in enumerate(VERSIONS):
        counts=np.histogram(boot[:,j],bins=edges)[0]
        assert counts.sum()==10000
        frame[model]=counts/100
    half=np.diff(np.quantile(boot,[.025,.975],axis=0),axis=0)[0]/2
    ticks=list(np.arange(np.ceil(edges[0]/2)*2,edges[-1],2))
    return Plate("配对不确定性","第三问把名义工期延长至 1.5 倍。曲线显示两版平均按时率变化在配对重抽样中的分布。",
                 "长工期减名义工期的按时率差（百分点）","每组重抽结果占比（%）",rows("",frame),ticks,[f"{t:.0f}" for t in ticks],
                 half,"95% 区间半宽（百分点）","30 个共同工期种子，三个案例等权，配对重抽 10,000 次。每 0.5 个百分点分一组，曲线经过各组的实测频率。",True)


def tail(q):
    nom=q.query("alpha==1 and beta==.5")
    profiles,deciles=[],[]
    k=np.arange(1,31)
    for case in CASES:
        part=nom.query("case_id==@case")
        assert len(part)==30
        frame=pd.DataFrame({m:np.sort(part[m]).cumsum()/k for m in VERSIONS},index=k)
        profiles.extend(rows(case,frame))
        deciles.append(frame.loc[3])
    return Plate("尾部风险","第三问在各案例内按加权按时率由低到高排序。曲线表示从最差 1 次逐步纳入到全部 30 次的平均按时率。",
                 "纳入最差的 k 次","平均加权按时率（%）",profiles,[1,3,10,20,30],["1","3","10","20","30"],np.mean(deciles,axis=0),
                 "最差一成 · 三案例均值（%）","名义工况，每个案例的两版共用 30 个工期种子。k = 3 对应最差一成，两版分别按各自的结果取尾部。",True)


def seed_spread():
    r=HERE.parent
    old=pd.concat([pd.read_csv(HERE/"previous_nominal_trials.csv.gz"),
                   pd.read_csv(r/"q3_baseline_seed_control/trials.csv.gz").query("alpha==1.5 and beta==.5")])
    final=pd.concat([pd.read_csv(path).assign(batch=batch) for batch,path in [
        ("original",r/"final_v1_stability/seed_trials.csv.gz"),("replication",r/"final_v1_evidence_extension/internal_replication.csv.gz")
    ]]).query("alpha in [1.,1.5] and beta==.5")
    keys=["alpha","beta","batch","case_id","seed","internal_seed"]
    a,b=[f.set_index(keys).sort_index() for f in (old,final)]
    assert a.index.is_unique and a.index.equals(b.index) and len(a)==7200 and a.instance_hash.equals(b.instance_hash)
    sds={}
    for model,f in zip(VERSIONS,[a,b]):
        blocks=f.groupby(level=keys[:-1]).weighted_on_time_rate
        assert blocks.size().eq(20).all()
        sds[model]=blocks.agg(lambda s:(s-s.iloc[0]).std(ddof=1)*100)
    sd=pd.DataFrame(sds)
    saved=pd.read_csv(HERE/"matched_seed_sd.csv").set_index(["alpha","batch"])[VERSIONS].sort_index()
    np.testing.assert_allclose(sd.groupby(level=["alpha","batch"]).mean(),saved,atol=1e-10)
    profiles=[]
    # Keep the observed mass at exactly zero visible, then group positive SDs in 1-pp bins.
    last=int(np.ceil(sd.to_numpy().max()))
    for alpha,cond in [(1.,"名义"),(1.5,"长工期")]:
        part=sd.xs(alpha,level="alpha")
        assert len(part)==180
        frame=pd.DataFrame(index=[0.,*list(np.arange(.5,last,1.))])
        for m in VERSIONS:
            vals=part[m].to_numpy()
            positive=vals[vals>0]
            # Right-closed intervals (0,1], (1,2], ... ; bin index ceil(v)-1.
            counts=np.bincount(np.ceil(positive).astype(int)-1,minlength=last)
            frame[m]=100*np.r_[(vals==0).sum(),counts]/len(vals)
            assert np.isclose(frame[m].sum(),100)
        profiles.extend(rows(cond,frame))
    ticks=list(range(0,last+1,2))
    return Plate("内部抽样种子稳定性","固定任务与实际工期，逐实例计算 20 个内部种子的按时率标准差。零点的高度表示换种子后按时率始终相同的比例。",
                 "实例内标准差（百分点）","实例与批次组合占比（%）",profiles,ticks,[str(t) for t in ticks],saved.mean().to_numpy(),
                 "四组平均标准差（百分点）","每种工况 90 个实例 × 两批。零值单列，正值按 1 个百分点分组。蓝层使用上一正式版的名义补测和长工期配对回放。")


def render(p):
    fig=canvas(p.title+" · 填充曲线瀑布图",p.description,height=6.8)
    groups=list(dict.fromkeys(g for g,*_ in p.profiles))
    ax=fig.add_axes([.055,.185,.86,.565],projection="3d",computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=23,azim=-57)
    ax.set_box_aspect((2.65,1.9,1.35),zoom=1.60)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.set_facecolor((.97,.98,.99,.18))
        axis.pane.set_edgecolor(RULE)
        axis._axinfo["grid"].update(color="#DFE6EA",linewidth=.65)
    ymax=0
    layers=[]
    for group,model,x,z in p.profiles:
        y=groups.index(group)*2.8+(1. if model=="基线" else 0.)
        layers.append((y,model,x,z))
        ymax=max(ymax,max(z))
    for j,(y,model,x,z) in enumerate(sorted(layers,key=lambda s:-s[0])):
        if p.smooth:
            xx=np.linspace(x[0],x[-1],max(180,len(x)*5))
            zz=PchipInterpolator(x,z)(xx).clip(0)
        else:
            xx,zz=x,z
        color=np.array(to_rgb(COLORS[model]))
        # Build a continuous vertical gradient inside a single silhouette; all vertices
        # share the same depth. The upper boundary interpolates only the measured values.
        bands=72
        faces=[]
        colors=[]
        for band in range(bands):
            lo,hi=zz*band/bands,zz*(band+1)/bands
            faces.append([(u,y,v) for u,v in zip(xx,lo)]+[(u,y,v) for u,v in zip(xx[::-1],hi[::-1])])
            amount=.045+.44*(band+.5)/bands
            colors.append(1-amount+amount*color)
        area=Poly3DCollection(faces,facecolors=colors,edgecolors=colors,linewidths=.35,
                              antialiased=False,zorder=2*j+1)
        ax.add_collection3d(area)
        ax.plot(xx,np.full(len(xx),y),zz,color=COLORS[model],lw=1.85,zorder=2*j+2)
        if len(x)<=8:
            ax.plot(x,np.full(len(x),y),z,ls="none",marker="o" if model=="基线" else "D",
                    color=COLORS[model],ms=3.4,zorder=2*j+2)
        ax.plot([x[0],x[-1]],[y,y],[0,0],color=COLORS[model],lw=.55,alpha=.5,zorder=2*j+1)
    locations=[s[0] for s in layers]
    xlo=min(x[0] for _,_,x,_ in layers)
    xhi=max(x[-1] for _,_,x,_ in layers)
    ax.set(xlim=(xlo,xhi),ylim=(min(locations)-.28,max(locations)+.3),zlim=(0,ymax*1.10),
           xticks=p.ticks,xticklabels=p.ticklabels)
    if groups==[""]:
        ax.set_yticks([0,1],labels=["终稿","基线"])
    else:
        ax.set_yticks([i*2.8+.5 for i in range(len(groups))],labels=groups)
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
    ax.tick_params(axis="x",labelsize=9,pad=0)
    ax.tick_params(axis="y",labelsize=9,pad=0)
    ax.tick_params(axis="z",labelsize=9,pad=0)
    for artist in [*ax.collections,*ax.lines]:
        artist.set_clip_on(False)
    fig.text(.065,.778,p.height,color=SUB,fontsize=10)
    fig.text(.48,.096,p.xlabel,ha="center",fontsize=9.6,color=SUB)
    name_precision={"参数敏感性":2,"跨环境稳定性":5,"配对不确定性":3,"尾部风险":2,"内部抽样种子稳定性":3}
    digits=name_precision[p.title]
    fig.text(.94,.938,p.headline,ha="right",va="center",fontsize=9.2,color=SUB)
    for x,m,v in zip([.80,.94],VERSIONS,p.overall):
        fig.text(x,.896,f"{m}  {v:.{digits}f}",ha="right",fontsize=11.5,weight="bold",color=COLORS[m])
    note(fig,p.foot)
    return fig


def main():
    q=q3_results()
    ref=pd.read_csv(HERE/"waterfall_values.csv").groupby("metric").first()
    OUT.mkdir(parents=True,exist_ok=True)
    with fitz.open() as combined,ZipFile(OUT/"五张填充曲线瀑布图_高清PNG.zip","w",ZIP_DEFLATED) as archive:
        for i,p in enumerate([perturbation(q),sites(),uncertainty(q),tail(q),seed_spread()],1):
            np.testing.assert_allclose(p.overall,ref.loc[p.title,["baseline","final"]].to_numpy(float),rtol=0,atol=1e-10)
            fig=render(p)
            assert len(fig.axes)==1 and fig.axes[0].name=="3d" and len(fig.axes[0].collections)==len(p.profiles)
            name=f"{i:02d}_{p.title}_填充曲线式三维瀑布图"
            pdf,png=OUT/(name+".pdf"),OUT/(name+".png")
            fig.savefig(pdf,metadata={"Title":name})
            fig.savefig(png,dpi=450)
            plt.close(fig)
            with fitz.open(pdf) as page:
                combined.insert_pdf(page)
            archive.write(png,arcname=png.name)
        combined.save(OUT/"五维填充曲线式三维瀑布图_审核稿.pdf",garbage=4,deflate=True)


if __name__=="__main__":
    main()
