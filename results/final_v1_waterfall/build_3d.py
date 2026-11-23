"""Render paired, stepped 3-D waterfalls from the five frozen comparisons."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import fitz
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from build import CASES, HERE, RESULTS, ROOT, VERSIONS, paired, q3_results
from plot_five_dimensions import BLUE, RED, NAVY, SUB, RULE, canvas, note

OUT = ROOT / "figures/final_v1_waterfalls_3d"


def datasets(q):
    # Each group's weighted contribution is expressed in the overall metric's unit.
    changes = []
    for name, x, before, after in [("q1_horizons.csv","horizon",.2,1),("q2_cost.csv","budget",2,0)]:
        rows = pd.read_csv(RESULTS/name).query("split=='test'").pivot(index=x,columns="model",values="nade")
        changes.append((rows.loc[after]-rows.loc[before]).rename({"original":"基线","B":"终稿"}))
    rates = q.query("beta==.5").groupby("alpha")[VERSIONS].mean()
    changes.append(rates.loc[1]-rates.loc[1.5])
    losses = pd.DataFrame(changes,index=["一问时长","二问预算","三问工期"])[VERSIONS]
    sensitivity = (100*losses.div(losses["基线"],axis=0)/3).cumsum()
    yield ("参数敏感性",sensitivity,"相对损失指数",2,"",
           "三问各以基线的扰动损失为 100，等权汇总。阶梯越低，整体损失越小。",
           "一问预测时长 0.2 → 1 秒，二问请求 2 → 0 次，三问工期倍率 1 → 1.5。前两问取 NADE 增量，第三问取按时率降幅。")

    keys = ["split","environment_id","episode_uuid","side","window_id"]
    windows = pd.read_csv(RESULTS/"q2_windows.csv.gz").query("split in ['test','validation']")
    assert windows.requests.eq(2).all() and np.allclose(windows.waiting_s,.2)
    sites = windows.pivot(index=keys,columns="model",values="nade")[["original","B"]]
    assert sites.shape==(602,2) and sites.notna().all().all()
    for levels in [keys[:-1],keys[:-2],keys[:2]]:
        sites = sites.groupby(level=levels).mean()
    sites = pd.concat([sites.loc["validation"],sites.loc["test"]]).rename(columns={"original":"基线","B":"终稿"})
    sites.index = [s.split("_")[-1] for s in sites.index]
    yield ("跨环境稳定性",(sites/8).cumsum(),"地点等权 NADE",5,"",
           "第二问按地点等权汇总误差。每一步加入一个地点的 NADE × 1/8，整体越低越好。",
           "横轴为地点编号。前四地为开发集（212 窗），后四地为既有测试集（390 窗）。两版在每个地点使用相同窗口。")

    data = q.set_index(["alpha","beta","case_id","seed"])[VERSIONS].sort_index()
    delta = data.loc[1.5,.5]-data.loc[1.,.5]
    widths = []
    for n in range(1,4):
        chosen = delta.loc[CASES[:n]].groupby(level="seed").mean()
        assert len(chosen)==30 and chosen.notna().all().all()
        widths.append({v:paired("long",chosen[v]/100)["halfwidth_pp"] for v in VERSIONS})
    uncertainty = pd.DataFrame(widths,index=["S1","加入 S2","加入 S3"])
    yield ("配对不确定性",uncertainty,"95% 区间半宽（百分点）",3," 个百分点",
           "从 S1 起逐个纳入案例，按案例等权重算长工期相对名义工期的配对区间。整体半宽越小，估计越精确。",
           "各阶段均按相同的 30 个工期种子配对重抽 10,000 次。浮动柱表示加入下一个案例后区间半宽的变化。")

    nominal = q.query("alpha==1 and beta==.5")
    assert nominal.groupby("case_id").size().eq(30).all()
    tail = nominal.groupby("case_id")[VERSIONS].agg(lambda x:x.nsmallest(3).mean()).loc[CASES]
    yield ("尾部风险",(tail/3).cumsum(),"加权按时率（%）",2,"%",
           "每个案例选取按时率最低的 3 次，再对三个案例等权汇总。整体越高，困难实例的交付表现越好。",
           "名义工况，每例 30 个官方工期种子。每个浮动柱的高度等于该版本本案例最差一成平均按时率的三分之一。")

    seeds = pd.read_csv(HERE/"matched_seed_sd.csv").set_index(["alpha","batch"])[VERSIONS]
    seeds = seeds.loc[[(a,b) for a in [1.,1.5] for b in ["original","replication"]]]
    seeds.index = ["名义首批","名义复验","长工期首批","长工期复验"]
    yield ("内部抽样种子稳定性",(seeds/4).cumsum(),"实例内标准差（百分点）",3," 个百分点",
           "固定实例，只换内部种子。名义与长工期各含两批，四组等权汇总。整体标准差越小，种子波动越小。",
           "每组 90 个实例，每例 20 个内部种子。红层为冻结终稿 K = 64，蓝层使用已完成名义和长工期复验的上一正式版。")


def render(title,levels,metric,digits,suffix,description,foot):
    fig = canvas(title+" · 三维瀑布图",description,height=6.4)
    ax = fig.add_axes([.055,.108,.882,.682],projection="3d",computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=23,azim=-68)
    ax.set_box_aspect((3.35,1.45,1.45),zoom=1.45)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.set_facecolor((.96,.975,.99,.28))
        axis.pane.set_edgecolor(RULE)
        axis._axinfo["grid"].update(color="#DFE6EB",linewidth=.55)
    n = len(levels)
    x = np.arange(n+1)
    width,depth = .62,.23
    for y,version,color in [(1,"基线",BLUE),(0,"终稿",RED)]:
        values = levels[version].to_numpy(float)
        starts = np.r_[0.,values[:-1],0.]
        ends = np.r_[values,values[-1]]
        order = 3 if y==0 else 1
        for i,(start,end) in enumerate(zip(starts,ends)):
            bottom,h = min(start,end),abs(end-start)
            if h>1.e-12:
                ax.bar3d(i-width/2,y-depth/2,bottom,width,depth,h,color=color,edgecolor=color,
                         linewidth=.45,alpha=.72 if i==n else .43,shade=False,zorder=order)
            else:
                ax.plot([i-width/2,i+width/2],[y-depth/2]*2,[end]*2,color=color,lw=2,zorder=order+1)
            ax.plot([i-width/2,i+width/2],[y-depth/2]*2,[end]*2,color=color,lw=1.65,zorder=order+1)
            if i<n:
                ax.plot([i+width/2,i+1-width/2],[y-depth/2]*2,[end]*2,
                        color=color,lw=1.05,ls="--",zorder=order+1)
    ax.set(xlim=(-.55,n+.55),ylim=(-.35,1.34),zlim=(0,1.10*levels.to_numpy().max()),
           xticks=x,xticklabels=[*levels.index,"整体"],yticks=[0,1],yticklabels=["终稿","基线"])
    ax.tick_params(axis="x",labelsize=9,pad=0)
    ax.tick_params(axis="y",labelsize=9,pad=0)
    ax.tick_params(axis="z",labelsize=9,pad=1)
    # The 3-D viewport is square; long projected layers extend past its 2-D clip box.
    for artist in [*ax.collections,*ax.lines]:
        artist.set_clip_on(False)
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
    fig.text(.066,.774,"高度  /  "+metric,color=SUB,fontsize=10)
    totals=levels.iloc[-1]
    fig.text(.947,.908,"基线  "+f"{totals['基线']:.{digits}f}"+suffix+"     →     终稿  "
             +f"{totals['终稿']:.{digits}f}"+suffix,ha="right",fontsize=13,weight="bold",color=NAVY)
    note(fig,foot)
    return fig


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ref=pd.read_csv(HERE/"waterfall_values.csv").groupby("metric").first()
    records=[]
    with fitz.open() as pdf,ZipFile(OUT/"五张三维瀑布图_高清PNG.zip","w",ZIP_DEFLATED) as archive:
        for i,plate in enumerate(datasets(q3_results()),1):
            title,levels=plate[:2]
            np.testing.assert_allclose(levels.iloc[-1][VERSIONS].to_numpy(),ref.loc[title,["baseline","final"]].to_numpy(float),rtol=0,atol=1e-10)
            records.append(levels.rename_axis("阶段").reset_index().assign(维度=title))
            fig=render(*plate)
            assert len(fig.axes)==1 and fig.axes[0].name=="3d"
            stem=f"{i:02d}_{title}_三维瀑布图"
            single,png=OUT/(stem+".pdf"),OUT/(stem+".png")
            fig.savefig(single,metadata={"Title":stem})
            fig.savefig(png,dpi=450)
            plt.close(fig)
            with fitz.open(single) as page:
                pdf.insert_pdf(page)
            archive.write(png,arcname=png.name)
        pdf.save(OUT/"五维三维瀑布图_审核稿.pdf",garbage=4,deflate=True)
    pd.concat(records).to_csv(HERE/"waterfall_3d_levels.csv",index=False)


if __name__=="__main__":
    main()
