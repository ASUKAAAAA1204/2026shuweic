"""Five single-axis line comparisons and an integrated directional heatmap."""
from zipfile import ZipFile, ZIP_DEFLATED

import fitz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd

from build import CASES, HERE, ROOT, VERSIONS, paired, q3_results
from filled_profiles import perturbation, sites
from plot_baseline_stability_lines import STYLE, legend
from plot_five_dimensions import NAVY, BLUE, RED, BLUSH, PALE, RULE, SUB, INK, canvas, note, style

OUT = ROOT/"figures/final_v1_simple_lines_review"


def comparisons(q):
    a = perturbation(q)
    frame = pd.DataFrame([{m:z[-1] for g,m,x,z in a.profiles if g==group}
                          for group in ["一问","二问","三问"]],
                         index=["一问 · 预测时长","二问 · 观测预算","三问 · 工期倍率"])
    yield ("参数敏感性","指定扰动的损失指数","指数",2,False,frame,frame,frame.mean(),
           "各问基线在指定扰动下的损失设为 100。终稿数值越低，表示该项扰动造成的损失越小。",
           "预测时长 0.2 → 1 秒，观测次数 2 → 0 次，工期倍率 1 → 1.5。三问等权，前两问以固定阻尼为基线，第三问以上一正式版为基线。")
    b = sites()
    frame = pd.DataFrame({m:z for _,m,x,z in b.profiles},index=b.ticklabels)
    splits = pd.DataFrame([frame.iloc[:4].mean(),frame.iloc[4:].mean()],index=["开发四地","测试四地"])
    yield ("跨环境稳定性","地点平均 NADE","NADE",5,False,frame,splits,frame.mean(),
           "第二问在同一地点比较固定阻尼与终稿。两版使用相同的测试窗口和观测预算，NADE 越低越好。",
           "前四地为开发集（212 窗），后四地为既有测试集（390 窗）。横轴为地点编号，整体按八个地点等权计算。")
    idx=q.set_index(["alpha","beta","case_id","seed"])[VERSIONS].sort_index()
    delta=idx.loc[1.5,.5]-idx.loc[1.,.5]
    def widths(part):
        return {m:paired("long",part.sort_index()[m]/100)["halfwidth_pp"] for m in VERSIONS}
    frame=pd.DataFrame([widths(delta.xs(c)) for c in CASES],index=CASES)
    total=pd.Series(widths(delta.groupby(level="seed").mean()))
    yield ("配对不确定性","配对 95% 区间半宽","百分点",3,False,frame,frame,total,
           "第三问将工期倍率从 1 提高至 1.5，比较按时率变化的估计精度。区间半宽越小，估计越精确。",
           "每个案例取同一组 30 个工期种子，配对重抽 10,000 次。整体半宽由三案例等权结果重抽得到。")
    nominal=q.query("alpha==1 and beta==.5")
    assert nominal.groupby("case_id").size().eq(30).all()
    frame=nominal.groupby("case_id")[VERSIONS].agg(lambda x:x.nsmallest(3).mean()).loc[CASES]
    yield ("尾部风险","最差一成平均按时率","%",2,True,frame,frame,frame.mean(),
           "第三问名义工况。每个案例取按时率最低的 3 次，比较这些困难实例的平均加权按时率。越高越好。",
           "每个案例共 30 个官方工期种子，两版分别选取各自的最差一成。整体按 S1、S2、S3 等权计算。")
    frame=pd.read_csv(HERE/"matched_seed_sd.csv").set_index(["alpha","batch"])[VERSIONS]
    frame=frame.loc[[(a,b) for a in (1.,1.5) for b in ("original","replication")]]
    frame.index=["名义 · 首批","名义 · 复验","长工期 · 首批","长工期 · 复验"]
    yield ("内部抽样种子稳定性","实例内标准差","百分点",3,False,frame,frame,frame.mean(),
           "第三问固定任务和实际工期，每个实例只换内部抽样种子。按时率标准差越小，表示结果越稳定。",
           "每组 90 个实例，每例 20 个种子。基线使用上一正式版补测结果，终稿固定 K = 64。长工期下基线两批标准差均为零。")


def line_chart(p):
    title,metric,unit,digits,higher,frame,_,total,description,foot = p
    fig=canvas(title,description)
    legend(fig)
    ax=fig.add_axes([.117,.224,.79,.484])
    style(ax)
    x=np.arange(len(frame))
    for m in VERSIONS:
        ax.plot(x,frame[m],**STYLE[m],ms=6.5,lw=2)
    ax.set(xticks=x,xticklabels=frame.index,xlim=(-.19,len(frame)-.81))
    ax.set_ylabel(metric+(f"（{unit}）" if unit not in ["指数","NADE"] else ""))
    if title=="内部抽样种子稳定性":
        ax.set_ylim(-.23,6.5)
    if title=="跨环境稳定性":
        ax.axvline(3.5,color=RULE,ls=":",lw=1)
    fig.text(.117,.76,f"整体   基线 {total['基线']:.{digits}f}   →   终稿 {total['终稿']:.{digits}f}  {unit}",
             color=NAVY,fontsize=11,weight="bold")
    note(fig,foot)
    return fig


def heatmap(plates):
    fig=canvas("五维稳定性对照","每格按“基线 → 终稿”列出原值。颜色表示该指标下哪一版表现更好，右列为各维度的整体结果。",height=7)
    ax=fig.add_axes([.055,.137,.89,.64])
    ax.set(xlim=(0,1),ylim=(5,0))
    ax.axis("off")
    for i,p in enumerate(plates):
        title,metric,unit,digits,higher,frame,small,total,description,foot=p
        ax.text(0,i+.35,title,va="center",fontsize=11.7,weight="bold",color=NAVY)
        ax.text(0,i+.68,unit+(" ↑" if higher else " ↓"),va="center",fontsize=9.4,color=SUB)
        w=.565/len(small)
        cells=[(.209+j*w,w-.003,s,r) for j,(s,r) in enumerate(small.iterrows())]
        cells.append((.791,.209,"整体",total))
        for left,width,s,r in cells:
            change=(r["终稿"]-r["基线"])*(1 if higher else -1)
            fill=RULE if np.isclose(change,0,atol=1e-12) else BLUSH if change>0 else PALE
            ax.add_patch(Rectangle((left,i+.035),width,.90,facecolor=fill,
                                   edgecolor=NAVY if s=="整体" else "white",lw=1.1))
            ax.text(left+width/2,i+.32,s,ha="center",va="center",fontsize=9.3,color=SUB)
            ax.text(left+width/2,i+.65,f"{r['基线']:.{digits}f} → {r['终稿']:.{digits}f}",
                    ha="center",va="center",fontsize=10.2,color=INK,weight="bold" if s=="整体" else "normal")
    fig.legend(handles=[Patch(facecolor=BLUSH,edgecolor=RED,label="终稿占优"),
                        Patch(facecolor=PALE,edgecolor=BLUE,label="基线占优")],
               loc="center right",bbox_to_anchor=(.951,.925),ncol=2)
    note(fig,"↑ 表示越高越好，↓ 表示越低越好。整体区间半宽由配对重抽计算，其余维度按图示分组等权汇总。")
    return fig


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plates=list(comparisons(q3_results()))
    refs=pd.read_csv(HERE/"waterfall_values.csv").groupby("metric").first()
    for p in plates:
        np.testing.assert_allclose(p[7][VERSIONS].to_numpy(),refs.loc[p[0],["baseline","final"]].to_numpy(float),atol=1e-10,rtol=0)
    charts=[(p[0],line_chart(p)) for p in plates]+[("五维整合热力图",heatmap(plates))]
    with fitz.open() as book,ZipFile(OUT/"六张高清PNG.zip","w",ZIP_DEFLATED) as archive:
        for i,(name,fig) in enumerate(charts,1):
            stem=f"{i:02d}_{name}"
            pdf,png=OUT/(stem+".pdf"),OUT/(stem+".png")
            fig.savefig(pdf,metadata={"Title":name})
            fig.savefig(png,dpi=450)
            plt.close(fig)
            with fitz.open(pdf) as page:
                book.insert_pdf(page)
            archive.write(png,arcname=png.name)
        book.save(OUT/"五维折线图与整合热力图.pdf",garbage=4,deflate=True)


if __name__=="__main__":
    main()
