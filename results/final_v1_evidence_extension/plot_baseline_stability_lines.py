"""Five stability comparisons on the release's matched evaluation support."""
import fitz
from zipfile import ZipFile, ZIP_DEFLATED
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from plot_final_baseline import RESULTS, SPACE, load_comparison
from plot_five_dimensions import (
    BLUE, DIVERGE, EXT, INK, NAVY, RED, ROOT, RULE, SUB, canvas, note, style,
)
from schedule_statistics import paired

OUT = ROOT / "figures/final_v1_baseline_stability_lines"
RATE = "weighted_on_time_rate"
KEY = ["case_id", "seed", "alpha", "beta"]
VERSIONS = ["基线", "终稿"]
CASES = ["S1", "S2", "S3"]
STYLE = {
    "基线": dict(color=BLUE, marker="o", mfc="white", ls="--"),
    "终稿": dict(color=RED, marker="D", ls="-"),
}


def export(fig, name):
    fig.savefig(OUT / f"{name}.pdf", metadata={"Title": name})
    fig.savefig(OUT / f"{name}.png", dpi=450)
    plt.close(fig)


def panels(fig, titles):
    grid = fig.add_gridspec(1, len(titles), left=.085, right=.964, bottom=.235,
                            top=.686, wspace=.39 if len(titles) == 3 else .29)
    axes = []
    for i, title in enumerate(titles):
        ax = fig.add_subplot(grid[0, i])
        style(ax)
        ax.set_title(title, fontsize=12, color=NAVY, weight="bold", pad=12, loc="left")
        axes.append(ax)
    return axes


def lines(ax, frame):
    for model in VERSIONS:
        ax.plot(frame.index, frame[model], lw=1.9, ms=5.2, label=model, **STYLE[model])
    ax.set_xticks(frame.index)


def legend(fig):
    handles = [Line2D([], [], lw=1.9, ms=5.5, label=m, **STYLE[m]) for m in VERSIONS]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(.958,.919),
               ncol=2, fontsize=10)


def q3_results():
    load_comparison()
    ref = pd.read_csv(SPACE / "正式方案/reference/q3_official_B_and_previous.csv.gz")
    new = pd.read_csv(RESULTS / "q3/b_k64/trials.csv.gz").set_index(KEY).sort_index()
    old = ref.query("method=='previous_local'").set_index(KEY).sort_index()
    frozen = ref.query("method=='B_K64'").set_index(KEY).sort_index()
    assert new.index.equals(old.index) and new.index.equals(frozen.index)
    np.testing.assert_allclose(new[RATE], frozen[RATE], rtol=0, atol=1e-12)
    for data in (new, old):
        assert data.valid_protocol.all() and data.invalid_actions.eq(0).all()
    return pd.DataFrame({"基线": 100*old[RATE], "终稿": 100*new[RATE]}).reset_index()


def cell(label, values, digits=2):
    return label, float(values["基线"]), float(values["终稿"]), digits


def sensitivity(q):
    fig = canvas("参数敏感性", "第一、二问取同一批 390 个既有测试窗。第三问取三个案例各 30 个官方工期种子，β 固定为 0.5 秒。")
    legend(fig)
    axes = panels(fig, ["第一问 · 预测时长", "第二问 · 观测预算", "第三问 · 工期倍率"])
    cells = []
    for ax, file, index, xlabel, label in zip(
        axes, ["q1_horizons.csv","q2_cost.csv"], ["horizon","budget"],
        ["预测时长（秒）","允许请求次数"], ["一问 · 0.2 → 1 秒","二问 · 2 → 0 次请求"],
    ):
        table = pd.read_csv(RESULTS / file).query("split=='test'").pivot(
            index=index, columns="model", values="nade").rename(columns={"original":"基线","B":"终稿"})
        lines(ax, table)
        ax.set(xlabel=xlabel, ylabel="归一化位移误差（NADE）")
        cells.append(cell(label, (table.iloc[-1]-table.iloc[0]).abs(), 4))
    rate = q.query("beta==.5").groupby("alpha")[VERSIONS].mean()
    lines(axes[-1], rate)
    axes[-1].set(xlabel="工期倍率 α", ylabel="加权按时率（%）")
    cells.append(cell("三问 · α 1 → 1.5", rate.loc[1]-rate.loc[1.5]))
    note(fig, "误差越低、按时率越高越好。热力图摘录端点变化，分别表示时长延伸、撤去两次观测、工期由名义档延长带来的损失。")
    export(fig, "01_参数敏感性")
    return cells


def environments():
    raw = pd.read_csv(RESULTS / "q2_windows.csv.gz").query("split in ['validation','test']")
    assert raw.requests.eq(2).all() and np.allclose(raw.waiting_s,.2)
    keys = ["split","environment_id","episode_uuid","side","window_id"]
    pairs = raw.pivot(index=keys, columns="model", values="nade")[["original","B"]]
    assert pairs.shape == (602,2) and pairs.notna().all().all()
    sites = (pairs.groupby(level=keys[:-1]).mean().groupby(level=keys[:-2]).mean()
             .groupby(level=keys[:2]).mean().rename(columns={"original":"基线","B":"终稿"}))
    names = {"restaurant":"餐厅","supermarket":"超市","station":"快递站"}
    fig = canvas("跨环境稳定性", "第二问在每个地点使用完全相同的窗口与两次观测。地点内依次对窗口、手侧和记录取平均。")
    legend(fig)
    axes = panels(fig, ["开发集 · 4 地 / 212 窗", "既有测试集 · 4 地 / 390 窗"])
    cells = []
    for ax, split in zip(axes, ["validation","test"]):
        frame = sites.loc[split].copy()
        labels = [names[s.split("_")[-2]]+" "+s.split("_")[-1] for s in frame.index]
        frame.index = range(len(labels))
        lines(ax, frame)
        ax.set(xticks=frame.index, xticklabels=labels, ylabel="归一化位移误差（NADE）")
        if split=="test":
            cells = [cell(lab, row, 4) for lab, (_,row) in zip(labels,frame.iterrows())]
    bounds = [a.get_ylim() for a in axes]
    for ax in axes:
        ax.set_ylim(min(x[0] for x in bounds),max(x[1] for x in bounds))
    note(fig, "横轴列出地点，连线便于对照各地表现。开发和既有测试地点均已参与此前审核，结论限于这八个地点。")
    export(fig, "02_跨环境稳定性")
    return cells


def uncertainty(q):
    delta = (q.query("alpha==1.5 and beta==.5").set_index(["case_id","seed"])[VERSIONS]
             - q.query("alpha==1 and beta==.5").set_index(["case_id","seed"])[VERSIONS])
    assert delta.shape==(90,2) and delta.notna().all().all()
    fig = canvas("配对不确定性", "第三问比较长工期与名义工期的按时率差。每次重抽同一组工期种子，保留两版、两档和三个案例的对应关系。")
    legend(fig)
    axes = panels(fig, ["三案例等权 · 工期延长后的按时率变化", "同一配对差的 95% 区间半宽"])
    sizes = [10,15,20,25,30]
    overall = delta.groupby(level="seed").mean().sort_index()
    records = []
    for n in sizes:
        for model in VERSIONS:
            records.append(dict(model=model, **paired("long",overall[model].iloc[:n]/100)))
    stats = pd.DataFrame(records)
    for key, ax in zip(["delta_pp","halfwidth_pp"], axes):
        lines(ax,stats.pivot(index="n",columns="model",values=key))
        ax.set(xlabel="累计纳入的官方工期种子数", ylabel="百分点")
    for model in VERSIONS:
        g = stats[stats.model==model]
        axes[0].fill_between(g.n.to_numpy(),g.lower_pp.to_numpy(),g.upper_pp.to_numpy(),
                             color=STYLE[model]["color"],alpha=.13,linewidth=0)
    cells = []
    for case in CASES+["三案例等权"]:
        part = overall if case=="三案例等权" else delta.xs(case).sort_index()
        cells.append(cell(case,{m:paired("long",part[m]/100)["halfwidth_pp"] for m in VERSIONS}))
    note(fig, "按种子编号取嵌套前缀，每档有放回重抽 10,000 次。阴影为百分位区间；区间半宽衡量估计精度，单位为百分点。")
    export(fig, "03_配对不确定性")
    return cells


def tails(q):
    fig = canvas("尾部风险", "第三问名义工况。每个案例共享 30 个官方工期种子，各版本按加权按时率从低到高排序。")
    legend(fig)
    axes = panels(fig, CASES)
    cells = []
    for ax, case in zip(axes, CASES):
        part = q.query("case_id==@case and alpha==1 and beta==.5")
        k = np.arange(1,31)
        frame = pd.DataFrame({m:np.sort(part[m]).cumsum()/k for m in VERSIONS}, index=k)
        lines(ax, frame)
        for line in ax.lines:
            line.set_markevery([0,2,9,19,29])
        ax.axvline(3,color=SUB,lw=.8,ls=":",zorder=0)
        ax.set(xticks=[1,3,10,20,30], xlabel="纳入最差的 k 次", ylabel="平均加权按时率（%）", ylim=(42,102))
        cells.append(cell(case,frame.loc[3]))
        low = frame.min().min()
        if low<42:
            for a in axes:
                a.set_ylim(min(low-2,a.get_ylim()[0]),102)
    lo = min(a.get_ylim()[0] for a in axes)
    for ax in axes:
        ax.set_ylim(lo,102)
    note(fig, "虚线 k = 3 对应最差 10%，k = 30 对应全部实例。两版的困难实例可不同；同一排序位置表示相同尾部比例。")
    export(fig, "04_尾部风险")
    return cells


def seeds(q):
    wf = EXT.parent/"final_v1_waterfall"
    old = pd.concat([pd.read_csv(wf/"previous_nominal_trials.csv.gz"),
                     pd.read_csv(EXT.parent/"q3_baseline_seed_control/trials.csv.gz").query("alpha==1.5 and beta==.5")])
    new = pd.concat([pd.read_csv(path).assign(batch=batch) for batch,path in [
        ("original",EXT.parent/"final_v1_stability/seed_trials.csv.gz"),
        ("replication",EXT/"internal_replication.csv.gz")]]).query("alpha in [1.,1.5] and beta==.5")
    keys=["batch",*KEY,"internal_seed"]
    a,b=[d.set_index(keys).sort_index() for d in (old,new)]
    assert a.index.equals(b.index) and a.index.is_unique and len(a)==7200 and a.instance_hash.equals(b.instance_hash)
    batches=pd.read_csv(wf/"matched_seed_sd.csv").set_index(["batch","alpha"])
    fig=canvas("内部抽样种子稳定性","第三问固定任务与实际工期，终稿固定 K = 64，仅替换内部种子。每个点平均同一组 90 个实例。")
    axes=panels(fig,["名义工期 · α = 1","长工期 · α = 1.5"])
    for model,frame,color in zip(VERSIONS,(old,new),(BLUE,RED)):
        samples=frame.groupby(["batch",*KEY])[RATE]
        assert samples.size().eq(20).all()
        sd=samples.agg(lambda s:(s-s.iloc[0]).std(ddof=1)*100).groupby(level=["batch","alpha"]).mean()
        np.testing.assert_allclose(sd,batches[model].reindex(sd.index),atol=1e-10)
        blocks=frame.groupby(["alpha","batch","internal_seed"])[RATE]
        assert blocks.size().eq(90).all()
        means=100*blocks.mean()
        for ax,alpha in zip(axes,[1.,1.5]):
            official=q.query("alpha==@alpha and beta==.5")[model].mean()
            np.testing.assert_allclose(means.loc[(alpha,"original",20261122)],official)
            for batch,marker,ls,label in [("original","o","-","首批"),("replication","D",":","复验")]:
                values=means.loc[(alpha,batch)].sort_values()
                assert len(values)==20
                ax.plot(np.arange(1,21),values,ls=ls,lw=1.7,marker=marker,ms=4.8 if batch=="original" else 2.8,
                        mfc="white" if batch=="original" else color,color=color,label=model+" · "+label)
            ax.set(xlabel="批内种子按平均按时率排序的位置",ylabel="加权按时率（%）",xticks=[1,5,10,15,20])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="center right",bbox_to_anchor=(.961,.915),ncol=4,fontsize=9.4)
    note(fig,"基线为上一正式版。各版各批独立排序，长工期基线两批曲线重合。热力图报告每个固定实例跨 20 个种子的标准差均值。")
    export(fig,"05_内部抽样种子稳定性")
    return [cell(cond+" · "+label,batches.loc[(batch,alpha)],3) for alpha,cond in [(1.,"名义"),(1.5,"长工期")]
            for batch,label in [("original","首批"),("replication","复验批")]]


def heatmap(rows):
    fig = canvas("五维稳定性对照", "各格依次列出基线值和终稿值。颜色表示终稿相对基线的改善率；格内保留原始单位。",height=7.3)
    labels = [
        ("参数敏感性","端点损失 ↓","前两问 NADE / 三问百分点",False),
        ("跨环境稳定性","测试地点误差 ↓","第二问 · NADE",False),
        ("配对不确定性","95% 区间半宽 ↓","第三问 · 百分点",False),
        ("尾部风险","最差一成按时率 ↑","第三问 · %",True),
        ("内部种子稳定性","实例内标准差 ↓","第三问 · 百分点",False),
    ]
    ax = fig.add_axes([.052,.208,.899,.61])
    ax.set(xlim=(0,1),ylim=(5,0))
    ax.axis("off")
    cap = 30
    norm = Normalize(-cap,cap,clip=True)
    for i, ((title,stat,unit,higher),row) in enumerate(zip(labels,rows)):
        ax.text(0,i+.29,title,fontsize=12.2,color=NAVY,weight="bold",va="center")
        ax.text(0,i+.56,stat,fontsize=9.4,color=INK,va="center")
        ax.text(0,i+.79,unit,fontsize=8.1,color=SUB,va="center")
        w = .777/len(row)
        for j,(tag,old,new,digits) in enumerate(row):
            x = .223+j*w
            valid = np.isfinite(old) and old>0
            gain = (new-old)*(1 if higher else -1)/old*100 if valid else np.nan
            shade = DIVERGE(norm(gain)) if valid else (.925,.937,.947,1.)
            color = "white" if np.dot(shade[:3],[.213,.715,.072])<.58 else INK
            ax.add_patch(Rectangle((x,i+.025),w-.003,.935,facecolor=shade,edgecolor=RULE,lw=.65))
            mid = x+(w-.003)/2
            ax.text(mid,i+.24,tag,ha="center",va="center",fontsize=9.4,color=color)
            assert np.isfinite([old,new]).all()
            old_label = f"{round(old,10):.{digits}f}"
            ax.text(mid,i+.51,f"{old_label} → {round(new,10):.{digits}f}",ha="center",va="center",
                    fontsize=11.1,color=color,weight="semibold")
            ax.text(mid,i+.78,(f"{gain:+.1f}%".replace("-","−") if valid else f"差 {new-old:+.3f} 个百分点"),
                    ha="center",va="center",fontsize=9,color=color)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DIVERGE),
                       cax=fig.add_axes([.31,.15,.45,.021]),orientation="horizontal",extend="both")
    bar.set_ticks([-30,-15,0,15,30],labels=["−30%","−15%","0","+15%","+30%"])
    bar.ax.tick_params(labelsize=8.5)
    fig.text(.296,.16,"基线占优",ha="right",va="center",color=BLUE,fontsize=9.5)
    fig.text(.775,.16,"终稿占优",va="center",color=RED,fontsize=9.5)
    note(fig,"改善率以基线值为分母，色阶截于 ±30%。灰格的实测基线为零，显示两版标准差的绝对差（百分点）。")
    export(fig,"06_五维整合热力图")


def radar(rows):
    names=["参数敏感性","跨环境稳定性","配对不确定性","尾部风险","内部种子稳定性"]
    metrics=["扰动损失指数 ↓","八地等权 NADE ↓","区间半宽（百分点）↓","尾部按时率（%）↑","实例内标准差（百分点）↓"]
    ref=pd.read_csv(EXT.parent/"final_v1_waterfall/waterfall_values.csv").groupby("metric").first()
    raw=ref.loc[names[:-1]+["内部抽样种子稳定性"],["baseline","final"]].to_numpy(float)
    np.testing.assert_allclose(raw[0],[100,100*np.mean([new/old for _,old,new,_ in rows[0]])])
    np.testing.assert_allclose(raw[2],rows[2][-1][1:3])
    for i in (3,4):
        np.testing.assert_allclose(raw[i],np.mean([row[1:3] for row in rows[i]],axis=0))
    assert np.isfinite(raw).all() and (raw>0).all()
    scores=100*raw.min(axis=1,keepdims=True)/raw
    scores[3]=100*raw[3]/raw[3].max()
    fig=canvas("基线与终稿的五维对比","每轴以两版中表现较好者为 100，越靠外表示该维度表现越好。右表列出五项整体指标的原值。",height=6.4)
    legend(fig)
    ax=fig.add_axes([.072,.185,.45,.525],projection="polar")
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    angles=np.linspace(0,2*np.pi,5,endpoint=False)
    closed=np.r_[angles,angles[0]]
    ax.set(xticks=angles,xticklabels=names,ylim=(0,105),yticks=[25,50,75,100])
    ax.tick_params(axis="x",pad=8,labelsize=10)
    ax.tick_params(axis="y",labelsize=8,colors=SUB)
    ax.spines["polar"].set_visible(False)
    ax.grid(color=RULE)
    for j,(m,col) in enumerate(zip(VERSIONS,[BLUE,RED])):
        y=np.r_[scores[:,j],scores[0,j]]
        ax.fill(closed,y,color=col,alpha=.08)
        ax.plot(closed,y,**STYLE[m],lw=1.8,ms=5)
    for x,label,align in [(.55,"整体指标","left"),(.835,"基线","right"),(.945,"终稿","right")]:
        fig.text(x,.696,label,ha=align,color=SUB,fontsize=10)
    for i,(label,digits) in enumerate(zip(metrics,[2,5,3,2,3])):
        y=.625-i*.076
        fig.text(.55,y,label,fontsize=9.6)
        for j,(x,color) in enumerate([(.835,BLUE),(.945,RED)]):
            fig.text(x,y,f"{raw[i,j]:.{digits}f}",ha="right",color=color,fontsize=10.6)
    note(fig,"↑ 项取本版值 ÷ 较大值 × 100，↓ 项取较小值 ÷ 本版值 × 100。五轴分别比较，不计算综合分。")
    export(fig,"07_基线与终稿五维对比雷达图")


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    q = q3_results()
    rows = [sensitivity(q),environments(),uncertainty(q),tails(q),seeds(q)]
    heatmap(rows)
    radar(rows)
    bundle = fitz.open()
    for pdf in sorted(OUT.glob("0?_*.pdf")):
        with fitz.open(pdf) as page:
            bundle.insert_pdf(page)
    assert len(bundle)==7
    temp=OUT/"终稿与基线_五维图册.partial.pdf"
    bundle.save(temp,garbage=4,deflate=True)
    bundle.close()
    temp.replace(OUT/"终稿与基线_五维图册.pdf")
    images=sorted(OUT.glob("0?_*.png"))
    for n,filename in [(6,"六张高清PNG.zip"),(7,"七张高清PNG.zip")]:
        with ZipFile(OUT/filename,"w",ZIP_DEFLATED) as archive:
            for file in images[:n]:
                archive.write(file,arcname=file.name)


if __name__=="__main__":
    main()
