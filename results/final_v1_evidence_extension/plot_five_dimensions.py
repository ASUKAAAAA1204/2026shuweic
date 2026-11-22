"""Render the five review figures from archived v1.0.0-final experiment outputs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

EXT = Path(__file__).resolve().parent
ROOT = EXT.parents[1]
SENS = EXT.parent / "final_v1_sensitivity"
TAIL = EXT.parent / "final_v1_s2_tail"
TAG = ROOT / "tmp/final_v1_20261122"
OUT = ROOT / "figures/final_v1_five_dimensions"

NAVY, BLUE, MID, PALE = "#1F4E89", "#2A73BA", "#87AEE6", "#E8EEF8"
RED, PINK, BLUSH = "#CE4844", "#F7A9A8", "#FDE6E6"
INK, SUB, RULE = "#20364D", "#4D6175", "#DCE4ED"
DIVERGE = LinearSegmentedColormap.from_list("deltas", [NAVY, MID, "#FFFFFF", PINK, RED], N=257)
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10,
                     "text.color": INK, "axes.labelcolor": SUB, "axes.labelsize": 10,
                     "xtick.color": SUB, "ytick.color": SUB, "axes.edgecolor": "#A4B2BF",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "axes.unicode_minus": True,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})


def canvas(title, subtitle, height=5.8):
    fig = plt.figure(figsize=(11.2, height), facecolor="white")
    fig.text(.064, 1-.32/height, title, fontsize=19, weight="bold", color=NAVY, va="top")
    fig.text(.065, 1-.77/height, subtitle, fontsize=10, color=SUB, va="top")
    rule(fig, 1-1.13/height)
    return fig


def rule(fig, y):
    fig.add_artist(Line2D([.065,.945], [y,y], transform=fig.transFigure, color=RULE, lw=.8))


def note(fig, text):
    rule(fig,.082)
    fig.text(.065,.045,text,color=SUB,fontsize=9,va="center")


def heading(fig, x, y, text):
    fig.text(x,y,text,fontsize=12,color=NAVY,weight="bold",va="center")


def style(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis,color=RULE,lw=.65)
    ax.tick_params(length=2.8,width=.6)
    ax.spines["left"].set_linewidth(.7)
    ax.spines["bottom"].set_linewidth(.7)


def export(fig, name):
    OUT.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUT / (name+".pdf"),metadata={"Title":name,"Subject":"v1.0.0-final"})
    fig.savefig(OUT / (name+".png"), dpi=450)
    plt.close(fig)


def draw_matrix(fig, rect, frame, rows, cols, limit, percent=False, invert=False, nominal=None):
    ax=fig.add_axes(rect)
    cmap=DIVERGE.reversed() if invert else DIVERGE
    norm=Normalize(-limit,limit)
    for i,row in enumerate(frame.to_numpy()):
        for j,value in enumerate(row):
            shade=cmap(norm(value))
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=shade,edgecolor="#D1DCE7",lw=.7))
            label=f"{value:+.2f}" if abs(value)>.00001 else "0.00"
            if percent: label+="%"
            light=np.dot(shade[:3],[.2126,.7152,.0722])>.55
            ax.text(j,i,label.replace("-","−"),ha="center",va="center",color=INK if light else "white",
                    fontsize=10.6,weight="semibold")
    if nominal is not None:
        r,c=nominal
        ax.add_patch(Rectangle((c-.47,r-.46),.94,.92,fill=False,edgecolor=NAVY,lw=1.8))
    ax.set(xlim=(-.5,len(cols)-.5),ylim=(len(rows)-.5,-.5),xticks=range(len(cols)),xticklabels=cols,
           yticks=range(len(rows)),yticklabels=rows)
    ax.tick_params(length=0,pad=5,labelsize=9)
    for edge in ax.spines.values(): edge.set_visible(False)


def series(ax, x, frame, labels):
    for (_,values), label, color, marker, shape in zip(frame.iterrows(),labels,[NAVY,BLUE,RED],
                                                        ["o","s","D"],["-","--",":"]):
        ax.plot(x,values,marker=marker,ls=shape,color=color,label=label,ms=5,mfc="white",lw=1.7,mew=1.3)
    style(ax)
    ax.legend(loc="upper left",bbox_to_anchor=(0,1.16),borderaxespad=0,ncol=3,fontsize=8.2,
              columnspacing=.9,handlelength=1.35,handletextpad=.4)


def sensitivity():
    hist=pd.read_csv(SENS/"q1_history.csv").pivot(index="history",columns="horizon",values="nade").loc[
        ["full","0.75s","0.30s"]]
    src=TAG/"C题工作区/独立方案B_旋转模型/第二问/delivery/evidence/position_sensitivity/summary.csv"
    delay=pd.read_csv(src).query("split=='validation' and method=='rotation_switch'").pivot(
        index="horizon",columns="delay",values="nade").sort_index()
    np.testing.assert_allclose(delay.loc[1.],pd.read_csv(SENS/"q2/summary.csv").nade)
    grid=100*pd.read_csv(EXT/"grid_means.csv").pivot(index="alpha",columns="beta",values="overall")
    baseline=grid.loc[1.,.5]
    fig=canvas("三问终稿模型的敏感性","左列为冻结模型实测响应，右列为相对名义条件的变化。红色表示性能变差，蓝色表示改善。",9.8)
    for y, label, subtitle in [
        (.851,"第一问   可见历史与质心预测","212 个开发窗"),
        (.589,"第二问   回传延迟与遮挡预测","同一批开发窗 · 计划请求时刻 0.3、0.6 秒"),
        (.325,"第三问   服务时长与按时交付","3 个案例等权 · 300 个配对工期种子 · K = 64")]:
        heading(fig,.065,y,label)
        fig.text(.942,y,subtitle,va="center",ha="right",fontsize=9,color=SUB)
    for y in [.62,.357]: rule(fig,y)
    for y in [.851,.589,.325]:
        fig.text(.582,y-.043,{.851:"NADE 相对完整历史的变化（%）",.589:"NADE 相对延迟 0.10 秒的变化（%）",
                              .325:"按时率相对名义工期的变化（百分点）"}[y],fontsize=9.8,weight="semibold")
    xvals=hist.columns.to_numpy(float)
    ax=fig.add_axes([.11,.686,.315,.112])
    series(ax,xvals,hist,["完整历史","0.75 秒","0.30 秒"])
    ax.set(xticks=xvals,xlabel="预测时长（秒）",ylabel="NADE",ylim=(0,.212))
    change=100*(hist.div(hist.loc["full"],axis=1)-1)
    draw_matrix(fig,[.591,.689,.346,.095],change,["完整历史","0.75 秒","0.30 秒"],["预测 0.2 秒","0.6 秒","1.0 秒"],3,True)
    ax=fig.add_axes([.11,.425,.315,.112])
    series(ax,delay.columns,delay,["遮挡 0.2 秒","0.6 秒","1.0 秒"])
    ax.set(xticks=[0,.1,.2],xticklabels=["0","0.10","0.20"],xlabel="单次回传延迟（秒）",ylabel="NADE",ylim=(0,.125))
    change=100*(delay.div(delay[.1],axis=0)-1)
    draw_matrix(fig,[.591,.428,.346,.095],change,["遮挡 0.2 秒","0.6 秒","1.0 秒"],["延迟 0 秒","0.10 秒","0.20 秒"],35,True)
    ax=fig.add_axes([.11,.16,.315,.117])
    series(ax,grid.columns,grid,["α = 0.75","α = 1.00","α = 1.50"])
    ax.scatter([.5],[baseline],s=95,facecolors="none",edgecolors=NAVY,lw=1.2,zorder=5)
    ax.set(xticks=[0,.5,1],xlabel="固定附加时长 β（秒）",ylabel="加权按时率（%）",ylim=(62,103),yticks=[65,75,85,95])
    draw_matrix(fig,[.591,.161,.346,.099],grid-baseline,["α = 0.75","α = 1.00","α = 1.50"],
                ["β = 0 秒","0.5 秒","1.0 秒"],22,invert=True,nominal=(1,1))
    note(fig,f"NADE 为归一化平均位移误差。第三问描边处为名义工期，按时率 {baseline:.2f}%；相同 β 下，α 升至 1.50 时为 {grid.loc[1.5,.5]:.2f}%。")
    export(fig,"01_敏感性")


def environments():
    data=pd.read_csv(EXT/"spaced_environments.csv").sort_values(["split","environment_id"],ascending=[False,True])
    np.testing.assert_allclose(data.change_pct,100*(data.nade_02/data.nade_01-1))
    assert len(data)==8 and data.n.sum()==308
    fig=canvas("八个已审核地点的延迟敏感性","第二问 · 可见历史 0.30 秒 · 同窗比较单次回传延迟 0.10 秒与 0.20 秒",6.05)
    heading(fig,.07,.752,"各地点在两档延迟下的 NADE")
    heading(fig,.555,.752,"同一地点的误差增幅")
    ax=fig.add_axes([.105,.217,.33,.458])
    ax.set_aspect("equal",adjustable="box")
    limits=[.033,.166]
    ax.plot(limits,limits,ls="--",color="#9CAABC",lw=1)
    ax.text(.131,.124,"等值线",color=SUB,fontsize=8,rotation=45)
    axis_ticks=[.04,.08,.12,.16]
    ax.set(xlim=limits,ylim=limits,xticks=axis_ticks,yticks=axis_ticks,xlabel="延迟 0.10 秒的 NADE",ylabel="延迟 0.20 秒的 NADE")
    style(ax,"both")
    right=fig.add_axes([.665,.217,.219,.458])
    style(right,"x")
    names=[]; ticks=[]; numbers=[]
    positions={"0011":(.091,.145),"0041":(.088,.157),"0078":(.066,.052),"0159":(.135,.158),
               "0004":(.126,.130),"0022":(.060,.087),"0074":(.062,.118),"0147":(.106,.100)}
    for k,row in enumerate(data.itertuples()):
        code=row.environment_id[-4:]
        place="超市" if "supermarket" in row.environment_id else ("餐厅" if "restaurant" in row.environment_id else "快递站")
        name=f"{place} {code}"
        color,mark=(BLUE,"o") if row.split=="validation" else (RED,"D")
        ax.scatter(row.nade_01,row.nade_02,s=38,marker=mark,color=color,zorder=4,edgecolor="white",lw=.7)
        ax.annotate(code,(row.nade_01,row.nade_02),xytext=positions[code],ha="center",va="center",fontsize=8.2,
                    color=color,arrowprops=dict(arrowstyle="-",lw=.55,color=color,shrinkA=1,shrinkB=4))
        y=7-k+(.32 if k<4 else 0)
        right.hlines(y,0,row.change_pct,color=PALE,lw=2)
        right.scatter(row.change_pct,y,marker=mark,s=40,color=color,zorder=3)
        right.annotate(f"{row.change_pct:.2f}%",(row.change_pct,y),xytext=(6,0),textcoords="offset points",va="center",fontsize=8.3)
        names.append(name); ticks.append(y); numbers.append(row.n)
    right.set(xlim=(0,47),xticks=[0,10,20,30,40],ylim=(-.45,7.8),yticks=ticks,yticklabels=names,xlabel="NADE 相对增幅（%）")
    right.spines["left"].set_visible(False);right.tick_params(axis="y",length=0)
    for y,n in zip(ticks,numbers):
        right.text(1.15,y,str(n),transform=right.get_yaxis_transform(),ha="right",va="center",color=SUB,fontsize=9)
    right.text(1.15,1.04,"窗数",transform=right.transAxes,ha="right",fontsize=8.6,color=SUB)
    markers=[Line2D([],[],marker=m,ls="",color=c,label=l) for m,c,l in [("o",BLUE,"开发地点"),("D",RED,"既有测试地点")]]
    fig.legend(handles=markers,loc="upper left",bbox_to_anchor=(.066,.736),ncol=2,fontsize=9,handletextpad=.15)
    note(fig,"同一记录、同一只手的相邻保留窗至少间隔 2 秒。开发 110 窗，既有测试 198 窗；八个地点的误差均随延迟增加。")
    export(fig,"02_跨环境稳定性")


def uncertainty():
    official=pd.read_csv(SENS/"q3_mapping.csv").query("alpha==1.5 and beta==.5").iloc[0]
    extra=pd.read_csv(EXT/"paired_grid.csv").query("contrast=='long-minus-nominal'").sort_values("n")
    rows=[(official.delta_pp,official.low_pp,official.high_pp)]+list(extra[["delta_pp","lower_pp","upper_pp"]].itertuples(index=False,name=None))
    fig=canvas("长工期使平均按时率下降多少？","第三问 · 长工期减名义工期 · β 固定为 0.5 秒，α 从 1.00 增至 1.50",5.0)
    ax=fig.add_axes([.219,.263,.373,.366]);style(ax,"x")
    ax.set(ylim=(-.5,2.5),yticks=[2,1,0],yticklabels=["官方 30 个","扩展前 30 个","扩展全部 300 个"],
           xlim=(-22.5,-14.9),xticks=[-22,-20,-18,-16],xlabel="平均加权按时率差（百分点）")
    ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0,pad=9)
    for pos,txt in [(.075,"工期种子"),(.222,"配对 95% 区间"),(.626,"点估计［95% 区间］"),(.906,"半宽")]:
        fig.text(pos,.671,txt,fontsize=10.4,color=NAVY,weight="bold")
    for i,(v,lo,hi) in enumerate(rows):
        y=2-i; color=RED if i==2 else BLUE
        ax.hlines(y,lo,hi,color=color,lw=2.2)
        ax.plot([lo,hi],[y,y],"|",color=color,ms=9)
        ax.scatter(v,y,s=57,color=color,zorder=4,marker="D" if i==2 else "o")
        yf=.263+.366*(y+.5)/3
        fig.text(.626,yf,f"{v:.2f}  [{lo:.2f}, {hi:.2f}]".replace("-","−"),va="center",color=color,fontsize=11)
        fig.text(.937,yf,f"{(hi-lo)/2:.2f}",ha="right",va="center",fontsize=11,weight="bold",color=color)
    fig.text(.075,.134,"扩展批次的配对区间半宽从 2.33 缩至 0.69 个百分点，达到预设的 ≤2 个百分点。",fontsize=10.5,weight="semibold")
    note(fig,"按工期种子成组重抽 10,000 次，保留三个案例的配对。扩展前 30 个包含在全部 300 个中；区间以现有案例和工期池为条件。")
    export(fig,"03_配对不确定性")


def tail():
    data=pd.read_csv(TAIL/"trials.csv.gz");summary=pd.read_csv(TAIL/"batches.csv").set_index("batch")
    old=pd.read_csv(EXT/"tail_replication.csv").query("case_id=='S2' and alpha==1").iloc[0]
    fig=canvas("S2 名义工况的尾部风险","第三问 · 两个独立工期种子批次，每批 300 次 · 固定任务、内部种子和 K = 64",6.2)
    heading(fig,.073,.756,"按批内排名观察最低一成")
    heading(fig,.702,.756,"两次尾部复核")
    for batch,bottom in [("A",.476),("B",.212)]:
        g=data[data.batch==batch].sort_values("rank")
        rank=g["rank"].to_numpy(); pct=100*g.weighted_on_time_rate.to_numpy()
        assert np.array_equal(rank,np.arange(1,301)) and np.all(np.diff(pct)>=0)
        cutoff=rank<=30; observed=pct[cutoff].mean()
        np.testing.assert_allclose(observed,summary.at[batch,"tail_pct"])
        ax=fig.add_axes([.106,bottom,.508,.19]);style(ax)
        ax.axvspan(.5,30.5,color=BLUSH)
        ax.scatter(rank[~cutoff],pct[~cutoff],color=MID,s=8,lw=0)
        ax.scatter(rank[cutoff],pct[cutoff],color=RED,s=10,lw=0)
        ax.set(xlim=(-4,305),ylim=(min(45,100*data.weighted_on_time_rate.min()-3),104),
               yticks=[50,75,100],xticks=[1,30,100,200,300],ylabel="按时率（%）")
        ax.text(0,1.095,f"{batch} 批",transform=ax.transAxes,weight="bold",color=NAVY,fontsize=10.5)
        ax.text(.23,1.095,f"最低 30 次均值   {observed:.2f}%",transform=ax.transAxes,color=RED,fontsize=10.5)
        if batch=="A": ax.tick_params(labelbottom=False)
        else: ax.set_xlabel("批内加权按时率排名（由低到高）")
    gap=abs(summary.at["A","tail_pct"]-summary.at["B","tail_pct"])
    for y,label,v,desc,col in [(.644,"首轮分半",old.gap_pp,"每半 150 次，各取最低 15 次",RED),
                                (.398,"独立两批",gap,"每批 300 次，各取最低 30 次",NAVY)]:
        fig.text(.702,y,label,weight="bold",fontsize=11)
        fig.text(.702,y-.072,f"{v:.2f}",fontsize=23,color=col,weight="bold")
        fig.text(.812,y-.070,"个百分点",fontsize=10,color=SUB)
        fig.text(.702,y-.132,desc,fontsize=9,color=SUB)
    fig.text(.702,.174,"预设批间绝对差 ≤5 个百分点",fontsize=10,color=INK)
    note(fig,"红点为批内最低 30 次，蓝点为其余 270 次。两批第 30 名均为 75%；同分按种子升序取足 30 次。")
    export(fig,"04_尾部风险")


def packed(values,spacing):
    # A deterministic beeswarm: only displace points orthogonally to the measured axis.
    placed=[]
    for value in sorted(values):
        offsets=[0]+[s*k for k in range(1,21) for s in (1,-1)]
        for offset in offsets:
            if all(abs(value-prev)>=spacing or offset!=lane for prev,lane in placed):
                placed.append((value,offset));break
    return np.array(placed).T


def seeds():
    from figure_input.seed_plot_data import load_seed_points
    points=load_seed_points()
    summary=pd.read_csv(EXT/"seed_batches.csv").set_index(["alpha","batch"])
    fig=canvas("内部抽样种子的得分与首步决策","第三问 · 每档固定同一批 90 个实例 · 两批各 20 个内部种子，K = 64",6.35)
    heading(fig,.091,.757,"单个种子的平均加权按时率")
    heading(fig,.643,.757,"同一实例内的首步一致率")
    for alpha,bottom,title,limits,ticks in [
            (1.,.49,"名义工期",(88.8,91.7),[89,90,91]),(1.5,.195,"长工期",(68.7,73.5),[69,70,71,72,73])]:
        ax=fig.add_axes([.17,bottom,.327,.174])
        style(ax,"x"); ax.set(xlim=limits,xticks=ticks,ylim=(-.61,1.63),yticks=[1,0],yticklabels=["原批","新批"])
        ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
        ax.text(-.245,1.12,title,transform=ax.transAxes,fontsize=10.6,weight="bold",color=NAVY)
        if alpha==1.5: ax.set_xlabel("平均按时率（%）")
        right=fig.add_axes([.651,bottom,.243,.174]);style(right,"x")
        right.set(xlim=(39,60),ylim=(-.6,1.6),yticks=[],xticks=[40,45,50,55,60])
        right.spines["left"].set_visible(False)
        if alpha==1.5: right.set_xlabel("首步一致率（%）")
        original=summary.loc[(alpha,"original")];new=summary.loc[(alpha,"replication")]
        right.plot([original.first_match_pct,new.first_match_pct],[1,0],color="#A0B3C9",lw=1.2)
        for batch,y,col in [("original",1,BLUE),("replication",0,RED)]:
            vals=points.loc[(points.alpha==alpha)&(points.batch==batch),"rate_pct"]
            assert len(vals)==20
            mean=vals.mean(); np.testing.assert_allclose(mean,summary.at[(alpha,batch),"mean_rate_pct"])
            x,lanes=packed(vals,.038*(limits[1]-limits[0]))
            ax.scatter(x,y+.14*lanes,s=23,color=col,edgecolor="white",linewidth=.4,zorder=3)
            ax.plot([mean-.04*(limits[1]-limits[0]),mean+.04*(limits[1]-limits[0])],[y-.40]*2,color=col,lw=2.6)
            ax.text(1.045,y,f"{mean:.2f}%",transform=ax.get_yaxis_transform(),va="center",color=col,fontsize=10)
            action=summary.at[(alpha,batch),"first_match_pct"]
            right.scatter(action,y,s=50,color=col,zorder=4)
            right.annotate(f"{action:.2f}%",(action,y),xytext=(7,0),textcoords="offset points",color=col,va="center",fontsize=9.3)
    note(fig,"左图每个点代表一个种子，短横线表示该批均值，两行横轴分别放大。右图按同一实例内不同种子的全部配对计算，再对实例等权平均。")
    export(fig,"05_内部抽样种子稳定性")


if __name__=="__main__":
    sensitivity()
    environments()
    uncertainty()
    tail()
    seeds()
