"""Single-page summary of the five frozen-model stability analyses."""
from plot_five_dimensions import EXT, SENS, TAIL, NAVY, BLUE, RED, PALE, RULE, SUB, INK, export, style
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def panel(fig, left, top, title, method):
    fig.text(left, top, title, weight="bold", color=NAVY, fontsize=13)
    fig.text(left, top-.034, method, color=SUB, fontsize=8.2)


def guide(fig, left, right, y):
    fig.add_artist(Line2D([left,right],[y,y], transform=fig.transFigure,color=RULE,lw=.9))


def main():
    fig=plt.figure(figsize=(12.6,8.75),facecolor="white")
    fig.text(.044,.936,"三问终稿模型的五维稳定性分析",fontsize=23,weight="bold",color=NAVY)
    fig.text(.956,.942,"v1.0.0-final",fontsize=11.5,color=SUB,ha="right")
    guide(fig,.044,.956,.905)

    panel(fig,.045,.851,"A   敏感性","第一、二问各 212 个开发窗；第三问 300 个配对工期种子。")
    hist=pd.read_csv(SENS/"q1_history.csv").query("horizon==1").set_index("history").nade
    delay=pd.read_csv(SENS/"q2/summary.csv").set_index("delay").nade
    ontime=pd.read_csv(EXT/"grid_means.csv").query("beta==.5").set_index("alpha").overall
    head=[(.046,"工况变化"),(.233,"名义"),(.30,"扰动"),(.38,"变化量")]
    for x,label in head:
        fig.text(x,.759,label,ha="left" if x==.046 else "right",color=SUB,fontsize=9.3)
    guide(fig,.046,.381,.744)
    rows=[("一问  历史截短至 0.30 秒",hist["full"],hist["0.30s"],False),
          ("二问  延迟 0.10 → 0.20 秒",delay[.1],delay[.2],False),
          ("三问  工期 α 从 1 升至 1.5",ontime[1],ontime[1.5],True)]
    for i,(name,base,stress,rate) in enumerate(rows):
        y=.702-i*.071
        fig.text(.046,y,name,fontsize=9.5)
        fig.text(.233,y,f"{base:.2%}" if rate else f"{base:.5f}",ha="right",fontsize=10.4,color=NAVY)
        fig.text(.30,y,f"{stress:.2%}" if rate else f"{stress:.5f}",ha="right",fontsize=10.4)
        delta=(f"{100*(stress-base):+.2f} 点" if rate else f"{100*(stress/base-1):+.2f}%").replace("-","−")
        fig.text(.38,y,delta,ha="right",color=RED,weight="bold",fontsize=10.7)
    fig.text(.046,.499,"前两行以 1 秒 NADE 计；第三行以加权按时率计，β = 0.5 秒。",fontsize=8.1,color=SUB)

    panel(fig,.413,.851,"B   跨环境稳定性","短历史下延迟翻倍；8 个已审核地点，308 个间隔窗。")
    sites=pd.read_csv(EXT/"spaced_environments.csv").sort_values(["split","environment_id"],ascending=[False,True])
    ax=fig.add_axes([.481,.553,.138,.212])
    style(ax,"x")
    labels=[];ys=[]
    for i,row in enumerate(sites.itertuples()):
        val=row.change_pct
        assert np.isclose(val,100*(row.nade_02/row.nade_01-1))
        y=7-i+(.4 if i<4 else 0)
        dev=row.split=="validation"
        place="超市" if "supermarket" in row.environment_id else "快递站" if "courier" in row.environment_id else "餐厅"
        labels.append(f"{place} {row.environment_id[-4:]}");ys.append(y)
        ax.hlines(y,0,val,color=RULE,lw=1.1)
        ax.scatter(val,y,s=23,marker="o" if dev else "D",color=BLUE if dev else RED,zorder=4)
        ax.annotate(f"{val:.1f}",(val,y),xytext=(4,0),textcoords="offset points",va="center",fontsize=7.7)
    ax.set(xlim=(0,46),xticks=[0,20,40],yticks=ys,yticklabels=labels,ylim=(-.6,8),xlabel="NADE 增幅（%）")
    ax.tick_params(labelsize=8,length=0);ax.spines["left"].set_visible(False)
    fig.text(.413,.782,"● 开发地点",fontsize=8.3,color=BLUE)
    fig.text(.505,.782,"◆ 既有测试地点",fontsize=8.3,color=RED)

    panel(fig,.662,.851,"C   配对不确定性","第三问长工期减名义工期，按种子配对重抽 10,000 次。")
    official=pd.read_csv(SENS/"q3_mapping.csv").query("alpha==1.5 and beta==.5").iloc[0]
    extra=pd.read_csv(EXT/"paired_grid.csv").query("contrast=='long-minus-nominal'").sort_values("n")
    intervals=[(official.delta_pp,official.low_pp,official.high_pp)]+list(extra[["delta_pp","lower_pp","upper_pp"]].itertuples(index=False,name=None))
    ax=fig.add_axes([.738,.571,.203,.195]);style(ax,"x")
    for i,(v,lo,hi) in enumerate(intervals):
        y=2-i; col=RED if i==2 else BLUE
        ax.plot([lo,hi],[y,y],color=col,lw=1.8,marker="|",ms=7)
        ax.scatter(v,y,color=col,s=32,zorder=4)
        ax.text(v,y+.18,f"{v:.2f}  [{lo:.2f}, {hi:.2f}]".replace("-","−"),ha="center",fontsize=7.6,color=col)
    ax.set(xlim=(-22.8,-14.3),xticks=[-22,-20,-18,-16],ylim=(-.38,2.6),yticks=[2,1,0],
           yticklabels=["官方 30 个","扩展前 30 个","扩展 300 个"],xlabel="按时率差及 95% 区间（百分点）")
    ax.tick_params(labelsize=8,length=0);ax.spines["left"].set_visible(False)
    fig.text(.662,.499,"后两行样本嵌套；300 个种子的区间半宽为 0.69 个百分点。",fontsize=8.1,color=SUB)
    guide(fig,.044,.956,.472)

    panel(fig,.045,.423,"D   尾部风险","S2 名义工况；每批取加权按时率最低一成，比较其均值。")
    initial=pd.read_csv(EXT/"tail_replication.csv").query("case_id=='S2' and alpha==1").iloc[0]
    tails=pd.read_csv(TAIL/"batches.csv").set_index("batch").tail_pct
    pairs=[(initial.first150*100,initial.last150*100),(tails.A,tails.B)]
    ax=fig.add_axes([.166,.20,.205,.133]);style(ax,"x")
    for y,(first,second) in zip([1,0],pairs):
        ax.plot([first,second],[y,y],color="#9EACBB",lw=1.7)
        ax.plot(first,y,marker="o",ms=6,mfc="white",mec=BLUE,mew=1.5)
        ax.plot(second,y,marker="D",ms=4,color=RED)
        ax.text(first-.23,y+.21,f"{first:.2f}%",ha="right",fontsize=8.4,color=NAVY)
        ax.text(second+.23,y+.21,f"{second:.2f}%",ha="left",fontsize=8.4,color=RED)
        gap=abs(second-first)
        ax.text(1.06,y,f"{gap:.2f} 点 · "+("通过" if gap<=5 else "超差"),transform=ax.get_yaxis_transform(),
                va="center",fontsize=9.4,weight="bold",color=NAVY if gap<=5 else RED)
    ax.set(xlim=(62,77),xticks=[62,66,70,74],ylim=(-.38,1.64),yticks=[1,0],
           yticklabels=["首次分半\n各取 15 / 150 次","独立复验\n各取 30 / 300 次"],xlabel="最低一成平均按时率（%）")
    ax.tick_params(labelsize=8.4,length=0);ax.spines["left"].set_visible(False)
    fig.text(.045,.118,"圆点与菱形分别为前后两个样本组。预设两组绝对差 ≤5 个百分点。",fontsize=8.2,color=SUB)

    panel(fig,.528,.423,"E   内部抽样种子稳定性","每档固定相同的 90 个实例与 K = 64；原批、新批各有 20 个内部种子。")
    batches=pd.read_csv(EXT/"seed_batches.csv").set_index(["alpha","batch"])
    for pos,metric,title,limits,ticks in [
        (.592,"mean_rate_pct","平均加权按时率（%）",(65,95),[70,80,90]),
        (.803,"first_match_pct","首步动作一致率（%）",(37,63),[40,50,60])]:
        ax=fig.add_axes([pos,.20,.142,.133]);style(ax,"x")
        for y,a in [(1,1.),(0,1.5)]:
            v=batches.at[(a,"original"),metric];w=batches.at[(a,"replication"),metric]
            ax.plot([v,w],[y,y],color="#9EACBB",lw=1.6)
            ax.plot(v,y,marker="o",ms=6,mfc="white",mec=BLUE,mew=1.5)
            ax.plot(w,y,marker="D",ms=4,color=RED)
            ax.text(v,y+.20,f"{v:.2f}",ha="center",fontsize=8.1,color=NAVY)
            ax.text(w,y-.32,f"{w:.2f}",ha="center",fontsize=8.1,color=RED)
        ax.set(xlim=limits,xticks=ticks,ylim=(-.52,1.6),yticks=[1,0],yticklabels=["名义工期","长工期"] if pos<.7 else [],xlabel=title)
        ax.tick_params(labelsize=8.2,length=0);ax.spines["left"].set_visible(False)
    fig.text(.528,.118,"蓝色空心圆为原批，红色菱形为新批；首步一致率按同一实例内的种子对计算。",fontsize=8.2,color=SUB)
    guide(fig,.044,.956,.087)
    fig.text(.045,.044,"NADE 为归一化平均位移误差，越低越好。图中“点”表示百分点。",color=SUB,fontsize=8.4)
    fig.text(.956,.044,"各维度证据评分（A–E，满分 10）   8.5  /  8.0  /  8.5  /  8.5  /  9.0",ha="right",color=NAVY,fontsize=8.6)
    export(fig,"五维稳定性_单页总图")


if __name__=="__main__":
    main()
