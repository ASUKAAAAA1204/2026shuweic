"""Draw the three-question comparison using the release's historical reference rows."""
import subprocess
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_five_dimensions import TAG, ROOT, NAVY, BLUE, RED, RULE, SUB, INK, style

SPACE = TAG / "C题工作区"
RESULTS = SPACE / "正式方案/results"
OUTPUT = ROOT / "figures/final_v1_baseline_comparison"
SPLITS = ["train", "validation", "test"]
COUNTS = [1527, 212, 390]


def load_comparison():
    sha = subprocess.check_output(["git","-C",str(TAG),"rev-parse","HEAD"],text=True).strip()
    assert sha == "9f8bc05fae9d8ce32da7fa7c5e28a82960085819"
    cmp = pd.read_csv(RESULTS/"comparison.csv")
    traj = {}
    for question in ["q1","q2"]:
        subset = cmp.query("question==@question and baseline=='historical_damping' and metric=='NADE'")
        frame = subset.set_index("split").loc[SPLITS]
        windows = pd.read_csv(RESULTS/f"{question}_windows.csv.gz")
        keys = ["split","window_id"] + (["horizon"] if question=="q1" else [])
        paired = windows.pivot(index=keys,columns="model",values="nade")[["original","B"]]
        assert paired.notna().all().all()
        assert windows.groupby("split").window_id.nunique().loc[SPLITS].tolist()==COUNTS
        groups = ["split","model"] + (["horizon"] if question=="q1" else []) + ["dataset","environment_id","episode_uuid"]
        if question=="q2":
            groups += ["side"]
        rolled = windows
        for levels in range(len(groups),1,-1):
            rolled = rolled.groupby(groups[:levels],as_index=False).nade.mean()
        values = rolled.pivot(index="split",columns="model",values="nade").loc[SPLITS,["original","B"]]
        np.testing.assert_allclose(values, frame[["old","new"]],rtol=0,atol=1e-12)
        traj[question] = frame
    keys = ["case_id","seed","alpha","beta"]
    ref = next(SPACE.rglob("q3_official_B_and_previous.csv.gz"))
    old = pd.read_csv(ref).query("method=='previous_local'").set_index(keys).sort_index()
    new = pd.read_csv(RESULTS/"q3/b_k64/trials.csv.gz").set_index(keys).sort_index()
    assert new.index.equals(old.index) and old.index.is_unique and len(old)==450
    case = pd.read_csv(RESULTS/"q3/b_k64/comparison.csv").query("metric=='weighted_on_time_rate'").set_index(["scope","case_id"])
    trade = cmp.query("question=='q3' and baseline=='previous_local'").set_index(["split","metric"])
    for src,col in [(old,"old"),(new,"new")]:
        nominal = src.query("alpha==1 and beta==0.5")
        assert len(nominal)==90
        for (scope,metric), row in trade.iterrows():
            values = nominal if scope=="validation_nominal" else src
            np.testing.assert_allclose(values[metric].mean(),row[col],atol=1e-10)
        for (scope,c), row in case.iterrows():
            rows = nominal if scope=="nominal" else src
            if c!="overall":
                rows = rows.xs(c,level="case_id")
            np.testing.assert_allclose(rows.weighted_on_time_rate.mean(),row["previous_local" if col=="old" else "B_K64"])
    return traj,case,trade


def header(fig,x,y,title,sub):
    fig.text(x,y,title,fontsize=13.8,weight="bold",color=NAVY)
    fig.text(x,y-.035,sub,fontsize=9,color=SUB)


def horizontal(fig,y):
    fig.add_artist(Line2D([.045,.954],[y,y],transform=fig.transFigure,color=RULE,lw=.85))


def trajectory(fig,x,title,detail,frame):
    header(fig,x,.855,title,detail)
    ax=fig.add_axes([x+.096,.596,.237,.158])
    style(ax,"x")
    ax.set(xlim=(.070,.136),xticks=[.08,.10,.12],ylim=(-.5,2.6),yticks=[2,1,0],
           yticklabels=["训练留一 · 1,527 窗","开发 · 212 窗","既有测试 · 390 窗"],xlabel="归一化平均位移误差（NADE，越低越好）")
    ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0,labelsize=8.6)
    for y,row in zip([2,1,0],frame.itertuples()):
        ax.plot([row.new,row.old],[y,y],color="#ACBCCE",lw=1.7)
        ax.plot(row.old,y,"o",mfc="white",mec=BLUE,mew=1.5,ms=6)
        ax.plot(row.new,y,"D",color=RED,ms=4.5)
        ax.text(row.old,y+.17,f"{row.old:.5f}",ha="left",fontsize=8.3,color=BLUE)
        ax.text(row.new,y-.10,f"{row.new:.5f}",ha="right",va="top",fontsize=8.3,color=RED)
        ax.text(1.28,y,f"{row.gain_pct:.2f}%",transform=ax.get_yaxis_transform(),ha="right",va="center",color=INK,fontsize=10.5,weight="bold")
    ax.text(1.28,1.03,"误差降幅",transform=ax.transAxes,ha="right",color=SUB,fontsize=8.7)


def main():
    traj,case,trade = load_comparison()
    fig=plt.figure(figsize=(12.6,8.4),facecolor="white")
    fig.text(.045,.939,"三问终稿与基线的整体表现",fontsize=22,weight="bold",color=NAVY)
    handles=[Line2D([],[],marker="o",ls="",mfc="white",mec=BLUE,mew=1.5,label="对照基线"),
             Line2D([],[],marker="D",ls="",color=RED,label="终稿 v1.0.0-final")]
    fig.legend(handles=handles,loc="center right",bbox_to_anchor=(.953,.95),ncol=2,fontsize=10.5,handletextpad=.3)
    horizontal(fig,.911)
    trajectory(fig,.045,"A   第一问｜手部质心预测","固定阻尼 → 冻结旋转残差预测；0.2、0.6、1.0 秒三个时长等权。",traj["q1"])
    trajectory(fig,.536,"B   第二问｜遮挡期间的付费观测","固定阻尼 → 旋转起步、回传后更新；两版均请求 2 次、累计等待 0.20 秒。",traj["q2"])
    horizontal(fig,.517)
    header(fig,.045,.465,"C   第三问｜不同案例的按时交付","上一正式版 → B_K64；三个案例均使用相同的 30 个官方验证种子。")
    header(fig,.536,.465,"D   第三问｜交付、拖期与费用","名义工况（α = 1，β = 0.5 秒），三案例等权，差值为终稿减基线。")
    ax=fig.add_axes([.150,.159,.231,.221]);style(ax,"x")
    ids=[("nominal","S1"),("nominal","S2"),("nominal","S3"),("nominal","overall"),("all_five","overall")]
    ys=[4.2,3.2,2.2,.95,-.05]
    for (scope,c),y in zip(ids,ys):
        r=case.loc[(scope,c)]
        a,b=100*r.previous_local,100*r.B_K64
        ax.plot([a,b],[y,y],color="#ACBCCE",lw=1.9)
        ax.plot(a,y,"o",mfc="white",mec=BLUE,mew=1.5,ms=6)
        ax.plot(b,y,"D",color=RED,ms=4.4)
        ax.text(a,y+.18,f"{a:.2f}",ha="center",color=BLUE,fontsize=8.4)
        ax.text(b,y-.1,f"{b:.2f}",ha="center",va="top",color=RED,fontsize=8.4)
        ax.text(1.22,y,f"{b-a:+.2f}".replace("-","−"),transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=10.6,weight="bold")
    ax.text(1.22,1.04,"差（百分点）",transform=ax.transAxes,ha="right",color=SUB,fontsize=8.5)
    ax.set(xlim=(82,101),xticks=[85,90,95,100],ylim=(-.63,4.85),yticks=ys,
           yticklabels=["S1 · 名义","S2 · 名义","S3 · 名义","名义工况平均","五工况平均"],xlabel="加权按时率（%）")
    ax.axhline(1.62,color=RULE,lw=.7);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
    cols=[.537,.755,.85,.945]
    for x,s,ha in zip(cols,["指标","上一正式版","终稿","差值"],["left","right","right","right"]):
        fig.text(x,.369,s,ha=ha,fontsize=9.2,color=SUB)
    for y,metric,label,rate in [(.318,"weighted_on_time_rate","加权按时率 ↑",True),(.266,"weighted_completion_rate","加权完成率 ↑",True),
                                (.214,"weighted_tardiness_s","加权总拖期（秒）↓",False),(.162,"cost","平均费用 ↓",False)]:
        r=trade.loc[("validation_nominal",metric)]
        fig.text(cols[0],y,label,fontsize=10.5)
        for x,v,col in [(cols[1],r.old,BLUE),(cols[2],r.new,RED)]:
            fig.text(x,y,f"{v:.2%}" if rate else f"{v:.2f}",ha="right",fontsize=11,color=col)
        delta=(r.new-r.old)*(100 if rate else 1)
        fig.text(cols[3],y,(f"{delta:+.2f}"+(" 点" if rate else "")).replace("-","−"),ha="right",fontsize=10.5,weight="bold")
    horizontal(fig,.083)
    fig.text(.045,.044,"第一、二问按相同窗口及来源尺度比较。第三问五工况共 450 个配对实例，包含名义工况的 90 个实例。“点”表示百分点。",color=SUB,fontsize=9)
    OUTPUT.mkdir(exist_ok=True,parents=True)
    for ext in ("pdf","png"):
        fig.savefig(OUTPUT/f"终稿与基线_三问整体对比.{ext}",dpi=450)
    plt.close(fig)


if __name__=="__main__":
    main()
