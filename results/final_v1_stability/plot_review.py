"""Review figure from the frozen-model window and trial records."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
SENS=OUT.parent/"final_v1_sensitivity"
FROZEN=OUT.parents[1]/"tmp/final_v1_20261122/C题工作区/正式方案/results"
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"font.size":9,
                     "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
BLUE,RED="#1F4E89","#CE4844"
fig,axes=plt.subplots(2,2,figsize=(9.4,6.1),layout="constrained")
q1=pd.read_csv(SENS/"q1_history_windows.csv.gz").query("horizon==1")
q2=pd.read_csv(SENS/"q2/per_window.csv.gz")
for ax,d,key,names,labels,title in [
    (axes[0,0],q1,"history",["full","0.30s"],["完整历史","历史 0.30 秒"],"A   历史截短后的 1 秒预测误差"),
    (axes[0,1],q2,"delay",[.1,.2],["延迟 0.10 秒","延迟 0.20 秒"],"B   延迟增加后的 1 秒遮挡误差")]:
    for i,(name,label) in enumerate(zip(names,labels)):
        y=d.loc[d[key]==name,"nade"].quantile([.5,.9,.95])
        ax.plot(range(3),y,color=[BLUE,RED][i],ls=["-","--"][i],marker=["o","D"][i],label=label)
    ax.set(xticks=[0,1,2],xticklabels=["中位数","P90","P95"],ylabel="逐窗 NADE",ylim=(0,None))
    ax.set_title(title,loc="left",weight="bold")
    ax.legend(frameon=False)

trials=pd.read_csv(FROZEN/"q3/b_k64/trials.csv.gz")
ax=axes[1,0]
cases=["S1","S2","S3"]
for alpha,shift,color in [(1,-.11,BLUE),(1.5,.11,RED)]:
    sub=trials.query("alpha==@alpha and beta==.5")
    for i,case in enumerate(cases):
        vals=100*sub.loc[sub.case_id==case,"weighted_on_time_rate"]
        tail=vals.nsmallest(3).mean()
        ax.plot([i+shift]*2,[tail,vals.mean()],color=color,lw=1.6)
        ax.plot(i+shift,vals.mean(),"o",color=color,ms=5)
        ax.plot(i+shift,tail,"v",mfc="white",mec=color,ms=6)
    ax.plot([],[],color=color,label=f"α = {alpha:.1f}")
ax.set(xticks=range(3),xticklabels=cases,ylabel="加权按时率（%）",ylim=(40,104),xlabel="圆点为 30 例平均，倒三角为最差 3 例平均")
ax.legend(frameon=False,ncol=2,loc="lower left")
ax.set_title("C   名义及长工期下的困难调度实例",loc="left",weight="bold")

ax=axes[1,1]
fixed=pd.read_csv(OUT/"fixed_instance_stability.csv").query("alpha==1 and beta==.5")
for i,case in enumerate(cases):
    vals=fixed.loc[fixed.case_id==case,"span_pp"]
    for val,count in vals.round(6).value_counts().items():
        ax.scatter(i+(np.arange(count)-(count-1)/2)*.018,[val]*count,s=13,
                   color="#87AEE6",edgecolor=BLUE,linewidth=.3)
    ax.plot([i-.24,i+.24],[vals.median()]*2,color=RED,lw=2)
ax.set(xticks=range(3),xticklabels=cases,ylabel="20 个内部种子的按时率极差（百分点）",
       xlabel="每点为一个固定实例，红线为案例内中位数",xlim=(-.42,2.42),ylim=(-2,32))
ax.set_title("D   相同任务与实际工期下的种子波动",loc="left",weight="bold")
for ax in axes.flat:
    ax.grid(axis="y",lw=.55,color="#E1E7EE")
    ax.set_axisbelow(True)
for ext in ("pdf","png"):
    fig.savefig(OUT/f"两项稳定性补验_审核图.{ext}",dpi=260)
