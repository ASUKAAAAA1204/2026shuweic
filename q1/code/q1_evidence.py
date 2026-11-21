"""Additional grouped evidence and figures, after running q1.py."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from q1 import ROOT, OLD, OUT


def main():
    quality = pd.read_csv(OUT/'quality_predictions.csv')
    train = quality.query("split=='train_oof'")
    counts = train.groupby(['environment_id','current_bad']).label.agg(['sum','count']).unstack(fill_value=0)
    draws=np.random.default_rng(20261120).integers(len(counts),size=(2000,len(counts)))
    intervals=[]
    for state in (False,True):
        events=counts['sum',state].to_numpy()
        total=counts['count',state].to_numpy()
        rates=(events[draws].sum(axis=1)+1)/(total[draws].sum(axis=1)+2)
        intervals.append(dict(current_bad=state,p=(events.sum()+1)/(total.sum()+2),lo=np.quantile(rates,.025),hi=np.quantile(rates,.975)))
    pd.DataFrame(intervals).to_csv(OUT/'risk_intervals.csv',index=False)
    usage=pd.read_csv(OLD/'baseline_v2/data/sample_usage.csv')
    usage.groupby(['dataset','split']).agg(records=('episode_uuid','nunique'),hand_frames=('total_samples','sum'),eligible=('reference_samples','sum'),excluded=('excluded_samples','sum'),windows=('windows','sum')).to_csv(OUT/'retention.csv')
    env=pd.read_csv(OUT/'motion_environments.csv').query("split=='validation'")
    comparisons=[]
    for (src,h),d in env.groupby(['dataset','horizon_s']):
        wide=d.pivot(index='environment_id',columns='model',values='ade')
        for m in ('ridge','robust'):
            diff=(wide[m]-wide.damped).to_numpy()
            limits=[np.nan,np.nan]
            if len(diff)>1:
                sample=np.random.default_rng(20261120).choice(diff,size=(2000,len(diff))).mean(axis=1)
                limits=np.quantile(sample,[.025,.975])
            comparisons.append(dict(dataset=src,horizon_s=h,model=m,delta=diff.mean(),lo=limits[0],hi=limits[1],environments=len(diff)))
    pd.DataFrame(comparisons).to_csv(OUT/'motion_paired.csv',index=False)
    scores=pd.read_csv(OUT/'motion_summary.csv').query("split=='validation'")
    plt.rcParams.update({'font.family':'Microsoft YaHei','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(9,3.1),sharey=True,layout='constrained')
    for ax,(src,group) in zip(axs,scores.groupby('dataset')):
        for m,label,marker in [('damped','原阻尼','o'),('robust','多帧稳健速度','s'),('ridge','位移回归','^')]:
            d=group[group.model==m]
            ax.plot(d.horizon_s,d.ade,marker=marker,label=label)
        ax.set(title=src,xlabel='遮罩时长（秒）',xticks=[.2,.6,1.])
    axs[0].set_ylabel('平均位移误差（各源坐标单位）')
    axs[0].legend(frameon=False,fontsize=8)
    fig.savefig(ROOT/'figures/q1_motion.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
