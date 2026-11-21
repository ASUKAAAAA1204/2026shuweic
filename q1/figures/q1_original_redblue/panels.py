import json
import numpy as np
from matplotlib.lines import Line2D
from style import DATA, NAVY, BLUE, RED, LIGHT, PALEBLUE, PALERED, PINK, SKY, panel, read


def observation(a,b):
    window_id='08f1a932-5d51-40cb-8bf5-18d008259415_right_hand_289'
    w=read('window_manifest.csv').set_index('window_id').loc[window_id]
    q=read('quality_predictions.csv.gz')
    q=q[(q.episode_uuid==w.record)&(q.side==w.side)&(q['index']==w.last_visible)].iloc[0]
    with np.load(DATA/'records'/f'{w.record}.npz') as z:
        t,x,bad,good=[z[f'{w.side}__{k}'] for k in ('t_s','position','bad','reference_candidate')]
    j=int(w.last_visible)
    names=json.loads((DATA/'q1_feature_names.json').read_text(encoding='utf-8'))[w.record]['metadata']['info']['features']['observation.state.hand_right_world']['names'][0]
    palm=x[j]
    center=palm.mean(axis=0)
    vertical=palm[names.index('rightMiddleTip')]-palm[names.index('rightWrist')]
    vertical/=np.linalg.norm(vertical)
    lateral=palm[names.index('rightLittleMCP')]-palm[names.index('rightIndexMCP')]
    lateral-=np.dot(lateral,vertical)*vertical
    basis=np.stack([lateral/np.linalg.norm(lateral),vertical],axis=1)
    xy=(palm-center)@basis
    panel(a,'A','21 个关键点与手部初速度')
    for finger in ['Thumb','Index','Middle','Ring','Little']:
        joints=[names.index('rightWrist')]+[names.index(f'right{finger}{part}') for part in ['MCP','PIP','DIP','Tip']]
        a.plot(*xy[joints].T,color=SKY,lw=1.7,zorder=1)
    a.scatter(*xy.T,s=25,fc=PALEBLUE,ec=BLUE,lw=1,zorder=3)
    prev=(x[j-1].mean(0)-center)@basis
    a.scatter(*prev,s=46,facecolors='white',edgecolors=RED,zorder=4)
    a.scatter(0,0,s=46,c=NAVY,marker='D',zorder=5)
    a.annotate('',xy=-2*prev,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=1.8,color=RED))
    a.legend(handles=[Line2D([],[],ls='',marker='o',mfc='white',mec=RED,label='前一帧质心'),
                      Line2D([],[],ls='',marker='D',color=NAVY,label='最后可见质心'),
                      Line2D([],[],color=RED,marker='>',markevery=[1],label='初速方向')],loc='upper left',
             bbox_to_anchor=(-.03,1.02))
    a.set_aspect('equal',adjustable='datalim')
    a.axis('off')
    panel(b,'B','位置预测和可靠性预测的时间窗')
    rel=t-t[j]
    inds=np.arange(len(t))
    hist=inds[(inds>=w.history_start)&(inds<=j)]
    future=inds[(inds>=w.first_hidden)&(inds<w.end_exclusive)&(rel<=.6+1e-12)]
    past=inds[(rel>=-.2)&(rel<=0)]
    labels=inds[(rel>0)&(rel<=.2)]
    assert len(future)>0 and good[np.r_[hist,future]].all()
    assert q.state=='recent_recovery' and bad[past].any() and not bad[j] and q.label==bad[labels].any()
    for y,h,f in [(1,hist,future),(0,past,labels)]:
        b.hlines(y,rel[h[0]],0,color=LIGHT,lw=13,zorder=1)
        b.hlines(y,0,rel[f[-1]],color=PALERED,lw=13,zorder=1)
        b.scatter(rel[h],np.full(len(h),y),s=18,color=BLUE,zorder=3)
        b.scatter(rel[f],np.full(len(f),y),s=16,facecolors='white',edgecolors=RED,zorder=3)
    b.scatter(rel[past[bad[past]]],np.zeros(bad[past].sum()),s=42,marker='x',color=RED,zorder=4)
    b.axvline(0,ls='--',color=NAVY,lw=1,zorder=2)
    for x0,y,txt in [(-.095,1.19,'可见历史'),(.29,1.19,'未来位置评分窗'),(-.101,.18,'历史质量'),(.15,.18,'未来标坏标签窗')]:
        b.text(x0,y,txt,ha='center',fontsize=9)
    b.text(0,1.56,'最后可见帧 / 质量锚点',ha='center',fontsize=8.5,color=NAVY)
    b.text(.32,-.16,'本例标签 = 0',fontsize=8.5,color=RED)
    b.set(yticks=[0,1],yticklabels=['可靠性','位置'],ylim=(-.48,1.78),xlim=(-.25,.64),
          xticks=[-.2,0,.2,.4,.6],xlabel='相对锚点的时间（秒）')
    b.spines['left'].set_visible(False)
    b.tick_params(axis='y',length=0)


def quality(a,b):
    panel(a,'A','各状态的未来标坏风险')
    r=read('quality_rates.csv').query("held=='development'").set_index('quality_state')
    keys=['clean_history','recent_recovery','flagged']
    for i,(key,col,mark) in enumerate(zip(keys,[BLUE,RED,NAVY],['o','D','s'])):
        row=r.loc[key]
        assert np.isclose(row.p,(row.events+1)/(row.samples+2))
        pct=100*row.p
        a.hlines(i,0,pct,color=col,alpha=.35)
        a.scatter(pct,i,c=col,s=46,marker=mark,zorder=3)
        a.annotate(f'{pct:.2f}%',(pct,i),xytext=(0,9),textcoords='offset points',
                   ha='left' if i==0 else 'center',fontsize=9,color=col)
    a.set(yticks=range(3),yticklabels=[f'{name}\n{r.loc[key,"events"]:,} / {r.loc[key,"samples"]:,}' for key,name in zip(keys,['持续未标坏','刚恢复','当前标坏'])],
          ylim=(2.6,-.7),xlim=(-2,102),xticks=[0,25,50,75,100],xlabel='未来 0.2 秒至少一次源标坏的概率（%）')
    a.tick_params(axis='y',length=0)
    a.grid(axis='x',alpha=.16)
    panel(b,'B','可靠性模型的 Brier 分数')
    scores=read('quality_scores.csv').query("subset=='all'").set_index('split')
    for i,(key,label,color,marker,shift) in enumerate([
            ('State','二状态',RED,'s',-.12),('State3','三状态',NAVY,'o',0),
            ('Motion','运动提升树',BLUE,'D',.12)]):
        v=scores.loc[['train','validation'],key].to_numpy()*1000
        xs=np.array([0,1])+shift
        b.scatter(xs,v,s=43,marker=marker,edgecolors=color,facecolors=color if i==1 else 'white',
                  label=label,zorder=3)
        for x0,val in zip(xs,v):
            b.annotate(f'{val:.3f}',(x0,val),xytext=(0,[11,-17,11][i]),ha='center',
                       textcoords='offset points',fontsize=7.4,color=color)
    b.set(xticks=[0,1],xticklabels=['训练环境留一','开发集'],xlim=(-.37,1.37),
          ylabel='Brier 分数（千分值）',ylim=(0,36))
    b.legend(loc='center',bbox_to_anchor=(.51,.55),ncol=1)
    b.grid(axis='y',alpha=.16)


def recovery(a,b):
    panel(a,'A','自然恢复时的归一化位置误差')
    df=read('recovery_natural_summary.csv').query("split=='validation' and scope=='common'")
    for key,name,color,marker,shift in [('recent6_hold','最近六帧＋末值回退',BLUE,'^',-.18),
                                       ('hold_two','等待两帧',RED,'s',-.06),
                                       ('zero_velocity','首帧更新＋零速',SKY,'D',.06),
                                       ('decay','首帧更新＋衰减旧速',NAVY,'o',.18)]:
        d=df[df.model==key].set_index('phase').loc[['pre_first','pre_second']]
        assert list(d.common)==[50,44]
        a.scatter(np.arange(2)+shift,d.norm_error,s=50,marker=marker,label=name,
                  edgecolors=color,facecolors=color if key=='decay' else 'white',lw=1.5,zorder=3)
    a.set(xlim=(-.37,1.37),ylim=(0,.4),xticks=[0,1],xticklabels=['接收首帧前\n50 / 52 段','接收第二帧前\n44 / 52 段'],
          ylabel='归一化位置误差')
    a.grid(axis='y',alpha=.15)
    a.legend(loc='center',ncol=1,fontsize=8)
    panel(b,'B','开发集误差球的实际覆盖率')
    ticks=[]
    for i,(interface,h,title,color) in enumerate([('continuous',1.,'连续遮罩\n1 秒内',BLUE),
                                                ('pre_first',0.,'自然恢复\n首帧前',RED),
                                                ('pre_second',0.,'自然恢复\n第二帧前',RED)]):
        row=read(f'cal_{interface}_summary.csv').query("split=='validation' and source=='ALL' and horizon==@h").iloc[0]
        pct=row.coverage*100
        b.scatter(i,pct,s=55,color=color,marker='D' if i==0 else 'o',zorder=3)
        b.annotate(f'{pct:.2f}%',(i,pct),xytext=(0,-16),ha='center',textcoords='offset points',fontsize=9)
        ticks.append(f'{title}\n{row.eligible:,} / {row.n:,}')
    b.axhline(90,color=NAVY,ls='--',lw=1.15)
    b.text(2.32,92,'标称 90%',ha='right',color=NAVY,fontsize=9)
    b.set(xlim=(-.5,2.5),ylim=(0,105),yticks=[0,20,40,60,80,100],xticks=range(3),xticklabels=ticks,
          ylabel='实际覆盖率（%）',xlabel='可估半径点数 / 可评分点数')
    b.grid(axis='y',alpha=.15)
