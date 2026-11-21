"""Render four independent vector figures, previews, and a captioned review copy."""
from pathlib import Path
import fitz
import matplotlib.pyplot as plt
from matplotlib import font_manager
from style import HERE, NAVY, start
from panels import observation, quality, recovery


CAPTIONS=[
    ('手部状态与观测窗口',
     'A 取 EgoDemo 同一只右手的真实关键点，在掌面上作等比例投影。手部代表点为 21 个三维位置的算术均值。箭头按末两帧位移的两倍绘制，以显示后向差分初速的投影方向。',
     'B 两项任务共用第 288 帧作为最后可见帧和质量锚点。位置示例展示 0.6 秒评分窗，空心点表示供事后评分的观测。质量回看 0.2 秒，叉号是历史源标坏；该锚点刚恢复，未来 0.2 秒未再标坏。'),
    ('固定阻尼与位置预测基线',
     'A 在相同的 212 个开发窗上评估四个模型。固定阻尼和沿向残差 MLP 的曲线接近。归一化 ADE 使用训练集同来源的头手距离尺度，数值越低，预测误差越小。',
     'B 正值代表误差下降。固定阻尼相对恒速在训练环境留一和开发集分别改善 21.55% 和 26.34%。残差 MLP 相对固定阻尼的训练改善为 0.73%，95% 区间为 −2.21% 至 3.51%。开发集仅 4 个环境，图中空心点表示其点估计。'),
    ('三状态模型对后续标坏的预测',
     'A 每行列出后续阳性数 / 状态锚点数，风险取全训练集加一平滑估计。持续未标坏与刚恢复的锚点当前都未标坏，后者在过去 0.2 秒出现过源标坏。预测目标为未来 0.2 秒内至少一次源标坏。',
     'B Brier 分数按记录、环境及来源逐层汇总，数值越低越好。三状态比二状态的训练折外分数改善 3.73%，95% 区间为 −0.96% 至 5.14%；运动提升树在三状态基础上进一步改善 0.083%。'),
    ('自然恢复时的预测误差与误差球覆盖',
     'A 在开发集 3 个环境的 52 个自然坏段中，各时机分别有 50 和 44 个事件供四种策略共同评分。第二帧接收前，等待两帧策略的归一化位置误差为 0.35898，首帧更新并衰减旧速为 0.03720。',
     'B 在可估半径的点上按记录、环境及来源等权计算覆盖率。连续遮罩采用 1 秒内的评分点。自然恢复首帧前有 8 个可评分点缺少可用半径；图示 47.16% 对应其余 42 个点。'),
]


def draw(fn, i):
    fig,(a,b)=plt.subplots(1,2,figsize=(10,3.85),gridspec_kw={'width_ratios':[1,1.05]})
    fig.subplots_adjust(left=.115 if i==3 else .08,right=.97,bottom=.23,top=.87,
                        wspace=.5 if i==2 else .31)
    fn(a,b)
    for ext in ('pdf','png'):
        fig.savefig(HERE/f'fig_0{i}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.14)
    plt.close(fig)


def pack():
    font=Path(font_manager.findfont('Microsoft YaHei'))
    archive=fitz.Archive(str(font.parent))
    css=f"@font-face {{font-family: han; src:url('{font.name}');}} * {{font-family:han;}}"
    with fitz.open() as out:
        for i,(title,p1,p2) in enumerate(CAPTIONS,1):
            with fitz.open(HERE/f'fig_0{i}.pdf') as src:
                height=660*src[0].rect.height/src[0].rect.width
                bottom=63+height
                page=out.new_page(width=720,height=bottom+102)
                for rect,html in [
                    ((28,14,690,48),f"<span style='color:{NAVY};font-size:17pt'>图 {i}　{title}</span>"),
                    ((30,bottom+8,690,bottom+97),''.join(f"<p style='font-size:9.8pt;color:#253954;margin:4pt 0'>{p}</p>" for p in (p1,p2)))]:
                    spare,_=page.insert_htmlbox(fitz.Rect(*rect),html,css=css,archive=archive,scale_low=1)
                    assert spare>=0,title
                page.show_pdf_page(fitz.Rect(30,55,690,55+height),src,0)
        out.set_toc([[1,f'图 {i}　{c[0]}',i] for i,c in enumerate(CAPTIONS,1)])
        out.subset_fonts()
        out.save(HERE/'第一问论文图_初版结构_红蓝审核稿.pdf',garbage=4,deflate=True)


if __name__=='__main__':
    from position.plot_position import draw_position
    start()
    for i,fn in enumerate([observation,draw_position,quality,recovery],1):
        draw(fn,i)
    pack()
