"""Generate the approved five-page Q1 review from the frozen experiment outputs."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import fitz
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "q1_original_redblue"))
from q1_original_redblue import render as original
from q1_visual_candidates import render as candidates
from position.plot_position import draw_position
from panels import observation, quality, recovery
from style import NAVY, start


TITLES = [
    "正文图 1　固定阻尼的预测收益与环境差异",
    "正文图 2　恢复历史与后续标坏风险",
    "正文图 3　自然恢复时的位置误差与误差球覆盖",
    "附图 A1　从手部关键点到预测窗口",
    "附图 A2　同一开发窗内的模型误差比较",
]
DEST = HERE / "第一问论文图_正文三图与附录两图_红蓝审核稿.pdf"


def build():
    start()
    font = Path(font_manager.findfont("Microsoft YaHei"))
    archive = fitz.Archive(str(font.parent))
    css = f"@font-face {{font-family:paper;src:url('{font.name}');}} * {{font-family:paper;}}"
    with TemporaryDirectory(dir=HERE) as folder, fitz.open() as doc:
        temp = Path(folder)
        original.HERE = candidates.HERE = temp
        for fn, number in [(quality,3),(recovery,4),(observation,1)]:
            original.draw(fn,number)
        candidates.heatmap()
        wins = candidates.scatter()
        fig_a,a = plt.subplots(figsize=(4.9,2.65))
        fig_b,b = plt.subplots(figsize=(4.9,2.5))
        draw_position(a,b)
        a.set_title("A   开发集 · 误差随预测时长的变化",loc="left")
        b.set_title("B   相对参考模型的误差降幅",loc="left")
        b.set_xlabel("归一化 ADE 降幅（%）")
        for fig,key in [(fig_a,"curve"),(fig_b,"forest")]:
            fig.savefig(temp/f"{key}.pdf",bbox_inches="tight",pad_inches=.07)
            plt.close(fig)

        for number,title in enumerate(TITLES,1):
            page = doc.new_page(width=760,height=580 if number==1 else 490)
            def label(box, html, size, color="#233951"):
                spare,_=page.insert_htmlbox(fitz.Rect(*box),
                    f"<div style='font-size:{size}pt;color:{color}'>{html}</div>",
                    css=css,archive=archive,scale_low=1)
                assert spare>=0, (number,html)

            def chart(key,box):
                with fitz.open(temp/f"{key}.pdf") as src:
                    page.show_pdf_page(fitz.Rect(*box),src,0)

            label((26,16,736,47),title,19,NAVY)
            if number==1:
                chart("curve",(24,60,379,256))
                chart("forest",(24,270,379,457))
                label((393,60,737,86),"C　训练留出环境 × 预测时长",12,NAVY)
                chart("heatmap",(391,87,736,461))
                notes=[
                    "A　四个模型使用相同的 212 个开发窗。以末帧可见时刻为时间原点，归一化平均位移误差（ADE）越低越好。",
                    "B　训练集按 16 个环境留一评估，区间由配对环境重采样得到。固定阻尼相对恒速的训练降幅为 21.55%（95% 区间 12.03% 至 30.79%），4 个开发环境的点估计为 26.34%。",
                    "C　各格为固定阻尼相对恒速的 ADE 降幅，环境内各来源等权，按 1 秒降幅排序。蓝色为改善，红色为退化。"
                ]
                y=472
            elif number in (2,3,4):
                table={2:("fig_03",2),3:("fig_04",3),4:("fig_01",0)}
                key,idx=table[number]
                chart(key,(25,60,735,370))
                notes=list(original.CAPTIONS[idx][1:])
                if number==2:
                    notes[1]+=" 训练与开发分别包含 15,488 和 2,236 个质量锚点。"
                y=383
            else:
                chart("scatter",(119,53,641,390))
                notes=[f"每点为一个开发窗，横纵轴分别是恒速外推和固定阻尼在 1 秒内的归一化 ADE。两轴采用相同对数刻度，虚线表示两种模型误差相等。212 个窗口中，线下 {wins} 个固定阻尼更优，线上 {212-wins} 个恒速更优。",
                       "上方和右侧的直方图分别给出恒速外推和固定阻尼的逐窗误差分布。窗口胜负按窗计数；正文图 1 的总体改善率按实验规定的记录、环境、来源和时长逐层汇总。"]
                y=394
            page.draw_line((28,y-5),(732,y-5),color=(.83,.87,.91),width=.55)
            label((28,y+2,732,page.rect.height-12),
                  "".join(f"<p style='margin:2pt 0'>{s}</p>" for s in notes),9.4)
        doc.set_toc([[1,title,i] for i,title in enumerate(TITLES,1)])
        doc.set_metadata({"title":"第一问论文图 · 红蓝审核稿"})
        doc.subset_fonts()
        doc.save(DEST,garbage=4,deflate=True)


if __name__=="__main__":
    build()
