"""Build the figure revision; preserve the accepted timeline panel and scatter page."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import fitz
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[1:1]=[str(HERE.parent/"q1_original_redblue"),str(HERE.parent/"q1_redblue_3d")]
from style import start,read,DATA,NAVY
from position import position_waterfall,environment_boxes,radar,recovery_bars,NAMES
from reliability.bars import risks,brier_bars
from geometry import development_points
from point_clouds import sphere_example,all_clouds

OLD=HERE.parent/"q1_redblue_review/第一问论文图_正文三图与附录两图_红蓝审核稿.pdf"
OUT=HERE/"第一问论文图_红蓝精修审核稿_v5.pdf"
TITLES=["正文图 1　四种位置预测模型的误差与计算开销","正文图 2　恢复历史与未来标坏风险",
        "正文图 3　自然恢复误差与开发集三维误差球","附图 A1　手掌三维示意与预测窗口",
        "附图 A2　同一开发窗内的模型误差比较"]


def build():
    start()
    font=Path(font_manager.findfont("Microsoft YaHei"))
    archive=fitz.Archive(str(font.parent))
    css=f"@font-face {{font-family:q;src:url('{font.name}');}} * {{font-family:q;color:#213A53;}}"
    with TemporaryDirectory(dir=HERE) as folder,fitz.open(OLD) as old,fitz.open() as out:
        def place(fig,box,name):
            path=Path(folder)/f"{name}.pdf"
            fig.savefig(path,bbox_inches="tight",pad_inches=.12,dpi=300)
            plt.close(fig)
            with fitz.open(path) as f: page.show_pdf_page(fitz.Rect(*box),f,0)

        def text(box,content,size=10):
            spare,_=page.insert_htmlbox(fitz.Rect(*box),f"<div style='font-size:{size}pt'>{content}</div>",
                                        css=css,archive=archive,scale_low=1)
            assert spare>=0,(content,box)

        def title(n):
            text((24,12,page.rect.width-24,58),TITLES[n],20)

        def heading(x,y,w,content):
            text((x,y,x+w,y+25),content,13)

        page=out.new_page(width=940,height=725)
        title(0)
        heading(30,63,440,"A　开发集误差随预测时长的变化")
        heading(490,63,420,"B　四模型的五项开发集指标")
        place(position_waterfall(),(22,88,462,356),"position_waterfall")
        radar_fig,raw=radar()
        place(radar_fig,(479,83,919,365),"radar")
        heading(30,378,440,"C　16 个留出环境的误差降幅分布")
        place(environment_boxes(),(25,407,465,627),"box")
        heading(490,378,420,"雷达图对应的原始指标")
        headers=["模型","平均\nADE","末帧\nFDE","1 秒\nP90","最差\n环境","耗时\nμs"]
        rows=[]
        for name,(_,r) in zip(NAMES,raw.iterrows()):
            vals=[name,*[f"{v:.4f}" for v in r.iloc[:4]],f"{r.iloc[4]:.1f}"]
            rows.append(vals)
        table_fig,table_ax=plt.subplots(figsize=(6.2,2.4))
        table_ax.axis("off")
        table=table_ax.table(cellText=rows,colLabels=headers,cellLoc="center",bbox=[.005,.005,.99,.99],
                             colWidths=[.27,.15,.15,.15,.15,.13])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor("#D9E2EC")
            cell.set_linewidth(.6)
            cell.set_facecolor("#E8EEF8" if r==0 else "#FFFFFF")
        place(table_fig,(487,410,917,565),"metrics")
        text((492,569,914,646),"雷达各轴均为 100 × 本轴最小值 ÷ 该模型原值，越向外越好。前四项为归一化误差，耗时为同机单窗中位数。开发集固定阻尼相对恒速的平均 ADE 下降 26.34%；残差 MLP 相对固定阻尼再降 3.17%。",10)
        text((29,646,911,719),"A　四种模型共用 212 个开发窗。末值保持位于恒速外推的上层，误差大小由各点的 ADE 高度读取。B　平均 ADE 与末帧 FDE 跨三个时长汇总；1 秒 P90 取逐窗第 90 百分位；最差环境取四个开发环境平均 ADE 的最大值。<br>C　每个预测时长均保留全部 16 个训练留出环境，每个蓝点代表一个环境。零线上方表示固定阻尼的误差低于恒速外推，下方表示高于恒速外推。",9.4)

        page=out.new_page(width=940,height=440)
        title(1)
        heading(28,70,437,"A　训练拟合风险与开发集实测阳性率")
        heading(493,70,430,"B　Brier 分数（局部放大）")
        place(risks(),(25,109,458,330),"risk")
        place(brier_bars(),(485,105,916,331),"brier")
        text((28,351,914,430),"A　分子为未来 0.2 秒内至少一次源标坏的阳性锚点数，分母为该状态锚点数。训练集取加一平滑概率，开发集取实际频率。刚恢复组的 8 个开发集阳性都来自同一环境。<br>B　左轴对应训练环境留一，右轴对应开发集。两轴等跨度、各自截断，数值向下增大；同组内柱越短，Brier 分数越低。标注值为原始分数的 1,000 倍。",10)

        page=out.new_page(width=940,height=748)
        title(2)
        heading(28,63,442,"A　自然恢复时的归一化位置误差")
        heading(491,63,430,"B　覆盖成功窗口中的球心与半径")
        place(recovery_bars(),(22,87,462,341),"recovery")
        points=development_points()
        example,row=sphere_example(points)
        place(example,(487,85,915,343),"one_ball")
        text((29,349,459,411),"开发集共 52 段自然坏段。接收首帧前有 50 段可共同评分；第二帧接收前有 44 段。两组内四种策略共用可评分事件。",9.7)
        text((491,344,919,415),f"覆盖成功示例：各点均在对应时刻的球内，且整段参考轨迹位于所示 0.6 秒球内。为便于辨认，选取合格窗口中参考轨迹包围盒对角线与球半径之比最大者。半径 {row.radius:.4f}，该时刻误差 {row.error:.4f}（原坐标单位）。",9.4)
        heading(28,421,880,"C　开发集三维残差：连续遮罩点云抽样 50%")
        for x,rgb,legend in [(645,(42/255,115/255,186/255),"球内（覆盖）"),(793,(206/255,72/255,68/255),"球外（未覆盖）")]:
            for j in range(17):
                t=j/17
                color=tuple(v*(.70+.30*np.sin(np.pi*t))+(.72*t*t)*(1-v) for v in rgb)
                page.draw_circle((x-2.2*t,432-2.2*t),6.4*(1-t),color=None,fill=color)
            text((x+10,422,x+132,445),legend,10.5)
        place(all_clouds(points),(22,447,920,661),"all_errors")
        text((29,662,916,738),"连续遮罩从 6,326 个评分点中以固定种子无放回抽取 3,163 个显示；84.34% 仍为全量加权覆盖率。其余三组显示全部观测。前三幅按各点半径归一化，再将径向距离 r 映射为 log₂(1+r)，保持单位球边界；蓝点在球内，红点在球外。右图 8 个无可用半径的点使用原始三轴残差。",10)

        from registered_hand import make_hand
        w=read("window_manifest.csv").set_index("window_id").loc["08f1a932-5d51-40cb-8bf5-18d008259415_right_hand_289"]
        with np.load(DATA/"records"/f"{w.record}.npz") as z: xyz=z["right_hand__position"]
        names=json.loads((DATA/"q1_feature_names.json").read_text(encoding="utf-8"))[w.record]["metadata"]["info"]["features"]["observation.state.hand_right_world"]["names"][0]
        page=out.new_page(width=760,height=490)
        title(3)
        heading(26,69,353,"A　掌面插画上的 21 个关键点")
        place(make_hand(HERE/"palm_anatomy.png",xyz[w.last_visible],xyz[w.last_visible-1],names),
              (26,95,380,377),"hand")
        # Keep panel B at its original size and coordinates.
        clip=fitz.Rect(390,60,746,368)
        timeline=old[3]
        label,=timeline.search_for("本例标签 = 0",clip=clip)
        timeline.add_redact_annot(label,fill=False)
        timeline.apply_redactions(images=0,graphics=0)
        page.show_pdf_page(clip,old,3,clip=clip)
        text((28,393,733,481),"A　以生成的掌面插画为底图，人工定位腕部及五指的对应标志。第 288 帧的 21 个实测点经掌面投影与分片仿射配准后叠加。大红箭头沿前后两帧质心的图上位移方向放大至 3.5 倍，用于显示初速方向。插画上的间距仅作示意。<br>B　位置与质量任务均以第 288 帧为锚点。位置展示 0.6 秒评分窗；质量历史与未来标签窗各为 0.2 秒。",9.7)
        out.insert_pdf(old,from_page=4,to_page=4)
        out.set_toc([[1,t,i] for i,t in enumerate(TITLES,1)])
        out.set_metadata({"title":"第一问论文图 · 红蓝精修审核稿"})
        out.subset_fonts()
        out.save(OUT,garbage=4,deflate=True)


if __name__=="__main__":
    build()
