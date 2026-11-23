"""English edition of the A–E composite, using the original plotting functions."""
import re

from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator

from plot_paper_composite import OUT, comparisons, internal_seeds, plt, sensitivity


LABELS = {
    "输入扰动敏感性": "Sensitivity to input changes",
    "开发": "Development", "既有测试": "Test",
    "第一问  ·  历史截短": "Q1  Shorter history", "第二问  ·  延迟翻倍": "Q2  Doubled delay",
    "第三问  ·  加权按时率（%）": "Q3  Weighted on-time rate (%)",
    "预测时长（秒）": "Forecast horizon (s)", "ΔNADE（×10⁻³）": "ΔNADE (×10⁻³)",
    "完整": "Full", "0.30 秒": "0.30 s", "可见历史": "Available history",
    "固定增量 β（秒）": "Fixed offset β (s)", "工期倍率 α": "Duration multiplier α",
    "跨环境稳定性": "Cross-site stability", "配对不确定性": "Paired uncertainty",
    "S2 名义工况尾部": "S2 lower tail (nominal)",
    "NADE 增幅（%）": "Increase in NADE (%)", "地点编号": "Site ID",
    "官方 30": "Official 30", "扩展前 30": "First 30", "扩展 300": "All 300",
    "按时率差（百分点）": "On-time rate difference (pp)",
    "第一组": "Group 1", "第二组": "Group 2", "尾部平均按时率（%）": "Lower-tail mean (%)",
    "首轮分半": "Split halves", "独立复验": "New batches",
    "内部种子稳定性": "Internal seed stability",
    "每对点从左至右为首批、复验批": "Left to right: original and replication batches",
    "baseline": "Baseline", "终稿 B_K64": "Final B_K64",
    "平均加权按时率（%）": "Mean weighted on-time rate (%)",
    "单实例标准差均值（百分点）": "Mean within-instance SD (pp)",
    "首步动作一致率（%）": "First-action agreement (%)",
    "名义": "Nominal", "长工期": "Long duration",
}


def render():
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        fig = plt.figure(figsize=(180/25.4, 155/25.4), facecolor="white")
        sensitivity(fig)
        comparisons(fig)
        internal_seeds(fig)
        for ax in fig.axes:
            for axis in [ax.xaxis, ax.yaxis]:
                labels = [t.get_text() for t in axis.get_ticklabels()]
                if any(label in LABELS for label in labels):
                    axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
                    axis.set_major_formatter(FixedFormatter([LABELS.get(s, s) for s in labels]))
        for artist in fig.findobj(Text):
            s = artist.get_text()
            if s in LABELS:
                artist.set_text(LABELS[s])
            if s.startswith("第三问  ·"):
                artist.set_x(.853)
                artist.set_ha("center")
                artist.set_fontsize(7.2)
            if s.startswith("每对点"):
                artist.set_x(.284)
                artist.set_fontsize(6.2)
        fig.canvas.draw()
        assert not any(re.search(r"[\u3400-\u9fff]", t.get_text()) for t in fig.findobj(Text))
        fig.savefig(OUT / "Stability_Composite_EN.pdf", metadata={"Title": "Sensitivity and stability of the frozen models"})
        fig.savefig(OUT / "Stability_Composite_EN.png", dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    render()
