"""Two-panel position-forecast summary for the fixed 10 x 4 inch layout."""

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D

from style import BLUE, NAVY, PALEBLUE, PINK, RED, SKY, panel, read

CJK_FONTS = (
    "Microsoft YaHei",
    "Noto Sans SC",
    "SimHei",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
)

# Panel A: development-set normalized ADE by forecast horizon.
SERIES = {
    "B": {
        "label": "固定阻尼",
        "color": NAVY,
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": NAVY,
        "markersize": 5.6,
        "markeredgewidth": 1.0,
        "linewidth": 2.0,
        "zorder": 4,
    },
    "CV": {
        "label": "恒速外推",
        "color": RED,
        "linestyle": "--",
        "marker": "s",
        "markerfacecolor": RED,
        "markersize": 5.2,
        "markeredgewidth": 1.0,
        "linewidth": 1.5,
        "zorder": 3,
    },
    "Hold": {
        "label": "末值保持",
        "color": SKY,
        "linestyle": ":",
        "marker": "^",
        "markerfacecolor": SKY,
        "markersize": 6.0,
        "markeredgewidth": 1.0,
        "linewidth": 1.5,
        "zorder": 3,
    },
    # The extension tracks B closely, so keep it hollow and a little larger.
    "C": {
        "label": "阻尼＋沿向残差 MLP",
        "color": BLUE,
        "linestyle": "-",
        "marker": "D",
        "markerfacecolor": "none",
        "markersize": 7.0,
        "markeredgewidth": 1.5,
        "linewidth": 1.1,
        "zorder": 5,
    },
}

# Panel B: display order is B vs CV, B vs Hold, C vs B (top to bottom).
PAIRS = (
    ("B", "CV", "固定阻尼\nvs 恒速外推"),
    ("B", "Hold", "固定阻尼\nvs 末值保持"),
    ("C", "B", "阻尼＋沿向残差 MLP\nvs 固定阻尼"),
)


def _ensure_cjk_font():
    """Keep Chinese labels legible when the style module has not set a CJK font."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    configured = list(rcParams.get("font.sans-serif", []))
    family = rcParams.get("font.family", [])
    configured.extend(family if isinstance(family, (list, tuple)) else [family])
    if any(name in available for name in configured if name in CJK_FONTS):
        return
    for name in CJK_FONTS:
        if name in available:
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [name, *configured]
            return


def _development(frame):
    """Use the project's validation rows as the development split."""
    split = "validation" if frame["split"].eq("validation").any() else "development"
    return frame.loc[frame["split"].eq(split)]


def _paired_row(frame, model, reference):
    row = frame.loc[frame["model"].eq(model) & frame["reference"].eq(reference)]
    if len(row) != 1:
        raise ValueError(
            f"expected one paired row for {model} vs {reference}, found {len(row)}"
        )
    return row.iloc[0]


def draw_position(a, b):
    """Draw panel A (development ADE curves) and panel B (paired gains)."""
    _ensure_cjk_font()
    horizons = read("position_horizons.csv")
    need = {"split", "model", "horizon", "nade"}
    if not need.issubset(horizons.columns):
        raise ValueError(
            f"position_horizons.csv missing columns: {sorted(need - set(horizons.columns))}"
        )
    development = _development(horizons)

    for model in ("B", "CV", "Hold", "C"):
        spec = SERIES[model]
        curve = development.loc[development["model"].eq(model)].sort_values("horizon")
        if curve.empty:
            raise ValueError(f"no development rows for model {model}")
        a.plot(
            curve["horizon"],
            curve["nade"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            marker=spec["marker"],
            markersize=spec["markersize"],
            markerfacecolor=spec["markerfacecolor"],
            markeredgecolor=spec["color"],
            markeredgewidth=spec["markeredgewidth"],
            linewidth=spec["linewidth"],
            label=spec["label"],
            zorder=spec["zorder"],
        )

    ticks = np.sort(development["horizon"].unique())
    a.set_xticks(ticks)
    a.set_xticklabels([f"{tick:g}" for tick in ticks])
    a.set_xlim(ticks[0] - 0.06, ticks[-1] + 0.06)
    a.set_ylim(0, float(development["nade"].max()) * 1.12)
    a.set_xlabel("预测时长（秒）")
    a.set_ylabel("归一化平均位移误差（ADE）")
    a.grid(axis="y", alpha=0.17)
    a.legend(loc="upper left", fontsize=8.5, handlelength=2.0, labelspacing=0.3)
    panel(a, "A", "开发集 · 归一化 ADE 随预测时长变化")

    paired = read("position_paired.csv")
    need = {"model", "reference", "split", "gain_pct", "lo95", "hi95"}
    if not need.issubset(paired.columns):
        raise ValueError(
            f"position_paired.csv missing columns: {sorted(need - set(paired.columns))}"
        )
    training = paired.loc[paired["split"].eq("train")]
    holdout = _development(paired)
    rows = np.arange(len(PAIRS) - 1, -1, -1, dtype=float)

    b.axvline(0, color=NAVY, linewidth=0.9, alpha=0.55, zorder=1)
    for index, ((model, reference, label), ypos) in enumerate(zip(PAIRS, rows)):
        if index % 2 == 0:
            b.axhspan(
                ypos - 0.42,
                ypos + 0.42,
                color=PALEBLUE,
                alpha=0.22,
                linewidth=0,
                zorder=0,
            )

        train = _paired_row(training, model, reference)
        dev = _paired_row(holdout, model, reference)
        color = BLUE if model == "C" else NAVY
        xerr = np.array(
            [
                [float(train["gain_pct"]) - float(train["lo95"])],
                [float(train["hi95"]) - float(train["gain_pct"])],
            ]
        )
        b.errorbar(
            float(train["gain_pct"]),
            ypos,
            xerr=xerr,
            fmt="o",
            color=color,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=6.2,
            markeredgewidth=1.0,
            elinewidth=1.5,
            capsize=3.2,
            capthick=1.2,
            zorder=4,
        )
        b.plot(
            float(dev["gain_pct"]),
            ypos,
            linestyle="none",
            marker="o",
            markersize=6.2,
            markerfacecolor="white",
            markeredgecolor=PINK,
            markeredgewidth=1.8,
            zorder=5,
        )

    b.set_yticks(rows)
    b.set_yticklabels([pair[2] for pair in PAIRS])
    for tick in b.get_yticklabels():
        tick.set_verticalalignment("center")
        tick.set_horizontalalignment("right")
        tick.set_multialignment("right")
        tick.set_linespacing(1.0)
    b.tick_params(axis="y", length=0, labelsize=8.8)
    b.set_ylim(rows[-1] - 0.55, rows[0] + 0.55)
    b.set_xlim(-7, 38)
    b.set_xticks([-5, 0, 10, 20, 30])
    b.set_xlabel("ADE 降幅（%）；正值表示前者更优")
    b.grid(axis="x", alpha=0.17)
    handles = [
        Line2D(
            [],
            [],
            color=NAVY,
            marker="o",
            linestyle="none",
            markerfacecolor=NAVY,
            markeredgecolor=NAVY,
            markersize=6.0,
            label="训练环境留一（95% CI）",
        ),
        Line2D(
            [],
            [],
            color=PINK,
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=PINK,
            markeredgewidth=1.7,
            markersize=6.0,
            label="开发集（点估计）",
        ),
    ]
    b.legend(
        handles=handles,
        loc="lower right",
        fontsize=8.5,
        handlelength=1.3,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.25,
    )
    panel(b, "B", "训练环境留一与开发集 · 配对 ADE 降幅")
