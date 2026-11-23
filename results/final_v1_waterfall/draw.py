"""Reusable, single-axis academic waterfall chart.

The function in this module only builds and returns a matplotlib ``Figure``.
It never reads data and never writes a file.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

__all__ = ["waterfall"]


# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------
BLUE = "#2A73BA"          # baseline
RED = "#CE4844"           # final
POS_FILL = "#B9D6EE"      # light blue for an increase
POS_EDGE = "#2A73BA"
POS_TEXT = "#12578F"
NEG_FILL = "#F3C4CD"      # light pink for a decrease
NEG_EDGE = "#CE4844"
NEG_TEXT = "#A33039"
ZERO_FILL = "#D9E1E7"

INK = "#1F2933"
MUTED = "#5F6B76"
RULE = "#DDE3E7"
GRID = "#E7EBEE"
CONNECTOR = "#98A4AF"
WHITE = "#FFFFFF"

FIG_SIZE = (11.2, 5.8)
LEFT = 0.075
RIGHT = 0.963


def _configure_matplotlib() -> None:
    """Apply the requested Chinese academic style and PDF font embedding."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Microsoft YaHei UI",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#CBD3D9",
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.use14corefonts": False,
            "axes.unicode_minus": True,
        }
    )


_configure_matplotlib()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_number(value: float, digits: int, *, signed: bool = False) -> str:
    """Format a value with the display precision and typographic minus sign."""
    quantum = 10.0 ** (-digits)
    if abs(value) < quantum * 0.5:
        value = 0.0
    if value < 0:
        body = f"−{abs(value):,.{digits}f}"
    elif signed and value > 0:
        body = f"+{value:,.{digits}f}"
    else:
        body = f"{value:,.{digits}f}"
    return body


def _format_quantity(value: float, digits: int, unit: str) -> str:
    number = _format_number(value, digits)
    if not unit:
        return number
    if unit in {"%", "‰", "°", "℃"} or unit.startswith("°"):
        return f"{number}{unit}"
    return f"{number} {unit}"


def _display_cells(text: str) -> int:
    """Approximate typographic width in half-em cells (CJK glyphs count two)."""
    width = 0
    for char in text:
        if char in "\r\n":
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _wrap_text(text: str, max_cells: int) -> list[str]:
    """Wrap text at a visual width while preserving explicitly supplied breaks."""
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    result: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            result.append("")
            continue
        current = ""
        current_cells = 0
        for char in paragraph:
            char_cells = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
            if current and current_cells + char_cells > max_cells:
                break_at = current.rfind(" ")
                if break_at >= max(2, max_cells // 3) and break_at < len(current) - 1:
                    result.append(current[:break_at].rstrip())
                    current = current[break_at + 1 :].lstrip() + char
                    current_cells = _display_cells(current)
                else:
                    result.append(current.rstrip())
                    current = char
                    current_cells = char_cells
            else:
                current += char
                current_cells += char_cells
        result.append(current.rstrip())
    return result


def _step_decimals(step: float) -> int:
    """Choose just enough decimals for the tick spacing."""
    step = abs(float(step))
    if step == 0.0:
        return 0
    for decimals in range(7):
        if abs(round(step, decimals) - step) <= max(1e-12, step * 1e-9):
            return decimals
    return 6


def _tick_formatter(decimals: int) -> FuncFormatter:
    def formatter(value: float, _position: float) -> str:
        if abs(value) < 1e-12:
            return "0"
        label = f"{value:,.{decimals}f}"
        if decimals > 0:
            label = label.rstrip("0").rstrip(".")
        return label.replace("-", "−")

    return FuncFormatter(formatter)


def _looks_like(label: str, words: Sequence[str]) -> bool:
    lines = str(label).splitlines()
    first_line = lines[0].strip().lower() if lines else ""
    return any(word.lower() in first_line for word in words)


_START_WORDS = (
    "基线",
    "基准",
    "现状",
    "当前",
    "初始",
    "起始",
    "起点",
    "原有",
    "baseline",
)
_END_WORDS = (
    "终稿",
    "终版",
    "最终",
    "目标",
    "结果",
    "合计",
    "final",
)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def waterfall(
    title: str,
    subtitle: str,
    labels: Sequence[str],
    changes: Sequence[float],
    baseline: float,
    final: float,
    unit: str = "",
    digits: int = 1,
    footnote: str = "",
) -> Figure:
    """Draw a one-axis waterfall of signed changes around a baseline.

    Parameters
    ----------
    title, subtitle:
        Figure title and explanatory deck.  The subtitle may contain newlines.
    labels:
        Three to eight category strings.  The preferred form supplies one label
        for each displayed bar, including the baseline and final bars; in that
        case ``len(labels) == len(changes) + 2``.  Labels may contain newlines.
        For convenience, step-only labels are also accepted when
        ``len(labels) == len(changes)``; baseline and final labels are then added.
    changes:
        Signed intermediate changes in the original unit.  Their sum must close
        exactly (to numerical precision) to ``final - baseline``.
    baseline, final:
        Absolute endpoint values in the original unit.  They are used for the
        compact ``基线 … → 终稿 …`` header.  The plotted y axis is the difference
        from the baseline, so the baseline is anchored at zero and the last bar
        spans zero to ``final - baseline``.
    unit:
        Original-unit suffix, e.g. ``"万元"`` or ``"%"``.
    digits:
        Number of decimal places for labels and endpoint values.
    footnote:
        Optional source/interpretation note.  Newlines are preserved.

    Returns
    -------
    matplotlib.figure.Figure
        The newly constructed figure.  The caller owns rendering/saving.
    """
    _configure_matplotlib()

    digits = int(digits)
    assert 0 <= digits <= 6, "digits must be an integer between 0 and 6"

    baseline_value = float(baseline)
    final_value = float(final)
    assert math.isfinite(baseline_value), "baseline must be finite"
    assert math.isfinite(final_value), "final must be finite"

    title_text = " ".join(str(title).split()) if title is not None else ""
    subtitle_text = "" if subtitle is None else str(subtitle)
    footnote_text = "" if footnote is None else str(footnote)
    unit_text = "" if unit is None else str(unit).strip()

    labels_list = [str(label) for label in labels]
    assert 3 <= len(labels_list) <= 8, "labels must contain 3–8 categories"

    try:
        raw_changes = [float(change) for change in changes]
    except (TypeError, ValueError) as exc:
        raise TypeError("changes must be a one-dimensional sequence of numbers") from exc
    assert raw_changes, "changes must not be empty"
    assert all(math.isfinite(change) for change in raw_changes), (
        "all changes must be finite"
    )

    n_labels = len(labels_list)
    n_changes = len(raw_changes)

    # Preferred form: labels describe every bar, changes describe the floating
    # bars only.  A per-bar vector with zero endpoint placeholders is also
    # accepted, as are labels that describe the steps only.
    if n_changes == n_labels - 2:
        bar_labels = labels_list
        steps = raw_changes
    elif n_changes == n_labels:
        internal = raw_changes[1:-1]
        endpoint_named = _looks_like(labels_list[0], _START_WORDS) and _looks_like(
            labels_list[-1], _END_WORDS
        )
        zero_tolerance = 0.5 * 10.0 ** (-digits)
        endpoint_placeholders = (
            abs(raw_changes[0]) <= zero_tolerance
            and abs(raw_changes[-1]) <= zero_tolerance
        )

        def closes(values: Sequence[float]) -> bool:
            return math.isclose(
                float(sum(values)),
                final_value - baseline_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )

        if endpoint_named or endpoint_placeholders or (
            closes(internal) and not closes(raw_changes)
        ):
            bar_labels = labels_list
            steps = internal
        else:
            bar_labels = ["基线", *labels_list, "终稿"]
            steps = raw_changes
    else:
        raise AssertionError(
            "changes must have len(labels)-2 entries (all bars are labelled), "
            "or len(labels) entries (labels cover the step bars)"
        )

    assert 3 <= len(bar_labels) <= 10, "the resolved waterfall is too crowded"

    total_delta = final_value - baseline_value
    assert math.isclose(
        float(sum(steps)),
        total_delta,
        rel_tol=1e-8,
        abs_tol=1e-8,
    ), (
        "additive closure failed: "
        f"基线 {baseline_value} + 中间变化 {float(sum(steps))} "
        f"≠ 终稿 {final_value}"
    )

    # ------------------------------------------------------------------
    # Figure and header
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=FIG_SIZE, facecolor=WHITE)

    title_cells = max(1, _display_cells(title_text))
    title_fontsize = min(19.5, max(15.0, 19.5 * 80.0 / title_cells))
    if title_text:
        fig.text(
            LEFT,
            0.960,
            title_text,
            ha="left",
            va="top",
            fontsize=title_fontsize,
            fontweight="bold",
            color=INK,
        )

    endpoint_text = (
        f"基线 {_format_quantity(baseline_value, digits, unit_text)} "
        f"→ 终稿 {_format_quantity(final_value, digits, unit_text)}"
    )
    endpoint_cells = max(1, _display_cells(endpoint_text))
    endpoint_fontsize = min(11.4, max(9.2, 11.4 * 142.0 / endpoint_cells))
    fig.text(
        LEFT,
        0.900,
        endpoint_text,
        ha="left",
        va="top",
        fontsize=endpoint_fontsize,
        fontweight="bold",
        color="#263746",
    )

    subtitle_lines = _wrap_text(subtitle_text, 142) if subtitle_text else []
    subtitle_lines = subtitle_lines or [""]
    subtitle_fontsize = 9.8
    subtitle_line_height = subtitle_fontsize * 1.30 / 72.0 / FIG_SIZE[1]
    if subtitle_text:
        fig.text(
            LEFT,
            0.858,
            "\n".join(subtitle_lines),
            ha="left",
            va="top",
            fontsize=subtitle_fontsize,
            color=MUTED,
            linespacing=1.30,
        )
    subtitle_bottom = 0.858 - subtitle_line_height * len(subtitle_lines)
    rule_y = max(0.145, subtitle_bottom - 0.012)

    footnote_lines = _wrap_text(footnote_text, 166) if footnote_text else []
    max_label_lines = max(label.count("\n") + 1 for label in bar_labels)
    bottom = 0.185 + 0.035 * (max_label_lines - 1)
    bottom += 0.020 * max(0, len(footnote_lines) - 1)
    bottom = min(0.355, bottom)
    top = max(bottom + 0.255, rule_y - 0.028)
    top = min(top, 0.800)

    fig.add_artist(
        Line2D(
            [LEFT, RIGHT],
            [rule_y, rule_y],
            transform=fig.transFigure,
            color=RULE,
            linewidth=0.8,
            zorder=0,
        )
    )

    ax = fig.add_axes([0.087, bottom, RIGHT - 0.087, top - bottom])

    # ------------------------------------------------------------------
    # Geometry and limits
    # ------------------------------------------------------------------
    count = len(bar_labels)
    positions = np.arange(count, dtype=float)
    bar_width = max(0.46, min(0.62, 5.4 / count))

    starts: list[float] = []
    ends: list[float] = []
    cursor = 0.0
    for step in steps:
        starts.append(cursor)
        cursor += step
        ends.append(cursor)

    levels = np.asarray([0.0, *starts, *ends, total_delta], dtype=float)
    low = float(np.min(levels))
    high = float(np.max(levels))
    span = high - low
    if span < max(1e-9, abs(baseline_value) * 1e-8):
        span = max(1.0, abs(total_delta), abs(float(sum(abs(v) for v in steps))))
    y_min = low - 0.18 * span
    y_max = high + 0.21 * span
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.55, count - 0.45)

    locator = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10], min_n_ticks=4)
    ticks = locator.tick_values(y_min, y_max)
    ticks = ticks[(ticks >= y_min) & (ticks <= y_max)]
    if y_min < 0 < y_max and not np.any(np.isclose(ticks, 0.0)):
        ticks = np.unique(np.append(ticks, 0.0))
    if len(ticks) < 2:
        ticks = np.linspace(y_min, y_max, 5)
    tick_step = float(np.min(np.diff(ticks))) if len(ticks) > 1 else 1.0
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(_tick_formatter(_step_decimals(tick_step)))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.72, zorder=0)
    ax.axhline(0.0, color="#74818C", linewidth=0.95, zorder=1.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#C8D0D6")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0, pad=8, labelsize=9.0, colors=MUTED)
    ax.tick_params(axis="x", length=0, pad=9, labelsize=9.6, colors="#35414D")
    ax.set_ylabel(
        f"相对基线的变化（{unit_text}）" if unit_text else "相对基线的变化",
        fontsize=9.4,
        color=MUTED,
        labelpad=11,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(bar_labels)
    for tick_label in ax.get_xticklabels():
        tick_label.set_linespacing(1.35)
        tick_label.set_horizontalalignment("center")
        tick_label.set_verticalalignment("top")

    # ------------------------------------------------------------------
    # Connectors sit behind the floating bars.
    # ------------------------------------------------------------------
    connector_ends = [0.0, *ends]
    for index in range(count - 1):
        level = connector_ends[index]
        ax.plot(
            [positions[index] + bar_width / 2.0, positions[index + 1] - bar_width / 2.0],
            [level, level],
            color=CONNECTOR,
            linewidth=0.9,
            linestyle=(0, (4.0, 3.0)),
            solid_capstyle="butt",
            dashes=(4.0, 3.0),
            zorder=1.6,
            clip_on=False,
        )

    # Baseline: a blue, zero-height reference marker rather than a fake value.
    baseline_cap = bar_width * 0.88
    ax.plot(
        [positions[0] - baseline_cap / 2.0, positions[0] + baseline_cap / 2.0],
        [0.0, 0.0],
        color=BLUE,
        linewidth=7.0,
        solid_capstyle="round",
        zorder=5.0,
        clip_on=False,
    )
    ax.plot(
        positions[0],
        0.0,
        marker="o",
        markersize=4.2,
        markerfacecolor=WHITE,
        markeredgecolor=BLUE,
        markeredgewidth=1.0,
        linestyle="none",
        zorder=5.2,
        clip_on=False,
    )

    # ------------------------------------------------------------------
    # Floating changes
    # ------------------------------------------------------------------
    for index, (start, end, change) in enumerate(zip(starts, ends, steps), start=1):
        if abs(change) <= 0.5 * 10.0 ** (-digits):
            ax.plot(
                positions[index],
                start,
                marker="o",
                markersize=4.2,
                color=ZERO_FILL,
                markeredgecolor=CONNECTOR,
                markeredgewidth=0.8,
                linestyle="none",
                zorder=4.0,
            )
            text_color = MUTED
            label = _format_number(0.0, digits)
        elif change > 0:
            ax.bar(
                positions[index],
                change,
                width=bar_width,
                bottom=start,
                color=POS_FILL,
                edgecolor=POS_EDGE,
                linewidth=0.85,
                zorder=3.0,
            )
            text_color = POS_TEXT
            label = _format_number(change, digits, signed=True)
        else:
            ax.bar(
                positions[index],
                change,
                width=bar_width,
                bottom=start,
                color=NEG_FILL,
                edgecolor=NEG_EDGE,
                linewidth=0.85,
                zorder=3.0,
            )
            text_color = NEG_TEXT
            label = _format_number(change, digits, signed=True)

        if change >= 0:
            ax.annotate(
                label,
                xy=(positions[index], end),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.6,
                fontweight="semibold",
                color=text_color,
                zorder=6.0,
                clip_on=False,
            )
        else:
            ax.annotate(
                label,
                xy=(positions[index], end),
                xytext=(0, -7),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=9.6,
                fontweight="semibold",
                color=text_color,
                zorder=6.0,
                clip_on=False,
            )

    # ------------------------------------------------------------------
    # Final bar, anchored at the baseline level (zero).
    # ------------------------------------------------------------------
    final_x = positions[-1]
    if abs(total_delta) <= 0.5 * 10.0 ** (-digits):
        ax.plot(
            [final_x - baseline_cap / 2.0, final_x + baseline_cap / 2.0],
            [0.0, 0.0],
            color=RED,
            linewidth=7.0,
            solid_capstyle="round",
            zorder=4.5,
            clip_on=False,
        )
    else:
        ax.bar(
            final_x,
            total_delta,
            width=bar_width,
            bottom=0.0,
            color=RED,
            edgecolor="#B33B38",
            linewidth=0.85,
            zorder=3.8,
        )
        ax.plot(
            final_x,
            total_delta,
            marker="D",
            markersize=4.2,
            markerfacecolor=WHITE,
            markeredgecolor=RED,
            markeredgewidth=1.0,
            linestyle="none",
            zorder=5.0,
            clip_on=False,
        )

    final_value_label = f"Δ {_format_number(total_delta, digits, signed=True)}"
    if total_delta >= 0:
        ax.annotate(
            final_value_label,
            xy=(final_x, total_delta),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="bold",
            color=RED,
            zorder=6.0,
            clip_on=False,
        )
    else:
        ax.annotate(
            final_value_label,
            xy=(final_x, total_delta),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10.2,
            fontweight="bold",
            color=RED,
            zorder=6.0,
            clip_on=False,
        )

    ax.annotate(
        "0",
        xy=(positions[0], 0.0),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=BLUE,
        zorder=6.0,
        clip_on=False,
    )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    if footnote_text:
        fig.text(
            LEFT,
            0.035,
            "\n".join(footnote_lines),
            ha="left",
            va="bottom",
            fontsize=8.4,
            color=MUTED,
            linespacing=1.35,
        )

    return fig
