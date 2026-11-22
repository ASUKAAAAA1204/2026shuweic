"""Render the paired, seed-level scheduling sensitivity figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


out = Path(__file__).resolve().parent
data = pd.read_csv(out/"q3_mapping.csv").query("alpha != 1.0 or beta != 0.5")
fig, ax = plt.subplots(figsize=(6.25, 2.25), layout="constrained")
plt.rcParams["pdf.fonttype"] = 42
for y, row in enumerate(data.itertuples()):
    color = "#205B78" if row.delta_pp > 0 else "#943F36"
    ax.plot([row.low_pp, row.high_pp], [y, y], color=color, lw=2.3, solid_capstyle="round")
    ax.plot(row.delta_pp, y, "o", color=color, ms=6)
    ax.annotate(f"{row.delta_pp:+.2f}", (row.high_pp, y), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center", fontsize=9)
ax.axvline(0, color="#6B7075", lw=.9)
ax.set_yticks(range(len(data)), [f"α = {r.alpha:.2f},  β = {r.beta:.2f} s" for r in data.itertuples()])
ax.set_ylim(len(data)-.45, -.5)
ax.set(xlim=(-21.5, 12.5), xticks=[-20, -15, -10, -5, 0, 5, 10])
ax.set_xlabel("相对名义映射的加权按时率变化（百分点）", fontfamily="Microsoft YaHei", fontsize=10)
ax.grid(axis="x", color="#D7DBDE", lw=.5)
ax.set_axisbelow(True)
ax.spines[["top", "left", "right"]].set_visible(False)
ax.tick_params(axis="both", length=0, pad=5, labelsize=9)
for ext in ("pdf", "png"):
    fig.savefig(out/f"q3_mapping_intervals.{ext}", dpi=220)
