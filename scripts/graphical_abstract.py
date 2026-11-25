"""Graphical abstract for EMS: landscape, readable at 13 cm, at least 1328 px wide.
Left: the workflow (inputs -> AI agent <-> MCP tools -> HBV_Lab). Right: agent against pipeline on 95 catchments.
Writes submission/graphical_abstract.pdf and .png (300 dpi, 13 cm wide)."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission")
os.makedirs(OUT, exist_ok=True)
INK, MUTED = "#1f1f1e", "#6b6a64"
BLUE, BLUE_L, ORANGE, ORANGE_L, GREY_L = "#2a78d6", "#eef4fd", "#eb6834", "#fdeee6", "#f4f3ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})

W_CM, H_CM = 13.0, 6.0
fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
ax = fig.add_axes([0, 0, 0.56, 1])
ax.set_xlim(0, 56), ax.set_ylim(0, 60), ax.axis("off")


def box(x, y, w, h, fc, ec, title, sub, tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=7.2, fontweight="bold", color=tcol)
    ax.text(x + w / 2, y + h - 8.6, sub, ha="center", va="top", fontsize=6.0, color=INK, linespacing=1.25)


def arrow(p0, p1, col=MUTED, lw=1.1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=col, lw=lw, shrinkA=0, shrinkB=0))


box(1, 36, 22, 20, GREY_L, "#b5b4ad", "Catchment", "CAMELS-GB data\n1970-1997 only\nfixed prompt")
box(31, 36, 23, 20, ORANGE_L, ORANGE, "AI agent", "chooses ranges,\nstates, calibrations,\nwrites a diagnosis", ORANGE)
box(31, 4, 23, 22, BLUE_L, BLUE, "MCP tools", "typed tools only\nfile sandbox\nrun budget, all logged", BLUE)
box(1, 4, 22, 22, "white", INK, "HBV-Lab", "open-source Python\nHBV model, DDS\n661 GB catchments")
arrow((23, 46), (31, 46))
arrow((40, 36), (40, 26), ORANGE, 1.3)
arrow((45, 26), (45, 36), BLUE, 1.3)
arrow((31, 15), (23, 15), BLUE, 1.3)

# right: result
rep = pd.read_csv(os.path.join(ROOT, "results", "agent_representative.csv"), index_col="gauge")
axr = fig.add_axes([0.68, 0.3, 0.29, 0.5])
for v, lab, col, ls in ((rep.pipeline_NSE, "pipeline", BLUE, "-"), (rep.agent_NSE, "agent", ORANGE, "--")):
    v = np.sort(v.dropna().to_numpy())
    axr.plot(v, (np.arange(1, len(v) + 1) - 0.5) / len(v), color=col, ls=ls, lw=1.6,
             label=f"{lab} ({np.median(v):.2f})")
axr.set_xlim(0.4, 1.0), axr.set_ylim(0, 1)
axr.set_xlabel("Test NSE, 1998-2008", fontsize=6.5)
axr.set_ylabel("Fraction of catchments", fontsize=6.5)
axr.tick_params(labelsize=6)
for sp in ("top", "right"):
    axr.spines[sp].set_visible(False)
axr.grid(color="#e4e3de", lw=0.5)
axr.legend(loc="center left", bbox_to_anchor=(-0.02, 0.62), frameon=False, fontsize=5.8, title="median NSE", title_fontsize=5.8)
fig.text(0.62, 0.97, "Agent matches automatic\ncalibration on 95 catchments", fontsize=7.0, fontweight="bold",
         color=INK, va="top")
fig.text(0.62, 0.03, "every modelling choice explained and auditable", fontsize=6.0, color=MUTED)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"graphical_abstract.{ext}"), dpi=300)
w_px = int(W_CM / 2.54 * 300)
print("wrote submission/graphical_abstract.pdf/.png,", w_px, "px wide")
