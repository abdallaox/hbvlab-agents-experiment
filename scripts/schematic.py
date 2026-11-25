"""Fig. 1 (concept): HBV_Lab, its MCP server and the agent experiment. Writes paper/figures/fig0_schematic.pdf/.png.

All icons are drawn with matplotlib patches (vector, no external images or fonts)."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
INK, MUTED, LINE = "#1f1f1e", "#6b6a64", "#b5b4ad"
BLUE, BLUE_L, BLUE_M, ORANGE, ORANGE_L, GREY_L = "#2a78d6", "#eef4fd", "#9ec5f4", "#eb6834", "#fdeee6", "#f4f3ef"
GREEN = "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans"})
W, H = 100, 76
fig, ax = plt.subplots(figsize=(7.4, 7.4 * H / W))
ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")
LS = 2.35
FS, TS = 6.3, 7.0


# ---------------------------------------------------------------- icons (centre cx, cy, size s in axis units)
def icon_document(cx, cy, s, col, check=False):
    w, h = 0.75 * s, s
    x0, y0 = cx - w / 2, cy - h / 2
    f = 0.28 * s
    ax.add_patch(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h - f), (x0 + w - f, y0 + h), (x0, y0 + h)],
                         closed=True, fc="white", ec=col, lw=0.9))
    ax.add_patch(Polygon([(x0 + w - f, y0 + h), (x0 + w - f, y0 + h - f), (x0 + w, y0 + h - f)], closed=True,
                         fc=col, ec=col, lw=0.6))
    if check:
        ax.plot([x0 + 0.2 * w, x0 + 0.42 * w, x0 + 0.8 * w], [cy, cy - 0.2 * h, cy + 0.18 * h], color=GREEN, lw=1.4,
                solid_capstyle="round")
    else:
        for i in range(3):
            yy = y0 + h * (0.25 + 0.17 * i)
            ax.plot([x0 + 0.18 * w, x0 + 0.82 * w - (0.2 * w if i == 2 else 0)], [yy, yy], color=col, lw=0.8)


def icon_landscape(cx, cy, s, col):
    x0, y0 = cx - s / 2, cy - 0.38 * s
    ax.add_patch(Polygon([(x0, y0), (x0 + 0.38 * s, y0 + 0.62 * s), (x0 + 0.62 * s, y0 + 0.3 * s),
                          (x0 + 0.78 * s, y0 + 0.5 * s), (x0 + s, y0)], closed=True, fc="#dfe9d8", ec=col, lw=0.8))
    ax.plot([x0 + 0.15 * s, x0 + 0.42 * s, x0 + 0.3 * s, x0 + 0.62 * s],
            [y0, y0 + 0.12 * s, y0 + 0.22 * s, y0 + 0.3 * s], color=BLUE, lw=1.1)
    ax.add_patch(Circle((x0 + 0.85 * s, y0 + 0.72 * s), 0.09 * s, fc="#eda100", ec="none"))


def icon_chart(cx, cy, s, col):
    x0, y0 = cx - s / 2, cy - s / 2
    ax.plot([x0, x0, x0 + s], [y0 + s, y0, y0], color=col, lw=0.9)
    xs = [x0 + s * f for f in (0.05, 0.2, 0.32, 0.45, 0.58, 0.72, 0.85, 0.97)]
    ys = [y0 + s * f for f in (0.2, 0.65, 0.3, 0.8, 0.35, 0.5, 0.25, 0.4)]
    ax.plot(xs, ys, color=BLUE, lw=1.0)


def icon_robot(cx, cy, s, col):
    w, h = 0.95 * s, 0.72 * s
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={0.18 * s}",
                                fc="white", ec=col, lw=1.1))
    ax.plot([cx, cx], [cy + h / 2, cy + h / 2 + 0.18 * s], color=col, lw=1.0)
    ax.add_patch(Circle((cx, cy + h / 2 + 0.22 * s), 0.07 * s, fc=col, ec="none"))
    for dx in (-0.2, 0.2):
        ax.add_patch(Circle((cx + dx * s, cy + 0.07 * s), 0.08 * s, fc=col, ec="none"))
    ax.plot([cx - 0.17 * s, cx + 0.17 * s], [cy - 0.17 * s, cy - 0.17 * s], color=col, lw=1.0,
            solid_capstyle="round")
    for dx in (-1, 1):
        ax.add_patch(Rectangle((cx + dx * (w / 2) - (0.08 * s if dx < 0 else 0), cy - 0.1 * s), 0.08 * s, 0.2 * s,
                               fc=col, ec="none"))


def icon_lock(cx, cy, s, col):
    w, h = 0.7 * s, 0.5 * s
    ax.add_patch(Arc((cx, cy), 0.44 * s, 0.6 * s, theta1=0, theta2=180, color=col, lw=1.1))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h), w, h, boxstyle=f"round,pad=0,rounding_size={0.06 * s}",
                                fc="white", ec=col, lw=1.0))
    ax.add_patch(Circle((cx, cy - 0.25 * s), 0.06 * s, fc=col, ec="none"))


def icon_server(cx, cy, s, col):
    w, h = 0.9 * s, 0.26 * s
    for i in range(3):
        y = cy - 0.45 * s + i * 0.31 * s
        ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, h, boxstyle=f"round,pad=0,rounding_size={0.05 * s}",
                                    fc="white", ec=col, lw=0.9))
        ax.add_patch(Circle((cx + 0.3 * s, y + h / 2), 0.045 * s, fc=GREEN, ec="none"))
        ax.plot([cx - 0.32 * s, cx + 0.05 * s], [y + h / 2, y + h / 2], color=col, lw=0.7)


def icon_gear(cx, cy, s, col):
    n, r1, r2 = 8, 0.36 * s, 0.5 * s
    pts = []
    for i in range(n):
        a = 2 * np.pi * i / n
        for da, r in ((-0.22, r1), (-0.12, r2), (0.12, r2), (0.22, r1)):
            pts.append((cx + r * np.cos(a + da), cy + r * np.sin(a + da)))
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=col, lw=0.9))
    ax.add_patch(Circle((cx, cy), 0.14 * s, fc="white", ec=col, lw=0.9))


def icon_target(cx, cy, s, col):
    for r, fc in ((0.5, "white"), (0.33, ORANGE_L), (0.16, ORANGE)):
        ax.add_patch(Circle((cx, cy), r * s, fc=fc, ec=col, lw=0.8))


# ---------------------------------------------------------------- boxes and arrows
def box_h(n, title=True):
    return (3.4 if title else 1.2) + LS * n + 0.8


def box(x, y_top, w, title, lines, fc, ec, tcol=INK, ls="-", lw=1.0, fs=FS, ts=TS, r=1.0, icon=None, h=None):
    h = h or box_h(len(lines), bool(title))
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls))
    ty = y_top - 1.3
    if title:
        ax.text(x + 1.3, ty, title, fontsize=ts, fontweight="bold", color=tcol, va="top")
        ty -= 3.0
    for ln in lines:
        ax.text(x + 1.3, ty, ln, fontsize=fs, color=INK, va="top")
        ty -= LS
    if icon:
        fn, col = icon
        fn(x + w - 2.6, y_top - 2.0, 2.6, col)
    return y_top - h


def arrow(p0, p1, col=MUTED, lw=1.0, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=col, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


for x, t in ((1, "Inputs"), (27.5, "Agent and tools"), (80, "Outputs")):
    ax.text(x, H - 0.5, t, fontsize=8.3, fontweight="bold", color=MUTED, va="top")

# inputs
xi, wi = 1, 22.5
y = H - 5
b1 = box(xi, y, wi, "Prompt", ["task: best daily flow in", "an unseen period (NSE)", "budget: calls and runs",
                                "submit with a diagnosis"], GREY_L, LINE, icon=(icon_document, MUTED))
b2 = box(xi, b1 - 2.5, wi, "Catchment", ["CAMELS-GB attributes:", "area, land cover, soils,", "hydrogeology, reservoirs",
                                          "climate indices 1970-1997"], GREY_L, LINE, icon=(icon_landscape, MUTED))
b3 = box(xi, b2 - 2.5, wi, "Training data", ["daily P, T, PET and Q", "1970-1997 only,", "test years withheld"],
         GREY_L, LINE, icon=(icon_chart, MUTED))

# agent and controls
xa, wa = 27.5, 24
ya = H - 5
ba = box(xa, ya, wa, "AI agent (LLM)", ["reads each reply, reasons,", "chooses the next tool call", "here Claude Opus 5.5"],
         ORANGE_L, ORANGE, tcol=ORANGE, icon=(icon_robot, ORANGE))
bc = box(53.5, ya, 24, "Controls", ["listed tools only, sandboxed", "50 calls, 10,000 model runs", "every call logged"],
         "none", MUTED, tcol=MUTED, ls="--", lw=0.9, icon=(icon_lock, MUTED))
bus = xi + wi + 1.6
mids = [(b1 + y) / 2, (b2 + b1 - 2.5) / 2, (b3 + b2 - 2.5) / 2]
for yy in mids:
    ax.plot([xi + wi, bus], [yy, yy], color=MUTED, lw=0.9)
ax.plot([bus, bus], [mids[-1], (ba + ya) / 2], color=MUTED, lw=0.9)
arrow((bus, (ba + ya) / 2), (xa, (ba + ya) / 2), lw=0.9)

# MCP server
ym = ba - 10
xm, wm = 27.5, 50
gap, cw = 1.2, 23.1
cols = [[("Set up", ["create_model", "load_data"]),
         ("Run and calibrate", ["run_model", "calibrate", "evaluate_uncertainty"]),
         ("Finish", ["submit_calibration"])],
        [("Configure", ["get/set_parameter_ranges", "set_parameters", "set_initial_conditions"]),
         ("Compare", ["clone_model", "copy_parameters", "get_metrics", "compare_models"])]]
inner_top = ym - 5.2
bottoms = []
for ci, groups in enumerate(cols):
    yy = inner_top
    for gname, items in groups:
        nb = box(xm + 1.3 + ci * (cw + gap), yy, cw, gname, items, "white", BLUE_M, tcol=BLUE, fs=5.9, ts=6.5, r=0.7)
        yy = nb - gap
    bottoms.append(yy)
bm = min(bottoms) - 0.6
ax.add_patch(FancyBboxPatch((xm, bm), wm, ym - bm, boxstyle="round,pad=0,rounding_size=1.2", fc=BLUE_L, ec=BLUE, lw=1.2,
                            zorder=0))
ax.text(xm + 1.3, ym - 1.4, "HBV-Lab MCP server: typed tools", fontsize=7.4, fontweight="bold", color=BLUE, va="top")
icon_server(xm + wm - 2.8, ym - 2.4, 2.8, BLUE)

# agent <-> server
mid = (ba + ym) / 2
arrow((33, ba), (33, ym), col=ORANGE, lw=1.3)
arrow((41, ym), (41, ba), col=BLUE, lw=1.3)
ax.text(32.2, mid, "tool calls", fontsize=6.0, color=ORANGE, ha="right", va="center")
ax.text(42, mid, "replies: metrics, water balance,\nparameters at a bound, store levels", fontsize=6.0,
        color=BLUE, va="center")

# library
yl = bm - 3.2
bl = box(xm, yl, wm, "HBV-Lab Python library", ["snow > soil > response > routing,   calibration,   uncertainty"],
         "white", INK, lw=1.1, fs=6.0, icon=(icon_gear, INK))
arrow((xm + wm / 2, bm), (xm + wm / 2, yl), col=INK)
ax.text(xi, (yl + bl) / 2 + 1.2, "Pipeline (no agent)", fontsize=6.6, fontweight="bold", color=MUTED, va="center")
ax.text(xi, (yl + bl) / 2 - 1.3, "a script runs DDS directly", fontsize=6.2, color=MUTED, va="center")
arrow((xi + 19.5, (yl + bl) / 2), (xm, (yl + bl) / 2), col=MUTED, ls="--")

# outputs
xo, wo = 80, 19.5
bs = box(xo, H - 5, wo, "Submission", ["calibrated parameters", "and ranges", "diagnosis of the", "catchment",
                                        "transcript with", "summarised reasoning"], GREY_L, LINE, fs=6.0,
         icon=(lambda cx, cy, s, c: icon_document(cx, cy, s, c, check=True), MUTED))
be = box(xo, bs - 7, wo, "Evaluation", ["test period 1998-2008,", "unseen by the agent", "NSE, KGE, FHV, FLV",
                                         "audit of every run", "compared with pipeline,", "LC-HBV and FUSE"],
         GREY_L, LINE, fs=6.0, icon=(icon_target, ORANGE))
arrow((xm + wm, ym - 6), (xo - 0.4, bs + 3), col=BLUE, lw=1.2)
arrow((xo + wo / 2, bs), (xo + wo / 2, bs - 7))
arrow((xm + wm, (yl + bl) / 2), (xo + wo / 2, be), ls="--", rad=0.25)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig0_schematic.{ext}"), dpi=300, bbox_inches="tight")
print("wrote", os.path.join(OUT, "fig0_schematic.pdf"))
