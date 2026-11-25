"""HBV model structure as used in HBV-Lab: stores, fluxes and the 14 parameters, with the symbols and parameter
descriptions of the paper (Section 2.1 and Table 1). Writes paper/figures/fig_hbv.pdf/.png."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
INK, MUTED, PAR = "#1f1f1e", "#6b6a64", "#2a78d6"
WATER, SNOW = "#d6e6fa", "#eef2f6"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.set_xlim(0, 110), ax.set_ylim(0, 92), ax.axis("off")
X0, W = 31, 25                      # tank column
FS, PS = 8.2, 7.8                   # flux and parameter font sizes
XL = X0 - 2.5                       # right edge of the parameter labels


def tank(y, h, fill, fill_col, name, store, store_at, closed=False):
    ax.add_patch(Rectangle((X0, y), W, h * fill, fc=fill_col, ec="none", zorder=1))
    xs, ys = ([X0, X0, X0 + W, X0 + W, X0], [y + h, y, y, y + h, y + h]) if closed else ([X0, X0, X0 + W, X0 + W], [y + h, y, y, y + h])
    ax.plot(xs, ys, color=INK, lw=1.4, zorder=3)
    ax.text(X0 + W / 2, y + h - 1.0, name, ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK, zorder=4)
    ax.text(X0 + W / 2, y + h * store_at, store, ha="center", va="center", fontsize=8.6, color=INK, zorder=4)


def level(y, label, ls, side):
    ax.plot([X0, X0 + W], [y, y], color=MUTED, lw=0.8, ls=ls, zorder=2)
    if side == "left":
        ax.text(X0 + 0.8, y + 0.4, label, fontsize=PS, color=PAR, va="bottom", zorder=4)
    else:
        ax.text(X0 + W - 0.8, y + 0.4, label, fontsize=PS, color=PAR, va="bottom", ha="right", zorder=4)


def arrow(p0, p1, col=INK, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color=col, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=5))


def params(y, lines):
    ax.text(XL, y, "\n".join(lines), ha="right", va="center", fontsize=PS, color=PAR, linespacing=1.55)


# positions (bottom, height) of the four stores
SNOW_Y, SNOW_H = 68, 12
SOIL_Y, SOIL_H = 43, 17.5
UZ_Y, UZ_H = 21, 14.5
LZ_Y, LZ_H = 1, 13

# inputs
ax.text(X0 + 8, 91, "$P$", ha="center", va="top", fontsize=9.5, color=INK)
ax.text(X0 + 17, 91, "$T$", ha="center", va="top", fontsize=9.5, color=MUTED)
arrow((X0 + 8, 87.2), (X0 + 8, SNOW_Y + SNOW_H))
arrow((X0 + 17, 87.2), (X0 + 17, SNOW_Y + SNOW_H), col=MUTED, lw=1.0, ls=":")
# potential evapotranspiration: a forcing of the soil routine
xe, ye = X0 + W + 16, SOIL_Y + SOIL_H * 0.88
ax.text(xe, 91, "$E_p$", ha="center", va="top", fontsize=9.5, color=MUTED)
ax.plot([xe, xe], [87.2, ye], color=MUTED, lw=1.0, ls=":", zorder=4)
arrow((xe, ye), (X0 + W, ye), col=MUTED, lw=1.0, ls=":")

# snow
tank(SNOW_Y, SNOW_H, 0.45, SNOW, "Snow", "$SP$, liquid water", 0.22, closed=True)
params(SNOW_Y + SNOW_H / 2, ["$TT$  threshold temperature", "$CFMAX$  degree-day factor", "$SFCF$  snowfall correction",
                             "$CFR$  refreezing coefficient", "$CWH$  water holding capacity"])
arrow((X0 + W / 2, SNOW_Y), (X0 + W / 2, SOIL_Y + SOIL_H))
ax.text(X0 + W / 2 + 1.2, (SNOW_Y + SOIL_Y + SOIL_H) / 2, "rain and snowmelt  $I$", fontsize=FS, color=INK,
        va="center")

# soil
tank(SOIL_Y, SOIL_H, 0.22, WATER, "Soil moisture", "$SM$", 0.11, closed=True)
level(SOIL_Y + SOIL_H * 0.6, "$FC$", "--", "right")
level(SOIL_Y + SOIL_H * 0.39, "$LP\\,FC$", ":", "right")
params(SOIL_Y + SOIL_H / 2, ["$FC$  field capacity", "$LP$  evapotranspiration limit", "$BETA$  recharge shape"])
# actual evapotranspiration: L-shaped, out of the soil to the right, then up beside the E_p line
xa, ya = xe + 6, SOIL_Y + SOIL_H * 0.15
ax.plot([X0 + W, xa], [ya, ya], color=INK, lw=1.2, zorder=4)
arrow((xa, ya), (xa, 87.2), lw=1.2)
ax.text(xa, 91, "$E_a$", ha="center", va="top", fontsize=9.5, color=INK)
arrow((X0 + W / 2, SOIL_Y), (X0 + W / 2, UZ_Y + UZ_H))
# water above FC goes directly to the upper zone
arrow((X0 + 4, SOIL_Y), (X0 + 4, UZ_Y + UZ_H), lw=1.1)
ax.text(X0 + 3, (SOIL_Y + UZ_Y + UZ_H) / 2, "excess above $FC$", fontsize=FS, color=INK, va="center", ha="right")
ax.text(X0 + W / 2 + 1.2, (SOIL_Y + UZ_Y + UZ_H) / 2, "recharge  $R = I\\,(SM/FC)^{BETA}$", fontsize=FS, color=INK,
        va="center")

# upper zone
tank(UZ_Y, UZ_H, 0.66, WATER, "Upper zone", "$S_{UZ}$", 0.18)
uzl = UZ_Y + UZ_H * 0.4
level(uzl, "$UZL$", "--", "right")
params(UZ_Y + UZ_H / 2, ["$K_0$  quick flow recession", "$K_1$  intermediate recession", "$UZL$  quick flow threshold"])
xc = X0 + W + 17
q0y, q1y = UZ_Y + UZ_H * 0.53, UZ_Y + UZ_H * 0.12
arrow((X0 + W, q0y), (xc, q0y))
ax.text(X0 + W + 1, q0y + 0.9, "$Q_0 = K_0\\,\\max(S_{UZ}-UZL,\\,0)$", fontsize=FS, color=INK, va="bottom")
arrow((X0 + W, q1y), (xc, q1y))
ax.text(X0 + W + 1, q1y + 0.9, "$Q_1 = K_1\\,S_{UZ}$", fontsize=FS, color=INK, va="bottom")
arrow((X0 + W / 2, UZ_Y), (X0 + W / 2, LZ_Y + LZ_H))
ax.text(X0 + W / 2 + 1.2, (UZ_Y + LZ_Y + LZ_H) / 2, "percolation $\\leq PERC$", fontsize=FS, color=INK,
        va="center")
params((UZ_Y + LZ_Y + LZ_H) / 2, ["$PERC$  percolation"])

# lower zone
tank(LZ_Y, LZ_H, 0.6, WATER, "Lower zone", "$S_{LZ}$", 0.3)
params(LZ_Y + LZ_H / 2, ["$K_2$  baseflow recession"])
q2y = LZ_Y + LZ_H * 0.3
arrow((X0 + W, q2y), (xc, q2y))
ax.text(X0 + W + 1, q2y + 0.9, "$Q_2 = K_2\\,S_{LZ}$", fontsize=FS, color=INK, va="bottom")

# collector and routing
ax.plot([xc, xc], [q2y, q0y], color=INK, lw=1.3, zorder=4)
ym = (q2y + q0y) / 2
# routing: a triangular weighting function, drawn as a triangle on the flow line
rx, rw, th = xc + 5, 13, 8
arrow((xc, ym), (rx, ym))
ax.add_patch(Polygon([(rx, ym), (rx + rw / 2, ym + th), (rx + rw, ym)], closed=True, fc=WATER, ec=INK, lw=1.2,
                     zorder=4))
ax.text(rx + rw / 2, ym + th + 1.2, "routing", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=INK)
ax.text(rx + rw / 2, ym - 1.4, "$MAXBAS$\nrouting base", ha="center", va="top", fontsize=PS, color=PAR,
        linespacing=1.4)
arrow((rx + rw, ym), (rx + rw + 11, ym))
ax.text(rx + rw + 6, ym + 1.0, "simulated\nflow  $Q$", ha="center", va="bottom", fontsize=FS, color=INK,
        linespacing=1.3)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_hbv.{ext}"), dpi=300, bbox_inches="tight")
print("wrote", os.path.join(OUT, "fig_hbv.pdf"))
