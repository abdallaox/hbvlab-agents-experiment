"""Paper figures, written to paper/figures/ as PDF and PNG.

fig1_maps:        pipeline test NSE across Great Britain, and its difference from LC-HBV and from FUSE Sacramento
fig2_cdf:         cumulative distributions of test NSE for the pipeline and the benchmarks
fig3_agent:       agent against pipeline on the 95 representative catchments: CDF and per-catchment scatter
fig4_hydrographs: observed and simulated flow in three showcase catchments, two test years

Reads results/benchmark_per_catchment.csv (compare_benchmarks.py), results/agent_per_catchment.csv
(analyse_agent.py) and results/runs_scored.csv (evaluate_runs.py).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from HBV_Lab import HBVModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

per = pd.read_csv(os.path.join(ROOT, "results", "benchmark_per_catchment.csv"), index_col="gauge")
topo = pd.read_csv(os.path.join(ROOT, "..", "camels_uk_exp", "data", "camels_gb", "data",
                                "CAMELS_GB_topographic_attributes.csv"), index_col="gauge_id")
_sub = pd.read_csv(os.path.join(ROOT, "config", "subset_ordered.csv"))
show = list(_sub[_sub["round"] <= 6].gauge_id)  # the 95 catchments of the agent experiment
xy = topo[["gauge_easting", "gauge_northing"]] / 1e3


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"{name}.pdf"))


from matplotlib import colors as mcolors

_z = np.load(os.path.join(ROOT, "..", "camels_uk_exp", "data", "processed", "outline_rings.npz"))
RINGS = [_z[k] for k in sorted(_z.files, key=lambda k: int(k[1:]))]  # land outline, as in the benchmark study


def map_frame(ax, title):
    for rx, ry in RINGS:
        ax.plot(rx, ry, color="#cccccc", lw=0.6, zorder=1)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=8.5, color=INK)


# Fig. 1: maps. Both panels use classes with counts; the 95 agent catchments get a thin red outline.
AGENT = set(show)


def class_map(ax, frame, classes, title, legend_title, agent=True):
    """frame: column v plus coordinates. classes: (lo, hi, colour, size, label). Agent catchments outlined."""
    for lo_, hi_, col, size, lab in classes:
        m = (frame.v > lo_) & (frame.v <= hi_)
        is_ag = frame.index.isin(AGENT) if agent else np.zeros(len(frame), bool)
        z = 3 if size > 6 else 2
        ax.scatter(frame.gauge_easting[m & ~is_ag], frame.gauge_northing[m & ~is_ag], color=col, s=size, lw=0, zorder=z,
                   label=f"{lab} ({m.sum()})")
        ax.scatter(frame.gauge_easting[m & is_ag], frame.gauge_northing[m & is_ag], color=col, s=max(size, 9) * 1.15,
                   lw=0.55, edgecolors=(0.89, 0.29, 0.28, 0.75), zorder=z + 1)
    if agent:
        ax.scatter([], [], color="white", s=12, lw=0.7, edgecolors=(0.89, 0.29, 0.28, 0.75),
                   label=f"agent catchment ({frame.index.isin(AGENT).sum()})")
    map_frame(ax, title)
    h, l = ax.get_legend_handles_labels()
    order_ = (list(range(len(h) - 1))[::-1] + [len(h) - 1]) if agent else list(range(len(h)))[::-1]
    ax.legend([h[i] for i in order_], [l[i] for i in order_], title=legend_title, loc="upper center",
              bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=7, title_fontsize=7.2, handletextpad=0.3,
              labelspacing=0.3, markerscale=1.3)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 5.8))
vir = plt.get_cmap("viridis_r")
d = per[["pipeline_NSE"]].rename(columns={"pipeline_NSE": "v"}).join(xy).dropna()
nse_classes = [(-np.inf, 0.6, vir(0.02), 11, "< 0.6"), (0.6, 0.7, vir(0.25), 10, "0.6 to 0.7"),
               (0.7, 0.8, vir(0.5), 9, "0.7 to 0.8"), (0.8, 0.9, vir(0.75), 9, "0.8 to 0.9"),
               (0.9, np.inf, vir(0.98), 9, "> 0.9")]
class_map(ax1, d, nse_classes, f"(a) Pipeline, test NSE (median {d.v.median():.2f})", "Test NSE (catchments)")

diff = (per.pipeline_NSE - per["FUSE-Sacramento_NSE"]).to_frame("v").join(xy).dropna()
diff_classes = [(-np.inf, -0.10, "#e34948", 13, "FUSE better by > 0.10"),
                (-0.10, -0.03, "#f2a3a2", 11, "FUSE better by 0.03 to 0.10"),
                (-0.03, 0.03, "#d9d8d3", 5, "similar (within 0.03)"),
                (0.03, 0.10, "#9ec5f4", 11, "HBV-Lab better by 0.03 to 0.10"),
                (0.10, np.inf, "#2a78d6", 13, "HBV-Lab better by > 0.10")]
class_map(ax2, diff, diff_classes, "(b) Pipeline against FUSE Sacramento", "Test NSE difference (catchments)",
          agent=False)
fig.tight_layout(w_pad=0.5)
save(fig, "fig1_maps")

# Fig. 2: CDFs
fig, ax = plt.subplots(figsize=(4.8, 4.2))
for m in ("FUSE-TOPMODEL", "FUSE-ARNO/VIC", "FUSE-PRMS"):
    v = np.sort(per[f"{m}_NSE"].dropna())
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=MAGENTA, lw=0.8, alpha=0.55, ls=(0, (5, 1, 1, 1)))
series = [("pipeline", "HBV-Lab pipeline", BLUE, "-", 2.0), ("LC-HBV", "LC-HBV", ORANGE, "--", 1.5),
          ("FUSE-Sacramento", "FUSE, four structures", MAGENTA, (0, (5, 1, 1, 1)), 1.5)]
for key, lab, col, ls, lw in series:
    v = np.sort(per[f"{key}_NSE"].dropna())
    txt = f"n={len(v)}, median {np.median(v):.2f}" if key != "FUSE-Sacramento" else f"n={len(v)}, medians 0.76-0.80"
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=lw, label=f"{lab}  ({txt})")
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
ax.set_xlabel("Test NSE, 1998-2008"), ax.set_ylabel("Fraction of catchments")
ax.grid(color=GRID, lw=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False, fontsize=6.8)
fig.tight_layout()
save(fig, "fig2_cdf")

# Fig. 3: agent against pipeline on the representative sample (95 catchments, one agent run each)
rep = pd.read_csv(os.path.join(ROOT, "results", "agent_representative.csv"), index_col="gauge")
sc = pd.read_csv(os.path.join(ROOT, "results", "runs_scored.csv"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))
for v, lab, col, ls in ((rep.pipeline_NSE, "Pipeline (median of 3 seeds)", BLUE, "-"),
                        (rep.agent_NSE, "Agent (one run)", ORANGE, "--"),
                        (rep["LC-HBV_NSE"], "LC-HBV", MUTED, ":")):
    v = np.sort(v.dropna().to_numpy())
    ax1.plot(v, (np.arange(1, len(v) + 1) - 0.5) / len(v), color=col, ls=ls, lw=1.6,
             label=f"{lab}, median {np.median(v):.2f}")
ax1.set_xlim(0.3, 1.0), ax1.set_ylim(0, 1)
ax1.set_xlabel("Test NSE, 1998-2008"), ax1.set_ylabel(f"Fraction of {len(rep)} catchments")
ax1.grid(color=GRID, lw=0.6)
ax1.legend(loc="upper left", frameon=False, fontsize=6.6)
ax1.set_title("(a) Distribution over catchments", loc="left", fontsize=8, color=INK)

lo, hi = 0.3, 1.0
ax2.fill_between([lo, hi], [lo - 0.01, hi - 0.01], [lo + 0.01, hi + 0.01], color=GRID, lw=0)
ax2.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
ax2.errorbar(rep.pipeline_NSE, rep.agent_NSE, xerr=[rep.pipeline_NSE - rep.pipeline_NSE_min,
             rep.pipeline_NSE_max - rep.pipeline_NSE], fmt="o", ms=3.5, color=ORANGE, ecolor="#9ec5f4",
             elinewidth=0.9, capsize=0, mec="white", mew=0.4)
off = rep[(rep.agent_NSE < lo) | (rep.pipeline_NSE < lo)]
for g_, lab, dx, dy in ((39014, "Ver", -0.015, 0.015), (54003, "Vyrnwy", -0.015, 0.012), (39037, "", 0, 0)):
    if g_ in rep.index and lab:
        ax2.annotate(lab, (rep.loc[g_, "pipeline_NSE"], rep.loc[g_, "agent_NSE"]),
                     (rep.loc[g_, "pipeline_NSE"] + dx, rep.loc[g_, "agent_NSE"] + dy), fontsize=7, color=INK,
                     ha="right" if dx < 0 else "left")
ax2.text(0.33, 0.95, "agent better", fontsize=7, color=MUTED)
ax2.text(0.72, 0.33, "pipeline better", fontsize=7, color=MUTED)
if len(off):
    ax2.text(0.33, 0.88, f"{len(off)} catchments below 0.3 not shown", fontsize=6.3, color=MUTED)
ax2.set_xlim(lo, hi), ax2.set_ylim(lo, hi), ax2.set_aspect("equal")
ax2.set_xlabel("Pipeline test NSE (bar: range of 3 seeds)"), ax2.set_ylabel("Agent test NSE")
ax2.grid(color=GRID, lw=0.6)
ax2.set_title("(b) Per catchment", loc="left", fontsize=8, color=INK)
fig.tight_layout()
save(fig, "fig3_agent")

# Fig. 4: hydrographs, run of median test NSE for agent and pipeline
with np.load(os.path.join(ROOT, "..", "camels_uk_exp", "data", "processed", "camels_gb.npz")) as z:
    D = {k: z[k] for k in ("forcing", "target", "basins", "dates", "forcing_names")}
fn = list(D["forcing_names"])
sc = pd.read_csv(os.path.join(ROOT, "results", "runs_scored.csv"))


def simulate(g, path):
    i = int(np.where(D["basins"] == g)[0][0])
    f = D["forcing"][i]
    df = pd.DataFrame({"Date": pd.to_datetime(D["dates"]), "P": f[:, fn.index("precipitation")].astype(float),
                       "T": f[:, fn.index("temperature")].astype(float), "PET": f[:, fn.index("pet")].astype(float),
                       "Q": D["target"][i].astype(float)})
    params = json.load(open(os.path.join(path, "submission.json")))["parameters"]
    m = HBVModel()
    m.load_data(data=df, date_column="Date", precip_column="P", temp_column="T", pet_column="PET", obs_q_column="Q",
                date_format="%Y-%m-%d", start_date="1970-10-01", warmup_end="1997-12-31", end_date="2008-12-31")
    m.set_parameters({grp: {k: {"default": params[k]} for k in m.params[grp]} for grp in m.params})
    m.run(verbose=False)
    return pd.Series(np.asarray(m.results["discharge"]), index=pd.to_datetime(m.results["dates"])), \
        pd.Series(np.asarray(m.results["observed_q"]), index=pd.to_datetime(m.results["dates"]))


def median_run(g, agent):
    r = sc[(sc.agent == agent) & (sc.gauge == g)].dropna(subset=["NSE_test"]).sort_values("NSE_test")
    row = r.iloc[len(r) // 2]
    kind = "agents" if agent == "claude-api" else "baseline"
    return os.path.join(ROOT, "runs", kind, agent, str(g), row["repeat"]), row.NSE_test


hydro = [(27009, "Ouse at Skelton, typical catchment"), (39014, "Ver at Hansteads, chalk, abstraction"),
         (54003, "Vyrnwy at Vyrnwy Reservoir, reservoir outflow")]
t0, t1 = "2000-10-01", "2002-09-30"
fig, axs = plt.subplots(len(hydro), 1, figsize=(7.2, 6.4), sharex=True)
for ax, (g, title) in zip(axs, hydro):
    pa, na = median_run(g, "claude-api")
    pp, npl = median_run(g, "pipeline")
    qa, obs = simulate(g, pa)
    qp, _ = simulate(g, pp)
    w = slice(t0, t1)
    ax.plot(obs[w].index, obs[w], color=INK, lw=1.0, label="Observed")
    ax.plot(qp[w].index, qp[w], color=BLUE, lw=1.1, label="Pipeline")
    ax.plot(qa[w].index, qa[w], color=ORANGE, lw=1.1, ls="--", label="Agent")
    nse_txt = f"test NSE: pipeline {npl:.2f}, agent {na:.2f}"
    ax.set_title(f"{title} ({g}).  {nse_txt}", loc="left", fontsize=7.5, color=INK)
    ax.set_ylabel("Flow (mm/day)")
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, lw=0.5)
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, 1.025))
fig.tight_layout()
save(fig, "fig4_hydrographs")
