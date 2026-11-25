"""Metric definitions copied verbatim from the benchmark study (camels-gb-benchmark, src/gbhydro/metrics.py), which
follow NeuralHydrology, so pipeline, agent, FUSE and LC-HBV are scored the same way. KGE is Gupta et al. (2009).
"""
import numpy as np


def _clean(sim, obs):
    m = np.isfinite(obs) & np.isfinite(sim)
    return sim[m], obs[m]


def nse(sim, obs):
    sim, obs = _clean(sim, obs)
    if len(obs) < 2 or obs.std() == 0:
        return np.nan
    return 1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(sim, obs):
    """Gupta et al. 2009: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), alpha = std ratio, beta = mean ratio."""
    sim, obs = _clean(sim, obs)
    if len(obs) < 2 or obs.std() == 0 or obs.mean() == 0:
        return np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def fdc_slice(x, lo, hi):
    """Values of x between flow-duration exceedance probabilities lo..hi (0 = highest flow)."""
    x = np.sort(x)[::-1]
    n = len(x)
    return x[int(lo * n):max(int(hi * n), int(lo * n) + 1)]


def fhv(sim, obs, h=0.02):
    """Peak-flow bias in %, Yilmaz et al. 2008: top 2% of the flow duration curve."""
    sim, obs = _clean(sim, obs)
    if len(obs) < 50:
        return np.nan
    s, o = fdc_slice(sim, 0, h), fdc_slice(obs, 0, h)
    return 100 * (s.sum() - o.sum()) / (o.sum() + 1e-6)


def flv(sim, obs, l=0.7):
    """Low-flow bias in % (log space), Yilmaz et al. 2008: bottom 30% of the flow duration curve."""
    sim, obs = _clean(sim, obs)
    if len(obs) < 50:
        return np.nan
    s, o = fdc_slice(sim, l, 1.0), fdc_slice(obs, l, 1.0)
    s, o = np.log(np.maximum(s, 1e-6)), np.log(np.maximum(o, 1e-6))
    s_min, o_min = s.min(), o.min()
    qsl, qol = np.sum(s - s_min), np.sum(o - o_min)
    return -100 * (qsl - qol) / (qol + 1e-6)


def fms(sim, obs, lo=0.2, hi=0.7):
    """Slope of the mid-section of the flow duration curve, bias in %, Yilmaz et al. 2008."""
    sim, obs = _clean(sim, obs)
    if len(obs) < 50:
        return np.nan
    def slope(x):
        x = np.sort(x)[::-1]
        a, b = x[int(lo * len(x))], x[int(hi * len(x))]
        return np.log(max(a, 1e-6)) - np.log(max(b, 1e-6))
    so, ss = slope(obs), slope(sim)
    return 100 * (ss - so) / (so + 1e-6)


def pearson_r(sim, obs):
    sim, obs = _clean(sim, obs)
    if len(obs) < 2 or obs.std() == 0 or sim.std() == 0:
        return np.nan
    return np.corrcoef(sim, obs)[0, 1]


def alpha(sim, obs):
    """KGE variability ratio: std(sim) / std(obs)."""
    sim, obs = _clean(sim, obs)
    return np.nan if len(obs) < 2 or obs.std() == 0 else sim.std() / obs.std()


def beta(sim, obs):
    """KGE bias ratio: mean(sim) / mean(obs)."""
    sim, obs = _clean(sim, obs)
    return np.nan if len(obs) < 2 or obs.mean() == 0 else sim.mean() / obs.mean()


def pbias(sim, obs):
    """Percent bias: 100 * (sum sim - sum obs) / sum obs."""
    sim, obs = _clean(sim, obs)
    return np.nan if len(obs) < 2 or obs.sum() == 0 else 100 * (sim.sum() - obs.sum()) / obs.sum()


ALL = {"NSE": nse, "KGE": kge, "FHV": fhv, "FLV": flv, "FMS": fms}
EXTRA = {"r": pearson_r, "alpha": alpha, "beta": beta, "PBIAS": pbias}


def all_metrics(sim, obs) -> dict:
    return {k: float(f(sim, obs)) for k, f in ALL.items()}
