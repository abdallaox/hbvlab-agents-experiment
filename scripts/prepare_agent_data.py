"""Write the inputs an agent gets for each catchment: the training-period data file and a catchment description.

For each catchment: data/agent_inputs/<gauge_id>/forcing.csv (1970-10-01 to 1997-12-31 only, so the test period
cannot leak) and catchment.json (name, location and static attributes; climate computed from the training data only). Flow-derived signatures (baseflow
index, runoff ratio and so on) are left out, as in the benchmark study's attribute set.

Usage: python scripts/prepare_agent_data.py [--n 64] [--subset config/subset_ordered.csv]
       [--npz ../camels_uk_exp/data/processed/camels_gb.npz] [--camels ../camels_uk_exp/data/camels_gb/data]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--subset", default="config/subset_ordered.csv")
p.add_argument("--n", type=int, default=64, help="first n catchments of the ordered subset")
p.add_argument("--gauges", type=int, nargs="*", help="explicit gauge ids instead of --n")
p.add_argument("--all", action="store_true", help="all catchments in config/basins_all.txt")
p.add_argument("--npz", default="../camels_uk_exp/data/processed/camels_gb.npz")
p.add_argument("--camels", default="../camels_uk_exp/data/camels_gb/data")
p.add_argument("--out", default="data/agent_inputs")
p.add_argument("--start", default="1970-10-01")
p.add_argument("--end", default="1997-12-31")
a = p.parse_args()

if a.gauges:
    gauges = a.gauges
elif a.all:
    gauges = pd.read_csv("config/basins_all.txt", header=None)[0].astype(int).tolist()
else:
    gauges = pd.read_csv(a.subset)["gauge_id"].head(a.n).astype(int).tolist()
with np.load(a.npz) as z:  # read each array once: indexing an npz decompresses the whole array every time
    d = {k: z[k] for k in ("forcing", "target", "basins", "dates", "forcing_names")}
names = list(d["forcing_names"])
dates = pd.to_datetime(d["dates"])
keep = (dates >= a.start) & (dates <= a.end)
read = lambda f: pd.read_csv(f"{a.camels}/CAMELS_GB_{f}_attributes.csv", index_col=0)
topo, clim, land, soil, geo, human = (read(f) for f in
                                      ("topographic", "climatic", "landcover", "soil", "hydrogeology", "humaninfluence"))

def climate(df):
    p, pet, t = df["Precipitation"], df["PotentialET"], df["Temperature"]
    pm = float(p.mean())
    years = len(df) / 365.25
    return {"mean_precipitation_mm_per_day": round(pm, 2),
            "mean_pet_mm_per_day": round(float(pet.mean()), 2),
            "aridity_pet_over_p": round(float(pet.mean()) / pm, 2),
            "fraction_of_precipitation_as_snow": round(float(p[t < 0].sum() / p.sum()), 3),
            "high_precipitation_days_per_year": round(float((p >= 5 * pm).sum()) / years, 2)}


for g in gauges:
    i = int(np.where(d["basins"] == g)[0][0])
    f = d["forcing"][i][keep]
    df = pd.DataFrame({
        "Date": dates[keep].strftime("%Y%m%d"),
        "Precipitation": f[:, names.index("precipitation")].round(3),
        "Temperature": f[:, names.index("temperature")].round(3),
        "PotentialET": f[:, names.index("pet")].round(3),
        "Discharge": d["target"][i][keep].round(4),
    })
    t, c, l, s, h, u = topo.loc[g], clim.loc[g], land.loc[g], soil.loc[g], geo.loc[g], human.loc[g]
    meta = {
        "gauge_id": int(g),
        "gauge_name": str(t["gauge_name"]),
        "location": {"latitude": float(t["gauge_lat"]), "longitude": float(t["gauge_lon"])},
        "topography": {"area_km2": float(t["area"]), "mean_elevation_m": float(t["elev_mean"]),
                       "gauge_elevation_m": float(t["gauge_elev"]), "max_elevation_m": float(t["elev_max"]),
                       "mean_slope_m_per_km": float(t["dpsbar"])},
        # computed from the training-period data the agent gets, not from CAMELS-GB (whose climate attributes cover
        # 1970-2015 and so include the test years). Definitions as in CAMELS-GB (Coxon et al. 2020).
        "climate_1970_1997": climate(df),
        "land_cover_percent": {"dominant": str(l["dom_land_cover"]), "deciduous_woodland": float(l["dwood_perc"]),
                               "evergreen_woodland": float(l["ewood_perc"]), "grass_and_pasture": float(l["grass_perc"]),
                               "shrub": float(l["shrub_perc"]), "crops": float(l["crop_perc"]),
                               "urban": float(l["urban_perc"])},
        "soils": {"sand_percent": float(s["sand_perc"]), "silt_percent": float(s["silt_perc"]),
                  "clay_percent": float(s["clay_perc"]), "organic_percent": float(s["organic_perc"]),
                  "total_available_water_capacity_mm": float(s["tawc"]), "root_depth_m": float(s["root_depth"]),
                  "depth_to_bedrock_m": float(s["soil_depth_pelletier"])},
        "hydrogeology_percent_of_area": {
            "highly_productive_fractured_aquifer": float(h["frac_high_perc"]),
            "moderately_productive_fractured_aquifer": float(h["frac_mod_perc"]),
            "low_productivity_fractured_aquifer": float(h["frac_low_perc"]),
            "highly_productive_intergranular_aquifer": float(h["inter_high_perc"]),
            "moderately_productive_intergranular_aquifer": float(h["inter_mod_perc"]),
            "low_productivity_intergranular_aquifer": float(h["inter_low_perc"]),
            "rocks_with_essentially_no_groundwater": float(h["no_gw_perc"])},
        "reservoirs": {"number": int(u["num_reservoir"]), "total_capacity_megalitres": float(u["reservoir_cap"])},
    }
    out = os.path.join(a.out, str(g))
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "forcing.csv"), index=False)
    with open(os.path.join(out, "catchment.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
print(f"wrote {len(gauges)} catchments to {a.out} ({df.Date.iloc[0]} to {df.Date.iloc[-1]}, {len(df)} days)")
