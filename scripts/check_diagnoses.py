"""Check agent diagnoses against CAMELS-GB records (representative sample, repeat 1: 95 catchments).

Three processes, each with a record in CAMELS-GB and a keyword test on the diagnosis text:
  reservoir   recorded: num_reservoir > 0                     named: reservoir, regulat, impound, dam, hydropower
  groundwater recorded: highly productive fractured aquifer   named: chalk, aquifer, limestone (not "groundwater",
              (chalk/limestone) > 30% of the area           which the agent uses for HBV's lower zone)
  abstraction recorded: surface + groundwater abstraction     named: abstract
              > 10% of mean flow (CAMELS-GB, 1999-2014)
Reservoirs and hydrogeology are in the agent's catchment description, abstraction is not, so naming abstraction is
an inference. Keyword tests are coarse: they show whether the process is named, not whether the reasoning is right.
Also reported: the water balance check the agent states (observed runoff below P - PET).
Writes results/diagnosis_check.csv and prints a summary.
"""
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "..", "camels_uk_exp", "data", "camels_gb", "data")
hum = pd.read_csv(os.path.join(D, "CAMELS_GB_humaninfluence_attributes.csv"), index_col="gauge_id")
geo = pd.read_csv(os.path.join(D, "CAMELS_GB_hydrogeology_attributes.csv"), index_col="gauge_id")
hyd = pd.read_csv(os.path.join(D, "CAMELS_GB_hydrologic_attributes.csv"), index_col="gauge_id")
rep = pd.read_csv(os.path.join(ROOT, "results", "agent_representative.csv"), index_col="gauge")
beh = pd.read_csv(os.path.join(ROOT, "results", "agent_behaviour.csv"))
diag = beh[beh["repeat"] == "r1"].set_index("gauge").diagnosis.reindex(rep.index).fillna("")

KEYS = {"reservoir": r"reservoir|regulat|impound|\bdams?\b|hydropower|hydro-electric",
        "groundwater": r"chalk|aquifer|limestone",
        "abstraction": r"abstract"}
abs_frac = (hum.surfacewater_abs.fillna(0) + hum.groundwater_abs.fillna(0)) / hyd.q_mean
recorded = pd.DataFrame({"reservoir": hum.num_reservoir.fillna(0) > 0,
                         "groundwater": geo.frac_high_perc.fillna(0) > 30,
                         "abstraction": abs_frac > 0.10}).reindex(rep.index)
named = pd.DataFrame({k: diag.str.contains(v, case=False, regex=True) for k, v in KEYS.items()})
out = recorded.add_prefix("rec_").join(named.add_prefix("named_"))
out["abs_frac_of_flow"] = abs_frac.reindex(rep.index).round(3)
out["diagnosis_words"] = diag.str.split().str.len()
out.to_csv(os.path.join(ROOT, "results", "diagnosis_check.csv"))

print(f"diagnoses: {len(diag)}, median words {out.diagnosis_words.median():.0f}")
for k in KEYS:
    r, n = recorded[k], named[k]
    print(f"{k:12s} recorded {r.sum():3d}: named in {(r & n).sum():3d} ({(r & n).sum() / max(r.sum(), 1):.0%}); "
          f"not recorded {(~r).sum():3d}: named in {(~r & n).sum():3d}")
print("named abstraction where not recorded:", list(out.index[(~recorded.abstraction) & named.abstraction]))
print("abstraction recorded but not named:", list(out.index[recorded.abstraction & ~named.abstraction]))
wb = diag.str.contains(r"P ?- ?PET|P minus PET|precipitation minus|water balance", case=False, regex=True)
print(f"diagnoses that discuss the water balance: {wb.sum()}")
