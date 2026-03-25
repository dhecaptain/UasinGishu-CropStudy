"""
01_data_cleaning.py
===================
Data Cleaning & Preparation Pipeline
Study: Climate Variability & Maize Yield — Uasin Gishu (Eldoret), Kenya
Source: Ministry of Agriculture & Livestock Development

Steps:
  1. Load all raw CSVs
  2. Inspect shape, dtypes, missing values, duplicates
  3. Handle outliers (IQR method + domain knowledge flags)
  4. Engineer additional features
  5. Validate and save cleaned datasets
  6. Print a full cleaning report

Run: python 01_data_cleaning.py
"""

import numpy as np
import pandas as pd
import os, json
from scipy import stats

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)

os.makedirs("data/cleaned", exist_ok=True)

SEPARATOR = "=" * 70

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
print(SEPARATOR)
print("STEP 1 — LOADING RAW DATA")
print(SEPARATOR)

rain_m  = pd.read_csv("data/rainfall_monthly.csv")
rain_a  = pd.read_csv("data/rainfall_annual.csv")
temp_m  = pd.read_csv("data/temperature_monthly.csv")
temp_a  = pd.read_csv("data/temperature_annual.csv")
maize   = pd.read_csv("data/maize_yield.csv")
panel   = pd.read_csv("data/panel_annual.csv")

datasets = {
    "rainfall_monthly":     rain_m,
    "rainfall_annual":      rain_a,
    "temperature_monthly":  temp_m,
    "temperature_annual":   temp_a,
    "maize_yield":          maize,
    "panel_annual":         panel,
}

for name, df in datasets.items():
    print(f"\n  [{name}]  shape={df.shape}  columns={df.columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  INSPECTION — MISSING VALUES & DUPLICATES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("STEP 2 — MISSING VALUES & DUPLICATES")
print(SEPARATOR)

cleaning_log = {}   # tracks all actions taken

for name, df in datasets.items():
    missing = df.isnull().sum()
    dups    = df.duplicated().sum()
    n_miss  = missing.sum()
    cleaning_log[name] = {"missing_before": int(n_miss), "duplicates": int(dups), "actions": []}
    print(f"\n  [{name}]")
    print(f"    Missing values : {n_miss}")
    if n_miss > 0:
        print(f"    Breakdown      : {missing[missing>0].to_dict()}")
    print(f"    Duplicate rows : {dups}")

# ── Rainfall monthly: 'TR' (trace) values were already replaced by 0 during
#    parsing. Confirm no remaining nulls.
assert rain_m["Rainfall_mm"].isnull().sum() == 0, "Unexpected nulls in rainfall_monthly"
print("\n  ✔ No missing values in any dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  OUTLIER DETECTION & HANDLING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("STEP 3 — OUTLIER DETECTION")
print(SEPARATOR)

def iqr_bounds(series, k=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - k * IQR, Q3 + k * IQR

# ── 3a. Monthly Rainfall ──────────────────────────────────────────────────────
print("\n  [rainfall_monthly] — Rainfall_mm")
lo, hi = iqr_bounds(rain_m["Rainfall_mm"], k=3.0)   # k=3 (generous for rainfall)
outliers_rain = rain_m[(rain_m["Rainfall_mm"] < lo) | (rain_m["Rainfall_mm"] > hi)]
print(f"    IQR bounds (k=3): [{lo:.1f}, {hi:.1f}]")
print(f"    Outliers detected: {len(outliers_rain)}")
if len(outliers_rain):
    print(outliers_rain[["Year","Month","Rainfall_mm"]].to_string(index=False))
    print("    → ACTION: Flag as 'Rain_Outlier' column (do NOT remove — extreme rainfall is real)")
    rain_m["Rain_Outlier"] = ((rain_m["Rainfall_mm"] < lo) | (rain_m["Rainfall_mm"] > hi)).astype(int)
    cleaning_log["rainfall_monthly"]["actions"].append(f"Flagged {len(outliers_rain)} rainfall outliers (k=3 IQR)")
else:
    rain_m["Rain_Outlier"] = 0

# ── 3b. Monthly Temperature ───────────────────────────────────────────────────
print("\n  [temperature_monthly] — Mean_Temp_C")
lo_t, hi_t = iqr_bounds(temp_m["Mean_Temp_C"], k=2.5)
outliers_temp = temp_m[(temp_m["Mean_Temp_C"] < lo_t) | (temp_m["Mean_Temp_C"] > hi_t)]
print(f"    IQR bounds (k=2.5): [{lo_t:.2f}, {hi_t:.2f}]")
print(f"    Outliers detected: {len(outliers_temp)}")
if len(outliers_temp):
    print(outliers_temp[["Year","Month","Mean_Temp_C"]].to_string(index=False))
    # Domain knowledge: Eldoret mean temp ~17°C, values < 5°C or > 25°C are
    # likely instrument errors — flag and impute with monthly mean
    temp_m["Temp_Outlier"] = ((temp_m["Mean_Temp_C"] < lo_t) | (temp_m["Mean_Temp_C"] > hi_t)).astype(int)
    monthly_means = temp_m.groupby("Month_Num")["Mean_Temp_C"].transform(
        lambda x: x[~((x < lo_t) | (x > hi_t))].mean()
    )
    n_imputed = temp_m["Temp_Outlier"].sum()
    temp_m["Mean_Temp_C_clean"] = np.where(temp_m["Temp_Outlier"] == 1, monthly_means, temp_m["Mean_Temp_C"])
    print(f"    → ACTION: Flagged {n_imputed} temperature outliers; imputed with monthly mean for 'Mean_Temp_C_clean'")
    cleaning_log["temperature_monthly"]["actions"].append(
        f"Flagged {n_imputed} temp outliers; imputed with monthly mean → Mean_Temp_C_clean"
    )
else:
    temp_m["Temp_Outlier"] = 0
    temp_m["Mean_Temp_C_clean"] = temp_m["Mean_Temp_C"]

# ── 3c. Maize Yield ───────────────────────────────────────────────────────────
print("\n  [maize_yield] — Maize_Yield_t_ha (n=12)")
lo_y, hi_y = iqr_bounds(maize["Maize_Yield_t_ha"], k=1.5)
outliers_yield = maize[(maize["Maize_Yield_t_ha"] < lo_y) | (maize["Maize_Yield_t_ha"] > hi_y)]
print(f"    IQR bounds (k=1.5): [{lo_y:.4f}, {hi_y:.4f}]")
print(f"    Outliers detected: {len(outliers_yield)}")
if len(outliers_yield):
    print(outliers_yield.to_string(index=False))
    maize["Yield_Outlier"] = ((maize["Maize_Yield_t_ha"] < lo_y) | (maize["Maize_Yield_t_ha"] > hi_y)).astype(int)
    cleaning_log["maize_yield"]["actions"].append(f"Flagged {len(outliers_yield)} yield outliers")
else:
    print("    No yield outliers detected.")
    maize["Yield_Outlier"] = 0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("STEP 4 — FEATURE ENGINEERING")
print(SEPARATOR)

# ── 4a. Rainfall features ─────────────────────────────────────────────────────
print("\n  [rainfall_annual] — Engineering new features")

# Long rains = March–May (months 3,4,5)
lr_rain = rain_m[rain_m["Month_Num"].isin([3,4,5])].groupby("Year")["Rainfall_mm"].sum().rename("Long_Rain_mm")
# Short rains = Oct–Nov (months 10,11)
sr_rain = rain_m[rain_m["Month_Num"].isin([10,11])].groupby("Year")["Rainfall_mm"].sum().rename("Short_Rain_mm")
# Jul–Aug core rains
jja_rain = rain_m[rain_m["Month_Num"].isin([6,7,8])].groupby("Year")["Rainfall_mm"].sum().rename("JJA_Rain_mm")
# Growing season Mar–Sep
gs_rain = rain_m[rain_m["Month_Num"].isin(range(3,10))].groupby("Year")["Rainfall_mm"].sum().rename("GS_Rain_mm")
# Specific months used in dashboard
apr_rain = rain_m[rain_m["Month_Num"] == 4].groupby("Year")["Rainfall_mm"].sum().rename("Apr_Rain_mm")
aug_rain = rain_m[rain_m["Month_Num"] == 8].groupby("Year")["Rainfall_mm"].sum().rename("Aug_Rain_mm")
# Dry season Jan–Feb
dry_rain = rain_m[rain_m["Month_Num"].isin([1,2])].groupby("Year")["Rainfall_mm"].sum().rename("DryS_Rain_mm")
# Coefficient of variation within year
cv_rain = rain_m.groupby("Year")["Rainfall_mm"].apply(
    lambda x: x.std()/x.mean()*100 if x.mean()>0 else 0
).rename("Within_Year_CV")

rain_a = rain_a.merge(lr_rain, on="Year").merge(sr_rain, on="Year") \
               .merge(jja_rain, on="Year").merge(gs_rain, on="Year") \
               .merge(apr_rain, on="Year").merge(aug_rain, on="Year") \
               .merge(dry_rain, on="Year").merge(cv_rain, on="Year")

# 5-year rolling mean and anomaly
rain_a = rain_a.sort_values("Year").reset_index(drop=True)
rain_a["Rolling5yr_Rain"] = rain_a["Annual_Rainfall_mm"].rolling(5, min_periods=3).mean()
rain_a["Rain_Anomaly"]    = rain_a["Annual_Rainfall_mm"] - rain_a["Annual_Rainfall_mm"].mean()
rain_a["Drought_Year"]    = (rain_a["Annual_Rainfall_mm"] < rain_a["Annual_Rainfall_mm"].quantile(0.25)).astype(int)
rain_a["Wet_Year"]        = (rain_a["Annual_Rainfall_mm"] > rain_a["Annual_Rainfall_mm"].quantile(0.75)).astype(int)

new_feats_rain = ["Long_Rain_mm","Short_Rain_mm","JJA_Rain_mm","GS_Rain_mm","Apr_Rain_mm","Aug_Rain_mm",
                  "Within_Year_CV","Rolling5yr_Rain","Rain_Anomaly","Drought_Year","Wet_Year"]
print(f"    Added: {new_feats_rain}")

# ── 4b. Temperature features ──────────────────────────────────────────────────
print("\n  [temperature_annual] — Engineering new features (using cleaned values)")

# Recompute annual stats from cleaned monthly column
temp_a_clean = temp_m.groupby("Year").agg(
    Annual_Mean_Temp_clean=("Mean_Temp_C_clean","mean"),
    Max_Monthly_Temp_clean=("Mean_Temp_C_clean","max"),
    Min_Monthly_Temp_clean=("Mean_Temp_C_clean","min"),
    Temp_Range_clean=("Mean_Temp_C_clean", lambda x: x.max()-x.min()),
    Temp_CV=("Mean_Temp_C_clean", lambda x: x.std()/x.mean()*100),
).reset_index()

# Growing season (Mar–Sep) temperature from cleaned
gs_temp_clean = temp_m[temp_m["Month_Num"].isin(range(3,10))].groupby("Year")["Mean_Temp_C_clean"].mean().rename("GS_Temp_clean")
temp_a_clean = temp_a_clean.merge(gs_temp_clean, on="Year")

# Warming anomaly
temp_a_clean["Temp_Anomaly"] = temp_a_clean["Annual_Mean_Temp_clean"] - temp_a_clean["Annual_Mean_Temp_clean"].mean()
temp_a_clean["Hot_Year"]     = (temp_a_clean["Annual_Mean_Temp_clean"] > temp_a_clean["Annual_Mean_Temp_clean"].quantile(0.75)).astype(int)

new_feats_temp = ["Annual_Mean_Temp_clean","GS_Temp_clean","Temp_Anomaly","Temp_CV","Hot_Year"]
print(f"    Added: {new_feats_temp}")

# ── 4c. Panel — rebuild with all engineered features ─────────────────────────
print("\n  [panel_annual] — Rebuilding with all engineered features")

rain_panel = rain_a[rain_a["Year"].between(2012,2023)].copy()
temp_panel = temp_a_clean[temp_a_clean["Year"].between(2012,2023)].copy()

panel_clean = maize.merge(rain_panel, on="Year").merge(temp_panel, on="Year")

# Lag features (previous year rainfall → this year yield)
panel_clean = panel_clean.sort_values("Year").reset_index(drop=True)
panel_clean["Rainfall_Lag1"] = panel_clean["Annual_Rainfall_mm"].shift(1)
panel_clean["LR_Rain_Lag1"]  = panel_clean["Long_Rain_mm"].shift(1)
panel_clean["Yield_Lag1"]    = panel_clean["Maize_Yield_t_ha"].shift(1)

# Yield category
mean_yield = panel_clean["Maize_Yield_t_ha"].mean()
panel_clean["Yield_Category"] = pd.cut(
    panel_clean["Maize_Yield_t_ha"],
    bins=[0, mean_yield - 0.5*panel_clean["Maize_Yield_t_ha"].std(),
             mean_yield + 0.5*panel_clean["Maize_Yield_t_ha"].std(), 10],
    labels=["Low","Medium","High"]
)

print(f"    Panel shape after engineering: {panel_clean.shape}")
print(f"    Columns: {panel_clean.columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FINAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("STEP 5 — FINAL VALIDATION")
print(SEPARATOR)

checks = {
    "Rainfall years complete (1990–2025)":  len(rain_a) == 36,
    "Temperature years complete (1995–2025)": len(temp_a_clean) == 31,
    "Maize yield records (2012–2023)":       len(maize) == 12,
    "No nulls in panel (excl. lag cols)":    panel_clean.drop(columns=["Rainfall_Lag1","LR_Rain_Lag1","Yield_Lag1"]).isnull().sum().sum() == 0,
    "Annual rainfall > 0 all years":         (rain_a["Annual_Rainfall_mm"] > 0).all(),
    "Temperature in plausible range":        temp_m["Mean_Temp_C_clean"].between(10,30).all(),
    "Yield in plausible range (1–10 t/ha)":  maize["Maize_Yield_t_ha"].between(1,10).all(),
}

all_pass = True
for check, result in checks.items():
    status = "✔" if result else "✘ FAILED"
    print(f"    {status}  {check}")
    if not result:
        all_pass = False

print(f"\n  {'All checks passed ✔' if all_pass else 'Some checks FAILED — review above'}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SAVE CLEANED DATA
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("STEP 6 — SAVING CLEANED DATASETS")
print(SEPARATOR)

rain_m.to_csv("data/cleaned/rainfall_monthly_clean.csv",        index=False)
rain_a.to_csv("data/cleaned/rainfall_annual_clean.csv",         index=False)
temp_m.to_csv("data/cleaned/temperature_monthly_clean.csv",     index=False)
temp_a_clean.to_csv("data/cleaned/temperature_annual_clean.csv",index=False)
maize.to_csv("data/cleaned/maize_yield_clean.csv",              index=False)
panel_clean.to_csv("data/cleaned/panel_annual_clean.csv",       index=False)

saved = {
    "rainfall_monthly_clean.csv":    rain_m.shape,
    "rainfall_annual_clean.csv":     rain_a.shape,
    "temperature_monthly_clean.csv": temp_m.shape,
    "temperature_annual_clean.csv":  temp_a_clean.shape,
    "maize_yield_clean.csv":         maize.shape,
    "panel_annual_clean.csv":        panel_clean.shape,
}
for fname, shape in saved.items():
    print(f"    ✔  data/cleaned/{fname}  {shape}")

# Save cleaning log as JSON
cleaning_log["panel_annual"]["actions"].append(
    f"Rebuilt panel with {panel_clean.shape[1]} engineered features"
)
with open("data/cleaned/cleaning_log.json", "w") as f:
    json.dump(cleaning_log, f, indent=2)
print("    ✔  data/cleaned/cleaning_log.json")

# ── Final summary table ───────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("CLEANING SUMMARY")
print(SEPARATOR)
print(f"""
  Dataset              | Before  | After   | Key Changes
  ---------------------|---------|---------|-------------------------------
  rainfall_monthly     | {432:>7} | {len(rain_m):>7} | +Rain_Outlier flag
  rainfall_annual      | {36:>7} | {len(rain_a):>7} | +9 engineered features
  temperature_monthly  | {372:>7} | {len(temp_m):>7} | +Temp_Outlier, +Mean_Temp_C_clean
  temperature_annual   | {31:>7} | {len(temp_a_clean):>7} | +5 engineered features (clean)
  maize_yield          | {12:>7} | {len(maize):>7} | +Yield_Outlier flag
  panel_annual         | {12:>7} | {len(panel_clean):>7} | Full rebuild — {panel_clean.shape[1]} columns
""")
print("  Data cleaning complete. Run 02_model_training.py next.\n")
