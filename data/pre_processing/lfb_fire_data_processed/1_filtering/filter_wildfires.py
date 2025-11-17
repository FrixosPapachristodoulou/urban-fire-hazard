# filter_wildfires.py
# -----------------------
# Purpose: JUST FILTER the LFB incidents down to vegetation/open-land wildfires.
# Outputs:
#   - lfb_wildfires.csv       (incident-level; preserves ALL original columns)
#   - lfb_wildfires_daily.csv (daily count only: Date, NumWildfires)
#
# Notes:
# - We do NOT create derived columns (e.g., PumpHours, DamageGBP).
# - We keep your original column names and order in the incident-level CSV.

import pandas as pd
import re

# ==== CONFIG ====
INPUT_CSV = "data/raw/lfb_fire_data/lfb_incident_data_2024.csv"
OUT_INCIDENTS = "data/pre_processing/lfb_fire_data_processed/lfb_wildfires_2024.csv"
OUT_DAILY = "data/pre_processing/lfb_fire_data_processed/lfb_wildfires_daily_2024.csv"

# Columns from your file we need for filtering/grouping
DATE_COL = "DateOfCall"
GROUP_COL = "IncidentGroup"
PCAT_COL = "PropertyCategory"
PTYPE_COL = "PropertyType"

# ==== LOAD ====
df = pd.read_csv(INPUT_CSV)

# Keep the original column order for the incident-level export
original_cols = list(df.columns)

# ---- parse date; create pure-date for grouping (internal only) ----
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df["__Date"] = df[DATE_COL].dt.date

# ---- ensure string cols exist and are strings ----
for col in [GROUP_COL, PCAT_COL, PTYPE_COL]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(str)

# ==== FILTERS ====

# 1) keep only "Fire" incidents
is_fire = df[GROUP_COL].str.strip().str.lower().eq("fire")

# 2) outdoor categories (explicit + robust contains)
outdoor_ok = (
    df[PCAT_COL].isin(["Outdoor", "Outdoor Structure", "Other outdoor location"])
    | df[PCAT_COL].str.contains("outdoor", case=False, na=False)
)

# 3) vegetation/open-land terms to INCLUDE
include_terms = [
    r"vegetation(?!.*equipment)",
    r"garden\b(?!.*equipment)",
    r"\bgrass(es)?\b",
    r"\bhedge(s)?\b",
    r"\b(scrub ?land|tree scrub|scrub)\b",
    r"\b(wood(land)?|forest)\b",
    r"\bwaste ?land\b",
    r"\b(open ?land)\b",
    r"\bparks?\b",            # <— tightened
    r"\bcommon land\b",       # <— tightened
    r"\b(field|crop(s)?)\b",
    r"\bstraw/?stubble burning\b",
    r"\ballotment(s)?\b",
    r"\bheath(land)?\b",
    r"\bmoor(s)?\b",
]

inc_re = re.compile("|".join(include_terms), flags=re.IGNORECASE)

# 4) non-wildfire outdoor terms to EXCLUDE
exclude_terms = [
    r"\b(loose )?refuse\b",
    r"\brubbish\b",
    r"\b(skip|paladin)\b",
    r"\bcontainer\b",
    r"\bdump\b",
    r"\b(vehicle|car|lorry|bus|van|road vehicle)\b",
    r"\b(road surface|pavement|cycle path|footpath|bridleway|roadside)\b",
    r"\b(train station|airport|concourse|terminal)\b",
    r"\b(post box|kiosk)\b",
    r"\b(equipment|machinery)\b",
    r"\b(lake|pond|reservoir)\b",
    r"\b(refuse/?rubbish tip)\b",
    # New explicit bin-storage exclusions
    r"\bbin(s)?\b",
    r"\bbin storage\b",
    r"\bbin store\b",
    r"\bwheelie bin\b",
    r"\brefuse storage\b",
    r"\brefuse store\b",
    r"\b(bin|refuse) (storage|store) area\b",
    r"\b(common external bin storage area)\b",
]

exc_re = re.compile("|".join(exclude_terms), flags=re.IGNORECASE)

ptype = df[PTYPE_COL].fillna("")
incl_ptype = ptype.str.contains(inc_re)
excl_ptype = ptype.str.contains(exc_re)

# 5) BBQ-origin vegetation edge case (keep only if BBQ + vegetation both appear)
bbq_veg = ptype.str.contains(r"\bbbq|barbecue\b", case=False, na=False) & ptype.str.contains(inc_re)

# main mask: Fire + Outdoor + (Vegetation OR BBQ+Vegetation) + NOT in exclude list
mask = is_fire & outdoor_ok & (incl_ptype | bbq_veg) & (~excl_ptype)

wildfires = df[mask].copy()

# ==== DIAGNOSTICS ====
print(f"Total rows in input: {len(df)}")
print(f"Wildfire rows kept:  {len(wildfires)}")
print("\nTop PropertyType kept:")
print(wildfires[PTYPE_COL].value_counts().head(20))
print("\nTop PropertyCategory kept:")
print(wildfires[PCAT_COL].value_counts().head(10))

# ==== SAVE INCIDENT-LEVEL (ALL ORIGINAL COLUMNS) ====
# Do not keep internal __Date in the incident-level CSV
incident_out = wildfires[original_cols].sort_values(DATE_COL)
incident_out.to_csv(OUT_INCIDENTS, index=False)
print(f"\n✅ Saved incident-level wildfires (all original columns) to: {OUT_INCIDENTS}")

# ==== SAVE DAILY COUNTS ONLY ====
# Since we are "just filtering", we produce a clean daily count table.
daily = (
    wildfires.groupby("__Date")
    .size()
    .to_frame("NumWildfires")
    .reset_index()
    .rename(columns={"__Date": "Date"})
    .sort_values("Date")
)
daily.to_csv(OUT_DAILY, index=False)
print(f"✅ Saved daily counts to: {OUT_DAILY}")

print("\nSample of daily counts:")
print(daily.head(10))
