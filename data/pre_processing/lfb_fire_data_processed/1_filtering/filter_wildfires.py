# filter_wildfires_all_years.py
# -----------------------------
# Purpose: Filter LFB incidents (2009–2024) down to vegetation/open-land wildfires.
# Outputs (per year):
#   data/pre_processing/lfb_fire_data_processed/1_filtering/lfb_wildfires_YYYY.csv
#
# Notes:
# - Only filters. No derived columns are created.
# - Preserves *all original columns* from the input.
# -----------------------------

import pandas as pd
import re
from pathlib import Path


# ==== CONFIG ====
YEARS = range(2009, 2025)  # inclusive
INPUT_TEMPLATE = "data/raw/lfb_fire_data/lfb_incident_data_{year}.csv"
OUTPUT_TEMPLATE = (
    "data/pre_processing/lfb_fire_data_processed/1_filtering/lfb_wildfires_{year}.csv"
)

# Important IRS columns used for filtering
DATE_COL = "DateOfCall"
GROUP_COL = "IncidentGroup"
PCAT_COL = "PropertyCategory"
PTYPE_COL = "PropertyType"


# ==== REGEX EXCLUSION LIST ====
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
    r"\b(lake|pond|reservoir)\b",
    r"\b(refuse/?rubbish tip)\b",

    # explicit bin-storage exclusions
    r"\bbin(s)?\b",
    r"\bbin storage\b",
    r"\bbin store\b",
    r"\bwheelie bin\b",
    r"\brefuse storage\b",
    r"\brefuse store\b",
    r"\b(bin|refuse) (storage|store) area\b",
    r"\b(common external bin storage area)\b",

    # NEW — generic non-wildfire outdoor category
    r"\bother outdoor location\b",
]

exc_re = re.compile("|".join(exclude_terms), flags=re.IGNORECASE)


# ==== FUNCTION: PROCESS ONE YEAR ==== 
def process_year(year: int):
    print(f"\n==============================")
    print(f" Processing {year}")
    print(f"==============================")

    input_path = Path(INPUT_TEMPLATE.format(year=year))
    output_path = Path(OUTPUT_TEMPLATE.format(year=year))

    if not input_path.exists():
        print(f"⚠️  Missing file: {input_path}")
        return

    # Load
    df = pd.read_csv(input_path)
    original_cols = list(df.columns)

    # Date handling
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["__Date"] = df[DATE_COL].dt.date

    # Ensure required string cols exist
    for col in [GROUP_COL, PCAT_COL, PTYPE_COL]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    # Filters
    is_fire = df[GROUP_COL].str.strip().str.lower().eq("fire")
    outdoor_ok = df[PCAT_COL].isin(["Outdoor"])
    excl_ptype = df[PTYPE_COL].fillna("").str.contains(exc_re)

    mask = is_fire & outdoor_ok & (~excl_ptype)
    wildfires = df[mask].copy()

    # Diagnostics
    print(f"Total incidents: {len(df)}")
    print(f"Wildfires kept: {len(wildfires)}")

    print("\nTop PropertyType kept:")
    print(wildfires[PTYPE_COL].value_counts().head(10))

    print("\nTop PropertyCategory kept:")
    print(wildfires[PCAT_COL].value_counts().head(10))

    # Save (keeping column order)
    wildfires_out = wildfires[original_cols].sort_values(DATE_COL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wildfires_out.to_csv(output_path, index=False)

    print(f"✅ Saved to {output_path}")


# ==== MAIN LOOP ====
if __name__ == "__main__":
    for year in YEARS:
        process_year(year)
