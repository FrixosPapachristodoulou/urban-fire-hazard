# strip_midas_metadata.py
# ------------------------
# Removes metadata from a MIDAS BADC-CSV file (everything above the "data" line),
# keeps only useful meteorological columns, and saves a clean CSV.

from pathlib import Path
import pandas as pd
import csv

# ==== CONFIGURATION ====
INPUT_FILE = Path("data/raw/metoffice_midas/uk-daily-temperature-obs/00695_hampstead/midas-open_uk-daily-temperature-obs_dv-202507_greater-london_00695_hampstead_qcv-1_2009.csv")
OUTPUT_FILE = Path("data/pre_processing/metoffice_midas_processed/uk-daily-temperature-obs_processed/2009/daily_temperature_clean_00695_hampstead.csv")
# ========================


def strip_midas_metadata(input_path: Path, output_path: Path):
    # --- 1️⃣ Read file and find where 'data' appears ---
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("data"):
            data_line_idx = i
            break

    if data_line_idx is None:
        raise ValueError(f"'data' line not found in {input_path}")

    # The actual CSV header starts one line after 'data'
    header_line = lines[data_line_idx + 1].strip()

    # Detect delimiter (tab vs comma)
    try:
        dialect = csv.Sniffer().sniff(header_line, delimiters=[",", "\t"])
        sep = dialect.delimiter
    except csv.Error:
        sep = "," if "," in header_line else "\t"

    # --- 2️⃣ Read the data section into a DataFrame ---
    df = pd.read_csv(
        input_path,
        skiprows=data_line_idx + 1,
        sep=sep,
        dtype=str,
        na_values=["NA"],
        keep_default_na=False,
        engine="python",
    )

    print(f"✅ Loaded data from line {data_line_idx + 1} onward using sep='{sep}'")
    print(f"   Original columns: {list(df.columns)}")

    # --- 3️⃣ Select only useful columns ---
    keep_cols = [
        "ob_end_time",
        "id",
        "ob_hour_count",
        "max_air_temp",
        "min_air_temp",
        "min_grss_temp",
        "min_conc_temp",
        "max_air_temp_q",
        "min_air_temp_q",
    ]
    # Keep only columns that actually exist in this file
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # --- 4️⃣ Save cleaned version ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved reduced file to: {output_path}")
    print(f"   Columns kept: {keep_cols}")
    print(f"   Total rows: {len(df)}")


if __name__ == "__main__":
    strip_midas_metadata(INPUT_FILE, OUTPUT_FILE)
