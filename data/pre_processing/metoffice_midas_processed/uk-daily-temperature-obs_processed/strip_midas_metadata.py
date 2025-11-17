# strip_midas_metadata.py
# ------------------------
# Removes metadata from a MIDAS BADC-CSV file (everything above the "data" line),
# converts observations to meteorological days (handling 12h and 24h obs),
# aggregates to one row per day, keeps only useful columns, and saves a clean CSV.

from pathlib import Path
import pandas as pd
import csv

# ==== CONFIGURATION ====
STATIONS = [
    "00695_hampstead",
    "00697_london-st-jamess-park",
    "00708_heathrow",
    "00709_northolt",
    "00710_northwood",
    "00711_hampton-w-wks",
    "00723_kew-gardens",
    "00726_kenley-airfield",
    "01044_rothamsted-science-centre",
    "62084_battersea-heliport",
]

YEARS = list(range(2009, 2025))  # 2009 to 2024 inclusive
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

    print(f"   Loaded data from line {data_line_idx + 1} onward using sep='{sep}'")
    print(f"   Original columns: {list(df.columns)}")

    # --- 3️⃣ Convert ob_end_time to meteorological day (handling 12h & 24h) ---
    if "ob_end_time" in df.columns:
        # Convert to datetime
        df["ob_end_time"] = pd.to_datetime(df["ob_end_time"], errors="coerce")

        # Default: subtract 1 day (covers 24h obs at 09:00 and 12h obs at 09:00)
        df["met_day"] = (df["ob_end_time"] - pd.Timedelta(days=1)).dt.date

        if "ob_hour_count" in df.columns:
            # Ensure ob_hour_count is numeric
            df["ob_hour_count"] = pd.to_numeric(df["ob_hour_count"], errors="coerce")

            hours = df["ob_end_time"].dt.hour

            # For 12-hour observations that end in the evening (e.g. 21:00),
            # the meteorological day is the SAME calendar day
            mask_12 = df["ob_hour_count"] == 12
            mask_evening_12 = mask_12 & (hours > 12)
            df.loc[mask_evening_12, "met_day"] = df.loc[
                mask_evening_12, "ob_end_time"
            ].dt.date

        # Drop the original ob_end_time column
        df = df.drop(columns=["ob_end_time"])

        print(f"   Converted ob_end_time to meteorological day (met_day)")

    # --- 4️⃣ Convert key numeric columns for aggregation ---
    temp_cols = [
        "max_air_temp",
        "min_air_temp",
        "min_grss_temp",
        "min_conc_temp",
    ]
    flag_cols = [
        "max_air_temp_q",
        "min_air_temp_q",
    ]

    for col in temp_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 5️⃣ Aggregate to one row per (station, met_day) ---
    if "met_day" in df.columns:
        group_cols = []
        if "id" in df.columns:
            group_cols.append("id")
        group_cols.append("met_day")

        def agg_ob_hour_count(x):
            vals = pd.to_numeric(x, errors="coerce")
            total = vals.sum(skipna=True)
            if pd.isna(total):
                return pd.NA
            # If we have two 12h obs or a 24h obs, treat as full 24h
            return 24 if total >= 24 else total

        agg_dict = {}

        if "ob_hour_count" in df.columns:
            agg_dict["ob_hour_count"] = agg_ob_hour_count
        if "max_air_temp" in df.columns:
            agg_dict["max_air_temp"] = "max"  # true daily max
        if "min_air_temp" in df.columns:
            agg_dict["min_air_temp"] = "min"  # true daily min
        if "min_grss_temp" in df.columns:
            agg_dict["min_grss_temp"] = "min"
        if "min_conc_temp" in df.columns:
            agg_dict["min_conc_temp"] = "min"
        if "max_air_temp_q" in df.columns:
            agg_dict["max_air_temp_q"] = "max"  # worst quality
        if "min_air_temp_q" in df.columns:
            agg_dict["min_air_temp_q"] = "max"

        if agg_dict:
            df = (
                df.groupby(group_cols, dropna=False)
                .agg(agg_dict)
                .reset_index()
            )

            print(
                f"   Aggregated to one row per {group_cols} "
                f"(rows after aggregation: {len(df)})"
            )

    # --- 6️⃣ Select only useful columns ---
    keep_cols = [
        "met_day",
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

    # Ensure met_day is first if present
    if "met_day" in df.columns:
        cols = ["met_day"] + [c for c in df.columns if c != "met_day"]
        df = df[cols]

    # --- 7️⃣ Save cleaned version ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   ✅ Saved to: {output_path}")
    print(f"   Columns kept: {keep_cols}")
    print(f"   Total rows: {len(df)}")


if __name__ == "__main__":
    total_processed = 0
    total_skipped = 0

    print("=" * 80)
    print("🚀 Starting MIDAS data processing")
    print("=" * 80)

    for station in STATIONS:
        print(f"\n📍 Processing station: {station}")

        for year in YEARS:
            input_file = Path(
                f"data/raw/metoffice_midas/uk-daily-temperature-obs/"
                f"{station}/midas-open_uk-daily-temperature-obs_dv-202507_"
                f"greater-london_{station}_qcv-1_{year}.csv"
            )
            output_file = Path(
                f"data/pre_processing/metoffice_midas_processed/"
                f"uk-daily-temperature-obs_processed/{year}/"
                f"daily_temperature_clean_{station}.csv"
            )

            if not input_file.exists():
                print(f"   ⏭️  Skipping {year}: File not found")
                total_skipped += 1
                continue

            try:
                print(f"   📅 Processing year {year}...")
                strip_midas_metadata(input_file, output_file)
                total_processed += 1
            except Exception as e:
                print(f"   ❌ Error processing {station} - {year}: {e}")
                total_skipped += 1
                continue

    print("\n" + "=" * 80)
    print(f"✅ Processing complete!")
    print(f"   Files processed: {total_processed}")
    print(f"   Files skipped: {total_skipped}")
    print("=" * 80)
