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
    "19144_london-weather-centre",
    "62084_battersea-heliport",
]

YEARS = list(range(2009, 2025))  # 2009 to 2024 inclusive
# ========================


def strip_midas_metadata(input_path: Path, output_path: Path):
    """
    Strip MIDAS metadata from an hourly-weather CSV, keep only useful columns
    for VPD / fire hazard work, and save a cleaned CSV.
    """

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
        skiprows=data_line_idx + 1,  # skip metadata + 'data' line
        sep=sep,
        dtype=str,
        na_values=["NA"],
        keep_default_na=False,
        engine="python",
    )

    print(f"      Loaded from line {data_line_idx + 1} using sep='{sep}'")
    print(f"      Original columns: {list(df.columns)}")

    # --- 3️⃣ Select only useful columns for VPD / fire work ---
    # Core time + station identity
    keep_cols = [
        "ob_time",
        "id",
        "id_type",
        "src_id",
    ]

    # Core meteorology for VPD / DFMC / hazard:
    keep_cols += [
        "air_temperature",
        "dewpoint",
        "wetb_temp",
        "rltv_hum",
        "msl_pressure",
        "stn_pres",
        "alt_pres",
        "wind_direction",
        "wind_speed",
        "visibility",
        "q10mnt_mxgst_spd",
        "cs_hr_sun_dur",
        "wmo_hr_sun_dur",
        "drv_hr_sun_dur",
        "snow_depth",
        "ground_state_id",
    ]

    # QC flags for key variables (could be useful later)
    keep_cols += [
        "air_temperature_q",
        "dewpoint_q",
        "wetb_temp_q",
        "rltv_hum_q",          # may or may not exist; we'll intersect below
        "msl_pressure_q",
        "stn_pres_q",
        "alt_pres_q",
        "wind_direction_q",
        "wind_speed_q",
        "visibility_q",
        "q10mnt_mxgst_spd_q",
        "cs_hr_sun_dur_q",
        "wmo_hr_sun_dur_q",
        "drv_hr_sun_dur_q",
        "snow_depth_q",
        "ground_state_id_q",
    ]

    # Justified (QC-adjusted) versions for key variables (optional but future-proof)
    keep_cols += [
        "air_temperature_j",
        "dewpoint_j",
        "wetb_temp_j",
        "rltv_hum_j",
        "msl_pressure_j",
        "stn_pres_j",
        "alt_pres_j",
        "wind_direction_j",
        "wind_speed_j",
        "visibility_j",
        "q10mnt_mxgst_spd_j",
    ]

    # Timestamps from MIDAS system (could be handy for debugging)
    keep_cols += [
        "meto_stmp_time",
        "midas_stmp_etime",
    ]

    # Keep only those columns that actually exist in this particular file
    keep_cols_present = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols_present].copy()

    print(f"      Columns kept: {keep_cols_present}")
    print(f"      Total rows: {len(df)}")

    # --- 4️⃣ Save cleaned version ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"      ✅ Saved cleaned file to: {output_path}")


if __name__ == "__main__":
    total_processed = 0
    total_skipped = 0

    print("=" * 80)
    print("🚀 Starting MIDAS hourly weather data processing")
    print("=" * 80)

    for station in STATIONS:
        print(f"\n📍 Processing station: {station}")

        for year in YEARS:
            input_file = Path(
                f"data/raw/metoffice_midas/uk-hourly-weather-obs/"
                f"{station}/midas-open_uk-hourly-weather-obs_dv-202507_"
                f"greater-london_{station}_qcv-1_{year}.csv"
            )
            output_file = Path(
                f"data/pre_processing/metoffice_midas_processed/"
                f"uk-hourly-weather_different_averaging/1_cleaned_by_year/{year}/"
                f"hourly-weather_clean_{station}.csv"
            )

            if not input_file.exists():
                print(f"   ⏭️  Skipping {year}: File not found ({input_file})")
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
    print("✅ Processing complete!")
    print(f"   Files processed: {total_processed}")
    print(f"   Files skipped:   {total_skipped}")
    print("=" * 80)
