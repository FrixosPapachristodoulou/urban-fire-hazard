from pathlib import Path
import pandas as pd

# ===== CONFIGURATION =====
BASE_DATA = Path("data")

WEATHER_DIR = BASE_DATA / "pre_processing" / "metoffice_midas_processed" / "uk-hourly-weather_different_averaging" / "3_averaged_by_year"
FIRE_DIR = BASE_DATA / "pre_processing" / "lfb_fire_data_processed" / "2_number_fires"

OUTPUT_DIR = BASE_DATA / "graph_generation" / "3_num_fires_vs_vpd_different_averaging" / "merged_weather_and_fire_counts_differently"

YEARS = list(range(2009, 2025))  # 2009–2024 inclusive
# =========================


def merge_weather_and_fire(year: int):
    weather_file = WEATHER_DIR / f"daily_weather_averaged_{year}.csv"
    fire_file = FIRE_DIR / f"daily_fire_counts_{year}.csv"

    if not weather_file.exists():
        print(f"⏭️ Weather file missing for {year}: {weather_file}")
        return
    if not fire_file.exists():
        print(f"⏭️ Fire-count file missing for {year}: {fire_file}")
        return

    print(f"\n📅 Year {year}")
    print(f"   🌤️ Weather: {weather_file}")
    print(f"   🔥 Fires:   {fire_file}")

    # --- Load weather (includes VPD_mean from daily mean T & RH) ---
    df_w = pd.read_csv(weather_file)
    if "met_day" not in df_w.columns:
        raise ValueError(f"'met_day' column missing in {weather_file}")

    # Ensure met_day is datetime for robust merging
    df_w["met_day"] = pd.to_datetime(df_w["met_day"], errors="coerce")

    # --- Load fire counts ---
    df_f = pd.read_csv(fire_file)
    if "date" not in df_f.columns or "fire_count" not in df_f.columns:
        raise ValueError(f"'date' or 'fire_count' missing in {fire_file}")

    df_f["date"] = pd.to_datetime(df_f["date"], errors="coerce")

    # --- Merge on date ---
    merged = df_w.merge(
        df_f,
        left_on="met_day",
        right_on="date",
        how="left",
        validate="one_to_one",
    )

    # Drop extra 'date' column (we keep met_day)
    merged = merged.drop(columns=["date"])

    # If any fire_count missing → 0
    merged["fire_count"] = merged["fire_count"].fillna(0).astype(int)

    # --- Reorder: put fire_count right after met_day ---
    cols = list(merged.columns)
    cols.insert(1, cols.pop(cols.index("fire_count")))
    merged = merged[cols]

    # Round all float columns (including VPD_mean) to 3 decimal places
    float_cols = merged.select_dtypes(include="float").columns
    merged[float_cols] = merged[float_cols].round(3)

    # Ensure met_day written as date only (YYYY-MM-DD)
    merged["met_day"] = merged["met_day"].dt.date

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"daily_weather_fire_{year}_differently.csv"
    merged.to_csv(out_file, index=False)

    print(f"   💾 Saved merged file: {out_file}")
    print(f"   Rows: {len(merged)} | Columns: {len(merged.columns)}")


def main():
    print("=" * 80)
    print("🔗 Merging daily weather (with VPD_mean) and wildfire counts per year")
    print("=" * 80)

    for year in YEARS:
        try:
            merge_weather_and_fire(year)
        except Exception as e:
            print(f"   ❌ Error for {year}: {e}")

    print("\n" + "=" * 80)
    print("✅ Finished creating merged weather + fire CSVs")
    print("=" * 80)


if __name__ == "__main__":
    main()
