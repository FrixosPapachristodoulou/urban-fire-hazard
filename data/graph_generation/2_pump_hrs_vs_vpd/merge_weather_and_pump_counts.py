from pathlib import Path
import pandas as pd

# ===== CONFIGURATION =====
BASE_DATA = Path("data")

WEATHER_DIR = BASE_DATA / "pre_processing" / "metoffice_midas_processed" / "uk-hourly-weather-obs_processed" / "3_averaged_by_year"
FIRE_DIR = BASE_DATA / "pre_processing" / "lfb_fire_data_processed" / "3_pump_hours"

OUTPUT_DIR = BASE_DATA / "graph_generation" / "pump_hrs_vs_vpd"

YEARS = list(range(2009, 2025))  # 2009–2024 inclusive
# =========================


def merge_weather_and_pumps(year: int):
    weather_file = WEATHER_DIR / f"daily_weather_averaged_{year}.csv"
    pump_file = FIRE_DIR / f"daily_pump_hours_{year}.csv"

    if not weather_file.exists():
        print(f"⏭️ Weather file missing for {year}: {weather_file}")
        return
    if not pump_file.exists():
        print(f"⏭️ Pump-count file missing for {year}: {pump_file}")
        return

    print(f"\n📅 Year {year}")
    print(f"   🌤️ Weather: {weather_file}")
    print(f"   🚒 Pumps:   {pump_file}")

    # --- Load weather ---
    df_w = pd.read_csv(weather_file)
    if "met_day" not in df_w.columns:
        raise ValueError(f"'met_day' column missing in {weather_file}")

    # Ensure met_day is datetime for robust merging
    df_w["met_day"] = pd.to_datetime(df_w["met_day"], errors="coerce")

    # --- Load pump counts ---
    df_p = pd.read_csv(pump_file)
    if "date" not in df_p.columns or "pump_count" not in df_p.columns:
        raise ValueError(f"'date' or 'pump_count' missing in {pump_file}")

    df_p["date"] = pd.to_datetime(df_p["date"], errors="coerce")

    # --- Merge on date ---
    merged = df_w.merge(
        df_p,
        left_on="met_day",
        right_on="date",
        how="left",
        validate="one_to_one",
    )

    # Drop extra 'date' column (we keep met_day)
    merged = merged.drop(columns=["date"])

    # If any pump_count missing (should not happen, but safe) → 0
    merged["pump_count"] = merged["pump_count"].fillna(0).astype(int)

    # --- Reorder: put pump_count right after met_day ---
    cols = list(merged.columns)
    cols.insert(1, cols.pop(cols.index("pump_count")))
    merged = merged[cols]

    # Ensure met_day written as date only (YYYY-MM-DD)
    merged["met_day"] = merged["met_day"].dt.date

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"daily_weather_pump_{year}.csv"
    merged.to_csv(out_file, index=False)

    print(f"   💾 Saved merged file: {out_file}")
    print(f"   Rows: {len(merged)} | Columns: {len(merged.columns)}")


def main():
    print("=" * 80)
    print("🔗 Merging daily weather (with VPD) and daily pump counts per year")
    print("=" * 80)

    for year in YEARS:
        try:
            merge_weather_and_pumps(year)
        except Exception as e:
            print(f"   ❌ Error for {year}: {e}")

    print("\n" + "=" * 80)
    print("✅ Finished creating merged weather + pump CSVs")
    print("=" * 80)


if __name__ == "__main__":
    main()
