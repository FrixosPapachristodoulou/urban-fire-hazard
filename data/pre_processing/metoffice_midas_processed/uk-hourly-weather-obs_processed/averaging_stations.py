from pathlib import Path
import pandas as pd

# ===== CONFIGURATION =====
BASE_DIR = Path("data/pre_processing/metoffice_midas_processed/uk-hourly-weather-obs_processed")
INPUT_ROOT = BASE_DIR / "2_daily_by_year"
OUTPUT_ROOT = BASE_DIR / "3_averaged_by_year"

YEARS = list(range(2009, 2025))  # 2009–2024 inclusive
# =========================


def average_daily_across_stations(year: int):
    year_dir = INPUT_ROOT / f"{year}"
    if not year_dir.exists():
        print(f"   ⏭️  Year folder not found: {year_dir}")
        return

    files = sorted(year_dir.glob("hourly-weather_daily_*.csv"))
    if not files:
        print(f"   ⏭️  No daily station files found for {year}")
        return

    print(f"   📂 Found {len(files)} station files for {year}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                print(f"      ⚠️  Empty file, skipping: {f.name}")
                continue

            # Ensure met_day is datetime-like (for safe grouping / sorting)
            if "met_day" in df.columns:
                df["met_day"] = pd.to_datetime(df["met_day"], errors="coerce")
            else:
                raise ValueError(f"'met_day' column missing in {f}")

            dfs.append(df)
        except Exception as e:
            print(f"      ❌ Error reading {f.name}: {e}")
            continue

    if not dfs:
        print(f"   ⚠️  No valid data frames for {year}, skipping.")
        return

    # Concatenate all station data for this year
    combined = pd.concat(dfs, ignore_index=True)

    # Group by date
    grouped = combined.groupby("met_day", dropna=False)

    # Columns we do NOT average (ID-like / categorical)
    exclude_from_mean = {"met_day", "id_type", "src_id"}

    # We actually want to use id for station_count, so don't average it
    # but we will exclude it from the numeric averaging
    if "id" in combined.columns:
        exclude_from_mean.add("id")

    # Numeric columns to average across stations
    numeric_cols = [
        c
        for c in combined.columns
        if c not in exclude_from_mean and pd.api.types.is_numeric_dtype(combined[c])
    ]

    # Mean across stations for each numeric column
    daily_means = grouped[numeric_cols].mean().reset_index()

    # Station count = number of unique station IDs contributing that day
    if "id" in combined.columns:
        station_counts = (
            grouped["id"].nunique()
            .reset_index()
            .rename(columns={"id": "station_count"})
        )
        daily_out = daily_means.merge(station_counts, on="met_day", how="left")
    else:
        daily_out = daily_means
        daily_out["station_count"] = grouped.size().values

    # Sort by date for neatness
    daily_out = daily_out.sort_values("met_day").reset_index(drop=True)

    # 🔹 Round all float columns to 3 decimal places
    float_cols = daily_out.select_dtypes(include="float").columns
    daily_out[float_cols] = daily_out[float_cols].round(3)

    # Optionally ensure met_day is written as YYYY-MM-DD (no time)
    daily_out["met_day"] = daily_out["met_day"].dt.date

    # Save
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_ROOT / f"daily_weather_averaged_{year}.csv"
    daily_out.to_csv(out_file, index=False)

    print(f"   ✅ Saved averaged daily file for {year} to {out_file}")
    print(f"      Rows (days): {len(daily_out)} | Columns: {len(daily_out.columns)}")


def main():
    print("=" * 80)
    print("🚀 Averaging daily weather across stations (per year)")
    print("=" * 80)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        print(f"\n📅 Year: {year}")
        average_daily_across_stations(year)

    print("\n" + "=" * 80)
    print("✅ Finished averaging daily weather across stations")
    print("=" * 80)


if __name__ == "__main__":
    main()
