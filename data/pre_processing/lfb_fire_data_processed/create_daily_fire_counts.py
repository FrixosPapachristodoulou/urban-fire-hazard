from pathlib import Path
import pandas as pd

# ===== CONFIGURATION =====
BASE_DIR = Path("data/pre_processing/lfb_fire_data_processed")
INPUT_DIR = BASE_DIR / "1_filtering"
OUTPUT_DIR = BASE_DIR / "2_number_fires"

YEARS = list(range(2009, 2025))  # 2009–2024 inclusive
# =========================


def process_year(year: int):
    input_file = INPUT_DIR / f"lfb_wildfires_{year}.csv"
    output_file = OUTPUT_DIR / f"daily_fire_counts_{year}.csv"

    if not input_file.exists():
        print(f"⏭️ Missing file for {year}: {input_file}")
        return

    print(f"\n📅 Year {year} → {input_file.name}")

    # Load wildfire dataset
    df = pd.read_csv(input_file)

    if df.empty:
        print("   ⚠️ File is empty — generating full year of zeros.")
        df = pd.DataFrame(columns=["DateOfCall"])

    # Ensure DateOfCall exists
    if "DateOfCall" in df.columns:
        df["DateOfCall"] = pd.to_datetime(df["DateOfCall"], errors="coerce")
        df = df.dropna(subset=["DateOfCall"])

    # Create daily counts only for days with fires
    if not df.empty:
        daily_counts = (
            df.groupby(df["DateOfCall"].dt.date)
              .size()
              .reset_index(name="fire_count")
              .rename(columns={"DateOfCall": "date"})
        )
    else:
        daily_counts = pd.DataFrame(columns=["date", "fire_count"])

    # ==== NEW: Generate full date range for the entire year ====
    full_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
    full_df = pd.DataFrame({"date": full_range.date})

    # Merge actual fire counts into full date index
    full_df = full_df.merge(daily_counts, on="date", how="left")

    # Fill missing days with zero fires
    full_df["fire_count"] = full_df["fire_count"].fillna(0).astype(int)
    # ===========================================================

    # Print summary
    for date, count in full_df.values:
        print(f"   {date} → {count} fires")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save final CSV
    full_df.to_csv(output_file, index=False)

    print(f"   💾 Saved: {output_file}")


def main():
    print("=" * 80)
    print("🔥 Generating DAILY wildfire counts (including zero-fire days)")
    print("=" * 80)

    for year in YEARS:
        process_year(year)

    print("\n" + "=" * 80)
    print("✅ Finished generating daily wildfire count CSVs")
    print("=" * 80)


if __name__ == "__main__":
    main()
