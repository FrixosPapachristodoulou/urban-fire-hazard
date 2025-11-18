from pathlib import Path
import pandas as pd

# ===== CONFIGURATION =====
BASE_DIR = Path("data/pre_processing/lfb_fire_data_processed")
INPUT_DIR = BASE_DIR / "1_filtering"
OUTPUT_DIR = BASE_DIR / "3_pump_hours"

YEARS = list(range(2009, 2025))  # 2009–2024 inclusive
# =========================


def process_year(year: int):
    input_file = INPUT_DIR / f"lfb_wildfires_{year}.csv"
    output_file = OUTPUT_DIR / f"daily_pump_hours_{year}.csv"

    if not input_file.exists():
        print(f"⏭️ Missing file for {year}: {input_file}")
        return

    print(f"\n📅 Year {year} → {input_file.name}")

    # Load wildfire dataset
    df = pd.read_csv(input_file)

    if df.empty:
        print("   ⚠️ File is empty — generating full year of zeros.")
        df = pd.DataFrame(columns=["DateOfCall", "PumpCount"])

    # Ensure DateOfCall exists and is valid
    if "DateOfCall" in df.columns:
        df["DateOfCall"] = pd.to_datetime(df["DateOfCall"], errors="coerce")
        df = df.dropna(subset=["DateOfCall"])

    # Ensure PumpCount column exists
    if "PumpCount" not in df.columns:
        print("   ⚠️ No 'PumpCount' column found — treating as zero for all days.")
        df["PumpCount"] = 0

    # Convert PumpCount to numeric
    df["PumpCount"] = pd.to_numeric(df["PumpCount"], errors="coerce").fillna(0)

    # Create daily sum of PumpCount for days with wildfires
    if not df.empty:
        df["date"] = df["DateOfCall"].dt.date
        daily_counts = (
            df.groupby("date", as_index=False)["PumpCount"]
              .sum()
              .rename(columns={"PumpCount": "pump_count"})
        )
    else:
        daily_counts = pd.DataFrame(columns=["date", "pump_count"])

    # ==== Full date range for the entire year ====
    full_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
    full_df = pd.DataFrame({"date": full_range.date})

    # Merge daily PumpCount into full date index
    full_df = full_df.merge(daily_counts, on="date", how="left")

    # Fill missing days with zero pumps
    full_df["pump_count"] = full_df["pump_count"].fillna(0).astype(int)
    # ============================================

    # Print summary
    for date, count in full_df.values:
        print(f"   {date} → {count} pumps")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save final CSV
    full_df.to_csv(output_file, index=False)

    print(f"   💾 Saved: {output_file}")


def main():
    print("=" * 80)
    print("🚒 Generating DAILY PumpCount totals (including zero-fire days)")
    print("=" * 80)

    for year in YEARS:
        process_year(year)

    print("\n" + "=" * 80)
    print("✅ Finished generating daily PumpCount CSVs")
    print("=" * 80)


if __name__ == "__main__":
    main()
