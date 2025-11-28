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
        df = pd.DataFrame(columns=["DateOfCall", "PumpMinutesRounded"])

    # Ensure DateOfCall exists and is valid
    if "DateOfCall" in df.columns:
        df["DateOfCall"] = pd.to_datetime(df["DateOfCall"], errors="coerce")
        df = df.dropna(subset=["DateOfCall"])
    else:
        print("   ⚠️ No 'DateOfCall' column found — generating full year of zeros.")
        df = pd.DataFrame(columns=["DateOfCall", "PumpMinutesRounded"])

    # Ensure PumpMinutesRounded column exists
    if "PumpMinutesRounded" not in df.columns:
        print("   ⚠️ No 'PumpMinutesRounded' column found — treating as zero.")
        df["PumpMinutesRounded"] = 0

    # Convert PumpMinutesRounded to numeric
    df["PumpMinutesRounded"] = (
        pd.to_numeric(df["PumpMinutesRounded"], errors="coerce")
          .fillna(0)
    )

    # ---- incident-level pump-hours (total across all pumps) ----
    df["pump_hours_incident"] = df["PumpMinutesRounded"] / 60.0

    # Daily sum of pump-hours for days with wildfires
    if not df.empty:
        df["date"] = df["DateOfCall"].dt.date
        daily_hours = (
            df.groupby("date", as_index=False)["pump_hours_incident"]
              .sum()
              .rename(columns={"pump_hours_incident": "pump_hours"})
        )
    else:
        daily_hours = pd.DataFrame(columns=["date", "pump_hours"])

    # ==== Full date range for the entire year ====
    full_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
    full_df = pd.DataFrame({"date": full_range.date})

    # Merge daily pump-hours into full date index
    full_df = full_df.merge(daily_hours, on="date", how="left")

    # Fill missing days with zero pump-hours
    full_df["pump_hours"] = full_df["pump_hours"].fillna(0.0)
    # ============================================

    # Print summary
    for date, hours in full_df.values:
        print(f"   {date} → {hours:.2f} pump-hours")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save final CSV
    full_df.to_csv(output_file, index=False)

    print(f"   💾 Saved: {output_file}")


def main():
    print("=" * 80)
    print("🚒 Generating DAILY pump-hours totals (including zero-fire days)")
    print("=" * 80)

    for year in YEARS:
        process_year(year)

    print("\n" + "=" * 80)
    print("✅ Finished generating daily pump-hours CSVs")
    print("=" * 80)


if __name__ == "__main__":
    main()
