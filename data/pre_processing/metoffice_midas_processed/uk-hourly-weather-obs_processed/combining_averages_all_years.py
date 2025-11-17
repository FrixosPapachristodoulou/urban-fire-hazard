from pathlib import Path
import pandas as pd

# === CONFIGURATION ===
INPUT_DIR = Path("data/pre_processing/metoffice_midas_processed/averaged")
OUTPUT_DIR = Path("data/pre_processing/metoffice_midas_processed/combined")
OUTPUT_FILE = OUTPUT_DIR / "daily_temperature_2008_2024_combined.csv"

YEARS = list(range(2009, 2025))

def load_all_years(input_dir, years):
    dfs = []
    for y in years:
        file = input_dir / f"daily_temperature_averaged_{y}.csv"
        print(f"📄 Loading {file}")
        df = pd.read_csv(file)
        dfs.append(df)
    return dfs

def merge_duplicate_days(df):

    # Ensure date is datetime
    df["met_day"] = pd.to_datetime(df["met_day"])

    # Convert numeric cols
    numeric_cols = [
        "max_air_temp",
        "min_air_temp",
        "min_grss_temp",
        "min_conc_temp",
        "station_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregation rules
    agg_dict = {
        "max_air_temp": "max",
        "min_air_temp": "min",
        "min_grss_temp": "min",
        "min_conc_temp": "min",
        "station_count": "max",
    }

    # Perform aggregation
    df = df.groupby("met_day", as_index=False).agg(agg_dict)

    return df

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("📂 Loading all yearly averaged temperature files...")
    dfs = load_all_years(INPUT_DIR, YEARS)

    print("\n📚 Concatenating all years into one DataFrame...")
    df_all = pd.concat(dfs, ignore_index=True)

    print("🧹 Merging duplicate 31/12 days...")
    df_final = merge_duplicate_days(df_all)

    print("📅 Sorting by date...")
    df_final = df_final.sort_values("met_day").reset_index(drop=True)

    # === NEW STEP: REMOVE 2008 observation ===
    print("🗑 Removing 2008-12-31 (first combined observation)...")
    df_final = df_final[df_final["met_day"] >= pd.to_datetime("2009-01-01")].reset_index(drop=True)

    print(f"💾 Saving combined file to: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Done!")
    print(f"   Range: {df_final['met_day'].min().date()} → {df_final['met_day'].max().date()}")
    print(f"   Total rows: {len(df_final)}")



if __name__ == "__main__":
    main()
