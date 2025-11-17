# average_stations.py
# -------------------
# Averages daily temperature observations across multiple stations for each day

from pathlib import Path
import pandas as pd
import numpy as np

# ==== CONFIGURATION ====
YEAR = "2009"
INPUT_DIR = Path(f"data/pre_processing/metoffice_midas_processed/uk-daily-temperature-obs_processed/{YEAR}")
OUTPUT_FILE = Path(f"data/pre_processing/metoffice_midas_processed/averaged/daily_temperature_averaged_{YEAR}.csv")

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
# ========================


def average_station_data(input_dir: Path, output_file: Path, year: str, stations: list):
    """
    Reads all station CSV files for a given year and averages the temperature
    readings for each meteorological day.
    """
    
    all_dfs = []
    
    print(f"📂 Reading station files from: {input_dir}")
    
    # --- 1️⃣ Load all station files ---
    for station in stations:
        station_file = input_dir / f"daily_temperature_clean_{station}.csv"
        
        if not station_file.exists():
            print(f"   ⏭️  Skipping {station}: File not found")
            continue
        
        try:
            df = pd.read_csv(station_file)
            
            # Convert met_day to datetime for proper sorting/grouping
            df["met_day"] = pd.to_datetime(df["met_day"])
            
            # Convert temperature columns to numeric (they might be strings)
            temp_cols = ["max_air_temp", "min_air_temp", "min_grss_temp", "min_conc_temp"]
            for col in temp_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            all_dfs.append(df)
            print(f"   ✅ Loaded {station}: {len(df)} rows")
            
        except Exception as e:
            print(f"   ❌ Error loading {station}: {e}")
            continue
    
    if not all_dfs:
        raise ValueError("No station files were successfully loaded!")
    
    # --- 2️⃣ Concatenate all dataframes ---
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 Combined data: {len(combined_df)} total rows from {len(all_dfs)} stations")
    
    # --- 3️⃣ Group by met_day and calculate averages ---
    print(f"\n🧮 Calculating daily averages...")
    
    # Define which columns to average
    avg_cols = ["max_air_temp", "min_air_temp", "min_grss_temp", "min_conc_temp"]
    avg_cols = [col for col in avg_cols if col in combined_df.columns]
    
    # Group by met_day and calculate mean, ignoring NaN values
    averaged_df = combined_df.groupby("met_day")[avg_cols].mean().reset_index()
    
    # Add a column showing how many stations contributed to each day's average
    station_counts = combined_df.groupby("met_day").size().reset_index(name="station_count")
    averaged_df = averaged_df.merge(station_counts, on="met_day")
    
    # Sort by date
    averaged_df = averaged_df.sort_values("met_day").reset_index(drop=True)
    
    # Round temperature values to 1 decimal place
    for col in avg_cols:
        averaged_df[col] = averaged_df[col].round(1)
    
    print(f"   ✅ Calculated averages for {len(averaged_df)} unique days")
    print(f"   Columns averaged: {avg_cols}")
    
    # --- 4️⃣ Save averaged data ---
    output_file.parent.mkdir(parents=True, exist_ok=True)
    averaged_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved averaged data to: {output_file}")
    print(f"   Total days: {len(averaged_df)}")
    print(f"\n📈 Sample of averaged data:")
    print(averaged_df.head(10))
    
    return averaged_df


if __name__ == "__main__":
    print("=" * 80)
    print(f"🚀 Averaging station data for year {YEAR}")
    print("=" * 80)
    
    averaged_df = average_station_data(INPUT_DIR, OUTPUT_FILE, YEAR, STATIONS)
    
    print("\n" + "=" * 80)
    print("✅ Averaging complete!")
    print("=" * 80)