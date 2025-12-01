"""
Process UK Met Office MIDAS daily rain observation data.

This script:
1. Strips MIDAS metadata from rain CSV files
2. Keeps only essential columns (date, station, precipitation amount)
3. Aggregates across stations to get a London-wide daily rain value
4. Computes "days since significant rain" for antecedent moisture analysis

Output will be used to investigate how antecedent rainfall modifies
the VPD-fire relationship in Greater London.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import csv

# ==== CONFIGURATION ====
# Rain station folders (based on your MIDAS structure)
# Format: "src_id_station-name"
RAIN_STATIONS = [
    "00695_hampstead",
    "00708_heathrow",
    "00709_northolt",
    "00710_northwood",
    "00711_hampton-w-wks",
    "05326_hornchurch-marshes-s-wks-no-2",
    "05536_walthamstow-coppermills-wks",
    "05571_buckingham-palace",
    "05574_chelsea-physic-garden",
    "05599_golders-hill-park",
    "05612_kingsbury-roe-green-park",
    "05656_isleworth-mogden-s-wks",
    "06570_hogsmill-valley-s-wks",
    "06586_cheam-p-sta-no-2",
    "06622_kenley-p-sta",
    "06625_purley-oaks-depot",
    "06633_beddington-new-s-wks",
    "06637_mitcham-london-road-cemetery",
    "06688_croydon-northampton-road",
    "06693_new-beckenham",
    "06704_deptford-p-sta",
    "06711_cross-ness-s-wks",
    "06759_danson-park",
    "18893_borehamwood-golf-club",
    "56945_norwood-resr",
    "56946_new-addington",
]

YEARS = list(range(2009, 2025))  # 2009 to 2024 inclusive

# Paths - adjust these to match your directory structure
RAW_DATA_DIR = Path("data/raw/metoffice_midas/uk-daily-rain-obs")
OUTPUT_DIR = Path("data/pre_processing/metoffice_midas_processed/uk-daily-rain-obs_processed")

# Threshold for "significant rain" in mm
SIGNIFICANT_RAIN_THRESHOLD = 1.0  # mm
# ========================


def strip_midas_rain_metadata(input_path: Path) -> pd.DataFrame:
    """
    Strip MIDAS metadata from a daily rain CSV, keep only useful columns.
    
    Returns a DataFrame with columns:
    - ob_date: observation date
    - src_id: station ID
    - prcp_amt: precipitation amount (mm)
    - prcp_amt_q: QC flag for precipitation
    """
    
    # --- 1. Read file and find where 'data' appears ---
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "data":
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

    # --- 2. Read the data section into a DataFrame ---
    df = pd.read_csv(
        input_path,
        skiprows=data_line_idx + 1,  # skip metadata + 'data' line
        sep=sep,
        dtype=str,
        na_values=["NA"],
        keep_default_na=True,
        engine="python",
    )
    
    # Remove the "end data" row if present
    if "end data" in df.iloc[:, 0].values:
        df = df[df.iloc[:, 0] != "end data"]
    
    # Also check for rows where ob_date contains "end"
    if "ob_date" in df.columns:
        df = df[~df["ob_date"].str.contains("end", case=False, na=False)]

    # --- 3. Select only useful columns ---
    keep_cols = ["ob_date", "src_id", "ob_day_cnt", "prcp_amt", "prcp_amt_q"]
    keep_cols_present = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols_present].copy()
    
    # --- 4. Filter to single-day observations only ---
    # ob_day_cnt > 1 indicates multi-day accumulations (e.g., monthly totals)
    # These are incorrectly recorded as single dates and must be excluded
    if "ob_day_cnt" in df.columns:
        df["ob_day_cnt"] = pd.to_numeric(df["ob_day_cnt"], errors="coerce")
        n_before = len(df)
        df = df[df["ob_day_cnt"] == 1].copy()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"      ⚠️  Removed {n_removed} multi-day accumulation records (ob_day_cnt > 1)")
        # Drop the column after filtering - no longer needed
        df = df.drop(columns=["ob_day_cnt"])
    
    # --- 5. Convert types ---
    df["ob_date"] = pd.to_datetime(df["ob_date"], errors="coerce")
    df["prcp_amt"] = pd.to_numeric(df["prcp_amt"], errors="coerce")
    df["src_id"] = pd.to_numeric(df["src_id"], errors="coerce").astype("Int64")
    
    if "prcp_amt_q" in df.columns:
        df["prcp_amt_q"] = pd.to_numeric(df["prcp_amt_q"], errors="coerce").astype("Int64")
    
    return df


def process_all_rain_stations():
    """
    Process all rain station files, combine into yearly aggregated files.
    
    Returns dict of {year: DataFrame} with station-level data.
    """
    
    print("=" * 80)
    print("🌧️  Processing MIDAS daily rain observation data")
    print("=" * 80)
    
    total_processed = 0
    total_skipped = 0
    
    # Collect all station data by year
    yearly_data = {year: [] for year in YEARS}
    
    for station in RAIN_STATIONS:
        print(f"\n📍 Processing station: {station}")
        
        for year in YEARS:
            # Construct input path - adjust pattern to match your files
            # Common patterns:
            # midas-open_uk-daily-rain-obs_dv-202507_greater-london_{station}_qcv-1_{year}.csv
            input_file = RAW_DATA_DIR / station / (
                f"midas-open_uk-daily-rain-obs_dv-202507_"
                f"greater-london_{station}_qcv-1_{year}.csv"
            )
            
            if not input_file.exists():
                # Try alternative naming patterns
                alt_patterns = [
                    f"midas-open_uk-daily-rain-obs_dv-*_greater-london_{station}_qcv-1_{year}.csv",
                ]
                found = False
                for pattern in alt_patterns:
                    matches = list((RAW_DATA_DIR / station).glob(pattern.replace("*", "*")))
                    if matches:
                        input_file = matches[0]
                        found = True
                        break
                
                if not found:
                    total_skipped += 1
                    continue
            
            try:
                df = strip_midas_rain_metadata(input_file)
                df["station"] = station
                yearly_data[year].append(df)
                total_processed += 1
                print(f"   ✅ {year}: {len(df)} records")
            except Exception as e:
                print(f"   ❌ Error processing {station} - {year}: {e}")
                total_skipped += 1
    
    print(f"\n✅ Station processing complete!")
    print(f"   Files processed: {total_processed}")
    print(f"   Files skipped:   {total_skipped}")
    
    return yearly_data


def aggregate_yearly_rain(yearly_data: dict) -> dict:
    """
    For each year, aggregate station-level rain to daily London-wide values.
    
    Saves yearly averaged files: daily_rain_averaged_{year}.csv
    
    Returns dict of {year: DataFrame} with daily averaged data.
    """
    
    print("\n" + "=" * 80)
    print("📊 Aggregating station data by year (averaging across stations)...")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    averaged_by_year_dir = OUTPUT_DIR / "averaged_by_year"
    averaged_by_year_dir.mkdir(parents=True, exist_ok=True)
    
    yearly_averaged = {}
    
    for year in YEARS:
        if not yearly_data[year]:
            print(f"   ⏭️  {year}: No data")
            continue
        
        # Combine all stations for this year
        df_year = pd.concat(yearly_data[year], ignore_index=True)
        
        # Extract date only (no time component)
        df_year["met_day"] = df_year["ob_date"].dt.date
        
        # Filter out invalid precipitation values
        df_valid = df_year[df_year["prcp_amt"].notna() & (df_year["prcp_amt"] >= 0)].copy()
        
        # Aggregate by day - AVERAGE across stations
        daily_rain = df_valid.groupby("met_day").agg(
            prcp_mean=("prcp_amt", "mean"),
            prcp_max=("prcp_amt", "max"),
            prcp_min=("prcp_amt", "min"),
            prcp_std=("prcp_amt", "std"),
            n_stations=("prcp_amt", "count"),
        ).reset_index()
        
        daily_rain["met_day"] = pd.to_datetime(daily_rain["met_day"])
        daily_rain = daily_rain.sort_values("met_day").reset_index(drop=True)
        
        # Add year column for convenience
        daily_rain["year"] = year
        
        # Round to 3 decimal places
        df_rounded = daily_rain.round(3)
        
        # Save yearly averaged file
        out_path = averaged_by_year_dir / f"daily_rain_averaged_{year}.csv"
        df_rounded.to_csv(out_path, index=False)
        
        yearly_averaged[year] = daily_rain
        
        print(f"   💾 {year}: {len(daily_rain)} days, "
              f"mean {daily_rain['n_stations'].mean():.1f} stations/day → {out_path.name}")
    
    return yearly_averaged


def combine_all_years(yearly_averaged: dict) -> pd.DataFrame:
    """
    Combine all yearly averaged DataFrames into one master DataFrame.
    
    Returns DataFrame with columns:
    - met_day: date
    - prcp_mean: mean precipitation across stations (mm)
    - prcp_max: max precipitation at any station (mm)
    - prcp_min: min precipitation at any station (mm)
    - prcp_std: std dev across stations (mm)
    - n_stations: number of stations reporting
    - year: year
    """
    
    print("\n" + "=" * 80)
    print("🌍 Combining all years into master dataset...")
    print("=" * 80)
    
    all_dfs = []
    for year in YEARS:
        if year in yearly_averaged and yearly_averaged[year] is not None:
            all_dfs.append(yearly_averaged[year])
    
    if not all_dfs:
        raise RuntimeError("No averaged rain data found!")
    
    daily_rain = pd.concat(all_dfs, ignore_index=True)
    daily_rain = daily_rain.sort_values("met_day").reset_index(drop=True)
    
    print(f"   Total days with rain data: {len(daily_rain)}")
    print(f"   Date range: {daily_rain['met_day'].min()} to {daily_rain['met_day'].max()}")
    print(f"   Mean stations per day: {daily_rain['n_stations'].mean():.1f}")
    
    return daily_rain


def compute_days_since_rain(daily_rain: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Compute "days since last significant rain" for each day.
    
    Parameters:
    -----------
    daily_rain : DataFrame with columns [met_day, prcp_mean, ...]
    threshold : minimum precipitation (mm) to count as "significant rain"
    
    Returns:
    --------
    DataFrame with additional columns:
    - significant_rain: bool, True if prcp_mean >= threshold
    - days_since_rain: int, days since last significant rain event
    - rain_category: categorical bin for stratified analysis
    """
    
    print(f"\n📐 Computing days since significant rain (threshold = {threshold} mm)...")
    
    df = daily_rain.copy()
    df = df.sort_values("met_day").reset_index(drop=True)
    
    # Flag significant rain days
    df["significant_rain"] = df["prcp_mean"] >= threshold
    
    # Compute days since last significant rain
    days_since = []
    last_rain_idx = -999  # Start with a large negative number
    
    for i, row in df.iterrows():
        if row["significant_rain"]:
            days_since.append(0)
            last_rain_idx = i
        else:
            if last_rain_idx < 0:
                # No rain yet in the series - use NaN or a large number
                days_since.append(np.nan)
            else:
                days_since.append(i - last_rain_idx)
    
    df["days_since_rain"] = days_since
    
    # Create categorical bins for stratified analysis
    def categorize_days(d):
        if pd.isna(d):
            return "unknown"
        elif d <= 2:
            return "0-2 days"
        elif d <= 7:
            return "3-7 days"
        elif d <= 14:
            return "8-14 days"
        else:
            return "15+ days"
    
    df["rain_category"] = df["days_since_rain"].apply(categorize_days)
    
    # Summary statistics
    print(f"\n   Rain category distribution:")
    print(df["rain_category"].value_counts().sort_index())
    
    return df


def main():
    """
    Main processing pipeline.
    """
    
    # Step 1: Process all station files (raw extraction)
    yearly_data = process_all_rain_stations()
    
    # Step 2: Aggregate by year (average across stations for each day within each year)
    yearly_averaged = aggregate_yearly_rain(yearly_data)
    
    # Step 3: Combine all years into master dataset
    daily_rain = combine_all_years(yearly_averaged)
    
    # Step 4: Compute days since significant rain
    daily_rain = compute_days_since_rain(daily_rain, threshold=SIGNIFICANT_RAIN_THRESHOLD)
    
    # Step 5: Save final output (rounded to 3 decimal places)
    daily_rain_rounded = daily_rain.round(3)
    output_path = OUTPUT_DIR / "daily_london_rain_with_antecedent.csv"
    daily_rain_rounded.to_csv(output_path, index=False)
    print(f"\n💾 Saved final rain dataset to: {output_path}")
    
    # Preview
    print("\n📋 Preview of output:")
    print(daily_rain.head(20).to_string())
    
    return daily_rain


if __name__ == "__main__":
    main()