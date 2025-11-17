from pathlib import Path
import pandas as pd
import numpy as np

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

YEARS = list(range(2009, 2025))  # 2009–2024 inclusive

BASE_DIR = Path("data/pre_processing/metoffice_midas_processed/uk-hourly-weather-obs_processed")
INPUT_ROOT = BASE_DIR / "1_cleaned_by_year"
OUTPUT_ROOT = BASE_DIR / "2_daily_by_year"
# ========================


def circular_mean_deg(series: pd.Series) -> float:
    """Circular mean of wind direction in degrees (0–360).
    Returns NaN if no valid values."""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    rad = np.deg2rad(vals.to_numpy())
    sin_m = np.sin(rad).mean()
    cos_m = np.cos(rad).mean()
    angle = np.arctan2(sin_m, cos_m)
    deg = np.rad2deg(angle)
    if deg < 0:
        deg += 360.0
    return deg


def ground_state_mode(series: pd.Series):
    """Most frequent ground_state_id; fall back to last observation if mode empty."""
    m = series.mode()
    if not m.empty:
        return m.iloc[0]
    if len(series) == 0:
        return np.nan
    return series.iloc[-1]


def aggregate_hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure ob_time as datetime and create date column
    df["ob_time"] = pd.to_datetime(df["ob_time"], errors="coerce")
    df = df.dropna(subset=["ob_time"])
    df["date"] = df["ob_time"].dt.date

    # Compute hourly VPD
    # --------------------------------------------
    T = pd.to_numeric(df["air_temperature"], errors="coerce")
    RH = pd.to_numeric(df["rltv_hum"], errors="coerce")

    esat = 610.7 * (10 ** (7.5 * T / (237.3 + T)))
    phi = RH / 100.0
    df["VPD_hourly"] = esat * (1 - phi)
    # --------------------------------------------

    # Convert numeric columns
    numeric_main = [
        "air_temperature","dewpoint","wetb_temp","rltv_hum",
        "msl_pressure","stn_pres","alt_pres","wind_speed",
        "visibility","q10mnt_mxgst_spd",
        "cs_hr_sun_dur","wmo_hr_sun_dur","drv_hr_sun_dur",
        "snow_depth","VPD_hourly"
    ]

    qc_cols = [
        "air_temperature_q","dewpoint_q","wetb_temp_q","msl_pressure_q",
        "stn_pres_q","alt_pres_q","wind_direction_q","wind_speed_q",
        "visibility_q","q10mnt_mxgst_spd_q","cs_hr_sun_dur_q",
        "wmo_hr_sun_dur_q","drv_hr_sun_dur_q","snow_depth_q",
        "ground_state_id_q"
    ]

    numeric_main = [c for c in numeric_main if c in df.columns]
    qc_cols = [c for c in qc_cols if c in df.columns]

    df[numeric_main + qc_cols] = df[numeric_main + qc_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Group
    group_cols = ["id", "src_id", "date"]
    if "id_type" in df.columns:
        group_cols.insert(1, "id_type")

    gb = df.groupby(group_cols, dropna=False)

    # ---- Aggregation spec ----
    agg_spec = {}

    # Temperatures: mean/max/min
    if "air_temperature" in df.columns:
        agg_spec["air_temperature"] = ["mean", "max", "min"]
    if "dewpoint" in df.columns:
        agg_spec["dewpoint"] = ["mean", "max", "min"]
    if "wetb_temp" in df.columns:
        agg_spec["wetb_temp"] = ["mean", "max", "min"]

    if "rltv_hum" in df.columns:
        agg_spec["rltv_hum"] = ["mean", "max", "min"]

    # Pressure means
    for col in ["msl_pressure", "stn_pres", "alt_pres"]:
        if col in df.columns:
            agg_spec[col] = ["mean"]

    # Wind, visibility, gust
    if "wind_speed" in df.columns:
        agg_spec["wind_speed"] = ["mean", "max"]
    if "visibility" in df.columns:
        agg_spec["visibility"] = ["mean", "min"]
    if "q10mnt_mxgst_spd" in df.columns:
        agg_spec["q10mnt_mxgst_spd"] = ["max"]

    # Sunshine sums
    for col in ["cs_hr_sun_dur","wmo_hr_sun_dur","drv_hr_sun_dur"]:
        if col in df.columns:
            agg_spec[col] = ["sum"]

    # Snow
    if "snow_depth" in df.columns:
        agg_spec["snow_depth"] = ["max"]

    # QC flags: worst (max)
    for col in qc_cols:
        agg_spec[col] = ["max"]

    # --- New: VPD hourly min/max/mean ---
    agg_spec["VPD_hourly"] = ["mean", "max", "min"]
    # ------------------------------------

    # Run aggregation
    daily_main = gb.agg(agg_spec)

    # Flatten columns
    daily_main.columns = [
        f"{var}_{stat}" for var, stat in daily_main.columns.to_flat_index()
    ]
    daily_main = daily_main.reset_index()

    # Wind direction circular mean
    if "wind_direction" in df.columns:
        daily_dir = gb["wind_direction"].agg(circular_mean_deg).reset_index()
        daily_dir = daily_dir.rename(columns={"wind_direction":"wind_direction_circmean"})
        daily_main = daily_main.merge(daily_dir, on=group_cols, how="left")

    # Ground state mode
    if "ground_state_id" in df.columns:
        daily_ground = gb["ground_state_id"].agg(ground_state_mode).reset_index()
        daily_ground = daily_ground.rename(columns={"ground_state_id":"ground_state_id_mode"})
        daily_main = daily_main.merge(daily_ground, on=group_cols, how="left")

    # Rename date → met_day
    daily_main = daily_main.rename(columns={"date":"met_day"})

    # Insert VPD_mean / VPD_max / VPD_min after met_day
    daily_main.insert(
        daily_main.columns.get_loc("met_day")+1,
        "VPD_mean", daily_main["VPD_hourly_mean"]
    )
    daily_main.insert(
        daily_main.columns.get_loc("met_day")+2,
        "VPD_max", daily_main["VPD_hourly_max"]
    )
    daily_main.insert(
        daily_main.columns.get_loc("met_day")+3,
        "VPD_min", daily_main["VPD_hourly_min"]
    )

    return daily_main



def process_station_year(station: str, year: int) -> None:
    input_file = (
        INPUT_ROOT
        / f"{year}"
        / f"hourly-weather_clean_{station}.csv"
    )
    output_dir = OUTPUT_ROOT / f"{year}"
    output_file = output_dir / f"hourly-weather_daily_{station}.csv"

    if not input_file.exists():
        print(f"   ⏭️  Skipping {year} - {station}: file not found ({input_file})")
        return

    print(f"   📄 Loading {input_file}")
    df_hourly = pd.read_csv(input_file)

    if df_hourly.empty:
        print(f"   ⚠️  Empty file, skipping {year} - {station}")
        return

    df_daily = aggregate_hourly_to_daily(df_hourly)

    output_dir.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(output_file, index=False)
    print(f"   ✅ Saved daily file to {output_file} (rows: {len(df_daily)})")


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Converting hourly MIDAS weather data to daily summaries")
    print("=" * 80)

    total_done = 0
    total_skipped = 0

    for year in YEARS:
        print(f"\n📅 Year: {year}")
        for station in STATIONS:
            try:
                process_station_year(station, year)
                total_done += 1
            except FileNotFoundError:
                total_skipped += 1
                continue
            except Exception as e:
                print(f"   ❌ Error processing {year} - {station}: {e}")
                total_skipped += 1
                continue

    print("\n" + "=" * 80)
    print("✅ Hourly → Daily conversion complete")
    print(f"   Station-year files attempted: {len(YEARS) * len(STATIONS)}")
    print(f"   Successful: {total_done}")
    print(f"   Skipped / errored: {total_skipped}")
    print("=" * 80)
