"""
Stratified VPD vs Fire Analysis by Antecedent Rainfall

Creates a 2x2 plot showing VPD vs fires for each rain category,
letting the data reveal if antecedent moisture affects the relationship.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/1_num_fires_vs_vpd")
RAIN_DATA_PATH = Path("data/pre_processing/metoffice_midas_processed/uk-daily-rain-obs_processed/daily_london_rain_with_antecedent.csv")
YEARS = list(range(2009, 2025))
SIGNIFICANT_RAIN_THRESHOLD = 1.0  # mm

RAIN_CATEGORIES = ["0-2 days", "3-7 days", "8-14 days", "15+ days"]
RAIN_COLORS = {
    "0-2 days": "#1f77b4",   # blue
    "3-7 days": "#2ca02c",   # green
    "8-14 days": "#ff7f0e",  # orange
    "15+ days": "#d62728",   # red
}
# =======================


def load_and_merge_data() -> pd.DataFrame:
    """Load weather-fire CSVs and merge with rain data."""
    
    # Load weather-fire data
    dfs = []
    for year in YEARS:
        file = BASE_DIR / f"daily_weather_fire_{year}.csv"
        if file.exists():
            df = pd.read_csv(file)
            df["met_day"] = pd.to_datetime(df["met_day"])
            dfs.append(df)
    
    if not dfs:
        raise RuntimeError("No weather-fire datasets found.")
    
    df_wf = pd.concat(dfs, ignore_index=True)
    
    # Load rain data
    if not RAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Rain data not found: {RAIN_DATA_PATH}")
    
    df_rain = pd.read_csv(RAIN_DATA_PATH)
    df_rain["met_day"] = pd.to_datetime(df_rain["met_day"])
    
    # Merge
    df = pd.merge(
        df_wf,
        df_rain[["met_day", "days_since_rain", "rain_category"]],
        on="met_day",
        how="left"
    )
    
    print(f"Merged: {len(df)} rows, {df['rain_category'].notna().sum()} with rain data")
    return df


def fit_power_law(vpd: np.ndarray, fires: np.ndarray):
    """Fit fires = a * VPD^b. Returns (a, b, r2)."""
    mask = (vpd > 0) & ~np.isnan(fires)
    x, y = vpd[mask], fires[mask]
    
    if len(x) < 10:
        return np.nan, np.nan, np.nan
    
    log_x, log_y = np.log(x), np.log(y + 1)
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)
    
    y_pred = log_a + b * log_x
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    return a, b, r2


def plot_stratified(df: pd.DataFrame, vpd_col: str = "VPD_mean"):
    """Create 2x2 stratified VPD vs fires plot."""
    
    df_valid = df[
        (df[vpd_col] > 0) & 
        df["fire_count"].notna() & 
        df["rain_category"].notna()
    ].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    
    # Optional: use these if you want global scaling from data
    vpd_max = df_valid[vpd_col].quantile(0.99)
    fire_max = df_valid["fire_count"].quantile(0.99)

    ref_vpd = 1000.0  # reference VPD level to highlight (Pa)
    
    for ax, cat in zip(axes, RAIN_CATEGORIES):
        df_cat = df_valid[df_valid["rain_category"] == cat]
        
        # Scatter points
        ax.scatter(
            df_cat[vpd_col], 
            df_cat["fire_count"],
            s=12, alpha=0.5, color=RAIN_COLORS[cat], edgecolors="none"
        )
        
        # Fit and plot power law
        a, b, r2 = fit_power_law(
            df_cat[vpd_col].values, 
            df_cat["fire_count"].values
        )
        if not np.isnan(a):
            vpd_curve = np.linspace(
                max(df_cat[vpd_col].min(), 1e-6),
                df_cat[vpd_col].max(),
                200
            )
            ax.plot(vpd_curve, a * vpd_curve ** b, "k-", lw=2, label="Power-law fit")
            
            # Highlight reference VPD point (e.g. 1000 Pa) if within range
            if df_cat[vpd_col].min() <= ref_vpd <= df_cat[vpd_col].max():
                ref_fire = a * ref_vpd ** b
                
                # Vertical reference line
                ax.axvline(
                    ref_vpd, 
                    color="0.4", 
                    linestyle="--", 
                    linewidth=1,
                    alpha=0.8
                )
                
                # Marker at (ref_vpd, ref_fire)
                ax.scatter(
                    [ref_vpd], [ref_fire],
                    s=45,
                    color=RAIN_COLORS[cat],
                    edgecolors="k",
                    zorder=5,
                    label=f"Model at {int(ref_vpd)} Pa"
                )
                
                # Annotation box
                ax.annotate(
                    f"{int(ref_vpd)} Pa\n{ref_fire:.1f} fires/day",
                    xy=(ref_vpd, ref_fire),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="0.7",
                        alpha=0.75
                    )
                )
        
        # Title
        n = len(df_cat)
        mean_f = df_cat["fire_count"].mean()

        title = rf"$\bf{{{cat}\ since\ rain}}$" + f"\nn={n}, mean={mean_f:.1f}"
        if not np.isnan(r2):
            title += f", b={b:.2f}, R²={r2:.2f}"

        ax.set_title(title, fontsize=12)
        
        # Styling
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 3050)
        ax.set_ylim(0, 120)

    # Shared labels and main title (outside loop)
    fig.supxlabel(f"{vpd_col} (Pa)", fontsize=12)
    fig.supylabel("Daily fire count", fontsize=12)
    fig.suptitle(
        f"VPD vs Fires by Antecedent Rainfall (threshold: {SIGNIFICANT_RAIN_THRESHOLD} mm)",
        fontsize=13,
        fontweight="bold",
        y=0.96
    )

    # Make room for suptitle
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])
    
    out_dir = BASE_DIR / "3_vpd_rain_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fires_vs_{vpd_col}_by_rain_category.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    
    plt.show()
    return fig


def main():
    print("Loading and merging data...")
    df = load_and_merge_data()
    
    print("Generating stratified plot...")
    plot_stratified(df, "VPD_mean")
    
    # Save merged dataset
    out_dir = BASE_DIR / "3_vpd_rain_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "daily_weather_fire_rain_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved merged dataset: {out_path}")
    
    print("Done!")



if __name__ == "__main__":
    main()