from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/1_num_fires_vs_vpd")
YEARS = list(range(2009, 2025))
# =======================


def is_spring_summer(date: datetime) -> bool:
    month = date.month
    return 3 <= month <= 8


def load_all_data() -> pd.DataFrame:
    dfs = []
    for year in YEARS:
        file = BASE_DIR / f"daily_weather_fire_{year}.csv"
        if not file.exists():
            continue
        df = pd.read_csv(file)
        df["met_day"] = pd.to_datetime(df["met_day"], errors="coerce")
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No input datasets found.")
    return pd.concat(dfs, ignore_index=True)


def fit_quadratic(vpd, fires):
    """Standard quadratic fit in log-log space."""
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]
    log_x = np.log(x)
    log_y = np.log(y + 1.0)
    gamma, beta, alpha = np.polyfit(log_x, log_y, 2)
    return alpha, beta, gamma


def analyze_thresholds(df: pd.DataFrame):
    """Analyze R² at different fire count thresholds."""
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )
    summer_df = df[df["season"] == "summer"]
    
    vpd = summer_df["VPD_mean"].values
    fires = summer_df["fire_count"].values.astype(float)
    
    # Fit the model
    alpha, beta, gamma = fit_quadratic(vpd, fires)
    
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]
    log_x = np.log(x)
    log_y = np.log(y + 1.0)
    y_pred = alpha + beta * log_x + gamma * (log_x ** 2)
    
    print("\n" + "=" * 80)
    print("FIRE COUNT DISTRIBUTION")
    print("=" * 80)
    percentiles = [50, 60, 70, 80, 90, 95]
    for p in percentiles:
        val = np.percentile(y, p)
        count = np.sum(y > val)
        print(f"  {p}th percentile: {val:.0f} fires/day ({count} days above)")
    
    print(f"\n  Max fires in a day: {np.max(y):.0f}")
    print(f"  Mean fires/day: {np.mean(y):.1f}")
    print(f"  Median fires/day: {np.median(y):.0f}")
    
    print("\n" + "=" * 80)
    print("R² BY FIRE COUNT THRESHOLD")
    print("=" * 80)
    
    thresholds = [10, 15, 20, 25, 30, 40, 50]
    results = []
    
    for thresh in thresholds:
        high_mask = y > thresh
        n_high = np.sum(high_mask)
        
        if n_high < 10:
            print(f"\n  Threshold {thresh}: Only {n_high} days, skipping...")
            continue
        
        # R² for high-fire days
        ss_res_high = np.sum((log_y[high_mask] - y_pred[high_mask]) ** 2)
        ss_tot_high = np.sum((log_y[high_mask] - np.mean(log_y[high_mask])) ** 2)
        r2_high = 1.0 - ss_res_high / ss_tot_high if ss_tot_high > 0 else np.nan
        
        # Mean absolute error in original scale for high-fire days
        pred_fires = np.exp(y_pred[high_mask]) - 1
        actual_fires = y[high_mask]
        mae = np.mean(np.abs(pred_fires - actual_fires))
        
        # Bias: is model under or over-predicting?
        bias = np.mean(pred_fires - actual_fires)
        
        results.append((thresh, n_high, r2_high, mae, bias))
        print(f"\n  Threshold > {thresh} fires/day:")
        print(f"    N days: {n_high}")
        print(f"    R²: {r2_high:.3f}")
        print(f"    MAE: {mae:.1f} fires")
        print(f"    Bias: {bias:+.1f} fires ({'under-predicting' if bias < 0 else 'over-predicting'})")
    
    return x, y, y_pred, alpha, beta, gamma, results


def plot_diagnostic(df: pd.DataFrame):
    """Create diagnostic plots to understand the fitting problem."""
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )
    summer_df = df[df["season"] == "summer"]
    
    vpd = summer_df["VPD_mean"].values
    fires = summer_df["fire_count"].values.astype(float)
    
    x, y, y_pred, alpha, beta, gamma, results = analyze_thresholds(df)
    
    log_x = np.log(x)
    log_y = np.log(y + 1.0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # === Plot 1: Scatter with residuals colored ===
    ax1 = axes[0, 0]
    residuals = log_y - y_pred
    scatter = ax1.scatter(x, y, c=residuals, cmap='RdBu_r', alpha=0.6, s=20,
                          vmin=-2, vmax=2)
    plt.colorbar(scatter, ax=ax1, label='Residual (log scale)')
    
    # Add fit curve
    vpd_range = np.linspace(np.min(x), np.max(x), 300)
    log_v = np.log(vpd_range)
    fires_fit = np.exp(alpha + beta * log_v + gamma * log_v**2) - 1
    ax1.plot(vpd_range, fires_fit, 'k-', linewidth=2, label='Quadratic fit')
    
    ax1.set_xlabel("VPD (Pa)")
    ax1.set_ylabel("Fire Count")
    ax1.set_title("Residuals: Blue=Over-predicted, Red=Under-predicted")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Predicted vs Actual (high-fire days highlighted) ===
    ax2 = axes[0, 1]
    pred_fires = np.exp(y_pred) - 1
    
    # Low fire days
    low_mask = y <= 30
    ax2.scatter(y[low_mask], pred_fires[low_mask], alpha=0.4, s=20, 
                color='gray', label='≤30 fires/day')
    
    # High fire days
    high_mask = y > 30
    ax2.scatter(y[high_mask], pred_fires[high_mask], alpha=0.7, s=40,
                color='red', label='>30 fires/day')
    
    # Perfect prediction line
    max_val = max(np.max(y), np.max(pred_fires))
    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect prediction')
    
    ax2.set_xlabel("Actual Fire Count")
    ax2.set_ylabel("Predicted Fire Count")
    ax2.set_title("Predicted vs Actual")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Fire count histogram ===
    ax3 = axes[1, 0]
    ax3.hist(y, bins=50, edgecolor='black', alpha=0.7)
    
    # Add percentile lines
    for p, color in [(70, 'orange'), (90, 'red')]:
        val = np.percentile(y, p)
        ax3.axvline(val, color=color, linestyle='--', linewidth=2, 
                    label=f'{p}th percentile ({val:.0f})')
    
    ax3.set_xlabel("Fire Count")
    ax3.set_ylabel("Frequency (days)")
    ax3.set_title("Distribution of Daily Fire Counts")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Plot 4: VPD distribution for different fire levels ===
    ax4 = axes[1, 1]
    
    fire_bins = [(0, 10), (10, 20), (20, 40), (40, 200)]
    colors = ['blue', 'green', 'orange', 'red']
    
    for (low, high), color in zip(fire_bins, colors):
        mask = (y >= low) & (y < high)
        if np.sum(mask) > 0:
            ax4.hist(x[mask], bins=30, alpha=0.5, color=color,
                     label=f'{low}-{high} fires (n={np.sum(mask)})')
    
    ax4.set_xlabel("VPD (Pa)")
    ax4.set_ylabel("Frequency (days)")
    ax4.set_title("VPD Distribution by Fire Count Level")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = BASE_DIR / "1_vpd_studies/fires_vs_vpd_diagnostic.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved diagnostic plot to: {out_path}")
    
    plt.show()
    
    # === Key insight ===
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    
    # Check VPD overlap between fire levels
    low_fire_vpd = x[y <= 15]
    high_fire_vpd = x[y > 30]
    
    print(f"\nVPD range for low-fire days (≤15): {np.min(low_fire_vpd):.0f} - {np.max(low_fire_vpd):.0f} Pa")
    print(f"VPD range for high-fire days (>30): {np.min(high_fire_vpd):.0f} - {np.max(high_fire_vpd):.0f} Pa")
    print(f"\nVPD median for low-fire days: {np.median(low_fire_vpd):.0f} Pa")
    print(f"VPD median for high-fire days: {np.median(high_fire_vpd):.0f} Pa")
    
    # Overlap
    overlap_min = max(np.min(low_fire_vpd), np.min(high_fire_vpd))
    overlap_max = min(np.max(low_fire_vpd), np.max(high_fire_vpd))
    print(f"\nOVERLAP ZONE: {overlap_min:.0f} - {overlap_max:.0f} Pa")
    print("→ Same VPD values can produce BOTH low and high fire counts!")
    print("→ This is why VPD alone cannot reliably predict high-fire days.")


def main():
    print("Loading data...")
    df = load_all_data()
    print("Running diagnostic analysis...")
    plot_diagnostic(df)


if __name__ == "__main__":
    main()