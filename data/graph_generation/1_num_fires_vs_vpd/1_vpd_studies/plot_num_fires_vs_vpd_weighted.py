from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.lines as mlines

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/1_num_fires_vs_vpd")
YEARS = list(range(2009, 2025))  # 2009–2024 inclusive
# =======================


def is_spring_summer(date: datetime) -> bool:
    """Return True if date is between 1 March and 31 August."""
    month = date.month
    return 3 <= month <= 8


def load_all_data() -> pd.DataFrame:
    """Load all yearly merged weather + fire CSV files into one DataFrame."""
    dfs = []

    for year in YEARS:
        file = BASE_DIR / f"daily_weather_fire_{year}.csv"
        if not file.exists():
            print(f"⚠️ Missing {file}")
            continue

        df = pd.read_csv(file)
        df["met_day"] = pd.to_datetime(df["met_day"], errors="coerce")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No input datasets found.")

    return pd.concat(dfs, ignore_index=True)


def compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma, threshold_percentile=70):
    """Compute R² for low and high FIRE COUNT regimes separately."""
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]
    
    log_x = np.log(x)
    log_y = np.log(y + 1.0)
    
    # Threshold based on FIRE COUNT, not VPD
    fire_threshold = np.percentile(y, threshold_percentile)
    y_pred = alpha + beta * log_x + gamma * (log_x ** 2)
    
    # Overall
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2_overall = 1.0 - ss_res / ss_tot
    
    # Low fire count days
    low_mask = y <= fire_threshold
    if np.sum(low_mask) > 2:
        ss_res_low = np.sum((log_y[low_mask] - y_pred[low_mask]) ** 2)
        ss_tot_low = np.sum((log_y[low_mask] - np.mean(log_y[low_mask])) ** 2)
        r2_low = 1.0 - ss_res_low / ss_tot_low if ss_tot_low > 0 else np.nan
    else:
        r2_low = np.nan
    
    # High fire count days (the dangerous ones!)
    high_mask = y > fire_threshold
    if np.sum(high_mask) > 2:
        ss_res_high = np.sum((log_y[high_mask] - y_pred[high_mask]) ** 2)
        ss_tot_high = np.sum((log_y[high_mask] - np.mean(log_y[high_mask])) ** 2)
        r2_high = 1.0 - ss_res_high / ss_tot_high if ss_tot_high > 0 else np.nan
    else:
        r2_high = np.nan
    
    return r2_overall, r2_low, r2_high, fire_threshold


def fit_quadratic_unweighted(vpd, fires):
    """Original unweighted quadratic fit."""
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]
    
    log_x = np.log(x)
    log_y = np.log(y + 1.0)
    
    gamma, beta, alpha = np.polyfit(log_x, log_y, 2)
    return alpha, beta, gamma


def fit_quadratic_fire_weighted(vpd, fires, power=1.0):
    """
    Weight by fire_count^power during fitting.
    Higher power = more emphasis on high-fire days.
    """
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]
    
    log_x = np.log(x)
    log_y = np.log(y + 1.0)
    
    # Weights proportional to (fire_count + 1)^power
    weights = (y + 1) ** power
    # Normalize
    weights = weights / np.mean(weights)
    
    gamma, beta, alpha = np.polyfit(log_x, log_y, 2, w=weights)
    return alpha, beta, gamma


def fit_quadratic_fire_oversampled(vpd, fires, threshold_percentile=70, oversample_factor=5):
    """
    Oversample high-FIRE days to give them more influence.
    """
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]
    
    fire_threshold = np.percentile(y, threshold_percentile)
    
    log_x = np.log(x)
    log_y = np.log(y + 1.0)
    
    # Separate low and high fire days
    low_mask = y <= fire_threshold
    high_mask = y > fire_threshold
    
    log_x_low = log_x[low_mask]
    log_y_low = log_y[low_mask]
    log_x_high = log_x[high_mask]
    log_y_high = log_y[high_mask]
    
    # Oversample high-fire days
    log_x_high_over = np.tile(log_x_high, oversample_factor)
    log_y_high_over = np.tile(log_y_high, oversample_factor)
    
    # Combine
    log_x_combined = np.concatenate([log_x_low, log_x_high_over])
    log_y_combined = np.concatenate([log_y_low, log_y_high_over])
    
    gamma, beta, alpha = np.polyfit(log_x_combined, log_y_combined, 2)
    return alpha, beta, gamma


def fit_quadratic_combined_weight(vpd, fires, vpd_power=0.5, fire_power=1.0):
    """
    Weight by both VPD and fire count.
    """
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]
    
    log_x = np.log(x)
    log_y = np.log(y + 1.0)
    
    # Combined weight
    weights = ((x / np.median(x)) ** vpd_power) * ((y + 1) ** fire_power)
    weights = weights / np.mean(weights)
    
    gamma, beta, alpha = np.polyfit(log_x, log_y, 2, w=weights)
    return alpha, beta, gamma


def plot_comparison(df: pd.DataFrame):
    """Compare all fitting approaches."""
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )

    summer_df = df[df["season"] == "summer"]
    winter_df = df[df["season"] == "winter"]

    vpd = summer_df["VPD_mean"].values
    fires = summer_df["fire_count"].values.astype(float)

    print("\n" + "=" * 80)
    print("COMPARING FIRE-COUNT WEIGHTING STRATEGIES")
    print("(Goal: Improve prediction accuracy for HIGH-FIRE days)")
    print("=" * 80)
    
    approaches = {}
    
    # 1. Unweighted (original)
    alpha, beta, gamma = fit_quadratic_unweighted(vpd, fires)
    r2_all, r2_low, r2_high, fire_threshold = compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma)
    approaches['Unweighted'] = (alpha, beta, gamma, r2_all, r2_low, r2_high)
    print(f"\nFire count threshold (70th percentile): {fire_threshold:.0f} fires/day")
    print(f"\n1. UNWEIGHTED (original):")
    print(f"   R² overall={r2_all:.3f}, low-fire={r2_low:.3f}, HIGH-FIRE={r2_high:.3f}")
    
    # 2. Fire-weighted (power=0.5)
    alpha, beta, gamma = fit_quadratic_fire_weighted(vpd, fires, power=0.5)
    r2_all, r2_low, r2_high, _ = compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma)
    approaches['Fire weight p=0.5'] = (alpha, beta, gamma, r2_all, r2_low, r2_high)
    print(f"\n2. FIRE-WEIGHTED (power=0.5):")
    print(f"   R² overall={r2_all:.3f}, low-fire={r2_low:.3f}, HIGH-FIRE={r2_high:.3f}")
    
    # 3. Fire-weighted (power=1)
    alpha, beta, gamma = fit_quadratic_fire_weighted(vpd, fires, power=1.0)
    r2_all, r2_low, r2_high, _ = compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma)
    approaches['Fire weight p=1'] = (alpha, beta, gamma, r2_all, r2_low, r2_high)
    print(f"\n3. FIRE-WEIGHTED (power=1):")
    print(f"   R² overall={r2_all:.3f}, low-fire={r2_low:.3f}, HIGH-FIRE={r2_high:.3f}")
    
    # 4. Fire-weighted (power=2)
    alpha, beta, gamma = fit_quadratic_fire_weighted(vpd, fires, power=2.0)
    r2_all, r2_low, r2_high, _ = compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma)
    approaches['Fire weight p=2'] = (alpha, beta, gamma, r2_all, r2_low, r2_high)
    print(f"\n4. FIRE-WEIGHTED (power=2):")
    print(f"   R² overall={r2_all:.3f}, low-fire={r2_low:.3f}, HIGH-FIRE={r2_high:.3f}")
    
    # 5. Oversample high-fire days 3x
    alpha, beta, gamma = fit_quadratic_fire_oversampled(vpd, fires, oversample_factor=3)
    r2_all, r2_low, r2_high, _ = compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma)
    approaches['Oversample fires 3x'] = (alpha, beta, gamma, r2_all, r2_low, r2_high)
    print(f"\n5. OVERSAMPLE HIGH-FIRE DAYS 3x:")
    print(f"   R² overall={r2_all:.3f}, low-fire={r2_low:.3f}, HIGH-FIRE={r2_high:.3f}")
    
    # 6. Oversample high-fire days 5x
    alpha, beta, gamma = fit_quadratic_fire_oversampled(vpd, fires, oversample_factor=5)
    r2_all, r2_low, r2_high, _ = compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma)
    approaches['Oversample fires 5x'] = (alpha, beta, gamma, r2_all, r2_low, r2_high)
    print(f"\n6. OVERSAMPLE HIGH-FIRE DAYS 5x:")
    print(f"   R² overall={r2_all:.3f}, low-fire={r2_low:.3f}, HIGH-FIRE={r2_high:.3f}")
    
    # 7. Oversample high-fire days 10x
    alpha, beta, gamma = fit_quadratic_fire_oversampled(vpd, fires, oversample_factor=10)
    r2_all, r2_low, r2_high, _ = compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma)
    approaches['Oversample fires 10x'] = (alpha, beta, gamma, r2_all, r2_low, r2_high)
    print(f"\n7. OVERSAMPLE HIGH-FIRE DAYS 10x:")
    print(f"   R² overall={r2_all:.3f}, low-fire={r2_low:.3f}, HIGH-FIRE={r2_high:.3f}")
    
    # 8. Combined VPD + fire weight
    alpha, beta, gamma = fit_quadratic_combined_weight(vpd, fires, vpd_power=0.5, fire_power=1.0)
    r2_all, r2_low, r2_high, _ = compute_r2_by_fire_regime(vpd, fires, alpha, beta, gamma)
    approaches['Combined weight'] = (alpha, beta, gamma, r2_all, r2_low, r2_high)
    print(f"\n8. COMBINED WEIGHT (VPD^0.5 × fire^1):")
    print(f"   R² overall={r2_all:.3f}, low-fire={r2_low:.3f}, HIGH-FIRE={r2_high:.3f}")
    
    # Find best
    best_high_fire = max(approaches.items(), key=lambda x: x[1][5])
    best_overall = max(approaches.items(), key=lambda x: x[1][3])
    
    print(f"\n{'=' * 80}")
    print(f"🔥 BEST HIGH-FIRE R²: {best_high_fire[0]} ({best_high_fire[1][5]:.3f})")
    print(f"📊 BEST OVERALL R²: {best_overall[0]} ({best_overall[1][3]:.3f})")
    print("=" * 80)
    
    # === PLOT ===
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter points - color by fire count intensity
    ax.scatter(winter_df["VPD_mean"], winter_df["fire_count"],
               color="blue", alpha=0.50, s=20, label="Autumn/Winter")
    ax.scatter(summer_df["VPD_mean"], summer_df["fire_count"],
               color="red", alpha=0.60, s=20, label="Spring/Summer")
    
    # Highlight high-fire days
    high_fire_mask = fires > fire_threshold
    ax.scatter(vpd[high_fire_mask], fires[high_fire_mask],
               facecolors='none', edgecolors='darkred', s=60, linewidths=1.5,
               label=f"High-fire days (>{fire_threshold:.0f})")
    
    # Generate curve x-values
    vpd_pos = vpd[vpd > 0]
    vpd_range = np.linspace(np.nanmin(vpd_pos), np.nanmax(vpd_pos), 300)
    log_v = np.log(vpd_range)
    
    # Original curve (gray dashed)
    orig = approaches['Unweighted']
    fires_orig = np.exp(orig[0] + orig[1] * log_v + orig[2] * log_v**2) - 1
    fires_orig = np.clip(fires_orig, 0, None)
    ax.plot(vpd_range, fires_orig, 'gray', linestyle='--', linewidth=2,
            label=f"Unweighted (R²={orig[3]:.2f}, high-fire={orig[5]:.2f})")
    
    # Best high-fire curve (black solid)
    best = best_high_fire[1]
    fires_best = np.exp(best[0] + best[1] * log_v + best[2] * log_v**2) - 1
    fires_best = np.clip(fires_best, 0, None)
    ax.plot(vpd_range, fires_best, 'black', linewidth=2.5,
            label=f"{best_high_fire[0]} (R²={best[3]:.2f}, high-fire={best[5]:.2f})")
    
    # Horizontal line at fire threshold
    ax.axhline(y=fire_threshold, color="orange", linestyle=":", alpha=0.7, linewidth=1.5,
               label=f"High-fire threshold ({fire_threshold:.0f} fires)")
    
    ax.set_title("Daily Wildfires vs VPD (London, 2009–2024)\nOptimized for High-Fire Day Prediction", fontsize=14)
    ax.set_xlabel("VPD_mean (Pa)", fontsize=12)
    ax.set_ylabel("Daily Wildfire Count", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add equation box
    textstr = (f"Best fit for high-fire days ({best_high_fire[0]}):\n"
               f"ln(NoF+1) = {best[0]:.2f} + {best[1]:.2f}·ln(VPD) + {best[2]:.3f}·[ln(VPD)]²\n"
               f"R² overall = {best[3]:.2f}\n"
               f"R² high-fire days = {best[5]:.2f}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.55, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    out_path = BASE_DIR / "1_vpd_studies/fires_vs_vpd_fire_weighted.png"
    plt.savefig(out_path, dpi=400)
    print(f"\nSaved plot to: {out_path}")
    
    plt.show()


def main():
    print("Loading data...")
    df = load_all_data()
    print("Comparing fire-count weighting strategies...")
    plot_comparison(df)


if __name__ == "__main__":
    main()