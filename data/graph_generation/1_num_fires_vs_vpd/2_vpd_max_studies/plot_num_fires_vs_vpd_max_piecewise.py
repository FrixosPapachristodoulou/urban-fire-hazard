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


def fit_piecewise_linear_loglog(vpd: np.ndarray, fires: np.ndarray, threshold_vpd: float = None):
    """
    Piecewise LINEAR fit in log-log space with separate slopes for low and high VPD.

    This allows the high-VPD regime to have its own optimized slope.
    """
    vpd = np.asarray(vpd, dtype=float)
    fires = np.asarray(fires, dtype=float)

    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    log_x = np.log(x)
    log_y = np.log(y + 1.0)

    # Auto-detect threshold at 70th percentile if not provided
    if threshold_vpd is None:
        threshold_vpd = np.percentile(x, 70)

    log_threshold = np.log(threshold_vpd)

    # LOW VPD regime: simple linear in log-log
    low_mask = x <= threshold_vpd
    log_x_low = log_x[low_mask]
    log_y_low = log_y[low_mask]

    beta_low, alpha_low = np.polyfit(log_x_low, log_y_low, 1)

    # Compute R² for low regime
    y_pred_low = alpha_low + beta_low * log_x_low
    ss_res_low = np.sum((log_y_low - y_pred_low) ** 2)
    ss_tot_low = np.sum((log_y_low - np.mean(log_y_low)) ** 2)
    r2_low = 1.0 - ss_res_low / ss_tot_low if ss_tot_low > 0 else np.nan

    # HIGH VPD regime: separate linear fit
    high_mask = x > threshold_vpd
    log_x_high = log_x[high_mask]
    log_y_high = log_y[high_mask]

    beta_high, alpha_high = np.polyfit(log_x_high, log_y_high, 1)

    # Compute R² for high regime
    y_pred_high = alpha_high + beta_high * log_x_high
    ss_res_high = np.sum((log_y_high - y_pred_high) ** 2)
    ss_tot_high = np.sum((log_y_high - np.mean(log_y_high)) ** 2)
    r2_high = 1.0 - ss_res_high / ss_tot_high if ss_tot_high > 0 else np.nan

    # Overall R² using piecewise predictions
    y_pred_all = np.where(
        x <= threshold_vpd,
        alpha_low + beta_low * log_x,
        alpha_high + beta_high * log_x
    )
    ss_res_all = np.sum((log_y - y_pred_all) ** 2)
    ss_tot_all = np.sum((log_y - np.mean(log_y)) ** 2)
    r2_overall = 1.0 - ss_res_all / ss_tot_all

    return {
        'threshold_vpd': threshold_vpd,
        'low': {'alpha': alpha_low, 'beta': beta_low, 'r2': r2_low},
        'high': {'alpha': alpha_high, 'beta': beta_high, 'r2': r2_high},
        'r2_overall': r2_overall
    }


def fit_piecewise_continuous(vpd: np.ndarray, fires: np.ndarray, threshold_vpd: float = None):
    """
    Piecewise linear fit in log-log space that is CONTINUOUS at the threshold.
    Uses constrained optimization to ensure the two segments meet.
    """
    vpd = np.asarray(vpd, dtype=float)
    fires = np.asarray(fires, dtype=float)

    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    log_x = np.log(x)
    log_y = np.log(y + 1.0)

    if threshold_vpd is None:
        threshold_vpd = np.percentile(x, 70)

    log_threshold = np.log(threshold_vpd)

    # Fit low regime first
    low_mask = x <= threshold_vpd
    log_x_low = log_x[low_mask]
    log_y_low = log_y[low_mask]
    beta_low, alpha_low = np.polyfit(log_x_low, log_y_low, 1)

    # Value at threshold (continuity point)
    y_at_threshold = alpha_low + beta_low * log_threshold

    # Fit high regime with constraint: passes through (log_threshold, y_at_threshold)
    high_mask = x > threshold_vpd
    log_x_high = log_x[high_mask]
    log_y_high = log_y[high_mask]

    # For line passing through (log_threshold, y_at_threshold):
    # log_y = y_at_threshold + beta_high * (log_x - log_threshold)
    # Solve for beta_high using least squares
    dx = log_x_high - log_threshold
    dy = log_y_high - y_at_threshold
    beta_high = np.sum(dx * dy) / np.sum(dx ** 2)
    alpha_high = y_at_threshold - beta_high * log_threshold

    # Compute R² values
    y_pred_low = alpha_low + beta_low * log_x_low
    ss_res_low = np.sum((log_y_low - y_pred_low) ** 2)
    ss_tot_low = np.sum((log_y_low - np.mean(log_y_low)) ** 2)
    r2_low = 1.0 - ss_res_low / ss_tot_low if ss_tot_low > 0 else np.nan

    y_pred_high = alpha_high + beta_high * log_x_high
    ss_res_high = np.sum((log_y_high - y_pred_high) ** 2)
    ss_tot_high = np.sum((log_y_high - np.mean(log_y_high)) ** 2)
    r2_high = 1.0 - ss_res_high / ss_tot_high if ss_tot_high > 0 else np.nan

    # Overall R²
    y_pred_all = np.where(
        x <= threshold_vpd,
        alpha_low + beta_low * log_x,
        alpha_high + beta_high * log_x
    )
    ss_res_all = np.sum((log_y - y_pred_all) ** 2)
    ss_tot_all = np.sum((log_y - np.mean(log_y)) ** 2)
    r2_overall = 1.0 - ss_res_all / ss_tot_all

    return {
        'threshold_vpd': threshold_vpd,
        'low': {'alpha': alpha_low, 'beta': beta_low, 'r2': r2_low},
        'high': {'alpha': alpha_high, 'beta': beta_high, 'r2': r2_high},
        'r2_overall': r2_overall
    }


def fit_single_quadratic(vpd: np.ndarray, fires: np.ndarray):
    """Original quadratic fit for comparison."""
    vpd = np.asarray(vpd, dtype=float)
    fires = np.asarray(fires, dtype=float)

    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    log_x = np.log(x)
    log_y = np.log(y + 1.0)

    gamma, beta, alpha = np.polyfit(log_x, log_y, 2)

    y_pred_log = alpha + beta * log_x + gamma * (log_x ** 2)
    ss_res = np.sum((log_y - y_pred_log) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    # Also compute regime-specific R² (by VPD_max percentile)
    threshold = np.percentile(x, 70)

    low_mask = x <= threshold
    high_mask = x > threshold

    y_pred_low = alpha + beta * log_x[low_mask] + gamma * (log_x[low_mask] ** 2)
    ss_res_low = np.sum((log_y[low_mask] - y_pred_low) ** 2)
    ss_tot_low = np.sum((log_y[low_mask] - np.mean(log_y[low_mask])) ** 2)
    r2_low = 1.0 - ss_res_low / ss_tot_low if ss_tot_low > 0 else np.nan

    y_pred_high = alpha + beta * log_x[high_mask] + gamma * (log_x[high_mask] ** 2)
    ss_res_high = np.sum((log_y[high_mask] - y_pred_high) ** 2)
    ss_tot_high = np.sum((log_y[high_mask] - np.mean(log_y[high_mask])) ** 2)
    r2_high = 1.0 - ss_res_high / ss_tot_high if ss_tot_high > 0 else np.nan

    return {
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'r2': r2, 'r2_low': r2_low, 'r2_high': r2_high,
        'threshold': threshold
    }


def plot_fires_vs_vpd(df: pd.DataFrame):
    # Determine colours by season
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )

    summer_df = df[df["season"] == "summer"]
    winter_df = df[df["season"] == "winter"]

    # *** USE VPD_max HERE ***
    vpd_summer = summer_df["VPD_max"].values
    fires_summer = summer_df["fire_count"].values.astype(float)

    # === Compare fitting approaches ===
    print("\n" + "=" * 70)
    print("COMPARING FITTING APPROACHES (VPD_max)")
    print("=" * 70)

    # 1. Original quadratic (on VPD_max)
    quad = fit_single_quadratic(vpd_summer, fires_summer)
    print(f"\n1. QUADRATIC (original, VPD_max):")
    print(f"   Overall R² = {quad['r2']:.3f}")
    print(f"   Low VPD_max R² = {quad['r2_low']:.3f}")
    print(f"   High VPD_max R² = {quad['r2_high']:.3f}")

    # 2. Piecewise (unconstrained - each segment independent)
    pw_unc = fit_piecewise_linear_loglog(vpd_summer, fires_summer)
    print(f"\n2. PIECEWISE UNCONSTRAINED (separate fits, VPD_max):")
    print(f"   Threshold VPD_max = {pw_unc['threshold_vpd']:.0f} Pa")
    print(f"   Overall R² = {pw_unc['r2_overall']:.3f}")
    print(f"   Low VPD_max: ln(NoF+1) = {pw_unc['low']['alpha']:.2f} + {pw_unc['low']['beta']:.2f}·ln(VPD_max), R² = {pw_unc['low']['r2']:.3f}")
    print(f"   High VPD_max: ln(NoF+1) = {pw_unc['high']['alpha']:.2f} + {pw_unc['high']['beta']:.2f}·ln(VPD_max), R² = {pw_unc['high']['r2']:.3f}")

    # 3. Piecewise continuous
    pw_con = fit_piecewise_continuous(vpd_summer, fires_summer)
    print(f"\n3. PIECEWISE CONTINUOUS (joined at threshold, VPD_max):")
    print(f"   Threshold VPD_max = {pw_con['threshold_vpd']:.0f} Pa")
    print(f"   Overall R² = {pw_con['r2_overall']:.3f}")
    print(f"   Low VPD_max: R² = {pw_con['low']['r2']:.3f}")
    print(f"   High VPD_max: R² = {pw_con['high']['r2']:.3f}")

    # Determine best approach
    approaches = [
        ("Quadratic", quad['r2'], quad['r2_high']),
        ("Piecewise Unconstrained", pw_unc['r2_overall'], pw_unc['high']['r2']),
        ("Piecewise Continuous", pw_con['r2_overall'], pw_con['high']['r2']),
    ]

    best_overall = max(approaches, key=lambda x: x[1])
    best_high = max(approaches, key=lambda x: x[2])

    print(f"\n{'=' * 70}")
    print(f"BEST OVERALL R² (VPD_max): {best_overall[0]} ({best_overall[1]:.3f})")
    print(f"BEST HIGH-VPD_max R²: {best_high[0]} ({best_high[2]:.3f})")
    print("=" * 70)

    # === PLOT with piecewise unconstrained as main fit ===
    vpd_pos = vpd_summer[vpd_summer > 0]
    vpd_min = np.nanmin(vpd_pos)
    vpd_max = np.nanmax(vpd_pos)
    threshold = pw_unc['threshold_vpd']

    # Generate curves
    vpd_low_range = np.linspace(vpd_min, threshold, 150)
    vpd_high_range = np.linspace(threshold, vpd_max, 150)

    # Low VPD_max curve
    log_v_low = np.log(vpd_low_range)
    fires_low = np.exp(pw_unc['low']['alpha'] + pw_unc['low']['beta'] * log_v_low) - 1
    fires_low = np.clip(fires_low, 0, None)

    # High VPD_max curve
    log_v_high = np.log(vpd_high_range)
    fires_high = np.exp(pw_unc['high']['alpha'] + pw_unc['high']['beta'] * log_v_high) - 1
    fires_high = np.clip(fires_high, 0, None)

    # Also plot quadratic for comparison
    vpd_quad = np.linspace(vpd_min, vpd_max, 300)
    log_v_quad = np.log(vpd_quad)
    fires_quad = np.exp(quad['alpha'] + quad['beta'] * log_v_quad + quad['gamma'] * log_v_quad**2) - 1
    fires_quad = np.clip(fires_quad, 0, None)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 8))

    # Winter = blue (VPD_max)
    ax.scatter(
        winter_df["VPD_max"],
        winter_df["fire_count"],
        color="blue",
        alpha=0.50,
        s=20,
        label="Autumn/Winter (Sep–Feb)",
    )

    # Summer = red (VPD_max)
    ax.scatter(
        summer_df["VPD_max"],
        summer_df["fire_count"],
        color="red",
        alpha=0.60,
        s=20,
        label="Spring/Summer (Mar–Aug)",
    )

    # Quadratic fit (gray, dashed) for comparison
    ax.plot(
        vpd_quad,
        fires_quad,
        color="gray",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label=f"Quadratic (R²={quad['r2']:.2f})",
    )

    # Piecewise fit (black, solid)
    ax.plot(vpd_low_range, fires_low, color="black", linewidth=2.5)
    ax.plot(vpd_high_range, fires_high, color="black", linewidth=2.5,
            label=f"Piecewise (R²={pw_unc['r2_overall']:.2f})")

    # Mark the threshold
    ax.axvline(x=threshold, color="green", linestyle=":", alpha=0.7, linewidth=1.5)

    # Titles and labels
    ax.set_title("Daily Wildfires vs VPD_max (London, 2009–2024)", fontsize=14)
    ax.set_xlabel("VPD_max (Pa)", fontsize=12)
    ax.set_ylabel("Daily Wildfire Count", fontsize=12)

    # Create detailed legend text
    low_eq = (f"Low VPD_max (≤{threshold:.0f} Pa):\n"
              f"  ln(NoF+1) = {pw_unc['low']['alpha']:.2f} + {pw_unc['low']['beta']:.2f}·ln(VPD_max)\n"
              f"  R² = {pw_unc['low']['r2']:.2f}")

    high_eq = (f"High VPD_max (>{threshold:.0f} Pa):\n"
               f"  ln(NoF+1) = {pw_unc['high']['alpha']:.2f} + {pw_unc['high']['beta']:.2f}·ln(VPD_max)\n"
               f"  R² = {pw_unc['high']['r2']:.2f}")

    textstr = f"{low_eq}\n\n{high_eq}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            family='monospace')

    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    out_path = BASE_DIR / "2_vpd_max_studies/fires_vs_vpd_max_piecewise.png"
    plt.savefig(out_path, dpi=400)
    print(f"\nSaved plot to: {out_path}")

    plt.show()


def main():
    print("Loading merged weather and fire datasets...")
    df = load_all_data()

    print("Fitting and comparing models (VPD_max)...")
    plot_fires_vs_vpd(df)


if __name__ == "__main__":
    main()
