from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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


def compute_r2_by_vpd_regime(
    vpd: np.ndarray,
    fires: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    p1: float = 33.3,
    p2: float = 66.7,
):
    """
    Compute R² overall and within three VPD regimes:
      - low VPD (0 to p1 percentile)
      - mid VPD (p1 to p2 percentile)
      - high VPD (p2 to 100 percentile)

    Uses log-space residuals, consistent with the fit.
    """
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    log_x = np.log(x)
    log_y = np.log(y + 1.0)

    # Predicted log fires
    y_pred = alpha + beta * log_x + gamma * (log_x ** 2)

    # Overall R²
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2_all = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Define VPD regime boundaries
    vpd_low_thr = np.percentile(x, p1)
    vpd_high_thr = np.percentile(x, p2)

    # Low VPD regime
    low_mask = x <= vpd_low_thr
    if np.sum(low_mask) > 2:
        ss_res_low = np.sum((log_y[low_mask] - y_pred[low_mask]) ** 2)
        ss_tot_low = np.sum((log_y[low_mask] - np.mean(log_y[low_mask])) ** 2)
        r2_low = 1.0 - ss_res_low / ss_tot_low if ss_tot_low > 0 else np.nan
    else:
        r2_low = np.nan

    # Mid VPD regime
    mid_mask = (x > vpd_low_thr) & (x <= vpd_high_thr)
    if np.sum(mid_mask) > 2:
        ss_res_mid = np.sum((log_y[mid_mask] - y_pred[mid_mask]) ** 2)
        ss_tot_mid = np.sum((log_y[mid_mask] - np.mean(log_y[mid_mask])) ** 2)
        r2_mid = 1.0 - ss_res_mid / ss_tot_mid if ss_tot_mid > 0 else np.nan
    else:
        r2_mid = np.nan

    # High VPD regime
    high_mask = x > vpd_high_thr
    if np.sum(high_mask) > 2:
        ss_res_high = np.sum((log_y[high_mask] - y_pred[high_mask]) ** 2)
        ss_tot_high = np.sum((log_y[high_mask] - np.mean(log_y[high_mask])) ** 2)
        r2_high = 1.0 - ss_res_high / ss_tot_high if ss_tot_high > 0 else np.nan
    else:
        r2_high = np.nan

    return (
        r2_all,
        r2_low,
        r2_mid,
        r2_high,
        vpd_low_thr,
        vpd_high_thr,
    )


def fit_quadratic_unweighted(vpd: np.ndarray, fires: np.ndarray):
    """Unweighted quadratic fit in log space."""
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    log_x = np.log(x)
    log_y = np.log(y + 1.0)

    gamma, beta, alpha = np.polyfit(log_x, log_y, 2)
    return alpha, beta, gamma


def fit_quadratic_vpd_weighted_logspace(
    vpd: np.ndarray,
    fires: np.ndarray,
    power: float = 1.0,
):
    """
    Quadratic fit with continuous weights that depend on log(VPD).

    1. Work in log(VPD) since the regression is in log space.
    2. Normalise log(VPD) to [0, 1].
    3. Raise to 'power' to emphasise the upper tail.
    """
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    log_x = np.log(x)
    log_y = np.log(y + 1.0)

    lx_min = np.min(log_x)
    lx_max = np.max(log_x)

    if lx_max == lx_min:
        gamma, beta, alpha = np.polyfit(log_x, log_y, 2)
        return alpha, beta, gamma

    # Normalise log(VPD) into [0, 1]
    lx_norm = (log_x - lx_min) / (lx_max - lx_min)

    eps = 1e-6
    weights = np.clip(lx_norm, eps, 1.0) ** power
    weights = weights / np.mean(weights)

    gamma, beta, alpha = np.polyfit(log_x, log_y, 2, w=weights)
    return alpha, beta, gamma


def fit_tail_powerlaw(vpd: np.ndarray, fires: np.ndarray, vpd_min_tail: float):
    """
    Fit a simple power-law model only on high-VPD days:

        ln(NoF + 1) = a + b * ln(VPD)

    using data where VPD >= vpd_min_tail.
    Returns (a, b, r2_tail, n_points).
    """
    mask = (vpd > 0) & ~np.isnan(fires) & (vpd >= vpd_min_tail)
    x = vpd[mask]
    y = fires[mask]

    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan, n

    log_x = np.log(x)
    log_y = np.log(y + 1.0)

    # Linear fit in log–log space
    b, a = np.polyfit(log_x, log_y, 1)

    y_pred = a + b * log_x
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2_tail = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return a, b, r2_tail, n


def tail_objective(stats, w_high: float = 0.7, w_all: float = 0.3) -> float:
    """
    Composite score prioritising high-VPD fit but still considering overall fit.

    stats = (alpha, beta, gamma, r2_all, r2_low, r2_mid, r2_high)
    """
    r2_all = stats[3]
    r2_high = stats[6]
    return w_high * r2_high + w_all * r2_all


def plot_comparison(df: pd.DataFrame):
    """Compare VPD based weighting approaches and plot the best tail-focused model."""
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )

    summer_df = df[df["season"] == "summer"]
    winter_df = df[df["season"] == "winter"]

    vpd = summer_df["VPD_mean"].values
    fires = summer_df["fire_count"].values.astype(float)

    print("\n" + "=" * 80)
    print("COMPARING VPD BASED WEIGHTING STRATEGIES")
    print("Fitting uses log(VPD) based continuous weights (0–1), no thresholds.")
    print("=" * 80)

    approaches: dict[str, tuple] = {}

    # 1. Unweighted fit
    alpha_u, beta_u, gamma_u = fit_quadratic_unweighted(vpd, fires)

    (
        r2_all_u,
        r2_low_u,
        r2_mid_u,
        r2_high_u,
        vpd_low_thr,
        vpd_high_thr,
    ) = compute_r2_by_vpd_regime(vpd, fires, alpha_u, beta_u, gamma_u)

    approaches["Unweighted"] = (
        alpha_u,
        beta_u,
        gamma_u,
        r2_all_u,
        r2_low_u,
        r2_mid_u,
        r2_high_u,
    )

    print("\n1. UNWEIGHTED:")
    print(
        f"   VPD regime boundaries: low ≤ {vpd_low_thr:.1f} Pa,"
        f" mid ≤ {vpd_high_thr:.1f} Pa, high > {vpd_high_thr:.1f} Pa"
    )
    print(
        f"   R² overall={r2_all_u:.3f}, low VPD={r2_low_u:.3f},"
        f" mid VPD={r2_mid_u:.3f}, high VPD={r2_high_u:.3f}"
    )

    # 2+. VPD-weighted fits with different powers in log-space
    power_values = [0.5, 1.0, 2.0, 4.0, 8.0]

    for p in power_values:
        alpha_p, beta_p, gamma_p = fit_quadratic_vpd_weighted_logspace(
            vpd, fires, power=p
        )
        r2_all, r2_low, r2_mid, r2_high, _, _ = compute_r2_by_vpd_regime(
            vpd, fires, alpha_p, beta_p, gamma_p
        )
        key = f"VPD weight p={p}"
        approaches[key] = (
            alpha_p,
            beta_p,
            gamma_p,
            r2_all,
            r2_low,
            r2_mid,
            r2_high,
        )
        print(f"\n{key}:")
        print(
            f"   R² overall={r2_all:.3f}, low VPD={r2_low:.3f},"
            f" mid VPD={r2_mid:.3f}, high VPD={r2_high:.3f}"
        )

    # Select champions
    best_high_vpd = max(approaches.items(), key=lambda x: x[1][6])
    best_overall = max(approaches.items(), key=lambda x: x[1][3])
    best_tail = max(approaches.items(), key=lambda x: tail_objective(x[1]))

    print("\n" + "=" * 80)
    print(
        f"Best high VPD R²: {best_high_vpd[0]} "
        f"(R²_high={best_high_vpd[1][6]:.3f})"
    )
    print(
        f"Best overall R²: {best_overall[0]} "
        f"(R²_all={best_overall[1][3]:.3f})"
    )
    print(
        f"Best tail objective (0.7·R²_high + 0.3·R²_all): {best_tail[0]} "
        f"(R²_high={best_tail[1][6]:.3f}, R²_all={best_tail[1][3]:.3f})"
    )

    # Tail-only power-law fit on high-VPD regime
    a_tail, b_tail, r2_tail, n_tail = fit_tail_powerlaw(vpd, fires, vpd_high_thr)
    if np.isnan(r2_tail):
        print("Tail-only power-law fit: not enough high-VPD points to fit.")
    else:
        print(
            f"Tail-only power-law (VPD ≥ {vpd_high_thr:.1f} Pa): "
            f"n={n_tail}, R²_highVPD={r2_tail:.3f}"
        )
    print("=" * 80)

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter points, colour by season
    ax.scatter(
        winter_df["VPD_mean"],
        winter_df["fire_count"],
        color="blue",
        alpha=0.50,
        s=20,
        label="Autumn and winter",
    )
    ax.scatter(
        summer_df["VPD_mean"],
        summer_df["fire_count"],
        color="red",
        alpha=0.60,
        s=20,
        label="Spring and summer",
    )

    # Get limits and shade VPD regimes (after scatter so y-lims are set)
    ymin, ymax = ax.get_ylim()
    vpd_min = np.nanmin(vpd[vpd > 0])
    vpd_max = np.nanmax(vpd[vpd > 0])

    ax.axvspan(vpd_min, vpd_low_thr, alpha=0.04)
    ax.axvspan(vpd_low_thr, vpd_high_thr, alpha=0.04)
    ax.axvspan(vpd_high_thr, vpd_max, alpha=0.04)

    # Vertical dashed lines for regime boundaries
    ax.axvline(
        x=vpd_low_thr,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="VPD regime boundaries",
    )
    ax.axvline(
        x=vpd_high_thr,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
    )

    # Generate curve x values
    vpd_pos = vpd[vpd > 0]
    vpd_range = np.linspace(np.nanmin(vpd_pos), np.nanmax(vpd_pos), 300)
    log_v = np.log(vpd_range)

    # Original curve (grey dashed)
    orig = approaches["Unweighted"]
    fires_orig = np.exp(
        orig[0] + orig[1] * log_v + orig[2] * (log_v ** 2)
    ) - 1.0
    fires_orig = np.clip(fires_orig, 0, None)
    ax.plot(
        vpd_range,
        fires_orig,
        "gray",
        linestyle="--",
        linewidth=2,
        label=f"Unweighted (R²_all={orig[3]:.2f}, R²_highVPD={orig[6]:.2f})",
    )

    # Tail-optimised quadratic curve (black solid)
    best = best_tail[1]
    fires_best = np.exp(
        best[0] + best[1] * log_v + best[2] * (log_v ** 2)
    ) - 1.0
    fires_best = np.clip(fires_best, 0, None)
    ax.plot(
        vpd_range,
        fires_best,
        "black",
        linewidth=2.5,
        label=(
            f"{best_tail[0]} "
            f"(R²_all={best[3]:.2f}, R²_highVPD={best[6]:.2f})"
        ),
    )

    # Tail-only power-law curve (drawn only for high VPD)
    if not np.isnan(a_tail):
        vpd_tail_range = np.linspace(vpd_high_thr, vpd_max, 150)
        log_v_tail = np.log(vpd_tail_range)
        fires_tail_curve = np.exp(a_tail + b_tail * log_v_tail) - 1.0
        fires_tail_curve = np.clip(fires_tail_curve, 0, None)
        ax.plot(
            vpd_tail_range,
            fires_tail_curve,
            linestyle="-.",
            linewidth=2,
            label=f"High-VPD tail-only power law (R²_highVPD={r2_tail:.2f})",
        )

    ax.set_title(
        "Daily wildfires vs VPD (London, 2009–2024)\n"
        "Quadratic fits with log(VPD) weighting and high-VPD tail power law",
        fontsize=14,
    )
    ax.set_xlabel("VPD_mean (Pa)", fontsize=12)
    ax.set_ylabel("Daily wildfire count", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Equation box for tail-optimised quadratic model
    textstr = (
        f"Tail-optimised quadratic: {best_tail[0]}\n"
        f"ln(NoF + 1) = {best[0]:.2f} + {best[1]:.2f} · ln(VPD)"
        f" + {best[2]:.3f} · [ln(VPD)]²\n"
        f"R² overall = {best[3]:.2f}\n"
        f"R² low VPD = {best[4]:.2f}\n"
        f"R² mid VPD = {best[5]:.2f}\n"
        f"R² high VPD = {best[6]:.2f}\n"
        f"VPD regimes: ≤ {vpd_low_thr:.1f} Pa, "
        f"{vpd_low_thr:.1f}–{vpd_high_thr:.1f} Pa, "
        f"> {vpd_high_thr:.1f} Pa"
    )
    if not np.isnan(a_tail):
        textstr += (
            f"\n\nHigh-VPD tail power law (VPD ≥ {vpd_high_thr:.1f} Pa):\n"
            f"ln(NoF + 1) = {a_tail:.2f} + {b_tail:.2f} · ln(VPD)\n"
            f"R² high-VPD = {r2_tail:.2f}, n = {n_tail}"
        )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.98,
        0.55,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    plt.tight_layout()

    out_path = BASE_DIR / "1_vpd_studies/fires_vs_vpd_vpd_weighted.png"
    plt.savefig(out_path, dpi=400)
    print(f"\nSaved plot to: {out_path}")

    plt.show()


def main():
    print("Loading data...")
    df = load_all_data()
    print("Comparing VPD based weighting strategies...")
    plot_comparison(df)


if __name__ == "__main__":
    main()
