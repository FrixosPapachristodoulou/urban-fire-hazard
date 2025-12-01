"""
Stratified VPD vs Fire Analysis by Mean Daily Wind Speed

Creates a 2x2 plot showing VPD vs fires for four physical wind-speed
regimes, letting the data reveal how wind modulates the VPD–fire relationship.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/1_num_fires_vs_vpd")
YEARS = list(range(2009, 2025))

# Physical wind-speed bins in m/s (daily mean)
WIND_BINS = [0, 3, 5, 7, np.inf]
WIND_LABELS = [
    "0–3 m/s",
    "3–5 m/s",
    "5–7 m/s",
    ">7 m/s",
]
WIND_COLORS = {
    "0–3 m/s": "#1f77b4",    # blue
    "3–5 m/s": "#2ca02c",  # green
    "5–7 m/s": "#ff7f0e",   # orange
    ">7 m/s": "#d62728",    # red
}

VPD_COL = "VPD_mean"
WIND_COL = "wind_speed_mean"
REF_VPD = 500.0  # Pa, reference VPD level to highlight
# =======================


def load_weather_fire_data() -> pd.DataFrame:
    """Load and concatenate daily weather–fire CSVs."""
    dfs = []
    for year in YEARS:
        file = BASE_DIR / f"daily_weather_fire_{year}.csv"
        if file.exists():
            df = pd.read_csv(file)
            df["met_day"] = pd.to_datetime(df["met_day"])
            dfs.append(df)
        else:
            print(f"Warning: file not found: {file}")

    if not dfs:
        raise RuntimeError("No weather-fire datasets found in BASE_DIR.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(dfs)} yearly files.")
    return df


def fit_power_law(vpd: np.ndarray, fires: np.ndarray):
    """
    Fit fires = a * VPD^b using log-log regression on (VPD, fires+1).
    Returns (a, b, r2). If insufficient data, returns (nan, nan, nan).
    """
    mask = (vpd > 0) & ~np.isnan(fires)
    x, y = vpd[mask], fires[mask]

    if len(x) < 10:
        return np.nan, np.nan, np.nan

    log_x, log_y = np.log(x), np.log(y + 1.0)
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)

    y_pred = log_a + b * log_x
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return a, b, r2


def plot_stratified_wind(df: pd.DataFrame,
                         vpd_col: str = VPD_COL,
                         wind_col: str = WIND_COL):
    """
    Create 2x2 stratified VPD vs fires plot using physically meaningful
    daily mean wind-speed bins.
    """

    # Filter to valid rows
    df_valid = df[
        (df[vpd_col] > 0) &
        df["fire_count"].notna() &
        df[wind_col].notna()
    ].copy()

    print(f"Valid rows for wind stratification: {len(df_valid)}")

    # Assign wind categories
    df_valid["wind_category"] = pd.cut(
        df_valid[wind_col],
        bins=WIND_BINS,
        labels=WIND_LABELS,
        include_lowest=True
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, cat in zip(axes, WIND_LABELS):
        df_cat = df_valid[df_valid["wind_category"] == cat]

        # Scatter points
        ax.scatter(
            df_cat[vpd_col],
            df_cat["fire_count"],
            s=12,
            alpha=0.5,
            color=WIND_COLORS[cat],
            edgecolors="none",
        )

        # Fit and plot power-law curve
        a, b, r2 = fit_power_law(
            df_cat[vpd_col].values,
            df_cat["fire_count"].values
        )

        if not np.isnan(a):
            vpd_min = max(df_cat[vpd_col].min(), 1e-6)
            vpd_max = df_cat[vpd_col].max()
            if vpd_max > vpd_min:
                vpd_curve = np.linspace(vpd_min, vpd_max, 200)
                ax.plot(
                    vpd_curve,
                    a * vpd_curve ** b,
                    "k-",
                    lw=2,
                )

                # Highlight reference VPD point if in range
                if vpd_min <= REF_VPD <= vpd_max:
                    ref_fire = a * REF_VPD ** b

                    ax.axvline(
                        REF_VPD,
                        color="0.4",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.8,
                    )

                    ax.scatter(
                        [REF_VPD],
                        [ref_fire],
                        s=45,
                        color=WIND_COLORS[cat],
                        edgecolors="k",
                        zorder=5,
                    )

                    ax.annotate(
                        f"{int(REF_VPD)} Pa\n{ref_fire:.1f} fires/day",
                        xy=(REF_VPD, ref_fire),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                        ha="left",
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="white",
                            ec="0.7",
                            alpha=0.75,
                        ),
                    )

        # Panel title
        n = len(df_cat)
        mean_f = df_cat["fire_count"].mean()

        title = rf"$\bf{{{cat}}}$" + f"\nn={n}, mean={mean_f:.1f}"
        if not np.isnan(r2):
            title += f", b={b:.2f}, R²={r2:.2f}"

        ax.set_title(title, fontsize=12)

        # Styling
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 3050)
        ax.set_ylim(0, 120)

    # Shared labels & suptitle
    fig.supxlabel(f"{vpd_col} (Pa)", fontsize=12)
    fig.supylabel("Daily Mean Wind (m/s)", fontsize=12)
    fig.suptitle(
        "VPD vs Fires Stratified by Mean Daily Wind Speed",
        fontsize=13,
        fontweight="bold",
        y=0.92,
    )

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])

    out_dir = BASE_DIR / "5_vpd_wind_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fires_vs_{vpd_col}_by_wind_category.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

    plt.show()

    return fig


def main():
    print("Loading weather–fire data...")
    df = load_weather_fire_data()

    print("Generating wind-stratified VPD–fire plot...")
    plot_stratified_wind(df, VPD_COL, WIND_COL)
    print("Done!")


if __name__ == "__main__":
    main()
