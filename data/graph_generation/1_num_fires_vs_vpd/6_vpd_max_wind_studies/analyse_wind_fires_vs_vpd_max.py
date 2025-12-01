"""
Stratified VPD vs Fire Analysis by Mean Daily Wind Speed (in m/s)

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

# Physical wind-speed bins in m/s (converted from knots)
# Conversion: 1 knot = 0.514444 m/s
WIND_BINS = [0, 2, 3, 4, np.inf]   # in m/s
WIND_LABELS = [
    "0–2 m/s",
    "2–3 m/s",
    "3–4 m/s",
    ">4 m/s",
]
WIND_COLORS = {
    "0–2 m/s": "#1f77b4",
    "2–3 m/s": "#2ca02c",
    "3–4 m/s": "#ff7f0e",
    ">4 m/s": "#d62728",
}

VPD_COL = "VPD_max"
WIND_COL = "wind_speed_mean"  # from CSV, in knots
REF_VPD = 1000.0  # Pa reference line
# =======================


def load_weather_fire_data() -> pd.DataFrame:
    """Load and concatenate daily weather–fire CSVs, convert wind knots → m/s."""
    dfs = []
    for year in YEARS:
        file = BASE_DIR / f"daily_weather_fire_{year}.csv"
        if file.exists():
            df = pd.read_csv(file)
            df["met_day"] = pd.to_datetime(df["met_day"])

            # Convert wind from knots → m/s
            if WIND_COL in df.columns:
                df["wind_speed_mps"] = df[WIND_COL] * 0.514444
            else:
                raise KeyError(f"{WIND_COL} not found in {file}")

            dfs.append(df)
        else:
            print(f"Warning: file not found: {file}")

    if not dfs:
        raise RuntimeError("No weather-fire datasets found in BASE_DIR.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(dfs)} yearly files.")
    return df


def fit_power_law(vpd: np.ndarray, fires: np.ndarray):
    """Fit fires = a * VPD^b. Returns (a, b, r2)."""
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
                         wind_col_mps: str = "wind_speed_mps"):
    """Create 2x2 stratified VPD vs fires plot using wind bins in m/s."""

    df_valid = df[
        (df[vpd_col] > 0) &
        df["fire_count"].notna() &
        df[wind_col_mps].notna()
    ].copy()

    print(f"Valid rows for wind stratification: {len(df_valid)}")

    # Assign wind categories
    df_valid["wind_category"] = pd.cut(
        df_valid[wind_col_mps],
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

        # Fit power law
        a, b, r2 = fit_power_law(df_cat[vpd_col].values,
                                 df_cat["fire_count"].values)

        if not np.isnan(a):
            vpd_min = max(df_cat[vpd_col].min(), 1e-6)
            vpd_max = df_cat[vpd_col].max()

            if vpd_max > vpd_min:
                vpd_curve = np.linspace(vpd_min, vpd_max, 200)
                ax.plot(vpd_curve,
                        a * vpd_curve ** b,
                        "k-",
                        lw=2)

                # Reference VPD annotation
                if vpd_min <= REF_VPD <= vpd_max:
                    ref_fire = a * REF_VPD ** b

                    ax.axvline(REF_VPD, color="0.4",
                               linestyle="--", lw=1)

                    ax.scatter([REF_VPD], [ref_fire],
                               s=45,
                               color=WIND_COLORS[cat],
                               edgecolors="k",
                               zorder=5)

                    ax.annotate(
                        f"{int(REF_VPD)} Pa\n{ref_fire:.1f} fires/day",
                        xy=(REF_VPD, ref_fire),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                        ha="left",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white",
                                  ec="0.7",
                                  alpha=0.75),
                    )

        # Title
        n = len(df_cat)
        mean_f = df_cat["fire_count"].mean()

        title = rf"$\bf{{{cat}}}$" + f"\nn={n}, mean={mean_f:.1f}"
        if not np.isnan(r2):
            title += f", b={b:.2f}, R²={r2:.2f}"

        ax.set_title(title, fontsize=12)

        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 6000)
        ax.set_ylim(0, 120)

    # Shared labels & suptitle
    fig.supxlabel(f"{vpd_col} (Pa)", fontsize=12)
    fig.supylabel("Daily fire count", fontsize=12)
    fig.suptitle(
        "VPD vs Fires Stratified by Mean Daily Wind Speed (m/s)",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])

    out_dir = BASE_DIR / "6_vpd_max_wind_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fires_vs_{vpd_col}_by_wind_category.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

    plt.show()
    return fig


def main():
    print("Loading weather–fire data...")
    df = load_weather_fire_data()

    print("Generating wind-stratified VPD–fire plot (m/s)...")
    plot_stratified_wind(df, VPD_COL, "wind_speed_mps")

    print("Done!")


if __name__ == "__main__":
    main()
