from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.lines as mlines  # for custom legend entries

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/3_num_fires_vs_vpd_different_averaging")
YEARS = list(range(2009, 2025))  # 2009–2024 inclusive
# =======================


def get_season(date: datetime) -> str:
    """Return meteorological season for a datetime."""
    m = date.month
    if m in (12, 1, 2):
        return "winter"
    elif m in (3, 4, 5):
        return "spring"
    elif m in (6, 7, 8):
        return "summer"
    else:  # 9,10,11
        return "autumn"


SEASON_COLORS = {
    "winter": "#1f77b4",  # blue
    "spring": "#2ca02c",  # green
    "summer": "#ff7f0e",  # orange
    "autumn": "#d62728",  # red
}


def load_all_data() -> pd.DataFrame:
    """Load all yearly merged weather + fire CSV files into one DataFrame."""
    dfs = []

    for year in YEARS:
        file = BASE_DIR / f"merged_weather_and_fire_counts_differently/daily_weather_fire_{year}_differently.csv"
        if not file.exists():
            print(f"⚠️ Missing {file}")
            continue

        df = pd.read_csv(file)
        df["met_day"] = pd.to_datetime(df["met_day"], errors="coerce")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No input datasets found.")

    df_all = pd.concat(dfs, ignore_index=True)

    # add season column
    df_all["season"] = df_all["met_day"].apply(get_season)

    return df_all


def fit_power_law(vpd: np.ndarray, fires: np.ndarray):
    """
    Fit a power law:

        NoF + 1 = a * VPD^b

    by doing linear regression in log–log space:

        ln(NoF + 1) = ln(a) + b * ln(VPD)

    Returns a, b, R2 (computed in log space).
    """
    vpd = np.asarray(vpd, dtype=float)
    fires = np.asarray(fires, dtype=float)

    # need positive VPD for the log; keep ALL fire counts (including zeros)
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    if len(x) < 2:
        raise RuntimeError("Not enough data points for power law fit.")

    log_x = np.log(x)
    log_y = np.log(y + 1.0)   # include zero days

    # ordinary least squares in log space
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)

    # compute R^2 in log space
    y_pred_log = log_a + b * log_x
    ss_res = np.sum((log_y - y_pred_log) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return a, b, r2


def plot_seasonal_power_laws(df: pd.DataFrame):
    vpd_all = df["VPD_mean"].values.astype(float)
    fires_all = df["fire_count"].values.astype(float)
    seasons = df["season"].values

    # define the four fitting scenarios
    scenarios = [
        ("All seasons", np.ones(len(df), dtype=bool)),
        ("Spring + Summer", df["season"].isin(["spring", "summer"]).values),
        ("Summer + Autumn", df["season"].isin(["summer", "autumn"]).values),
        ("Summer only", (df["season"] == "summer").values),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 12), sharex=True, sharey=True
    )
    axes = axes.ravel()
    
    # Set axis limits
    for ax in axes:
        ax.set_xlim(0, 3050)
        ax.set_ylim(0, 120)

    for ax, (title, subset_mask) in zip(axes, scenarios):
        # which points actually participate in the fit?
        fit_mask = subset_mask & (vpd_all > 0) & ~np.isnan(fires_all)

        # fit power law on the participating subset
        vpd_fit_data = vpd_all[fit_mask]
        fires_fit_data = fires_all[fit_mask]
        a_pl, b_pl, r2 = fit_power_law(vpd_fit_data, fires_fit_data)

        # X-range for curve in this subplot (based on used points)
        vpd_min = np.nanmin(vpd_fit_data)
        vpd_max = np.nanmax(vpd_fit_data)
        vpd_curve = np.linspace(vpd_min, vpd_max, 300)
        fires_curve = a_pl * vpd_curve ** b_pl - 1.0
        fires_curve = np.clip(fires_curve, 0, None)

        # ---- scatter points by season with different alpha depending on fit participation ----
        for season_name, color in SEASON_COLORS.items():
            season_mask = (seasons == season_name)

            used = fit_mask & season_mask
            unused = (~fit_mask) & season_mask

            # low-alpha for points not used in this fit
            ax.scatter(
                vpd_all[unused],
                fires_all[unused],
                s=10,
                alpha=0.15,
                color=color,
                edgecolors="none",
            )

            # high-alpha for points used in this fit
            ax.scatter(
                vpd_all[used],
                fires_all[used],
                s=10,
                alpha=0.95,
                color=color,
                edgecolors="none",
            )

        # plot power-law curve
        ax.plot(
            vpd_curve,
            fires_curve,
            color="black",
            linewidth=2,
        )

        ax.set_title(
            f"$\\mathbf{{{title.replace(' ', '~')}}}$\n"
            f"NoF + 1 = {a_pl:.2g} · VPD$^{{{b_pl:.2f}}}$\n"
            f"$\\mathbf{{R^2 = {r2:.2f}}}$",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)

    # shared labels
    fig.supxlabel("VPD_mean (Pa)", fontsize=13)
    fig.supylabel("Daily wildfire count", fontsize=13, fontweight="bold")
    # ----- global legend for seasons + curve -----
    season_handles = [
        mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                      label=season_name.capitalize())
        for season_name, color in SEASON_COLORS.items()
    ]
    curve_handle = mlines.Line2D([], [], color="black", label="Power-law fit")

    fig.legend(
        handles=season_handles + [curve_handle],
        loc="upper center",
        ncol=5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.97),
    )

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])

    out_path = BASE_DIR / "1_vpd_studies_differently/fires_vs_vpd_powerlaw.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=400)
    print(f"Saved seasonal power-law figure to: {out_path}")

    plt.show()


def main():
    print("Loading merged weather and fire datasets...")
    df = load_all_data()

    print("Plotting seasonal power-law fits...")
    plot_seasonal_power_laws(df)


if __name__ == "__main__":
    main()
