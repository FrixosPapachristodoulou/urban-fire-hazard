from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.lines as mlines   # for custom legend entries

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
    log_y = np.log(y + 1.0)   # <-- key change: include zero days

    # ordinary least squares in log space
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)

    # compute R^2 in log space
    y_pred_log = log_a + b * log_x
    ss_res = np.sum((log_y - y_pred_log) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return a, b, r2



def plot_fires_vs_vpd(df: pd.DataFrame):
    # determine colours by season
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )

    summer_df = df[df["season"] == "summer"]
    winter_df = df[df["season"] == "winter"]

    vpd_all = df["VPD_max"].values
    fires_all = df["fire_count"].values.astype(float)

    # ---- Fit power law: NoF + 1 = a * VPD^b ----
    a_pl, b_pl, r2 = fit_power_law(vpd_all, fires_all)
    print(
        f"Power law fit (NoF+1): NoF + 1 = {a_pl:.3g} * VPD^{b_pl:.3f}, R^2 = {r2:.3f}"
    )

    # x range for plotting the power law curve
    vpd_pos = vpd_all[vpd_all > 0]
    vpd_min = np.nanmin(vpd_pos)
    vpd_max = np.nanmax(vpd_pos)
    vpd_fit = np.linspace(vpd_min, vpd_max, 300)

    # Back to original scale: NoF = a * VPD^b - 1
    fires_fit = a_pl * vpd_fit ** b_pl - 1.0
    fires_fit = np.clip(fires_fit, 0, None)


    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 8))  # square

    # winter = blue
    ax.scatter(
        winter_df["VPD_max"],
        winter_df["fire_count"],
        color="blue",
        alpha=0.50,
        s=20,
        label="Autumn/Winter (Sep–Feb)",
    )

    # summer = red
    ax.scatter(
        summer_df["VPD_max"],
        summer_df["fire_count"],
        color="red",
        alpha=0.60,
        s=20,
        label="Spring/Summer (Mar–Aug)",
    )

    # power law fit curve (black)
    ax.plot(
        vpd_fit,
        fires_fit,
        color="black",
        linewidth=2,
        label="Power law fit",
    )

    # titles and labels
    ax.set_title("Daily Wildfires vs VPD (London, 2009–2024)", fontsize=14)
    ax.set_xlabel("VPD_max (Pa)", fontsize=12)
    ax.set_ylabel("Daily Wildfire Count", fontsize=12)

    # add equation and R^2 into the legend
    glm_label = (
        f"NoF = {a_pl:.2g} · VPD$^{{{b_pl:.2f}}}$\n"
        f"$R^2$ = {r2:.2f}"
    )


    dummy_handle = mlines.Line2D([], [], color="none")

    ax.legend(
        handles=[
            mlines.Line2D([], [], color="blue", marker="o", linestyle="None",
                          label="Autumn/Winter (Sep–Feb)"),
            mlines.Line2D([], [], color="red", marker="o", linestyle="None",
                          label="Spring/Summer (Mar–Aug)"),
            mlines.Line2D([], [], color="black", label="Power law fit"),
            dummy_handle,
        ],
        labels=[
            "Autumn/Winter (Sep–Feb)",
            "Spring/Summer (Mar–Aug)",
            "Power law fit",
            glm_label,
        ],
        loc="upper left",
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # high resolution PNG (same folder)
    out_path = BASE_DIR / "2_vpd_max_studies/fires_vs_vpd_max_scatter_power_law.png"
    plt.savefig(out_path, dpi=1000)
    print(f"Saved scatter plus power law fit to: {out_path}")

    plt.show()


def main():
    print("Loading merged weather and fire datasets...")
    df = load_all_data()

    print("Plotting fire_count vs VPD_max with power law fit...")
    plot_fires_vs_vpd(df)


if __name__ == "__main__":
    main()
