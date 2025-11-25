from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.lines as mlines   # for custom legend entries

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/num_fires_vs_vpd")
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
    Fit a *curved* power law in log–log space:

        ln(NoF + 1) = alpha + beta * ln(VPD) + gamma * [ln(VPD)]^2

    by doing a quadratic regression in log–log space.

    Returns alpha, beta, gamma, R2 (all computed in log space).
    """
    vpd = np.asarray(vpd, dtype=float)
    fires = np.asarray(fires, dtype=float)

    # need positive VPD for the log; keep ALL fire counts (including zeros)
    mask = (vpd > 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    if len(x) < 3:
        raise RuntimeError("Not enough data points for quadratic log–log fit.")

    log_x = np.log(x)
    log_y = np.log(y + 1.0)   # include zero-fire days

    # quadratic fit in log–log space: log_y ≈ gamma*log_x^2 + beta*log_x + alpha
    gamma, beta, alpha = np.polyfit(log_x, log_y, 2)

    # compute R^2 in log space
    y_pred_log = alpha + beta * log_x + gamma * (log_x ** 2)
    ss_res = np.sum((log_y - y_pred_log) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return alpha, beta, gamma, r2


def plot_fires_vs_vpd(df: pd.DataFrame):
    # determine colours by season
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )

    summer_df = df[df["season"] == "summer"]
    winter_df = df[df["season"] == "winter"]

    # === fit only on spring/summer (red dots) ===
    vpd_summer = summer_df["VPD_mean"].values
    fires_summer = summer_df["fire_count"].values.astype(float)

    # ---- Quadratic log–log fit: ln(NoF+1) = alpha + beta ln(VPD) + gamma [ln(VPD)]^2 ----
    alpha, beta, gamma, r2 = fit_power_law(vpd_summer, fires_summer)
    print(
        "Quadratic log–power-law fit on SPRING/SUMMER only (NoF+1):\n"
        f"ln(NoF + 1) = {alpha:.3f} + {beta:.3f} ln(VPD) + {gamma:.3f} [ln(VPD)]^2, "
        f"R^2 = {r2:.3f}"
    )

    # x range for plotting the curve based on summer VPD only
    vpd_pos = vpd_summer[vpd_summer > 0]
    vpd_min = np.nanmin(vpd_pos)
    vpd_max = np.nanmax(vpd_pos)
    vpd_fit = np.linspace(vpd_min, vpd_max, 300)

    # Back-transform to original scale:
    # ln(NoF+1) = alpha + beta ln(VPD) + gamma [ln(VPD)]^2
    # => NoF = exp(alpha + beta ln(VPD) + gamma [ln(VPD)]^2) - 1
    log_v = np.log(vpd_fit)
    log_fires_fit = alpha + beta * log_v + gamma * (log_v ** 2)
    fires_fit = np.exp(log_fires_fit) - 1.0
    fires_fit = np.clip(fires_fit, 0, None)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 8))  # square

    # winter = blue
    ax.scatter(
        winter_df["VPD_mean"],
        winter_df["fire_count"],
        color="blue",
        alpha=0.50,
        s=20,
        label="Autumn/Winter (Sep–Feb)",
    )

    # summer = red
    ax.scatter(
        summer_df["VPD_mean"],
        summer_df["fire_count"],
        color="red",
        alpha=0.60,
        s=20,
        label="Spring/Summer (Mar–Aug)",
    )

    # curved log–power-law fit (black), fitted on spring/summer only
    ax.plot(
        vpd_fit,
        fires_fit,
        color="black",
        linewidth=2,
        label="Quadratic log–power-law fit (spring/summer)",
    )

    # titles and labels
    ax.set_title("Daily Wildfires vs VPD (London, 2009–2024)", fontsize=14)
    ax.set_xlabel("VPD_mean (Pa)", fontsize=12)
    ax.set_ylabel("Daily Wildfire Count", fontsize=12)

    # add equation and R^2 into the legend (in log-space form)
    glm_label = (
        r"$\ln(\mathrm{NoF}+1)"
        rf" = {alpha:.2f}"
        rf" + {beta:.2f}\ln(\mathrm{{VPD}})"
        rf" + {gamma:.2f}[\ln(\mathrm{{VPD}})]^2$"
        "\n"
        rf"$R^2 = {r2:.2f}$ (spring/summer)"
    )

    dummy_handle = mlines.Line2D([], [], color="none")

    ax.legend(
        handles=[
            mlines.Line2D([], [], color="blue", marker="o", linestyle="None",
                          label="Autumn/Winter (Sep–Feb)"),
            mlines.Line2D([], [], color="red", marker="o", linestyle="None",
                          label="Spring/Summer (Mar–Aug)"),
            mlines.Line2D([], [], color="black",
                          label="Quadratic log–power-law fit (spring/summer)"),
            dummy_handle,
        ],
        labels=[
            "Autumn/Winter (Sep–Feb)",
            "Spring/Summer (Mar–Aug)",
            "Quadratic log–power-law fit (spring/summer)",
            glm_label,
        ],
        loc="upper left",
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # high resolution PNG (same folder)
    out_path = BASE_DIR / "only_s_s_fires_vs_vpd_scatter_quadratic_power_law.png"
    plt.savefig(out_path, dpi=400)
    print(f"Saved scatter plus quadratic log–power-law fit to: {out_path}")

    plt.show()


def main():
    print("Loading merged weather and fire datasets...")
    df = load_all_data()

    print("Plotting fire_count vs VPD_mean with quadratic log–power-law fit...")
    plot_fires_vs_vpd(df)


if __name__ == "__main__":
    main()
