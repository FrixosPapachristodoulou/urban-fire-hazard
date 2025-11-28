from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.lines as mlines   # for custom legend entries
from scipy.optimize import curve_fit

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/2_pump_hrs_vs_vpd")
YEARS = list(range(2009, 2025))  # 2009–2024 inclusive
# =======================


def is_spring_summer(date: datetime) -> bool:
    """Return True if date is between 1 March and 31 August."""
    month = date.month
    return 3 <= month <= 8


def load_all_data() -> pd.DataFrame:
    """Load all yearly merged weather + pump CSV files into one DataFrame."""
    dfs = []

    for year in YEARS:
        file = BASE_DIR / f"daily_weather_pump_{year}.csv"
        if not file.exists():
            print(f"⚠️ Missing {file}")
            continue

        df = pd.read_csv(file)
        df["met_day"] = pd.to_datetime(df["met_day"], errors="coerce")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No input datasets found.")

    return pd.concat(dfs, ignore_index=True)


# === Jamie-style helpers ===
def linlaw(x, a, b):
    """Linear model in log–log space: log10(y) = a + b * log10(x)."""
    return a + b * x


def fit_power_law(vpd: np.ndarray, pump_vals: np.ndarray):
    """
    Fit a power law:

        PH + 1 = A * VPD^B

    using SciPy's curve_fit on log10-transformed data:

        log10(PH + 1) = a + b * log10(VPD)
        A = 10^a, B = b

    Returns A, B, R2 (computed in log10 space).
    """
    vpd = np.asarray(vpd, dtype=float)
    pump_vals = np.asarray(pump_vals, dtype=float)

    # need positive VPD; allow zero pump_vals by shifting +1
    mask = (vpd > 0) & ~np.isnan(pump_vals)
    x = vpd[mask]
    y = pump_vals[mask] + 1.0   # PH + 1 so zeros are allowed

    # remove any non-positive (just in case)
    valid = y > 0
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        raise RuntimeError("Not enough data points for power law fit.")

    x_log = np.log10(x)
    y_log = np.log10(y)

    # Fit linear model in log–log space
    popt_log, pcov_log = curve_fit(linlaw, x_log, y_log)
    a_lin, b_lin = popt_log

    # Convert back to power law parameters
    A = 10 ** a_lin   # prefactor
    B = b_lin         # exponent

    # R^2 in log space
    y_pred_log = linlaw(x_log, *popt_log)
    ss_res = np.sum((y_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return A, B, r2


def plot_pump_hours_vs_vpd(df: pd.DataFrame):
    # determine colours by season
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )

    summer_df = df[df["season"] == "summer"]
    winter_df = df[df["season"] == "winter"]

    vpd_all = df["VPD_mean"].values
    pumps_all = df["pump_hours"].values.astype(float)

    # ---- Fit power law: PH + 1 = A * VPD^B ----
    A_pl, B_pl, r2 = fit_power_law(vpd_all, pumps_all)
    print(
        f"Power law fit (PH+1): PH + 1 = {A_pl:.3g} * VPD^{B_pl:.3f}, R^2 = {r2:.3f}"
    )

    # x range for plotting the power law curve
    vpd_pos = vpd_all[vpd_all > 0]
    vpd_min = np.nanmin(vpd_pos)
    vpd_max = np.nanmax(vpd_pos)
    vpd_fit = np.linspace(vpd_min, vpd_max, 300)

    # Back to original scale: PH = A * VPD^B - 1
    pumps_fit = A_pl * vpd_fit ** B_pl - 1.0
    pumps_fit = np.clip(pumps_fit, 0, None)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 8))  # square

    # winter = blue
    ax.scatter(
        winter_df["VPD_mean"],
        winter_df["pump_hours"],
        color="blue",
        alpha=0.50,
        s=20,
        label="Autumn/Winter (Sep–Feb)",
    )

    # summer = red
    ax.scatter(
        summer_df["VPD_mean"],
        summer_df["pump_hours"],
        color="red",
        alpha=0.60,
        s=20,
        label="Spring/Summer (Mar–Aug)",
    )

    # power law fit curve (black)
    ax.plot(
        vpd_fit,
        pumps_fit,
        color="black",
        linewidth=2,
        label="Power law fit",
    )

    # titles and labels
    ax.set_title("Daily Pump Hours vs VPD (London, 2009–2024)", fontsize=14)
    ax.set_xlabel("VPD_mean (Pa)", fontsize=12)
    ax.set_ylabel("Daily Pump Hours", fontsize=12)

    # add equation and R^2 into the legend
    glm_label = (
        f"PH = {A_pl:.2g} · VPD$^{{{B_pl:.2f}}}$ − 1\n"
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
    out_path = BASE_DIR / "pump_hours_vs_vpd_scatter_power_law.png"
    plt.savefig(out_path, dpi=400)
    print(f"Saved scatter plus power law fit to: {out_path}")

    plt.show()


def main():
    print("Loading merged weather and pump datasets...")
    df = load_all_data()

    print("Plotting pump_hours vs VPD_mean with power law fit...")
    plot_pump_hours_vs_vpd(df)


if __name__ == "__main__":
    main()
