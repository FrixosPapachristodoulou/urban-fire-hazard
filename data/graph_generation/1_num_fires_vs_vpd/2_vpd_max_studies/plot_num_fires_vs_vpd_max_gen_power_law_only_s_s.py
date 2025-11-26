from pathlib import Path 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.lines as mlines   # for custom legend entries

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/1_num_fires_vs_vpd")
YEARS = list(range(2009, 2025))  # 2009–2024 inclusive

# Range of exponent c in NoF = a * exp(b * VPD^c)
C_MIN = 0.2
C_MAX = 1.2
N_CANDIDATES = 10
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


def fit_generalised_exponential(vpd: np.ndarray, fires: np.ndarray, c: float = 0.5):
    """
    Fit a generalised exponential model:

        NoF = a * exp(b * VPD^c) - 1

    by doing linear regression in transformed space:

        ln(NoF + 1) = ln(a) + b * VPD^c

    Parameters
    ----------
    vpd : array-like
        VPD values (Pa).
    fires : array-like
        Daily fire counts.
    c : float
        Exponent applied to VPD.

    Returns
    -------
    a : float
        Scale parameter.
    b : float
        Exponential rate parameter.
    c : float
        Exponent used in the fit.
    r2 : float
        Coefficient of determination in log-space.
    """
    vpd = np.asarray(vpd, dtype=float)
    fires = np.asarray(fires, dtype=float)

    # need non-negative VPD; keep ALL fire counts (including zeros)
    mask = (vpd >= 0) & ~np.isnan(fires)
    x = vpd[mask]
    y = fires[mask]

    if len(x) < 2:
        raise RuntimeError("Not enough data points for exponential fit.")

    # transformed predictor and response
    z = np.power(x, c)
    log_y = np.log(y + 1.0)   # include zero-fire days

    # ordinary least squares: log_y ≈ log_a + b * z
    b, log_a = np.polyfit(z, log_y, 1)
    a = np.exp(log_a)

    # compute R^2 in log space
    log_y_pred = log_a + b * z
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return a, b, c, r2


def plot_fires_vs_vpd(df: pd.DataFrame):
    # determine colours by season
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )

    summer_df = df[df["season"] == "summer"]
    winter_df = df[df["season"] == "winter"]

    # --- use only SPRING/SUMMER (red) points for the fit, with VPD_max ---
    vpd_summer = summer_df["VPD_max"].values
    fires_summer = summer_df["fire_count"].values.astype(float)

    # ---- Grid search over c for generalised exponential fit ----
    candidate_cs = np.linspace(C_MIN, C_MAX, N_CANDIDATES)

    best_params = None
    best_r2 = -np.inf

    print("Testing generalised exponential exponents c in "
          f"[{C_MIN:.2f}, {C_MAX:.2f}] with {N_CANDIDATES} candidates "
          "for VPD_max:")

    for c in candidate_cs:
        a_exp, b_exp, c_exp, r2 = fit_generalised_exponential(
            vpd_summer, fires_summer, c=c
        )
        print(f"  c = {c:.3f}: R^2 = {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_params = (a_exp, b_exp, c_exp, r2)

    if best_params is None:
        raise RuntimeError("Failed to fit generalised exponential model for any c.")

    a_exp, b_exp, c_exp, r2 = best_params

    print(
        "\nBest generalised exponential fit on SPRING/SUMMER only (NoF+1, VPD_max):\n"
        f"  ln(NoF + 1) = ln({a_exp:.3g}) + {b_exp:.3e} * VPD_max^{c_exp:.3f}\n"
        f"  Equivalent: NoF = {a_exp:.3g} * exp({b_exp:.3e} * VPD_max^{c_exp:.3f}) - 1\n"
        f"  Best R^2 (log-space) = {r2:.4f}"
    )

    # x range for plotting the curve (based on summer VPD_max only)
    vpd_pos = vpd_summer[vpd_summer > 0]
    vpd_min = np.nanmin(vpd_pos)
    vpd_max = np.nanmax(vpd_pos)
    vpd_fit = np.linspace(vpd_min, vpd_max, 300)

    # Back-transform to original scale:
    # NoF = a * exp(b * VPD_max^c) - 1 (using best c)
    fires_fit = a_exp * np.exp(b_exp * np.power(vpd_fit, c_exp)) - 1.0
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

    # generalised exponential fit curve (black) – fitted only on summer points
    ax.plot(
        vpd_fit,
        fires_fit,
        color="black",
        linewidth=2,
        label="Best generalised exponential fit (spring/summer)",
    )

    # titles and labels
    ax.set_title("Daily Wildfires vs VPD_max (London, 2009–2024)", fontsize=14)
    ax.set_xlabel("VPD_max (Pa)", fontsize=12)
    ax.set_ylabel("Daily Wildfire Count", fontsize=12)

    # add equation and R^2 into the legend
    glm_label = (
        r"$\ln(\mathrm{NoF}+1)"
        rf" = \ln({a_exp:.2g})"
        rf" + {b_exp:.2e}\,\mathrm{{VPD_{{max}}}}^{{{c_exp:.2f}}}$"
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
                          label="Best generalised exponential fit (spring/summer)"),
            dummy_handle,
        ],
        labels=[
            "Autumn/Winter (Sep–Feb)",
            "Spring/Summer (Mar–Aug)",
            "Best generalised exponential fit (spring/summer)",
            glm_label,
        ],
        loc="upper left",
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # high resolution PNG (same folder)
    out_path = BASE_DIR / "2_vpd_max_studies/only_s_s_fires_vs_vpd_max_scatter_gen_exp_best_c.png"
    plt.savefig(out_path, dpi=1000)
    print(f"\nSaved scatter plus generalised exponential fit to: {out_path}")

    plt.show()


def main():
    print("Loading merged weather and fire datasets...")
    df = load_all_data()

    print("Plotting fire_count vs VPD_max with generalised exponential fit...")
    plot_fires_vs_vpd(df)


if __name__ == "__main__":
    main()
