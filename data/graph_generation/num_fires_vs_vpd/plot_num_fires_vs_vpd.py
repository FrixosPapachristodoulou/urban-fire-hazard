from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm

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


def fit_poisson_glm(vpd: np.ndarray, fires: np.ndarray):
    """
    Fit a Poisson GLM:

        log(E[NoF]) = beta0 + beta1 * log(VPD)

    which is equivalent to:

        E[NoF] = a * VPD^b

    Returns a, b, pseudo_R2, results_object.
    """
    # Use only positive VPD (log defined)
    mask = vpd > 0
    x = vpd[mask]
    y = fires[mask].astype(float)

    if len(x) < 2:
        raise RuntimeError("Not enough positive data points for Poisson GLM fit.")

    log_vpd = np.log(x)
    X = sm.add_constant(log_vpd)

    model = sm.GLM(y, X, family=sm.families.Poisson())
    results = model.fit()

    beta0, beta1 = results.params  # intercept, slope

    # Convert to power-law parameters
    a = np.exp(beta0)
    b = beta1

    # Pseudo-R^2 based on deviance
    pseudo_r2 = 1.0 - results.deviance / results.null_deviance

    return a, b, pseudo_r2, results


def plot_fires_vs_vpd(df: pd.DataFrame):
    # Determine colours by season
    df["season"] = df["met_day"].apply(
        lambda d: "summer" if is_spring_summer(d) else "winter"
    )

    summer_df = df[df["season"] == "summer"]
    winter_df = df[df["season"] == "winter"]

    vpd_all = df["VPD_mean"].values
    fires_all = df["fire_count"].values.astype(float)

    # ---- Fit Poisson GLM: log(E[NoF]) = beta0 + beta1 * log(VPD) ----
    a_glm, b_glm, pseudo_r2, glm_results = fit_poisson_glm(vpd_all, fires_all)
    print(glm_results.summary())
    print(
        f"\nPoisson GLM (log link): E[NoF] = {a_glm:.3g} * VPD^{b_glm:.3f}, "
        f"pseudo-R^2 = {pseudo_r2:.3f}"
    )

    # X-range for plotting the GLM fit curve
    vpd_pos = vpd_all[vpd_all > 0]
    vpd_min = np.nanmin(vpd_pos)
    vpd_max = np.nanmax(vpd_pos)
    vpd_fit = np.linspace(vpd_min, vpd_max, 300)
    fires_fit_glm = a_glm * vpd_fit ** b_glm

    # ---- Plot ----
    plt.figure(figsize=(8, 8))  # square

    # Winter = blue
    plt.scatter(
        winter_df["VPD_mean"],
        winter_df["fire_count"],
        color="blue",
        alpha=0.50,
        s=20,
        label="Autumn/Winter (Sep–Feb)",
    )

    # Summer = red
    plt.scatter(
        summer_df["VPD_mean"],
        summer_df["fire_count"],
        color="red",
        alpha=0.60,
        s=20,
        label="Spring/Summer (Mar–Aug)",
    )

    # Poisson GLM fit curve (black)
    plt.plot(
        vpd_fit,
        fires_fit_glm,
        color="black",
        linewidth=2,
        label="Poisson GLM fit",
    )

    # Titles + labels
    plt.title("Daily Wildfires vs VPD (London, 2009–2024)", fontsize=14)
    plt.xlabel("VPD_mean (Pa)", fontsize=12)
    plt.ylabel("Daily Wildfire Count", fontsize=12)

    # Text box with power-law form and pseudo-R^2
    eq_text = (
        f"E[NoF] = {a_glm:.2g} · VPD$^{{{b_glm:.2f}}}$\n"
        f"Poisson pseudo-$R^2$ = {pseudo_r2:.2f}"
    )
    ax = plt.gca()
    ax.text(
        0.05,
        0.95,
        eq_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # High-resolution PNG (same folder)
    out_path = BASE_DIR / "fires_vs_vpd_scatter_poisson_glm.png"
    plt.savefig(out_path, dpi=400)
    print(f"📈 Saved scatter + Poisson GLM fit to: {out_path} (1000 DPI)")

    plt.show()


def main():
    print("📥 Loading merged weather + fire datasets...")
    df = load_all_data()

    print("📊 Plotting fire_count vs VPD_mean with Poisson GLM fit...")
    plot_fires_vs_vpd(df)


if __name__ == "__main__":
    main()
