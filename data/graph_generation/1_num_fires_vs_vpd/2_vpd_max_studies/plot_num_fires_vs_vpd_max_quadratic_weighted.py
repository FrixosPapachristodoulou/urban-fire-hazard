from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.lines as mlines  # for custom legend entries
import mplcursors  # for interactive hover
from scipy.stats import rankdata  # for percentile-based weighting

# ==== CONFIGURATION ====
BASE_DIR = Path("data/graph_generation/1_num_fires_vs_vpd")
YEARS = list(range(2009, 2025))  # 2009–2024 inclusive

# Percentile that defines the "high VPD" tail region.
HIGH_VPD_PERCENTILE = 80  # top 20% of VPD days
HIGH_VPD_THRESHOLD = None  # will be computed after loading data

# Exponential weighting parameter (for prioritising high-VPD days in fit)
# w(q) = (exp(λq) - 1) / (exp(λ) - 1), where q = VPD percentile rank
# λ controls steepness: higher λ = more aggressive tail weighting
LAMBDA_WEIGHT = 2.0  # λ=3 → bottom 50% contributes ~30%, top 25% contributes ~60-100%
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
        file = BASE_DIR / f"daily_weather_fire_{year}.csv"
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


def compute_exponential_weights(vpd: np.ndarray, lam: float = LAMBDA_WEIGHT) -> np.ndarray:
    """
    Compute exponential weights based on VPD percentile rank.
    
    Formula: w(q) = (exp(λq) - 1) / (exp(λ) - 1)
    
    This gives:
    - w(0) = 0  (lowest VPD → zero weight)
    - w(1) = 1  (highest VPD → full weight)
    - Exponential growth mirrors how fire danger scales with dryness
    
    With λ=3:
        10th percentile → weight ≈ 0.05
        25th percentile → weight ≈ 0.10
        50th percentile → weight ≈ 0.29
        75th percentile → weight ≈ 0.58
        90th percentile → weight ≈ 0.82
    
    Args:
        vpd: array of VPD values
        lam: steepness parameter (higher = more aggressive tail weighting)
    
    Returns:
        weights: array of same shape as vpd, values in [0, 1]
    """
    # Convert VPD to percentile rank (0 to 1)
    q = rankdata(vpd, method='average') / len(vpd)
    
    # Exponential ramp: w(q) = (exp(λq) - 1) / (exp(λ) - 1)
    weights = (np.exp(lam * q) - 1.0) / (np.exp(lam) - 1.0)
    
    return weights


def fit_quadratic_loglog(vpd: np.ndarray, fires: np.ndarray, use_weighting: bool = True):
    """
    Fit a quadratic relation in log–log space:

        log(NoF + 1) = c0 + c1*log(VPD) + c2*[log(VPD)]^2

    With optional percentile-exponential weighting to prioritise high-VPD days.
    This biases the fit towards dangerous (high-VPD) conditions where
    accurate fire prediction matters most.

    Returns (c2, c1, c0), R²_all   [np.polyfit order].
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
    log_y = np.log(y + 1.0)   # include zero days

    # Compute percentile-exponential weights if requested
    if use_weighting:
        weights = compute_exponential_weights(x)
    else:
        weights = None

    # quadratic least squares in log space
    # returns [c2, c1, c0]
    coeffs = np.polyfit(log_x, log_y, 2, w=weights)
    c2, c1, c0 = coeffs

    # compute R² in log space over all data (UNWEIGHTED for fair evaluation)
    y_pred_log = c2 * log_x**2 + c1 * log_x + c0
    ss_res = np.sum((log_y - y_pred_log) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2_all = 1.0 - ss_res / ss_tot

    return coeffs, r2_all


def compute_r2_high_vpd(
    vpd: np.ndarray,
    fires: np.ndarray,
    c2: float,
    c1: float,
    c0: float,
) -> float:
    """
    Compute R² restricted to the high VPD regime, defined as VPD > HIGH_VPD_THRESHOLD.
    Computation is done in the same log space as the fit (unweighted for fair evaluation).
    """
    vpd = np.asarray(vpd, dtype=float)
    fires = np.asarray(fires, dtype=float)

    mask = (vpd > HIGH_VPD_THRESHOLD) & (vpd > 0) & ~np.isnan(fires)
    if mask.sum() < 3:
        return np.nan

    x = vpd[mask]
    y = fires[mask]

    log_x = np.log(x)
    log_y = np.log(y + 1.0)

    y_pred_log = c2 * log_x**2 + c1 * log_x + c0
    ss_res = np.sum((log_y - y_pred_log) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)

    return 1.0 - ss_res / ss_tot


def plot_seasonal_quadratic_loglog(df: pd.DataFrame):
    vpd_all = df["VPD_max"].values.astype(float)
    fires_all = df["fire_count"].values.astype(float)
    dates_all = df["met_day"].values  # store dates for hover
    seasons = df["season"].values

    # define the four fitting scenarios
    scenarios = [
        ("All seasons", np.ones(len(df), dtype=bool)),
        ("Spring + Summer", df["season"].isin(["spring", "summer"]).values),
        ("Summer + Autumn", df["season"].isin(["summer", "autumn"]).values),
        ("Summer only", (df["season"] == "summer").values),
    ]

    # ---- Exponential weighting info ----
    print("\n====== PERCENTILE-EXPONENTIAL WEIGHTING ======")
    print(f"  λ = {LAMBDA_WEIGHT:.1f}")
    print(f"  Formula: w(q) = (exp(λq) - 1) / (exp(λ) - 1)")
    print("  Weight at percentiles:")
    for pct in [10, 25, 50, 75, 90]:
        q = pct / 100.0
        w = (np.exp(LAMBDA_WEIGHT * q) - 1.0) / (np.exp(LAMBDA_WEIGHT) - 1.0)
        print(f"    {pct:3d}th → {w:.3f}")
    print("===============================================\n")

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 9), sharex=True, sharey=True
    )
    axes = axes.ravel()
    
    # Set axis limits
    for ax in axes:
        ax.set_xlim(0, 6000)
        ax.set_ylim(0, 120)

    # Store all scatter artists and their corresponding data for hover
    all_scatter_artists = []

    # Store secondary axes for later adjustment
    secondary_axes = []

    for ax, (title, subset_mask) in zip(axes, scenarios):
        # which points actually participate in the fit?
        fit_mask = subset_mask & (vpd_all > 0) & ~np.isnan(fires_all)

        # subset data for this scenario
        vpd_fit_data = vpd_all[fit_mask]
        fires_fit_data = fires_all[fit_mask]

        # ---- Percentile-exponential weighted quadratic log–log fit ----
        coeffs, r2_all = fit_quadratic_loglog(
            vpd_fit_data,
            fires_fit_data,
            use_weighting=True,
        )
        c2, c1, c0 = coeffs

        # R² in the high VPD regime (evaluation, still unweighted)
        r2_high = compute_r2_high_vpd(vpd_fit_data, fires_fit_data, c2, c1, c0)

        # X-range for curve in this subplot (based on used points)
        # Avoid tiny VPD values that cause log(VPD) → -∞ behaviour in quadratic fits
        vpd_min_safe = max(np.nanpercentile(vpd_fit_data, 5), 20.0)  # clip at 20 Pa minimum
        vpd_max = np.nanmax(vpd_fit_data)
        vpd_curve = np.linspace(vpd_min_safe, vpd_max, 300)

        log_vpd_curve = np.log(vpd_curve)
        log_fires_curve = c2 * log_vpd_curve**2 + c1 * log_vpd_curve + c0

        fires_curve = np.exp(log_fires_curve) - 1.0
        fires_curve = np.clip(fires_curve, 0, None)

        # ---- scatter points by season with different alpha depending on fit participation ----
        for season_name, color in SEASON_COLORS.items():
            season_mask = (seasons == season_name)

            used = fit_mask & season_mask
            unused = (~fit_mask) & season_mask

            # low-alpha for points not used in this fit
            if unused.any():
                sc_unused = ax.scatter(
                    vpd_all[unused],
                    fires_all[unused],
                    s=10,
                    alpha=0.15,
                    color=color,
                    edgecolors="none",
                    picker=True,  # enable picking for hover
                )
                # Store scatter artist with its metadata
                all_scatter_artists.append({
                    'artist': sc_unused,
                    'dates': dates_all[unused],
                    'vpd': vpd_all[unused],
                    'fires': fires_all[unused],
                    'season': season_name,
                })

            # high-alpha for points used in this fit
            if used.any():
                sc_used = ax.scatter(
                    vpd_all[used],
                    fires_all[used],
                    s=10,
                    alpha=0.95,
                    color=color,
                    edgecolors="none",
                    picker=True,  # enable picking for hover
                )
                # Store scatter artist with its metadata
                all_scatter_artists.append({
                    'artist': sc_used,
                    'dates': dates_all[used],
                    'vpd': vpd_all[used],
                    'fires': fires_all[used],
                    'season': season_name,
                })

        # plot quadratic log–log curve
        ax.plot(
            vpd_curve,
            fires_curve,
            color="black",
            linewidth=2,
        )

        # ---- Secondary axis: weight curve ----
        ax2 = ax.twinx()
        secondary_axes.append(ax2)
        
        # Compute weights across the full VPD range shown in subplot
        vpd_for_weights = np.linspace(1, 6000, 500)
        # Weights based on percentile rank (simulating the ranking)
        q_values = vpd_for_weights / 6000  # approximate percentile as fraction of max
        weights_curve = (np.exp(LAMBDA_WEIGHT * q_values) - 1.0) / (np.exp(LAMBDA_WEIGHT) - 1.0)
        
        ax2.fill_between(
            vpd_for_weights,
            0,
            weights_curve,
            color="purple",
            alpha=0.06,
        )
        ax2.plot(
            vpd_for_weights,
            weights_curve,
            color="purple",
            alpha=0.25,
            linewidth=1.5,
            linestyle="--",
        )
        ax2.set_ylim(0, 1.05)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.tick_params(axis='y', labelcolor="purple", labelsize=7, colors="purple")
        for label in ax2.get_yticklabels():
            label.set_alpha(0.4)
        
        # Only show y-label on right-side subplots
        if ax in [axes[1], axes[3]]:
            ax2.set_ylabel("Weight $w(q)$", fontsize=9, color="purple", alpha=0.4)
        else:
            ax2.set_yticklabels([])


        # mark the high VPD threshold with a vertical line (more transparent)
        ax.axvline(
            HIGH_VPD_THRESHOLD,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.3,  # lower opacity
        )

        # nice title with the quadratic relation in log–log space
        season_title = r"$\mathbf{" + title.replace(" ", r'\ ') + "}$"
        eq_line = (
            r"$\ln(\mathrm{NoF}+1) = "
            f"{c0:.2f}"
            r" + "
            f"{c1:.2f}"
            r"\ln(\mathrm{VPD}) + "
            f"{c2:.2f}"
            r"[\ln(\mathrm{VPD})]^2$"
        )
        r2_all_line = rf"$\mathbf{{R^2_{{all}} = {r2_all:.2f}}}$"
        r2_high_line = rf"$R^2_{{high}} = {r2_high:.2f}$"

        ax.set_title(
            f"{season_title}\n{eq_line}\n{r2_all_line}, {r2_high_line}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)

    # shared labels
    fig.supxlabel("VPD_max (Pa)", fontsize=13)
    fig.supylabel("Daily wildfire count", fontsize=13, fontweight="bold")

    # ----- global legend for seasons + curve + threshold -----
    season_handles = [
        mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                      label=season_name.capitalize())
        for season_name, color in SEASON_COLORS.items()
    ]
    curve_handle = mlines.Line2D([], [], color="black", label="Weighted fit (percentile-exponential)")
    threshold_handle = mlines.Line2D(
        [], [], color="black", linestyle="--", alpha=0.3,
        label=f"{HIGH_VPD_PERCENTILE}th pctl threshold"
    )

    weight_handle = mlines.Line2D(
        [], [], color="purple", alpha=0.3, linestyle="--",
        label="Weight $w(q)$"
    )

    fig.legend(
        handles=season_handles + [curve_handle, threshold_handle, weight_handle],
        loc="upper center",
        ncol=7,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.97),
    )

    # explanatory note about weighting scheme
    # compute weight at 90th percentile for annotation
    w_90 = (np.exp(LAMBDA_WEIGHT * 0.9) - 1.0) / (np.exp(LAMBDA_WEIGHT) - 1.0)
    w_10 = (np.exp(LAMBDA_WEIGHT * 0.1) - 1.0) / (np.exp(LAMBDA_WEIGHT) - 1.0)
    
    fig.text(
        0.5,
        0.06,
        (
            f"Exponential weighting: "
            f"$w(q) = (e^{{\\lambda q}} - 1) / (e^{{\\lambda}} - 1)$, "
            f"$\\lambda = {LAMBDA_WEIGHT:.1f}$, "
            f"where $q$ = VPD percentile. "
            f"90th pctl: $w={w_90:.2f}$, 10th pctl: $w={w_10:.2f}$."
        ),
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=[0.03, 0.04, 1, 0.93])

    # ===== ADD HOVER FUNCTIONALITY =====
    # Create cursor for all scatter artists
    scatter_artists_only = [item['artist'] for item in all_scatter_artists]
    cursor = mplcursors.cursor(scatter_artists_only, hover=True)

    # Create a mapping from artist to metadata
    artist_to_metadata = {item['artist']: item for item in all_scatter_artists}

    @cursor.connect("add")
    def on_add(sel):
        # Get the metadata for this artist
        metadata = artist_to_metadata.get(sel.artist)
        if metadata is None:
            return
        
        idx = sel.index
        date = metadata['dates'][idx]
        vpd = metadata['vpd'][idx]
        fires = metadata['fires'][idx]
        season = metadata['season']
        
        # Format the date nicely
        if pd.notna(date):
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        else:
            date_str = "Unknown"
        
        sel.annotation.set_text(
            f"Date: {date_str}\n"
            f"Season: {season.capitalize()}\n"
            f"VPD: {vpd:.1f} Pa\n"
            f"Fires: {int(fires)}"
        )
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    # ===================================

    out_path = BASE_DIR / "2_vpd_max_studies/fires_vs_vpd_max_quadratic_weighted.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=400)
    print(f"Saved seasonal quadratic log-log figure to: {out_path}")

    plt.show()


def main():
    print("Loading merged weather and fire datasets...")
    df = load_all_data()

    # --- USE ONLY FINITE, POSITIVE VPD VALUES ---
    vpd_all = df["VPD_max"].values.astype(float)
    mask = np.isfinite(vpd_all) & (vpd_all > 0)

    valid_vpd = vpd_all[mask]

    # --- COMPUTE 80th PERCENTILE THRESHOLD ---
    global HIGH_VPD_THRESHOLD
    HIGH_VPD_THRESHOLD = np.nanpercentile(valid_vpd, HIGH_VPD_PERCENTILE)

    # --- COUNT HOW MANY POINTS ARE ABOVE / BELOW ---
    n_total = valid_vpd.size
    n_above = np.sum(valid_vpd > HIGH_VPD_THRESHOLD)
    n_below = np.sum(valid_vpd <= HIGH_VPD_THRESHOLD)

    print("\n====== HIGH-VPD THRESHOLD ANALYSIS ======")
    print(f"Percentile used: {HIGH_VPD_PERCENTILE}th")
    print(f"Computed threshold: {HIGH_VPD_THRESHOLD:.2f} Pa")
    print(f"Total valid VPD datapoints: {n_total}")
    print(f"Below or equal threshold: {n_below}  ({100*n_below/n_total:.1f}%)")
    print(f"Above threshold       : {n_above}  ({100*n_above/n_total:.1f}%)")
    print("=========================================\n")

    print("Plotting seasonal quadratic log-log fits...")
    plot_seasonal_quadratic_loglog(df)




if __name__ == "__main__":
    main()