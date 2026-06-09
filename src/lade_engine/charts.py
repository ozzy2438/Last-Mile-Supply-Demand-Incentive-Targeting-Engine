from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Guard against headless environments — use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


_CITY_COLORS = {
    "Chongqing": "#e74c3c",
    "Hangzhou":  "#2ecc71",
    "Shanghai":  "#3498db",
    "Yantai":    "#9b59b6",
}

_FIG_DPI = 150


def plot_courier_load_vs_delay(
    courier_segments: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Chart 1: Courier daily frequency (X) vs 1-day delay rate (Y).

    Each dot = one courier.  Regression line per city shows the universal
    negative slope (more orders ↔ lower delay) and whether Chongqing's
    low-freq segment is outlier-level or just part of the same trend.

    courier_segments must be the output of build_courier_workload_segments()
    filtered to non-pilot cities.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cities = [c for c in _CITY_COLORS if c in courier_segments["city"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=_FIG_DPI)

    for city in cities:
        sub = courier_segments[courier_segments["city"] == city].copy()
        # Clip extreme x-axis outliers for readability
        sub = sub[sub["avg_daily_orders"] <= 60]

        color = _CITY_COLORS[city]
        ax.scatter(
            sub["avg_daily_orders"],
            sub["delay_rate_1d_pct"],
            color=color,
            alpha=0.25,
            s=12,
            linewidths=0,
            label="_nolegend_",
        )

        # OLS regression line
        x = sub["avg_daily_orders"].values
        y = sub["delay_rate_1d_pct"].values
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, m * x_line + b, color=color, linewidth=2, label=city)

    ax.set_xlabel("Avg daily orders per courier", fontsize=11)
    ax.set_ylabel("1-day delay rate (%)", fontsize=11)
    ax.set_title(
        "Courier workload vs. 1-day delay rate\n"
        "Low-frequency couriers drive disproportionate delays across all cities",
        fontsize=12,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(title="City", fontsize=9, title_fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_distance_vs_duration(
    df: pd.DataFrame,
    output_path: Path,
    sample_n: int = 15_000,
    max_dist_km: float = 20.0,
) -> None:
    """
    Chart 2: Accept-GPS → delivery-GPS distance (km) vs delivery_minutes.

    Near-zero regression slope kills the geography hypothesis for Chongqing
    and shows the same for all other cities.  Annotated with Pearson r.

    df must be the output of load_lade_parquet() (has delivery_minutes,
    accept_gps_lat/lng, delivery_gps_lat/lng).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Haversine vectorised
    def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                      lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    cities = [c for c in _CITY_COLORS if c in df["city"].unique()]
    n_cities = len(cities)
    fig, axes = plt.subplots(1, n_cities, figsize=(4.5 * n_cities, 4.5), dpi=_FIG_DPI,
                             sharey=True)
    if n_cities == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        sub = df[
            (df["city"] == city)
            & df["accept_gps_lat"].notna()
            & df["delivery_gps_lat"].notna()
            & (df["accept_gps_lat"] != 0)
            & (df["delivery_gps_lat"] != 0)
        ].copy()

        sub["dist_km"] = _haversine_km(
            sub["accept_gps_lat"].values, sub["accept_gps_lng"].values,
            sub["delivery_gps_lat"].values, sub["delivery_gps_lng"].values,
        )
        # Remove GPS outliers (> max_dist_km is clearly erroneous for urban delivery)
        sub = sub[sub["dist_km"] <= max_dist_km]

        if len(sub) > sample_n:
            sub = sub.sample(n=sample_n, random_state=42)

        x = sub["dist_km"].values
        y = sub["delivery_minutes"].values

        color = _CITY_COLORS[city]
        ax.scatter(x, y, color=color, alpha=0.08, s=6, linewidths=0)

        r = np.corrcoef(x, y)[0, 1] if len(x) >= 2 else float("nan")
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(0, max_dist_km, 200)
            ax.plot(x_line, m * x_line + b, color=color, linewidth=2.5)

        ax.text(0.97, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))
        ax.set_title(city, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Distance accept → delivery (km)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel("Delivery time (minutes)", fontsize=10)
    fig.suptitle(
        "Distance explains almost nothing about delivery duration\n"
        "Geography is not the root cause of Chongqing delays",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


_MONTH_LABELS = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}


def plot_parallel_trends(
    monthly_avg: pd.DataFrame,
    output_path: Path,
    treatment_boundary_month: float = 7.5,
    outcome_label: str = "Median delivery time (min)",
) -> None:
    """
    Chart 3: Pre/post monthly outcome trends for treated vs control zones.

    Parallel trends assumption is visible in the pre-period (left of dashed
    vertical line): if the two lines are roughly parallel there, the DiD
    identifying assumption holds.

    monthly_avg must have columns: month, group_label, mean_outcome
    (as returned by check_parallel_trends()["monthly_avg"]).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=_FIG_DPI)

    colors = {"Treated (supply ↑)": "#e74c3c", "Control (supply stable)": "#3498db"}
    markers = {"Treated (supply ↑)": "o", "Control (supply stable)": "s"}

    for label, grp in monthly_avg.groupby("group_label"):
        grp = grp.sort_values("month")
        ax.plot(
            grp["month"], grp["mean_outcome"],
            color=colors.get(label, "#555"),
            marker=markers.get(label, "^"),
            linewidth=2.2, markersize=7, label=label,
        )

    # Treatment boundary
    ax.axvline(x=treatment_boundary_month, color="#2c3e50", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.text(treatment_boundary_month + 0.05, ax.get_ylim()[1] * 0.98,
            "Treatment\nboundary", va="top", fontsize=8, color="#2c3e50", alpha=0.8)

    # Pre-period shading
    pre_end = treatment_boundary_month
    ax.axvspan(ax.get_xlim()[0], pre_end, alpha=0.06, color="#27ae60")
    ax.axvspan(pre_end, ax.get_xlim()[1], alpha=0.06, color="#e74c3c")
    ax.text(5.5, ax.get_ylim()[0] * 1.01, "Pre-period\n(parallel trends check)",
            fontsize=7.5, color="#27ae60", alpha=0.9)
    ax.text(8.2, ax.get_ylim()[0] * 1.01, "Post-period\n(treatment effect)",
            fontsize=7.5, color="#c0392b", alpha=0.9)

    ax.set_xticks(sorted(monthly_avg["month"].unique()))
    ax.set_xticklabels([_MONTH_LABELS.get(m, str(m)) for m in sorted(monthly_avg["month"].unique())])
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel(outcome_label, fontsize=11)
    ax.set_title(
        "Parallel trends check: treated vs control zones\n"
        "Pre-period lines should be parallel to validate DiD assumption",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
