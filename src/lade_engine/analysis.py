from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

# Prescriptions are data, not prose — keeping them close to the deconfound logic
# so diagnosis and recommendation stay in sync when thresholds change.
_CITY_PRESCRIPTIONS: dict[str, dict[str, str]] = {
    "Yantai": {
        "root_cause": "Pool composition — 42% of courier pool is low-freq (thin market)",
        "mechanism": "Regular couriers are fast (45 min/km); the problem is supply scarcity, not route difficulty",
        "prescription": "Supply-side fix: recruit/retain regular couriers to push low-freq pool share from 42% → ~20%. "
                        "Immediate bridge: hard-cap low-freq orders during peak hours until supply improves.",
        "priority": "HIGH",
        "kpi": "Low-freq pool share % (target: <20%)",
    },
    "Chongqing": {
        "root_cause": "Terrain × capability — mountainous routes (64 min/km) amplify performance gaps",
        "mechanism": "Regular couriers already operate at the highest difficulty baseline (2.93% delay rate); "
                     "low-freq couriers cannot absorb additional complexity",
        "prescription": "Zone-specialist assignment at peak hours (08:00, 14:00): route orders to couriers "
                        "with proven performance in the target sub-region. "
                        "Low-freq couriers receive short, low-complexity routes only.",
        "priority": "HIGH",
        "kpi": "1-day delay rate (target: <1.5%, from 3.37%)",
    },
    "Shanghai": {
        "root_cause": "Mild amplification (2.5×); absolute delay rates remain very low",
        "mechanism": "Flat urban terrain and deep courier pool absorb low-freq workers without systemic breakdown",
        "prescription": "No immediate action. Revisit if amplification exceeds 5× or 1-day delay rate climbs above 0.5%.",
        "priority": "LOW",
        "kpi": "1-day delay rate (monitor: currently 0.24%)",
    },
    "Hangzhou": {
        "root_cause": "No significant low-freq effect (1.1× amplification — effectively noise)",
        "mechanism": "Large courier pool and moderate routes; low-freq couriers perform on par with regular",
        "prescription": "No intervention needed. Current assignment logic is working.",
        "priority": "—",
        "kpi": "1-day delay rate (monitor: currently 0.60%)",
    },
}


def courier_frequency_crosscity(
    df: pd.DataFrame,
    low_freq_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Cross-city comparison of low-freq vs regular courier delay rates.

    Answers: is the "low-freq courier → high 1-day delay" pattern
    universal (system-wide assignment problem) or Chongqing-specific
    (local operational problem)?

    Returns one row per (city, workload_segment) with delay_rate_1d_pct
    and share_of_city_orders so you can see both severity and exposure.
    """
    con = duckdb.connect(database=":memory:")
    con.register("delivery", df)
    return con.sql(
        f"""
        with daily as (
            select city, courier_id, ds, count(*) as daily_orders
            from delivery
            group by city, courier_id, ds
        ),
        courier_avg as (
            select city, courier_id, avg(daily_orders) as avg_daily
            from daily
            group by city, courier_id
        ),
        labelled as (
            select
                d.*,
                case
                    when c.avg_daily < {low_freq_threshold} then 'low_freq'
                    else 'regular'
                end as workload_segment
            from delivery d
            join courier_avg c using (city, courier_id)
        ),
        city_totals as (
            select city, count(*) as city_total
            from delivery
            group by city
        )
        select
            l.city,
            l.workload_segment,
            count(distinct l.courier_id)                                      as couriers,
            count(*)                                                           as orders,
            100.0 * count(*) / ct.city_total                                  as share_of_city_orders_pct,
            median(l.delivery_minutes)                                        as median_delivery_min,
            100.0 * sum(case when l.delivery_minutes > 1440 then 1 else 0 end)
                  / count(*)                                                  as delay_rate_1d_pct
        from labelled l
        join city_totals ct using (city)
        where not l.is_pilot
        group by l.city, l.workload_segment, ct.city_total
        order by l.city, l.workload_segment desc
        """
    ).df()


def business_impact_scenario(
    df: pd.DataFrame,
    peak_hours: tuple[int, ...] = (8, 14),
    low_freq_threshold: float = 5.0,
    city: str = "Chongqing",
) -> dict[str, int | float | str]:
    """
    Upper-bound estimate of preventable 1-day delays if low-freq couriers
    are excluded from peak-hour assignment in `city`.

    KEY ASSUMPTION (upper-bound caveat): the reassigned orders are absorbed
    by regular couriers at their current delay rate.  This holds only if
    regular couriers have spare capacity at peak hours.  If capacity is
    tight, their delay rate will rise and the net gain will be smaller.
    Present this number as an upper-bound, not a point forecast.

    Logic:
    1. Identify low-freq couriers (avg daily orders < threshold).
    2. Find their peak-hour orders that resulted in a 1-day delay.
    3. Counterfactual: same volume handled by regular couriers at their rate.
    4. Net prevented = actual_low_freq_peak_delayed - counterfactual_delayed.

    Lead finding (not this number): low-freq couriers carry 3% of orders
    but produce 15.7% of 1-day delays — a 5x disproportionality that holds
    regardless of the capacity assumption.
    """
    city_df = df[(df["city"] == city) & (~df["is_pilot"])].copy()

    daily = city_df.groupby(["courier_id", "ds"]).size().reset_index(name="n")
    avg_daily = daily.groupby("courier_id")["n"].mean()
    low_freq_ids = set(avg_daily[avg_daily < low_freq_threshold].index)

    city_df["is_low_freq"] = city_df["courier_id"].isin(low_freq_ids)
    city_df["is_peak"] = city_df["accept_hour"].isin(peak_hours)
    city_df["delayed_1d"] = city_df["delivery_minutes"] > 1440

    total_orders = len(city_df)
    total_delayed = int(city_df["delayed_1d"].sum())
    baseline_delay_rate = total_delayed / total_orders

    low_freq_all = city_df[city_df["is_low_freq"]]
    low_freq_all_delayed = int(low_freq_all["delayed_1d"].sum())
    low_freq_order_share = len(low_freq_all) / total_orders
    low_freq_delay_share = low_freq_all_delayed / max(total_delayed, 1)
    disproportionality_ratio = low_freq_delay_share / max(low_freq_order_share, 1e-9)

    low_peak = city_df[city_df["is_low_freq"] & city_df["is_peak"]]
    low_peak_orders = len(low_peak)
    low_peak_delayed = int(low_peak["delayed_1d"].sum())

    regular = city_df[~city_df["is_low_freq"]]
    regular_delay_rate = regular["delayed_1d"].mean()

    # Upper-bound: assumes regular couriers absorb at current rate (no capacity pressure)
    counterfactual_delayed = round(low_peak_orders * regular_delay_rate)
    net_prevented_upper = low_peak_delayed - counterfactual_delayed

    return {
        "city": city,
        "low_freq_threshold_daily_orders": low_freq_threshold,
        "peak_hours": list(peak_hours),
        "total_orders": total_orders,
        "total_delayed_1d": total_delayed,
        "baseline_delay_rate_pct": round(baseline_delay_rate * 100, 2),
        # --- Lead finding: disproportionality (robust to capacity assumption) ---
        "low_freq_order_share_pct": round(low_freq_order_share * 100, 1),
        "low_freq_delay_share_pct": round(low_freq_delay_share * 100, 1),
        "disproportionality_ratio": round(disproportionality_ratio, 1),
        # --- Impact scenario (upper-bound, capacity-dependent) ---
        "low_freq_couriers": len(low_freq_ids),
        "low_freq_peak_orders": low_peak_orders,
        "low_freq_peak_delayed": low_peak_delayed,
        "low_freq_peak_delay_rate_pct": round(low_peak_delayed / max(low_peak_orders, 1) * 100, 2),
        "regular_delay_rate_pct": round(regular_delay_rate * 100, 2),
        "counterfactual_delayed_upper_bound": int(counterfactual_delayed),
        "net_prevented_delays_upper_bound": int(net_prevented_upper),
        "pct_of_total_delays_prevented_upper_bound": round(
            net_prevented_upper / max(total_delayed, 1) * 100, 1
        ),
        "capacity_caveat": (
            "Upper-bound assumes regular couriers absorb reassigned volume at their "
            "current delay rate. If peak capacity is tight, real gains will be lower. "
            "Validate with a pilot: A/B test peak-hour assignment filtering for "
            "2–4 weeks before committing to the full rollout."
        ),
        "lead_finding": (
            f"Low-freq couriers carry {round(low_freq_order_share*100,1)}% of {city} orders "
            f"but produce {round(low_freq_delay_share*100,1)}% of 1-day delays "
            f"({round(disproportionality_ratio,1)}x disproportionate). "
            f"This ratio is robust — it does not depend on any capacity assumption."
        ),
        "impact_headline": (
            f"Redirecting low-freq couriers away from peak-hour slots would prevent "
            f"an estimated {net_prevented_upper:,} 1-day delays "
            f"({round(net_prevented_upper/max(total_delayed,1)*100,1)}% of current total) "
            f"— upper-bound; actual gain depends on regular-courier spare capacity."
        ),
    }


def deconfound_terrain_vs_pool(
    df: pd.DataFrame,
    low_freq_threshold: float = 5.0,
    max_dist_km: float = 20.0,
) -> pd.DataFrame:
    """
    Separates two competing explanations for why Chongqing and Yantai show
    elevated low-freq courier delay rates while Hangzhou and Shanghai do not.

    Hypothesis A — terrain/difficulty: difficult routes punish low-freq couriers.
      Proxy: median min/km for REGULAR couriers (controls for courier skill).
      If A, cities with high min/km should also have high low-freq amplification.

    Hypothesis B — pool composition: a thin courier market means too many
      occasional workers accept orders they can't complete reliably.
      Proxy: % of courier pool that is low_freq.
      If B, cities with high low_freq pool share should show high amplification.

    Returns one row per non-pilot city with both proxies and the amplification
    factor (low_freq_delay_rate / regular_delay_rate), so the caller can
    reason about which hypothesis fits each city.
    """
    non_pilot = df[~df["is_pilot"]].copy()

    # Courier segmentation
    daily = non_pilot.groupby(["city", "courier_id", "ds"]).size().reset_index(name="n")
    avg_daily = daily.groupby(["city", "courier_id"])["n"].mean().reset_index(name="avg_daily")
    avg_daily["segment"] = avg_daily["avg_daily"].apply(
        lambda x: "low_freq" if x < low_freq_threshold else "regular"
    )

    # Pool composition (Hypothesis B proxy)
    pool = avg_daily.groupby("city").agg(
        total_couriers=("courier_id", "count"),
        low_freq_couriers=("segment", lambda s: (s == "low_freq").sum()),
    ).reset_index()
    pool["low_freq_pool_share_pct"] = (
        pool["low_freq_couriers"] / pool["total_couriers"] * 100
    ).round(1)

    # Delay rates per segment
    merged = non_pilot.merge(avg_daily[["courier_id", "segment"]], on="courier_id")
    delay = merged.groupby(["city", "segment"])["delivery_minutes"].agg(
        delay_rate=lambda x: (x > 1440).mean() * 100
    ).unstack("segment").reset_index()
    delay.columns = ["city", "low_freq_delay_rate_1d", "regular_delay_rate_1d"]
    delay["amplification_factor"] = (
        delay["low_freq_delay_rate_1d"] / delay["regular_delay_rate_1d"].clip(lower=0.01)
    ).round(1)

    # Terrain proxy: median min/km for regular couriers (Hypothesis A proxy)
    reg_df = merged[
        (merged["segment"] == "regular")
        & merged["accept_gps_lat"].notna()
        & merged["delivery_gps_lat"].notna()
        & (merged["accept_gps_lat"] != 0)
    ].copy()

    def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                      lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    reg_df["dist_km"] = _haversine_km(
        reg_df["accept_gps_lat"].values, reg_df["accept_gps_lng"].values,
        reg_df["delivery_gps_lat"].values, reg_df["delivery_gps_lng"].values,
    )
    reg_df = reg_df[(reg_df["dist_km"] > 0.1) & (reg_df["dist_km"] <= max_dist_km)]
    reg_df["min_per_km"] = reg_df["delivery_minutes"] / reg_df["dist_km"]

    terrain = reg_df.groupby("city").agg(
        median_dist_km=("dist_km", "median"),
        median_min_per_km_regular=("min_per_km", "median"),
    ).reset_index()

    result = pool.merge(delay, on="city").merge(terrain, on="city")

    # Human-readable diagnosis per city
    def _diagnose(row: pd.Series) -> str:
        high_pool = row["low_freq_pool_share_pct"] > 30
        high_terrain = row["median_min_per_km_regular"] > 58
        if high_pool and not high_terrain:
            return "Pool composition — thin market, too many occasional couriers"
        if high_terrain and not high_pool:
            return "Terrain/difficulty — routes are hard even for regular couriers"
        if high_pool and high_terrain:
            return "Both factors present — cannot isolate with this data"
        return "Neither factor dominant — low-freq effect mild, likely noise"

    result["diagnosis"] = result.apply(_diagnose, axis=1)

    result["diagnosis"] = result.apply(_diagnose, axis=1)

    out = result[[
        "city",
        "total_couriers",
        "low_freq_couriers",
        "low_freq_pool_share_pct",
        "regular_delay_rate_1d",
        "low_freq_delay_rate_1d",
        "amplification_factor",
        "median_dist_km",
        "median_min_per_km_regular",
        "diagnosis",
    ]].sort_values("amplification_factor", ascending=False).reset_index(drop=True)

    # Attach prescriptions from the lookup table
    out["root_cause"]    = out["city"].map(lambda c: _CITY_PRESCRIPTIONS.get(c, {}).get("root_cause", ""))
    out["prescription"]  = out["city"].map(lambda c: _CITY_PRESCRIPTIONS.get(c, {}).get("prescription", ""))
    out["priority"]      = out["city"].map(lambda c: _CITY_PRESCRIPTIONS.get(c, {}).get("priority", "—"))
    out["kpi"]           = out["city"].map(lambda c: _CITY_PRESCRIPTIONS.get(c, {}).get("kpi", ""))

    return out


def chongqing_regional_deep_dive(
    df: pd.DataFrame,
    max_dist_km: float = 20.0,
) -> pd.DataFrame:
    """
    Per-region analysis within Chongqing.

    Tests the terrain hypothesis at sub-city granularity:
    - Do delays cluster in specific regions?
    - Do high-delay regions also show slower min/km (terrain effect)?
    - Or is the delay pattern uniform (pointing to assignment/pool problems)?

    Returns one row per Chongqing region_id with operational metrics
    and a terrain difficulty classification.
    """
    cq = df[(df["city"] == "Chongqing") & (~df["is_pilot"])].copy()

    # Haversine distance
    def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                      lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    has_gps = cq[
        cq["accept_gps_lat"].notna()
        & cq["delivery_gps_lat"].notna()
        & (cq["accept_gps_lat"] != 0)
    ].copy()
    has_gps["dist_km"] = _haversine_km(
        has_gps["accept_gps_lat"].values, has_gps["accept_gps_lng"].values,
        has_gps["delivery_gps_lat"].values, has_gps["delivery_gps_lng"].values,
    )
    has_gps = has_gps[(has_gps["dist_km"] > 0.1) & (has_gps["dist_km"] <= max_dist_km)]
    has_gps["min_per_km"] = has_gps["delivery_minutes"] / has_gps["dist_km"]

    terrain_by_region = has_gps.groupby("region_id").agg(
        median_dist_km=("dist_km", "median"),
        median_min_per_km=("min_per_km", "median"),
    ).reset_index()

    # Core delivery metrics per region (all Chongqing rows)
    region_metrics = cq.groupby("region_id").agg(
        n_orders=("order_id", "count"),
        n_couriers=("courier_id", "nunique"),
        median_delivery_min=("delivery_minutes", "median"),
        delay_rate_1d=("delivery_minutes", lambda x: (x > 1440).mean() * 100),
        delay_rate_3h=("delivery_minutes", lambda x: (x > 180).mean() * 100),
    ).reset_index()

    result = region_metrics.merge(terrain_by_region, on="region_id", how="left")

    # Terrain difficulty flag: min/km > 75th percentile across Chongqing regions
    terrain_cutoff = result["median_min_per_km"].quantile(0.75)
    delay_cutoff   = result["delay_rate_1d"].quantile(0.75)

    def _classify(row: pd.Series) -> str:
        hard   = row["median_min_per_km"] >= terrain_cutoff
        delays = row["delay_rate_1d"]     >= delay_cutoff
        if hard and delays:
            return "Hard terrain + high delay — terrain hypothesis supported"
        if not hard and delays:
            return "Flat terrain + high delay — operational / assignment problem"
        if hard and not delays:
            return "Hard terrain + low delay — experienced couriers absorb difficulty"
        return "Flat terrain + low delay — normal operation"

    result["classification"] = result.apply(_classify, axis=1)

    # Concentration: what share of Chongqing's 1-day delays come from each region?
    total_delayed = result["delay_rate_1d"].mul(result["n_orders"]).div(100).sum()
    result["region_delayed_orders"] = result["delay_rate_1d"] * result["n_orders"] / 100
    result["pct_of_chongqing_delays"] = (
        result["region_delayed_orders"] / max(total_delayed, 1) * 100
    ).round(2)

    return result.sort_values("delay_rate_1d", ascending=False).reset_index(drop=True)


def build_city_prescriptions() -> pd.DataFrame:
    """
    Returns the static prescription table as a standalone DataFrame.
    Useful for rendering the one-pager without re-running the full deconfound.
    """
    rows = [
        {"city": city, **fields}
        for city, fields in _CITY_PRESCRIPTIONS.items()
    ]
    return pd.DataFrame(rows)[["city", "root_cause", "mechanism", "prescription", "priority", "kpi"]]
