from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def build_package_metrics(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.register("lade", df)
    return con.sql(
        """
        select
            order_id,
            courier_id,
            city,
            region_id,
            aoi_id,
            zone_id,
            ds,
            hour,
            accept_time,
            pickup_time,
            delivery_time,
            promise_end_time,
            date_diff('second', accept_time, pickup_time) / 60.0 as dispatch_latency_min,
            date_diff('second', pickup_time, delivery_time) / 60.0 as delivery_duration_min,
            date_diff('second', accept_time, delivery_time) / 60.0 as end_to_end_min,
            case when delivery_time <= promise_end_time then 1 else 0 end as sla_hit
        from lade
        where accept_time is not null
          and pickup_time is not null
          and delivery_time is not null
        """
    ).df()


def build_zone_hour_metrics(package_metrics: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.register("pkg", package_metrics)
    return con.sql(
        """
        with zone_hour as (
            select
                city,
                region_id,
                zone_id,
                ds,
                hour,
                count(*) as demand_packages,
                count(distinct courier_id) as supply_density,
                avg(dispatch_latency_min) as dispatch_latency_min,
                median(dispatch_latency_min) as dispatch_latency_p50_min,
                quantile_cont(dispatch_latency_min, 0.9) as dispatch_latency_p90_min,
                avg(delivery_duration_min) as delivery_duration_min,
                avg(end_to_end_min) as end_to_end_min,
                avg(sla_hit) as sla_hit_rate,
                count(*) * 1.0 / nullif(count(distinct courier_id), 0) as courier_load
            from pkg
            group by all
        )
        select
            *,
            demand_packages * 1.0 / nullif(supply_density, 0) as demand_supply_ratio,
            case
                when supply_density <= quantile_cont(supply_density, 0.25) over (partition by city) then 1
                else 0
            end as low_supply_flag,
            case
                when demand_packages >= quantile_cont(demand_packages, 0.75) over (partition by city) then 1
                else 0
            end as high_demand_flag
        from zone_hour
        order by city, region_id, ds, hour
        """
    ).df()


def build_data_quality_report(raw_parquet_path: Path) -> pd.DataFrame:
    """
    Per-city data quality audit from the RAW (unfiltered) parquet.

    Must run BEFORE load_lade_parquet() so all counts are pre-filter.
    This is the professional "olay yeri tutanağı" — every downstream cleaning
    decision should reference a row in this table.

    Columns
    -------
    city                  : city name
    raw_records           : total rows before any filter
    min_ds / max_ds       : earliest and latest date partition (integer MMDD)
    dataset_total_days    : distinct ds values across the whole dataset
    active_days           : distinct ds values for this city
    missing_days          : dataset_total_days - active_days
    coverage_pct          : active_days / dataset_total_days * 100
    zero_min_records      : rows where accept_time == delivery_time (string eq)
    zero_min_pct          : zero_min_records / raw_records * 100
    negative_min_records  : rows where delivery_time < accept_time after parsing
    gps_delivery_null_pct : % rows with null delivery_gps_lat
    unique_couriers       : distinct courier_id
    avg_daily_volume      : raw_records / active_days
    causal_eligible       : False for Jilin (coverage < 90% or hacim < 5% of median)
    """
    raw = pd.read_parquet(raw_parquet_path)

    # ── delivery duration (minutes) using string timestamps ──────────────────
    # Fast path: same string → zero; otherwise parse minimally
    same_ts = raw["accept_time"] == raw["delivery_time"]
    zero_count_by_city = raw[same_ts].groupby("city").size().rename("zero_min_records")

    # Negative durations require full parse — only ~3 records total
    raw["_accept_dt"]   = pd.to_datetime("2023-" + raw["accept_time"],   errors="coerce")
    raw["_delivery_dt"] = pd.to_datetime("2023-" + raw["delivery_time"], errors="coerce")
    raw["_dur_min"]     = (raw["_delivery_dt"] - raw["_accept_dt"]).dt.total_seconds() / 60
    neg_count_by_city   = (raw["_dur_min"] < 0).groupby(raw["city"]).sum().rename("negative_min_records")

    # ── dataset-wide date range ──────────────────────────────────────────────
    dataset_total_days = raw["ds"].nunique()

    # ── per-city aggregation via DuckDB ─────────────────────────────────────
    con = duckdb.connect(":memory:")
    con.register("raw", raw)
    per_city = con.sql(
        f"""
        select
            city,
            count(*)                                               as raw_records,
            min(ds)                                                as min_ds,
            max(ds)                                                as max_ds,
            {dataset_total_days}                                   as dataset_total_days,
            count(distinct ds)                                     as active_days,
            {dataset_total_days} - count(distinct ds)              as missing_days,
            round(100.0 * count(distinct ds) / {dataset_total_days}, 1)
                                                                   as coverage_pct,
            sum(case when delivery_gps_lat is null then 1 else 0 end) * 100.0
                / count(*)                                         as gps_delivery_null_pct,
            count(distinct courier_id)                             as unique_couriers,
            round(count(*) * 1.0 / count(distinct ds), 0)         as avg_daily_volume
        from raw
        group by city
        order by raw_records desc
        """
    ).df()

    # ── merge derived columns ────────────────────────────────────────────────
    per_city = (
        per_city
        .merge(zero_count_by_city.reset_index(), on="city", how="left")
        .merge(neg_count_by_city.reset_index(), on="city", how="left")
    )
    per_city["zero_min_records"]    = per_city["zero_min_records"].fillna(0).astype(int)
    per_city["negative_min_records"] = per_city["negative_min_records"].fillna(0).astype(int)
    per_city["zero_min_pct"]        = (
        per_city["zero_min_records"] / per_city["raw_records"] * 100
    ).round(2)
    per_city["gps_delivery_null_pct"] = per_city["gps_delivery_null_pct"].round(2)

    # ── eligibility flag for causal analysis ────────────────────────────────
    # A city is causal-eligible if it has ≥90% date coverage AND
    # its avg daily volume is ≥5% of the median across all cities.
    median_daily = per_city["avg_daily_volume"].median()
    per_city["causal_eligible"] = (
        (per_city["coverage_pct"] >= 90)
        & (per_city["avg_daily_volume"] >= 0.05 * median_daily)
    )

    # Drop internal parse columns
    raw.drop(columns=["_accept_dt", "_delivery_dt", "_dur_min"], inplace=True)

    return per_city[[
        "city", "raw_records", "min_ds", "max_ds",
        "dataset_total_days", "active_days", "missing_days", "coverage_pct",
        "zero_min_records", "zero_min_pct",
        "negative_min_records", "gps_delivery_null_pct",
        "unique_couriers", "avg_daily_volume", "causal_eligible",
    ]]


def build_real_zone_hour_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the real LaDe delivery data to zone × date × accept_hour grain.

    Analogous to build_zone_hour_metrics() but adapted for data without pickup_time.
    Jilin (is_pilot=True) is excluded automatically.

    Outcome variable: delivery_minutes (accept → delivery end-to-end).
    Supply proxy: distinct couriers active per zone-hour.
    Zone: city_regionid (e.g. "Chongqing_10").

    Returned columns
    ----------------
    zone_id, city, region_id, ds, hour, month,
    n_orders, n_couriers, courier_load,
    median_delivery_min, p90_delivery_min, mean_delivery_min,
    delay_rate_1d, delay_rate_3h
    """
    con = duckdb.connect(":memory:")
    con.register("delivery", df[~df["is_pilot"]])
    return con.sql(
        """
        select
            city || '_' || cast(region_id as varchar)  as zone_id,
            city,
            region_id,
            ds,
            accept_hour                                as hour,
            ds // 100                                  as month,
            count(*)                                   as n_orders,
            count(distinct courier_id)                 as n_couriers,
            count(*) * 1.0 / nullif(count(distinct courier_id), 0)
                                                       as courier_load,
            median(delivery_minutes)                   as median_delivery_min,
            quantile_cont(delivery_minutes, 0.90)      as p90_delivery_min,
            avg(delivery_minutes)                      as mean_delivery_min,
            100.0 * sum(case when delivery_minutes > 1440 then 1 else 0 end)
                  / count(*)                           as delay_rate_1d,
            100.0 * sum(case when delivery_minutes > 180  then 1 else 0 end)
                  / count(*)                           as delay_rate_3h
        from delivery
        group by city, region_id, ds, accept_hour
        having count(*) >= 5
        order by city, region_id, ds, accept_hour
        """
    ).df()


def build_all_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    package_metrics = build_package_metrics(df)
    zone_hour_metrics = build_zone_hour_metrics(package_metrics)
    return package_metrics, zone_hour_metrics


def build_delivery_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    City-level delay rates at 3-hour, 1-day, and 2-day thresholds.
    Input df must have delivery_minutes > 0 (use load_lade_parquet output).
    Jilin rows are included but flagged via is_pilot so callers can split.
    """
    con = duckdb.connect(database=":memory:")
    con.register("delivery", df)
    return con.sql(
        """
        select
            city,
            bool_or(is_pilot)                                                 as is_pilot,
            count(*)                                                           as total_orders,
            count(distinct courier_id)                                        as unique_couriers,
            median(delivery_minutes)                                          as median_delivery_min,
            quantile_cont(delivery_minutes, 0.75)                            as p75_delivery_min,
            quantile_cont(delivery_minutes, 0.90)                            as p90_delivery_min,
            avg(delivery_minutes)                                             as mean_delivery_min,
            sum(case when delivery_minutes > 180  then 1 else 0 end)         as delayed_3h,
            sum(case when delivery_minutes > 1440 then 1 else 0 end)         as delayed_1d,
            sum(case when delivery_minutes > 2880 then 1 else 0 end)         as delayed_2d,
            100.0 * sum(case when delivery_minutes > 180  then 1 else 0 end)
                  / count(*)                                                  as delay_rate_3h_pct,
            100.0 * sum(case when delivery_minutes > 1440 then 1 else 0 end)
                  / count(*)                                                  as delay_rate_1d_pct,
            100.0 * sum(case when delivery_minutes > 2880 then 1 else 0 end)
                  / count(*)                                                  as delay_rate_2d_pct
        from delivery
        group by city
        order by delay_rate_1d_pct desc
        """
    ).df()


def build_courier_workload_segments(
    df: pd.DataFrame,
    low_freq_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Per-courier daily average and delay rates.

    Couriers are segmented as:
    - low_freq  : avg daily orders < low_freq_threshold  (likely part-time / occasional)
    - regular   : avg daily orders >= low_freq_threshold

    Returns one row per courier with workload_segment label.
    """
    con = duckdb.connect(database=":memory:")
    con.register("delivery", df)
    return con.sql(
        f"""
        with daily as (
            select
                city,
                courier_id,
                ds,
                count(*) as daily_orders
            from delivery
            group by city, courier_id, ds
        ),
        courier_avg as (
            select
                city,
                courier_id,
                avg(daily_orders)   as avg_daily_orders,
                count(*)            as active_days,
                sum(daily_orders)   as total_orders
            from daily
            group by city, courier_id
        ),
        courier_delays as (
            select
                courier_id,
                sum(case when delivery_minutes > 1440 then 1 else 0 end) as delayed_1d,
                count(*)                                                  as n_orders
            from delivery
            group by courier_id
        )
        select
            a.city,
            a.courier_id,
            a.avg_daily_orders,
            a.active_days,
            a.total_orders,
            d.delayed_1d,
            100.0 * d.delayed_1d / d.n_orders                            as delay_rate_1d_pct,
            case
                when a.avg_daily_orders < {low_freq_threshold} then 'low_freq'
                else 'regular'
            end                                                           as workload_segment
        from courier_avg a
        join courier_delays d using (courier_id)
        order by a.city, a.avg_daily_orders
        """
    ).df()

