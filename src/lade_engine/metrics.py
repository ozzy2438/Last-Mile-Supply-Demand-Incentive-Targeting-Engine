from __future__ import annotations

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


def build_all_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    package_metrics = build_package_metrics(df)
    zone_hour_metrics = build_zone_hour_metrics(package_metrics)
    return package_metrics, zone_hour_metrics

