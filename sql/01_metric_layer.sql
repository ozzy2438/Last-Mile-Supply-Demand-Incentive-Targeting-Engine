-- Core metric layer for LaDe-style package events.
-- Register the raw dataframe/table as `lade` before running.

with package_metrics as (
    select
        order_id,
        courier_id,
        city,
        region_id,
        aoi_id,
        date_trunc('hour', accept_time) as accept_hour,
        date_diff('second', accept_time, pickup_time) / 60.0 as dispatch_latency_min,
        date_diff('second', pickup_time, delivery_time) / 60.0 as delivery_duration_min,
        case when delivery_time <= promise_end_time then 1 else 0 end as sla_hit
    from lade
)
select
    city,
    region_id,
    accept_hour,
    count(*) as demand_packages,
    count(distinct courier_id) as supply_density,
    avg(dispatch_latency_min) as dispatch_latency_min,
    quantile_cont(dispatch_latency_min, 0.9) as dispatch_latency_p90_min,
    avg(delivery_duration_min) as delivery_duration_min,
    avg(sla_hit) as sla_hit_rate,
    count(*) * 1.0 / nullif(count(distinct courier_id), 0) as courier_load
from package_metrics
group by all
order by city, region_id, accept_hour;

