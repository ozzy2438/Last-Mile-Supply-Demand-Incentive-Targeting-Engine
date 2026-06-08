from __future__ import annotations

from dataclasses import dataclass


REQUIRED_COLUMNS = {
    "courier_id",
    "accept_time",
    "pickup_time",
    "delivery_time",
    "lng",
    "lat",
    "aoi_id",
    "region_id",
    "city",
    "ds",
}

OPTIONAL_COLUMNS = {
    "accept_gps_time",
    "promise_start_time",
    "promise_end_time",
    "delivery_window_start",
    "delivery_window_end",
    "order_id",
}


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    grain: str
    formula: str
    decision_use: str


METRIC_DEFINITIONS = [
    MetricDefinition(
        "supply_density",
        "city x region x date x hour",
        "count(distinct courier_id)",
        "Proxy for courier supply available to absorb demand.",
    ),
    MetricDefinition(
        "dispatch_latency_min",
        "package",
        "(pickup_time - accept_time) in minutes",
        "Primary operational ETA lever.",
    ),
    MetricDefinition(
        "delivery_duration_min",
        "package",
        "(delivery_time - pickup_time) in minutes",
        "Captures route execution and batching pressure.",
    ),
    MetricDefinition(
        "sla_hit",
        "package",
        "delivery_time <= promise_end_time when a promise window exists",
        "Reliability metric for marketplace health.",
    ),
    MetricDefinition(
        "courier_load",
        "courier x hour",
        "active packages per courier-hour",
        "Utilization and batching pressure proxy.",
    ),
]

