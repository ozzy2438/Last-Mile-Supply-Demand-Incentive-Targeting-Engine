# Methodology

## Objective

This project estimates where incremental courier supply is most likely to improve last-mile ETA reliability. LaDe does not include an explicit incentive or earnings field, so the project treats courier supply density as the operational proxy for the outcome incentives are meant to change.

## Metric Layer

The core grain is `city x region_id x ds x hour`.

- Supply density: distinct active `courier_id` values per zone-hour.
- Demand: package count per zone-hour.
- Dispatch latency: minutes from `accept_time` to `pickup_time`.
- Delivery duration: minutes from `pickup_time` to `delivery_time`.
- SLA hit rate: share of orders delivered before `promise_end_time`.
- Courier load: packages per distinct courier in the zone-hour.

## Quasi-Experimental Design

The pipeline identifies treated zones as the top 30% of zones by post-period supply lift. Controls are zones without that relative supply shock. The default model estimates:

```text
dispatch_latency ~ treated + post + treated*post + demand + city fixed effects + hour fixed effects
```

The interaction coefficient is interpreted as the relative change in dispatch latency for supply-shock zones after controlling for time-of-day, city, and demand.

## A/B Pilot

The recommended pilot randomizes eligible low-supply, high-demand zone-hours into incentive and holdout arms. The primary metric is SLA hit rate. The included power calculation estimates required zone-hours per arm for a configurable minimum detectable effect.

## Targeting Rule

The production-style rule prioritizes zone-hours where four conditions overlap:

- high demand,
- low supply density,
- high dispatch latency,
- elevated demand/supply pressure.

This intentionally avoids broad incentives in cells where the marginal operational gain is likely weak.

