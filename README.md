# Last-Mile Supply-Demand & Incentive Targeting Engine

An end-to-end analytics project for last-mile marketplace operations. It turns LaDe-style package event data into courier supply metrics, causal estimates, A/B pilot sizing, and an incentive targeting dashboard.

## Project Pitch

Courier incentives should not be spread evenly across a city. This engine identifies the zone-hours where high demand, low courier supply, and elevated dispatch latency overlap, then estimates whether additional supply is likely to improve ETA reliability enough to justify intervention.

## What It Includes

- DuckDB SQL metric layer for supply density, demand, dispatch latency, delivery duration, SLA hit rate, and courier load.
- Difference-in-Differences analysis for supply-shock zones versus controls.
- Supply elasticity estimate to expose diminishing returns.
- A/B test power analysis for an incentive pilot.
- ROI-style targeting rule and prioritized zone-hour output.
- Streamlit dashboard for operations review.
- Deterministic demo data generator, so the project runs without private/raw LaDe files.

## Repository Structure

```text
src/lade_engine/      Python analytics package
dashboard/            Streamlit app
sql/                  DuckDB metric SQL
docs/                 Methodology and interview one-pager
tests/                End-to-end and unit tests
data/raw/             Put LaDe CSV files here
data/processed/       Generated pipeline outputs
reports/              Reserved for exported artifacts
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m lade_engine --demo
streamlit run dashboard/streamlit_app.py
```

The demo command writes:

- `data/processed/package_metrics.csv`
- `data/processed/zone_hour_metrics.csv`
- `data/processed/incentive_targets.csv`
- `data/processed/executive_summary.json`
- `data/processed/stakeholder_one_pager.md`

## Running With Real LaDe CSVs

Place LaDe-P and/or LaDe-D CSV files in `data/raw/`, then run:

```bash
python -m lade_engine --raw-dir data/raw --output-dir data/processed
```

Required columns:

```text
courier_id, accept_time, pickup_time, delivery_time, lng, lat, aoi_id, region_id, city, ds
```

Optional columns such as `accept_gps_time`, `promise_end_time`, `delivery_window_end`, and `order_id` are used when available. If no promise window exists, the pipeline creates a conservative 60-minute SLA fallback for reproducible analysis.

## Testing

```bash
pytest
```

## Decision Rule

Recommend incentives only when these conditions overlap:

1. Demand is high for the local city.
2. Supply density is low for the local city.
3. Dispatch latency is above the 70th percentile.
4. Demand/supply pressure is elevated.

This produces a focused operations queue instead of a broad, expensive incentive blanket.

