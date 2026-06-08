from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import METRIC_DEFINITIONS


def write_outputs(
    output_dir: Path,
    package_metrics: pd.DataFrame,
    zone_hour_metrics: pd.DataFrame,
    targeting: pd.DataFrame,
    did: dict[str, float | int],
    elasticity: dict[str, float],
    power: dict[str, float | int],
    targeting_stats: dict[str, float | int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_metrics.to_csv(output_dir / "package_metrics.csv", index=False)
    zone_hour_metrics.to_csv(output_dir / "zone_hour_metrics.csv", index=False)
    targeting.to_csv(output_dir / "incentive_targets.csv", index=False)
    summary = {
        "metrics": [definition.__dict__ for definition in METRIC_DEFINITIONS],
        "difference_in_differences": did,
        "supply_elasticity": elasticity,
        "ab_power_analysis": power,
        "targeting": targeting_stats,
    }
    (output_dir / "executive_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "stakeholder_one_pager.md").write_text(
        render_one_pager(did, elasticity, power, targeting_stats),
        encoding="utf-8",
    )


def render_one_pager(
    did: dict[str, float | int],
    elasticity: dict[str, float],
    power: dict[str, float | int],
    targeting_stats: dict[str, float | int],
) -> str:
    did_delta = did["did_dispatch_latency_min"]
    direction = "reduced" if did_delta < 0 else "increased"
    return f"""# Last-Mile Incentive Targeting One-Pager

## Decision
Target courier incentives only where high demand, low courier supply, and elevated dispatch latency overlap. This keeps incentives focused on zone-hours where extra supply is most likely to improve ETA reliability.

## Evidence
- Difference-in-differences estimate: treated supply-shock zones {direction} dispatch latency by {abs(did_delta):.2f} minutes versus controls (p={did["did_p_value"]:.3f}).
- Supply elasticity: a 1% increase in courier supply is associated with a {elasticity["elasticity"]:.3f}% change in dispatch latency after controlling for city, hour, and demand.
- Targeting rule selects {targeting_stats["zone_hours_targeted"]:,} of {targeting_stats["zone_hours_reviewed"]:,} zone-hours ({targeting_stats["targeting_share"]:.1%}) with an average expected latency gain of {targeting_stats["avg_expected_latency_gain_min"]:.2f} minutes.

## Pilot Design
Primary metric: {power["primary_metric"]}. To detect a {power["minimum_detectable_effect_pp"]:.1f} percentage point SLA lift with {power["power"]:.0%} power, run approximately {power["required_zone_hours_per_arm"]:,} zone-hours per arm, or {power["estimated_calendar_days"]} calendar days at the current sample rate.

## Operating Rule
Incentivize a region-hour when demand is high, supply density is in the local bottom quartile, dispatch latency is above the 70th percentile, and demand/supply pressure is elevated. Monitor all other cells unless they degrade for two consecutive peak windows.
"""

