from __future__ import annotations

from pathlib import Path

from .causal import estimate_supply_elasticity, run_difference_in_differences
from .config import PATHS
from .data import generate_demo_lade, load_lade_csvs
from .metrics import build_all_metrics
from .power import sla_power_analysis
from .reporting import write_outputs
from .targeting import build_targeting_table, targeting_summary


def run_pipeline(raw_dir: Path | None = None, output_dir: Path | None = None, demo: bool = False) -> dict:
    raw_dir = raw_dir or PATHS.raw_data
    output_dir = output_dir or PATHS.processed_data
    df = generate_demo_lade() if demo else load_lade_csvs(raw_dir)
    package_metrics, zone_hour_metrics = build_all_metrics(df)
    did = run_difference_in_differences(zone_hour_metrics)
    elasticity = estimate_supply_elasticity(zone_hour_metrics)
    power = sla_power_analysis(zone_hour_metrics)
    targeting = build_targeting_table(zone_hour_metrics)
    targeting_stats = targeting_summary(targeting)
    write_outputs(
        output_dir,
        package_metrics,
        zone_hour_metrics,
        targeting,
        did,
        elasticity,
        power,
        targeting_stats,
    )
    return {
        "orders": len(package_metrics),
        "zone_hours": len(zone_hour_metrics),
        "output_dir": str(output_dir),
        "difference_in_differences": did,
        "supply_elasticity": elasticity,
        "ab_power_analysis": power,
        "targeting": targeting_stats,
    }

