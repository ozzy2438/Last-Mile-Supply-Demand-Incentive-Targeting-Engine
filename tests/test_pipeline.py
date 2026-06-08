from __future__ import annotations

import json

from lade_engine.data import generate_demo_lade
from lade_engine.metrics import build_all_metrics
from lade_engine.pipeline import run_pipeline
from lade_engine.targeting import build_targeting_table


def test_demo_generator_has_lade_columns() -> None:
    df = generate_demo_lade(seed=7, days=3)
    assert {"courier_id", "accept_time", "pickup_time", "delivery_time", "city"}.issubset(df.columns)
    assert df["order_id"].is_unique
    assert df["city"].nunique() == 5


def test_metric_layer_builds_zone_hour_metrics() -> None:
    df = generate_demo_lade(seed=7, days=3)
    package_metrics, zone_hour = build_all_metrics(df)
    assert len(package_metrics) == len(df)
    assert zone_hour["supply_density"].min() > 0
    assert zone_hour["sla_hit_rate"].between(0, 1).all()
    assert zone_hour["dispatch_latency_min"].mean() > 0


def test_targeting_rule_selects_high_pressure_zone_hours() -> None:
    df = generate_demo_lade(seed=7, days=4)
    _, zone_hour = build_all_metrics(df)
    targets = build_targeting_table(zone_hour)
    selected = targets[targets["recommend_incentive"]]
    assert not selected.empty
    assert selected["target_score"].min() >= 0.75
    assert selected["dispatch_latency_min"].mean() >= targets["dispatch_latency_min"].median()


def test_pipeline_writes_end_to_end_outputs(tmp_path) -> None:
    result = run_pipeline(output_dir=tmp_path, demo=True)
    assert result["orders"] > 10_000
    assert (tmp_path / "package_metrics.csv").exists()
    assert (tmp_path / "zone_hour_metrics.csv").exists()
    assert (tmp_path / "incentive_targets.csv").exists()
    summary = json.loads((tmp_path / "executive_summary.json").read_text(encoding="utf-8"))
    assert "difference_in_differences" in summary
    assert "ab_power_analysis" in summary

