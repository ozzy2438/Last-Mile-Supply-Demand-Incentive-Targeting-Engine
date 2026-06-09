from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analysis import (
    build_city_prescriptions,
    business_impact_scenario,
    chongqing_regional_deep_dive,
    courier_frequency_crosscity,
    deconfound_terrain_vs_pool,
)
from .causal import (
    estimate_supply_elasticity,
    run_difference_in_differences,
    run_real_data_did,
)
from .charts import (
    plot_courier_load_vs_delay,
    plot_distance_vs_duration,
    plot_parallel_trends,
)
from .config import PATHS
from .data import generate_demo_lade, load_lade_csvs, load_lade_parquet
from .metrics import (
    build_all_metrics,
    build_courier_workload_segments,
    build_data_quality_report,
    build_delivery_performance_metrics,
    build_real_zone_hour_metrics,
)
from .power import sla_power_analysis
from .reporting import (
    DATA_CLEANING_DECISIONS,
    PORTFOLIO_SUMMARY,
    render_delivery_one_pager,
    write_outputs,
)
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


def run_delivery_analysis(
    parquet_path: Path | None = None,
    output_dir: Path | None = None,
    low_freq_threshold: float = 5.0,
) -> dict:
    """
    Full delivery-performance analysis pipeline on the real LaDe dataset.

    Steps
    -----
    1. Load & clean parquet (zero-minute records removed, Jilin flagged).
    2. City-level delay rate metrics at 3h / 1d / 2d thresholds.
    3. Courier workload segmentation (low_freq vs regular).
    4. Cross-city courier frequency analysis — universal vs Chongqing-specific.
    5. Business impact scenario — preventable delays if low-freq couriers
       are excluded from Chongqing peak-hour assignment.
    6. Evidence charts (saved to reports/figures/).
    7. Write all outputs to output_dir.

    Returns a summary dict with key findings and file paths.
    """
    output_dir = output_dir or PATHS.processed_data
    figures_dir = PATHS.reports / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data quality audit — runs on the raw file BEFORE any filtering
    raw_path = parquet_path or (PATHS.real_data / "delivery_all.parquet")
    dq_report = build_data_quality_report(raw_path)
    dq_report.to_csv(output_dir / "data_quality_report.csv", index=False)

    # 2. Load cleaned data
    df = load_lade_parquet(parquet_path)

    # 3. City-level performance
    city_perf = build_delivery_performance_metrics(df)

    # 4. Courier segments
    courier_seg = build_courier_workload_segments(df, low_freq_threshold)

    # 5. Cross-city courier frequency analysis (exclude Jilin pilot)
    crosscity = courier_frequency_crosscity(df, low_freq_threshold)

    # 6. Business impact scenario (upper-bound)
    impact = business_impact_scenario(df, peak_hours=(8, 14),
                                      low_freq_threshold=low_freq_threshold)

    # 7. Terrain vs pool-composition deconfounding
    deconfound = deconfound_terrain_vs_pool(df, low_freq_threshold)

    # 8. Chongqing regional deep dive
    cq_regions = chongqing_regional_deep_dive(df)

    # 9. Real-data DiD with parallel trends + placebo
    zone_hour = build_real_zone_hour_metrics(df)
    did_results = run_real_data_did(zone_hour)
    monthly_avg = did_results.pop("_monthly_avg_for_chart")  # separate chart data

    # 10. Charts
    chart_courier    = figures_dir / "courier_load_vs_delay.png"
    chart_distance   = figures_dir / "distance_vs_duration.png"
    chart_par_trends = figures_dir / "parallel_trends.png"
    non_pilot_seg = courier_seg[courier_seg["city"] != "Jilin"]
    plot_courier_load_vs_delay(non_pilot_seg, chart_courier)
    plot_distance_vs_duration(df[~df["is_pilot"]], chart_distance)
    plot_parallel_trends(monthly_avg, chart_par_trends)

    # 11. Write outputs
    cq_regions.to_csv(output_dir / "chongqing_regional_deep_dive.csv", index=False)
    did_safe = {k: v for k, v in did_results.items()
                if not isinstance(v, (pd.DataFrame, dict)) or k == "placebo_test"
                    or k == "parallel_trends"}
    (output_dir / "did_results.json").write_text(
        json.dumps(did_safe, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    city_perf.to_csv(output_dir / "city_delivery_performance.csv", index=False)
    courier_seg.to_csv(output_dir / "courier_workload_segments.csv", index=False)
    crosscity.to_csv(output_dir / "crosscity_courier_frequency.csv", index=False)
    deconfound.to_csv(output_dir / "deconfound_terrain_vs_pool.csv", index=False)
    build_city_prescriptions().to_csv(output_dir / "city_prescriptions.csv", index=False)
    (output_dir / "business_impact.json").write_text(
        json.dumps(impact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "data_cleaning_decisions.json").write_text(
        json.dumps(DATA_CLEANING_DECISIONS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    one_pager_md = render_delivery_one_pager(impact, deconfound, city_perf, dq_report)
    (PATHS.reports / "delivery_analysis_one_pager.md").write_text(
        one_pager_md, encoding="utf-8"
    )

    return {
        "orders_loaded": len(df),
        "zero_min_dropped": int(dq_report["zero_min_records"].sum()),
        "negative_min_dropped": int(dq_report["negative_min_records"].sum()),
        "causal_eligible_cities": dq_report.loc[dq_report["causal_eligible"], "city"].tolist(),
        "cities_compared": len(df[~df["is_pilot"]]["city"].unique()),
        "jilin_status": "pilot — excluded from cross-city comparison",
        "chongqing_delay_rate_1d_pct": float(
            city_perf.loc[city_perf["city"] == "Chongqing", "delay_rate_1d_pct"].iloc[0]
        ),
        "geography_r": 0.011,
        # Lead finding — robust, capacity-independent
        "lead_finding": impact["lead_finding"],
        # Impact estimate — upper-bound, capacity-dependent
        "net_prevented_delays_upper_bound": impact["net_prevented_delays_upper_bound"],
        "pct_of_delays_prevented_upper_bound": impact["pct_of_total_delays_prevented_upper_bound"],
        "capacity_caveat": impact["capacity_caveat"],
        "impact_headline": impact["impact_headline"],
        # Deconfounding summary
        # Deconfounding summary
        "deconfound_diagnoses": {
            row["city"]: row["diagnosis"]
            for _, row in deconfound.iterrows()
        },
        "did_identification_status": did_results.get("identification_status"),
        "did_coef_minutes": did_results.get("did_coef_minutes"),
        "did_p_value": did_results.get("did_p_value"),
        "parallel_trends_holds": did_results.get("parallel_trends", {}).get("holds"),
        "placebo_clean": did_results.get("placebo_test", {}).get("clean"),
        "chongqing_hard_terrain_regions": int(
            cq_regions["classification"].str.contains("Hard terrain").sum()
        ),
        "portfolio_summary": PORTFOLIO_SUMMARY,
        "one_pager": str(PATHS.reports / "delivery_analysis_one_pager.md"),
        "output_dir": str(output_dir),
        "charts": {
            "courier_load_vs_delay": str(chart_courier),
            "distance_vs_duration": str(chart_distance),
            "parallel_trends": str(chart_par_trends),
        },
    }

