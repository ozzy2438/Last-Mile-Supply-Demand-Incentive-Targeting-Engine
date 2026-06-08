from __future__ import annotations

import pandas as pd


def build_targeting_table(zone_hour: pd.DataFrame) -> pd.DataFrame:
    df = zone_hour.copy()
    latency_cutoff = df["dispatch_latency_min"].quantile(0.70)
    ratio_cutoff = df["demand_supply_ratio"].quantile(0.70)
    df["target_score"] = (
        0.35 * df["high_demand_flag"]
        + 0.25 * df["low_supply_flag"]
        + 0.25 * (df["dispatch_latency_min"] >= latency_cutoff).astype(int)
        + 0.15 * (df["demand_supply_ratio"] >= ratio_cutoff).astype(int)
    )
    df["recommend_incentive"] = df["target_score"] >= 0.75
    df["expected_latency_gain_min"] = (
        (df["dispatch_latency_min"] - df["dispatch_latency_min"].median()).clip(lower=0) * 0.45
    )
    df["priority"] = pd.cut(
        df["target_score"],
        bins=[-0.01, 0.35, 0.70, 1.01],
        labels=["monitor", "watch", "incentivize"],
    ).astype(str)
    return df.sort_values(
        ["recommend_incentive", "target_score", "expected_latency_gain_min"],
        ascending=[False, False, False],
    )


def targeting_summary(targeting: pd.DataFrame) -> dict[str, float | int]:
    selected = targeting[targeting["recommend_incentive"]]
    total = len(targeting)
    return {
        "zone_hours_reviewed": int(total),
        "zone_hours_targeted": int(len(selected)),
        "targeting_share": float(len(selected) / total if total else 0),
        "avg_expected_latency_gain_min": float(selected["expected_latency_gain_min"].mean() or 0),
        "avg_selected_sla_hit_rate": float(selected["sla_hit_rate"].mean() or 0),
    }

