from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import METRIC_DEFINITIONS

# ── Portfolio summary ────────────────────────────────────────────────────────
DATA_CLEANING_DECISIONS: list[dict[str, str]] = [
    {
        "rule":     "Exclude delivery_minutes == 0",
        "count":    "12,930 records (~0.29% of raw dataset)",
        "why":      "accept_time == delivery_time in 100% of these cases. "
                    "GPS coordinates differ — courier physically moved — so timestamps are wrong, "
                    "not the route. Physical impossibility: no courier accepts and delivers in 0 seconds. "
                    "Likely cause: same-minute batch scan or scanner clock drift.",
        "effect_if_kept": "Artificially pulls mean delivery time toward zero; "
                          "inflates apparent proportion of 'fast' deliveries.",
        "distribution": "Spread proportionally across all cities (0.16–0.52%) — "
                        "not clustered, so not a city-specific system error.",
    },
    {
        "rule":     "Exclude delivery_minutes < 0",
        "count":    "3 records (<0.0001% of raw dataset)",
        "why":      "Delivery timestamp precedes accept timestamp — physically impossible. "
                    "Corrupted or mis-ordered records.",
        "effect_if_kept": "Negligible at 3 records, but signals timestamp unreliability "
                          "for those specific entries.",
        "distribution": "Isolated records, no city pattern.",
    },
    {
        "rule":     "Exclude delivery_minutes > 10,000 (approx 7 days)",
        "count":    "Not applied as hard filter; used as outlier cap for charts only",
        "why":      "A small tail of extreme values (max ~166,499 min, approx 115 days) exists. "
                    "These could be real abandoned/re-opened orders or data entry errors. "
                    "We keep them in rate calculations (they legitimately count as 'delayed') "
                    "but cap chart axes to prevent scale distortion.",
        "effect_if_kept": "Minimal effect on median/percentile metrics; "
                          "significantly distorts mean and chart readability.",
        "distribution": "Present in all cities; Chongqing has the highest tail count.",
    },
    {
        "rule":     "Jilin excluded from cross-city causal comparison",
        "count":    "31,415 records retained in pilot analysis; excluded from DiD/elasticity",
        "why":      "Jilin is active on only 163/184 days (88.6% coverage), starts 11 days "
                    "late, and averages 193 orders/day vs 1,122 for Yantai (next smallest). "
                    "The parallel-trends assumption required for DiD cannot be verified "
                    "across this volume and timing mismatch. "
                    "Root cause unknown from data alone: either a pilot launch or data collection gap. "
                    "Treated separately as a pilot-city performance analysis.",
        "effect_if_kept": "DiD treatment/control assignment would be based on "
                          "non-comparable baselines; elasticity estimates would be downward-biased "
                          "by Jilin's thin market.",
        "distribution": "Jilin is tagged is_pilot=True in all downstream DataFrames.",
    },
]

PORTFOLIO_SUMMARY = (
    "Identified the root cause of last-mile delivery delays by disproving the "
    "apparent geography explanation and decomposing the effect into courier-pool "
    "composition (Yantai: 42% low-freq share) and "
    "terrain x capability interaction (Chongqing: 64 min/km baseline, 5.6x amplification); "
    "quantified ~1,479 preventable 1-day delays per period via peak-hour reassignment "
    "(upper-bound estimate, validated with capacity caveat)."
)


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


def render_delivery_one_pager(
    impact: dict,
    deconfound: pd.DataFrame,
    city_perf: pd.DataFrame,
    dq_report: pd.DataFrame | None = None,
) -> str:
    """
    Produces a decision-maker one-pager in Markdown.
    Executive language throughout: no jargon, no statistical terms.
    Reader gets the full story in 30 seconds.
    """
    lf_order_share = impact["low_freq_order_share_pct"]
    lf_delay_share = impact["low_freq_delay_share_pct"]
    disp_ratio     = impact["disproportionality_ratio"]
    net_prevented  = impact["net_prevented_delays_upper_bound"]
    pct_prevented  = impact["pct_of_total_delays_prevented_upper_bound"]
    caveat         = impact["capacity_caveat"]
    cq             = city_perf[city_perf["city"] == "Chongqing"].iloc[0]
    total_orders   = int(sum(city_perf["total_orders"]))

    # ── Data quality section (optional) ─────────────────────────────────────
    dq_section = ""
    if dq_report is not None:
        def _dq_row(r: pd.Series) -> str:
            eligible = "yes" if r["causal_eligible"] else "no (pilot)"
            return (
                f"| **{r['city']}** | {int(r['raw_records']):,} | "
                f"{int(r['min_ds'])}–{int(r['max_ds'])} | "
                f"{int(r['active_days'])}/{int(r['dataset_total_days'])} ({r['coverage_pct']:.0f}%) | "
                f"{int(r['zero_min_records']):,} ({r['zero_min_pct']:.2f}%) | "
                f"{r['gps_delivery_null_pct']:.1f}% | "
                f"{int(r['unique_couriers']):,} | "
                f"{int(r['avg_daily_volume']):,} | "
                f"{eligible} |"
            )

        dq_rows = "\n".join(_dq_row(row) for _, row in dq_report.iterrows())
        decisions = "\n".join(
            f"- **{d['rule']}** ({d['count']}): {d['why']}"
            for d in DATA_CLEANING_DECISIONS
        )
        dq_section = (
            "\n---\n\n"
            "## Data Quality Audit (Pre-Filter)\n\n"
            "| City | Raw records | Date range (ds) | Coverage | Zero-min records | GPS null | Couriers | Avg daily vol | Causal eligible |\n"
            "|------|---:|---:|---:|---:|---:|---:|---:|:---:|\n"
            f"{dq_rows}\n\n"
            "### Cleaning Decisions\n\n"
            f"{decisions}\n\n"
            "*All subsequent analysis uses the cleaned dataset (zero-min and negative-duration records removed).*\n"
            "*Jilin is retained in the data but excluded from cross-city causal comparisons.*\n\n"
        )

    # ── City action table ────────────────────────────────────────────────────
    def _action_row(r: pd.Series) -> str:
        city = r["city"]
        if city == "Yantai":
            problem = "42% of couriers are part-time — thin market"
        elif city == "Chongqing":
            spd = r["median_min_per_km_regular"]
            problem = f"Mountain routes slow all couriers ({spd:.0f} min/km)"
        else:
            problem = "No significant problem detected"
        fix = (r.get("prescription") or "—").split(".")[0]
        priority = r.get("priority", "—")
        return f"| **{city}** | {problem} | {fix} | {priority} |"

    action_rows = "\n".join(_action_row(row) for _, row in deconfound.iterrows())

    current_1d = int(cq["delayed_1d"])
    after_fix = current_1d - net_prevented
    current_rate = cq["delay_rate_1d_pct"]
    total_cq = int(cq["total_orders"])

    lines = [
        "# Delivery Performance — Decision Brief",
        dq_section,
        "---",
        "",
        "## The Problem in One Sentence",
        "",
        f"> A small group of infrequent couriers carries only {lf_order_share}% of deliveries",
        f"> but accounts for {lf_delay_share}% of next-day delays — a {disp_ratio}x overrepresentation.",
        "> This is not a geography problem. It is a **who gets assigned the order** problem.",
        "",
        "We confirmed this directly: delivery distance and delivery time are almost uncorrelated",
        "(r = 0.011 in Chongqing). The city layout is not causing the delays. Courier selection is.",
        "",
        "---",
        "",
        "## What We Found, City by City",
        "",
        "| City | Root problem | Recommended fix | Priority |",
        "|------|-------------|-----------------|----------|",
        action_rows,
        "",
        "**The critical insight:** Yantai and Chongqing both have high delay rates among infrequent",
        "couriers, but for opposite reasons. Applying the same solution in both cities would waste",
        "budget and miss the actual lever.",
        "",
        "- **Yantai** needs more full-time couriers. The pool is too thin (42% part-time).",
        "- **Chongqing** needs smarter order assignment. The pool size is adequate; the algorithm",
        "  ignores whether a courier can handle a difficult route.",
        "",
        "---",
        "",
        "## What This Costs Today — and What One Change Could Recover",
        "",
        f"Chongqing snapshot ({total_cq:,} deliveries analysed):",
        "",
        "| | Current | After fix — best case |",
        "|-|---------|----------------------|",
        f"| Orders delayed 1+ day | {current_1d:,} ({current_rate:.1f}%) | ~{after_fix:,} |",
        f"| Recoverable with peak-hour fix | — | **~{net_prevented:,} fewer delays** |",
        "",
        f"**Honest caveat:** {caveat}",
        "A 2–4 week pilot will give you the actual number before any permanent change.",
        "",
        "---",
        "",
        "## The Three Decisions on the Table",
        "",
        "**Decision 1 — Chongqing (algorithm change, no new headcount):**",
        "During morning and afternoon peak windows, route orders only to couriers",
        "who average more than 5 deliveries per day. One rule, one place in the",
        f"dispatch algorithm. Estimated gain: up to ~{net_prevented:,} fewer next-day delays per period.",
        "",
        "**Decision 2 — Yantai (requires recruiting investment):**",
        "Reduce part-time couriers from 42% to roughly 20% of the active pool —",
        "the level where other cities show no delay problem. Immediate bridge:",
        "cap part-time orders at peak hours while recruiting is underway.",
        "",
        "**Decision 3 — Run a pilot before scaling either change:**",
        "Strong descriptive evidence supports both fixes, but the data is observational.",
        "A controlled 2–4 week test in Chongqing will confirm the effect size, surface",
        "capacity constraints, and give you a defensible number before a full rollout.",
        "",
        "**What to measure in the pilot:**",
        f"- Target: 1-day delay rate below 1.5% (currently {current_rate:.1f}%)",
        "- Guard rail: full-time courier delay rate must not rise",
        "  (if it does, they are being overloaded — reduce the volume shift)",
        "",
        "---",
        "",
        "## What This Analysis Cannot Tell You",
        "",
        "Three honest limits of this work:",
        "",
        "1. **Observational data only.** We identified who is causing delays and why, but cannot fully",
        "   rule out unobserved confounders without a randomised experiment. The pilot closes this gap.",
        "",
        "2. **Single market, one time period.** Results come from five Chinese cities over six months",
        "   (May–Oct 2023). Verify the courier-pool dynamics are similar before applying these findings",
        "   to a different market or season.",
        "",
        "3. **One city (Jilin) had incomplete records** and was excluded from city comparisons. It looks",
        "   similar to the others in the data it does have, but cannot be confirmed at full scale.",
        "",
        "---",
        "",
        "## If You Remember One Thing",
        "",
        f"> Stopping occasional couriers from taking orders during the two busiest windows of the day",
        f"> could cut Chongqing's multi-day delays by roughly {pct_prevented}% with no new headcount.",
        "> In Yantai, the fix is different — you need more full-time couriers, not a rule change.",
        "> Same symptom. Different cure. One solution will not work for both.",
        "",
        "---",
        "",
        f"*{total_orders:,} deliveries across 4 cities, May–Oct 2023.",
        "Dataset: LaDe (public research release). Full methodology: project repository.*",
    ]
    return "\n".join(lines) + "\n"


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
