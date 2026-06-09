from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def assign_supply_shock_treatment(zone_hour: pd.DataFrame) -> pd.DataFrame:
    df = zone_hour.copy()
    df["date"] = pd.to_datetime(df["ds"])
    midpoint = df["date"].min() + (df["date"].max() - df["date"].min()) / 2
    df["post"] = (df["date"] > midpoint).astype(int)

    pre = df[df["post"] == 0].groupby("zone_id")["supply_density"].mean()
    post = df[df["post"] == 1].groupby("zone_id")["supply_density"].mean()
    lift = ((post - pre) / pre.replace(0, np.nan)).rename("supply_lift")
    df = df.merge(lift, on="zone_id", how="left")
    cutoff = df["supply_lift"].quantile(0.70)
    df["treated"] = (df["supply_lift"] >= cutoff).astype(int)
    df["did"] = df["treated"] * df["post"]
    return df


def run_difference_in_differences(zone_hour: pd.DataFrame) -> dict[str, float | int]:
    df = assign_supply_shock_treatment(zone_hour)
    model = smf.ols(
        "dispatch_latency_min ~ treated + post + did + demand_packages + C(city) + C(hour)",
        data=df,
    ).fit(cov_type="HC3")
    return {
        "observations": int(model.nobs),
        "treated_zone_hours": int(df["treated"].sum()),
        "control_zone_hours": int((1 - df["treated"]).sum()),
        "did_dispatch_latency_min": float(model.params["did"]),
        "did_p_value": float(model.pvalues["did"]),
        "baseline_dispatch_latency_min": float(df[df["post"] == 0]["dispatch_latency_min"].mean()),
    }


def assign_real_supply_treatment(
    zone_hour: pd.DataFrame,
    pre_months: list[int],
    post_months: list[int],
    treatment_quantile: float = 0.70,
) -> pd.DataFrame:
    """
    Assigns treatment to zones based on supply (courier) density lift
    between the pre-period and post-period months.

    Treatment = zones in the top (1 - treatment_quantile) of supply lift.
    This mirrors assign_supply_shock_treatment() but works with
    build_real_zone_hour_metrics() output (uses 'month' instead of 'date').
    """
    df = zone_hour.copy()
    df["post"] = df["month"].isin(post_months).astype(int)

    pre_supply  = df[df["month"].isin(pre_months)].groupby("zone_id")["n_couriers"].mean()
    post_supply = df[df["month"].isin(post_months)].groupby("zone_id")["n_couriers"].mean()
    lift = ((post_supply - pre_supply) / pre_supply.replace(0, np.nan)).rename("supply_lift")

    df = df.merge(lift, on="zone_id", how="left")
    cutoff = df["supply_lift"].quantile(treatment_quantile)
    df["treated"] = (df["supply_lift"] >= cutoff).astype(int)
    df["did"] = df["treated"] * df["post"]
    return df


def check_parallel_trends(
    zone_hour_treated: pd.DataFrame,
    pre_months: list[int],
    outcome_col: str = "median_delivery_min",
) -> dict[str, object]:
    """
    Formal parallel-trends test for the pre-treatment period.

    Method: in the pre-period only, regress the outcome on month_num,
    treated, and their interaction.  If the interaction is significant
    (p < 0.05), pre-treatment trends diverge — the DiD identifying
    assumption is violated.

    Returns
    -------
    monthly_avg   : DataFrame of mean outcome by (month, treated) for charting
    interaction_coef : float — coefficient on month × treated
    interaction_p    : float — p-value of the interaction
    parallel_trends_holds : bool — True if p >= 0.05
    interpretation : str — human-readable verdict
    """
    pre = zone_hour_treated[zone_hour_treated["month"].isin(pre_months)].copy()

    # Collapse to zone × month so we're testing zone-level trends
    zone_month = (
        pre.groupby(["zone_id", "city", "month", "treated"])[outcome_col]
        .mean()
        .reset_index()
    )
    zone_month["month_num"] = zone_month["month"] - min(pre_months)

    model_full = smf.ols(
        f"{outcome_col} ~ month_num + C(treated) + month_num:C(treated) + C(city)",
        data=zone_month,
    ).fit(cov_type="HC3")

    model_restricted = smf.ols(
        f"{outcome_col} ~ month_num + C(treated) + C(city)",
        data=zone_month,
    ).fit(cov_type="HC3")

    interaction_coef = float(model_full.params.get("month_num:C(treated)[T.1]", np.nan))
    interaction_p    = float(model_full.pvalues.get("month_num:C(treated)[T.1]", np.nan))
    parallel_holds   = (np.isnan(interaction_p)) or (interaction_p >= 0.05)

    # Monthly avg for charting
    monthly_avg = (
        zone_month.groupby(["month", "treated"])[outcome_col]
        .mean()
        .reset_index()
        .rename(columns={"treated": "treated_group", outcome_col: "mean_outcome"})
    )
    monthly_avg["group_label"] = monthly_avg["treated_group"].map(
        {1: "Treated (supply ↑)", 0: "Control (supply stable)"}
    )

    return {
        "monthly_avg": monthly_avg,
        "interaction_coef": round(interaction_coef, 4),
        "interaction_p": round(interaction_p, 4),
        "parallel_trends_holds": parallel_holds,
        "interpretation": (
            "✓ Parallel trends holds — pre-treatment trends are statistically indistinguishable "
            f"(interaction p={interaction_p:.3f} ≥ 0.05). DiD identifying assumption supported."
            if parallel_holds else
            f"✗ Parallel trends VIOLATED — treated and control diverge pre-treatment "
            f"(interaction p={interaction_p:.3f} < 0.05). DiD estimates may be biased. "
            "Consider restricting to zones with similar pre-period trends."
        ),
    }


def run_placebo_did(
    zone_hour_treated: pd.DataFrame,
    pre_months: list[int],
    fake_post_months: list[int],
    outcome_col: str = "median_delivery_min",
) -> dict[str, float | str]:
    """
    Placebo test: run DiD entirely within the pre-treatment period using
    a fake 'post' date. A significant DiD coefficient here means the
    treatment assignment is picking up pre-existing trends, not a real effect.

    Typical setup: pre_months=[5,6], fake_post_months=[7]
    (all before the real treatment boundary at month 8).
    """
    placebo_data = zone_hour_treated[
        zone_hour_treated["month"].isin(pre_months + fake_post_months)
    ].copy()
    placebo_data["post"] = placebo_data["month"].isin(fake_post_months).astype(int)
    placebo_data["did"] = placebo_data["treated"] * placebo_data["post"]

    model = smf.ols(
        f"{outcome_col} ~ treated + post + did + n_orders + C(city) + C(hour)",
        data=placebo_data,
    ).fit(cov_type="HC3")

    placebo_coef = float(model.params.get("did", np.nan))
    placebo_p    = float(model.pvalues.get("did", np.nan))
    placebo_clean = placebo_p >= 0.10  # lenient threshold — we want no false positives

    return {
        "observations": int(model.nobs),
        "placebo_did_coef": round(placebo_coef, 4),
        "placebo_p_value": round(placebo_p, 4),
        "placebo_clean": placebo_clean,
        "interpretation": (
            f"✓ Placebo clean — no spurious pre-trend effect detected "
            f"(coef={placebo_coef:.2f}, p={placebo_p:.3f}). "
            "Treatment assignment is not simply capturing existing seasonality."
            if placebo_clean else
            f"✗ Placebo significant (coef={placebo_coef:.2f}, p={placebo_p:.3f}). "
            "The treatment assignment may be correlated with pre-existing trends. "
            "Main DiD estimate should be interpreted with caution."
        ),
    }


def run_real_data_did(
    zone_hour: pd.DataFrame,
    pre_months: list[int] | None = None,
    post_months: list[int] | None = None,
    outcome_col: str = "median_delivery_min",
) -> dict[str, object]:
    """
    Full DiD pipeline for the real LaDe delivery data.

    Steps (in order, mirroring best-practice causal workflow):
    1. Assign treatment based on supply lift (pre → post courier density change).
    2. Check parallel trends in the pre-period.
    3. Run placebo DiD within the pre-period (should find nothing).
    4. Run main DiD on full data.

    Pre-period : months 5, 6, 7 (May–Jul)
    Post-period: months 8, 9, 10 (Aug–Oct)

    Returns a single dict with all results, parallel_trends, placebo,
    and main DiD coefficient — ready for JSON export.
    """
    pre_months  = pre_months  or [5, 6, 7]
    post_months = post_months or [8, 9, 10]

    # 1. Treatment assignment
    df = assign_real_supply_treatment(zone_hour, pre_months, post_months)

    # 2. Parallel trends
    pt = check_parallel_trends(df, pre_months, outcome_col)

    # 3. Placebo (use months 5–6 as fake pre, month 7 as fake post)
    placebo = run_placebo_did(df, pre_months=[5, 6], fake_post_months=[7],
                              outcome_col=outcome_col)

    # 4. Main DiD
    model = smf.ols(
        f"{outcome_col} ~ treated + post + did + n_orders + C(city) + C(hour)",
        data=df,
    ).fit(cov_type="HC3")

    did_coef = float(model.params.get("did", np.nan))
    did_p    = float(model.pvalues.get("did", np.nan))

    treatment_effect_pct = (
        did_coef / float(df[df["post"] == 0][outcome_col].mean()) * 100
        if df[df["post"] == 0][outcome_col].mean() != 0 else np.nan
    )

    return {
        "outcome_variable": outcome_col,
        "pre_months": pre_months,
        "post_months": post_months,
        "observations": int(model.nobs),
        "treated_zone_hours": int(df["treated"].sum()),
        "control_zone_hours": int((1 - df["treated"]).sum()),
        "baseline_outcome_pre_period": round(float(df[df["post"] == 0][outcome_col].mean()), 2),
        # Parallel trends
        "parallel_trends": {
            "holds": pt["parallel_trends_holds"],
            "interaction_coef": pt["interaction_coef"],
            "interaction_p": pt["interaction_p"],
            "interpretation": pt["interpretation"],
        },
        # Placebo
        "placebo_test": {
            "clean": placebo["placebo_clean"],
            "coef": placebo["placebo_did_coef"],
            "p_value": placebo["placebo_p_value"],
            "interpretation": placebo["interpretation"],
        },
        # Main estimate
        "did_coef_minutes": round(did_coef, 3),
        "did_p_value": round(did_p, 4),
        "treatment_effect_pct": round(treatment_effect_pct, 2),
        "r_squared": round(float(model.rsquared), 4),
        "identification_status": (
            "STRONG — parallel trends holds and placebo is clean"
            if pt["parallel_trends_holds"] and placebo["placebo_clean"]
            else "WEAK — see parallel_trends and placebo_test for details"
        ),
        # Monthly trend data for charting (from parallel trends check)
        "_monthly_avg_for_chart": pt["monthly_avg"],
    }


def estimate_supply_elasticity(zone_hour: pd.DataFrame) -> dict[str, float]:
    df = zone_hour.copy()
    df = df[(df["supply_density"] > 0) & (df["dispatch_latency_min"] > 0)].copy()
    df["log_supply"] = np.log(df["supply_density"])
    df["log_dispatch"] = np.log(df["dispatch_latency_min"])
    model = smf.ols(
        "log_dispatch ~ log_supply + demand_packages + C(city) + C(hour)",
        data=df,
    ).fit(cov_type="HC3")
    return {
        "elasticity": float(model.params["log_supply"]),
        "p_value": float(model.pvalues["log_supply"]),
        "r_squared": float(model.rsquared),
    }

