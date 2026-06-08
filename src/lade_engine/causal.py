from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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

