from __future__ import annotations

from math import ceil

import pandas as pd
from statsmodels.stats.power import TTestIndPower


def sla_power_analysis(
    zone_hour: pd.DataFrame,
    minimum_detectable_effect: float = 0.03,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float | int]:
    sla_std = float(zone_hour["sla_hit_rate"].std(ddof=1))
    if sla_std == 0:
        sla_std = 0.01
    effect_size = minimum_detectable_effect / sla_std
    required_per_arm = TTestIndPower().solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1,
        alternative="two-sided",
    )
    daily_zone_hours = zone_hour.groupby("ds").size().mean()
    return {
        "primary_metric": "sla_hit_rate",
        "minimum_detectable_effect_pp": minimum_detectable_effect * 100,
        "alpha": alpha,
        "power": power,
        "required_zone_hours_per_arm": int(ceil(required_per_arm)),
        "estimated_calendar_days": int(ceil((2 * required_per_arm) / daily_zone_hours)),
    }

