from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from lade_engine.config import PATHS
from lade_engine.pipeline import run_pipeline


st.set_page_config(page_title="Last-Mile Incentive Targeting", layout="wide")


@st.cache_data
def load_outputs(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = Path(output_dir)
    targeting = pd.read_csv(path / "incentive_targets.csv")
    zone_hour = pd.read_csv(path / "zone_hour_metrics.csv")
    summary = json.loads((path / "executive_summary.json").read_text(encoding="utf-8"))
    return targeting, zone_hour, summary


st.title("Last-Mile Supply-Demand & Incentive Targeting Engine")

output_dir = st.sidebar.text_input("Output directory", str(PATHS.processed_data))
if st.sidebar.button("Generate demo outputs"):
    run_pipeline(output_dir=Path(output_dir), demo=True)
    st.cache_data.clear()

try:
    targets, zone_hour_metrics, executive = load_outputs(output_dir)
except FileNotFoundError:
    st.info("No processed outputs found. Use the sidebar button to generate deterministic demo outputs.")
    st.stop()

cities = sorted(targets["city"].unique())
selected_cities = st.sidebar.multiselect("City", cities, default=cities)
priority = st.sidebar.multiselect(
    "Priority", ["incentivize", "watch", "monitor"], default=["incentivize", "watch"]
)
filtered = targets[targets["city"].isin(selected_cities) & targets["priority"].isin(priority)]

targeting = executive["targeting"]
did = executive["difference_in_differences"]
elasticity = executive["supply_elasticity"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Zone-hours reviewed", f"{targeting['zone_hours_reviewed']:,}")
col2.metric("Targeted share", f"{targeting['targeting_share']:.1%}")
col3.metric("DiD latency impact", f"{did['did_dispatch_latency_min']:.2f} min")
col4.metric("Supply elasticity", f"{elasticity['elasticity']:.3f}")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Incentive Priority Queue")
    columns = [
        "city",
        "region_id",
        "ds",
        "hour",
        "priority",
        "target_score",
        "demand_packages",
        "supply_density",
        "dispatch_latency_min",
        "sla_hit_rate",
        "expected_latency_gain_min",
    ]
    st.dataframe(filtered[columns].head(250), width="stretch", hide_index=True)

with right:
    st.subheader("Dispatch Latency by Supply Density")
    chart_data = zone_hour_metrics[zone_hour_metrics["city"].isin(selected_cities)]
    st.scatter_chart(
        chart_data,
        x="supply_density",
        y="dispatch_latency_min",
        color="city",
        size="demand_packages",
        width="stretch",
    )

st.subheader("Hourly SLA Reliability")
sla = (
    zone_hour_metrics[zone_hour_metrics["city"].isin(selected_cities)]
    .groupby(["hour"], as_index=False)["sla_hit_rate"]
    .mean()
)
st.line_chart(sla, x="hour", y="sla_hit_rate", width="stretch")

st.subheader("A/B Pilot Design")
st.json(executive["ab_power_analysis"], expanded=False)
