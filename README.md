# Last-Mile Supply-Demand & Incentive Targeting Engine

End-to-end analytics project on the public [LaDe dataset](https://huggingface.co/datasets/Cainiao-AI/LaDe) — 4.5 million last-mile delivery records across five Chinese cities. The project identifies why delivery delays concentrate in certain cities and couriers, disproves a geography explanation, and produces city-specific operational recommendations.

[dashboard_preview.webm](https://github.com/user-attachments/assets/0de7ead8-bd4c-4ba5-9515-258084fae8c9)



---

## Key Findings

**Low-frequency couriers (averaging fewer than 5 deliveries per day) carry 11% of orders but produce 57% of next-day delays** — a 5x overrepresentation. This is not explained by route difficulty.

Delivery distance and delivery time are almost uncorrelated (r = 0.011 in Chongqing). The root cause is courier selection, not city layout.

The mechanism differs by city:

| City | Problem | Recommended fix |
| ---- | ------- | --------------- |
| **Chongqing** | Terrain × capability mismatch (63 min/km for regular couriers) | Route peak-hour orders to couriers with >5 avg daily deliveries |
| **Yantai** | Thin market — 42% of active couriers are part-time | Recruit full-time couriers; cap part-time volume at peak hours |
| Shanghai, Hangzhou | Amplification ≤ 2.5× — no significant problem | Monitor only |

**Causal evidence is honest:** parallel trends holds (interaction p = 0.43) but the placebo DiD fails, meaning treatment assignment partially correlates with pre-existing trends. The DiD coefficient is -1.17 min (p = 0.43, not significant). Identification status: **WEAK**. This actually strengthens the recommendation to run a controlled A/B test before scaling any change.

---

## Data

**Source:** [LaDe-D (delivery) dataset](https://huggingface.co/datasets/Cainiao-AI/LaDe), Cainiao/Alibaba logistics research release.

**Coverage:** ~4.5M delivery records, May–October 2023, five cities: Hangzhou, Shanghai, Chongqing, Yantai, Jilin.

**Format:** Parquet (~190 MB), download as `real_data/delivery_all.parquet`.

**Jilin note:** 88.6% date coverage (below the 90% causal-eligibility threshold). Treated as a separate pilot-city analysis — excluded from cross-city DiD.

To download:

```python
from datasets import load_dataset
ds = load_dataset("Cainiao-AI/LaDe", "delivery_all")
ds["train"].to_parquet("real_data/delivery_all.parquet")
```

---

## Repository Structure

```text
src/lade_engine/
    config.py         Project paths
    data.py           Data loaders (demo + real LaDe parquet)
    metrics.py        DuckDB metric layer (zone-hour, delivery performance, data quality)
    analysis.py       Cross-city courier analysis, deconfounding, Chongqing deep dive
    causal.py         Parallel trends check, placebo DiD, main DiD
    charts.py         Evidence charts (courier load, distance vs duration, parallel trends)
    reporting.py      One-pager generator and data cleaning documentation
    pipeline.py       run_delivery_analysis() — orchestrates all steps
    targeting.py      Incentive targeting rule (demo pipeline)
    power.py          A/B test power analysis
data/processed/       Generated outputs (CSV, JSON)
reports/
    delivery_analysis_one_pager.md   Executive decision brief
    figures/                         Evidence charts (PNG)
real_data/            Place delivery_all.parquet here
tests/                End-to-end and unit tests
```

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Run the real-data pipeline** (requires `real_data/delivery_all.parquet`):

```python
from lade_engine.pipeline import run_delivery_analysis
results = run_delivery_analysis()
print(results["lead_finding"])
print(results["did_identification_status"])
```

**Run the demo pipeline** (no data required, uses synthetic data):

```bash
python -m lade_engine --demo
```

---

## Output Files

After running `run_delivery_analysis()`:

| File | Contents |
| ---- | -------- |
| `data/processed/data_quality_report.csv` | Per-city raw record counts, coverage, zero-min and GPS null rates, causal eligibility |
| `data/processed/data_cleaning_decisions.json` | Documented cleaning rules with counts and justification |
| `data/processed/city_delivery_performance.csv` | City-level delay rates at 3h / 1d / 2d thresholds |
| `data/processed/courier_workload_segments.csv` | Per-courier avg daily orders, delay rates, low_freq / regular label |
| `data/processed/crosscity_courier_frequency.csv` | Cross-city low_freq vs regular comparison |
| `data/processed/deconfound_terrain_vs_pool.csv` | Pool composition + terrain proxy per city, with diagnosis and prescription |
| `data/processed/city_prescriptions.csv` | Static per-city root cause and action plan |
| `data/processed/business_impact.json` | Upper-bound preventable delay estimate with capacity caveat |
| `data/processed/chongqing_regional_deep_dive.csv` | Per-region terrain + delay classification within Chongqing |
| `data/processed/did_results.json` | Parallel trends test, placebo DiD, main DiD coefficient + identification status |
| `reports/delivery_analysis_one_pager.md` | Executive decision brief (non-technical language) |
| `reports/figures/courier_load_vs_delay.png` | Scatter: courier daily frequency vs 1-day delay rate |
| `reports/figures/distance_vs_duration.png` | Scatter: route distance vs delivery time (geography null result) |
| `reports/figures/parallel_trends.png` | Monthly treated vs control trends (DiD pre-period check) |

---

## Methodology Notes

- **Terrain proxy:** median minutes per km for regular couriers (controls for courier skill variation)
- **Counterfactual framing:** all "preventable delay" estimates are upper-bound — assumes full-time couriers have spare capacity to absorb reassigned volume
- **Causal eligibility:** city must have ≥90% date coverage AND avg daily volume ≥5% of cross-city median
- **DiD treatment assignment:** top 30% of zones by courier density lift from pre-period (May–Jul) to post-period (Aug–Oct)
- **Placebo test:** DiD within the pre-period only (fake treatment at month 7); should yield null

---

## Testing

```bash
pytest
```

---

## Limitations

1. Observational data — descriptive evidence is strong, but a randomised pilot is needed to rule out confounders and confirm effect sizes.
2. Five Chinese cities, six-month window. External validity to other markets or seasons is unverified.
3. Jilin excluded from causal analysis due to incomplete date coverage.
