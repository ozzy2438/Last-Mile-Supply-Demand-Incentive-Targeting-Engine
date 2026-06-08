# Interview One-Pager

## Pitch

I built a last-mile supply-demand engine that quantifies how courier supply density affects dispatch latency and SLA reliability, then turns that causal estimate into an incentive targeting rule for marketplace operations.

## Why This Matters

Last-mile teams do not want to spend incentives everywhere. They need to know where extra courier supply changes ETA reliability enough to justify intervention. This project maps raw package events into zone-hour marketplace metrics, estimates supply impact with a quasi-experiment, and produces a target list for operations.

## What I Built

- DuckDB metric layer for courier supply, demand, dispatch latency, delivery duration, SLA hit rate, and courier load.
- Difference-in-differences model comparing supply-shock zones against control zones.
- Supply elasticity model to show diminishing returns.
- A/B pilot power analysis for an incentive experiment.
- Streamlit dashboard and executive one-pager for stakeholder handoff.

## Resume Bullet

Quantified the causal impact of courier supply on dispatch latency and SLA reliability using LaDe-style last-mile package data; built DuckDB metric definitions, Difference-in-Differences analysis, A/B power analysis, and a Streamlit incentive-targeting dashboard that prioritizes high-demand, low-supply, high-latency zone-hours.

