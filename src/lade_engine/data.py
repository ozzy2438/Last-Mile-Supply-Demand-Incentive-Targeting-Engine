from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PATHS
from .schema import REQUIRED_COLUMNS


def validate_lade_frame(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required LaDe columns: {', '.join(missing)}")


def load_lade_csvs(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = [pd.read_csv(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    validate_lade_frame(df)
    return normalize_lade_frame(df)


def normalize_lade_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "order_id" not in df.columns:
        df["order_id"] = np.arange(len(df)) + 1

    for column in [
        "accept_time",
        "accept_gps_time",
        "pickup_time",
        "delivery_time",
        "promise_start_time",
        "promise_end_time",
        "delivery_window_start",
        "delivery_window_end",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if "promise_end_time" not in df.columns:
        fallback = "delivery_window_end" if "delivery_window_end" in df.columns else None
        df["promise_end_time"] = df[fallback] if fallback else df["accept_time"] + pd.Timedelta(minutes=60)

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.date.astype(str)
    df["hour"] = df["accept_time"].dt.hour
    df["zone_id"] = df["city"].astype(str) + "_" + df["region_id"].astype(str)
    return df


def load_lade_parquet(path: Path | None = None) -> pd.DataFrame:
    """
    Load and clean the real LaDe delivery dataset (Hugging Face, ~190 MB parquet).

    Schema differences vs the synthetic pipeline:
    - Timestamps are 'MM-DD HH:MM:SS' strings (no year); we prepend 2023.
    - No pickup_time column; end-to-end = accept → delivery.
    - Jilin is a pilot city (~20% of Yantai's daily volume); flagged but kept.

    Rows with delivery_minutes <= 0 are dropped:
    - Zero: courier scanned accept and delivery in the same minute ("instant scan" artifact).
    - Negative: three records with corrupted timestamps.
    """
    path = path or (PATHS.real_data / "delivery_all.parquet")
    df = pd.read_parquet(path)

    for col in ("accept_time", "delivery_time"):
        df[col] = pd.to_datetime("2023-" + df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    df["delivery_minutes"] = (df["delivery_time"] - df["accept_time"]).dt.total_seconds() / 60
    df = df[df["delivery_minutes"] > 0].reset_index(drop=True)

    df["is_pilot"] = df["city"] == "Jilin"
    df["accept_hour"] = df["accept_time"].dt.hour

    return df


def generate_demo_lade(seed: int = 42, days: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cities = ["Shanghai", "Hangzhou", "Chongqing", "Jilin", "Yantai"]
    regions = list(range(1, 9))
    start = pd.Timestamp("2025-04-01")
    rows: list[dict[str, object]] = []
    order_id = 1

    for day in range(days):
        date = start + pd.Timedelta(days=day)
        for city in cities:
            for region in regions:
                base_supply = rng.integers(7, 18)
                treated_zone = city in {"Hangzhou", "Shanghai"} and region in {2, 3, 4}
                shock = day >= days // 2 and treated_zone
                for hour in range(7, 23):
                    peak = hour in {11, 12, 13, 18, 19, 20}
                    demand = int(rng.poisson(18 + 9 * peak + 4 * (region in {2, 3, 4})))
                    supply = int(base_supply + rng.normal(0, 2) + (5 if shock else 0) - (2 if peak else 0))
                    supply = max(3, supply)
                    couriers = [f"{city[:2]}-{region}-{idx}" for idx in range(supply)]
                    pressure = demand / supply
                    for _ in range(demand):
                        accept_time = date + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                        courier_id = rng.choice(couriers)
                        dispatch = max(2.0, rng.normal(9 + 3.4 * pressure - 0.55 * supply, 3.0))
                        duration = max(5.0, rng.normal(23 + 2.1 * pressure + 1.5 * peak, 6.0))
                        pickup_time = accept_time + pd.Timedelta(minutes=float(dispatch))
                        delivery_time = pickup_time + pd.Timedelta(minutes=float(duration))
                        promise_end = accept_time + pd.Timedelta(minutes=45 + int(8 * peak))
                        rows.append(
                            {
                                "order_id": order_id,
                                "courier_id": courier_id,
                                "accept_time": accept_time,
                                "accept_gps_time": accept_time + pd.Timedelta(minutes=1),
                                "pickup_time": pickup_time,
                                "delivery_time": delivery_time,
                                "promise_end_time": promise_end,
                                "lng": 120.0 + rng.normal(0, 0.08),
                                "lat": 30.0 + rng.normal(0, 0.08),
                                "aoi_id": f"AOI-{region}-{int(rng.integers(1, 5))}",
                                "region_id": region,
                                "city": city,
                                "ds": date.date().isoformat(),
                            }
                        )
                        order_id += 1

    return normalize_lade_frame(pd.DataFrame(rows))

