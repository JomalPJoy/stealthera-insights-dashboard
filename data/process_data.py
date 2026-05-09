"""
process_data.py
Cleans raw wearable data and engineers features used by the insight engine.
"""

import pandas as pd
import numpy as np
import json
import os


# ── helpers ──────────────────────────────────────────────────────────────────

def load_raw(path: str = "raw/all_users.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Sort by user + time.
    2. Forward-fill missing HR / SpO2 within the same user window (≤ 5 min gap).
    3. Smooth HR with a 5-minute rolling median to suppress motion artefacts.
    4. Clip extreme values to physiologically plausible ranges.
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    for uid, grp in df.groupby("user_id"):
        idx = grp.index

        # Forward-fill (limit = 5 samples = ~5 min)
        df.loc[idx, "heart_rate"] = (
            grp["heart_rate"].ffill(limit=5).bfill(limit=5)
        )
        df.loc[idx, "spo2"] = (
            grp["spo2"].ffill(limit=5).bfill(limit=5)
        )

        # Rolling-median smoothing (5-min window)
        df.loc[idx, "heart_rate"] = (
            df.loc[idx, "heart_rate"]
              .rolling(5, min_periods=1, center=True)
              .median()
        )

    # Clip
    df["heart_rate"] = df["heart_rate"].clip(30, 220)
    df["spo2"]       = df["spo2"].clip(70, 100)
    df["steps"]      = df["steps"].clip(0, 300)        # max ~300 steps/min

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hourly aggregates and personalised baselines."""
    df = df.copy()
    df["date"]    = df["timestamp"].dt.date
    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday   # 0=Mon … 6=Sun

    # ── hourly rollup ──────────────────────────────────────────────────────
    hourly = (
        df.groupby(["user_id", "date", "hour"])
          .agg(
              avg_hr    = ("heart_rate", "mean"),
              max_hr    = ("heart_rate", "max"),
              min_spo2  = ("spo2", "min"),
              avg_spo2  = ("spo2", "mean"),
              total_steps = ("steps", "sum"),
              sleep_mins  = ("is_sleep", "sum"),
          )
          .reset_index()
    )
    hourly["avg_hr"]   = hourly["avg_hr"].round(1)
    hourly["avg_spo2"] = hourly["avg_spo2"].round(2)

    # ── personalised baselines (rolling 7-day, per user) ──────────────────
    hourly = hourly.sort_values(["user_id", "date", "hour"])
    hourly["hr_baseline"]   = (
        hourly.groupby("user_id")["avg_hr"]
              .transform(lambda s: s.rolling(7 * 24, min_periods=24).mean())
    )
    hourly["hr_std_baseline"] = (
        hourly.groupby("user_id")["avg_hr"]
              .transform(lambda s: s.rolling(7 * 24, min_periods=24).std())
    )
    hourly["spo2_baseline"] = (
        hourly.groupby("user_id")["avg_spo2"]
              .transform(lambda s: s.rolling(7 * 24, min_periods=24).mean())
    )
    hourly["step_baseline"] = (
        hourly.groupby("user_id")["total_steps"]
              .transform(lambda s: s.rolling(7 * 24, min_periods=24).mean())
    )

    # Z-score deviation from rolling baseline
    hourly["hr_z"] = (
        (hourly["avg_hr"] - hourly["hr_baseline"])
        / hourly["hr_std_baseline"].replace(0, np.nan)
    ).round(2)

    return df, hourly


def daily_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily summaries per user."""
    daily = (
        hourly.groupby(["user_id", "date"])
              .agg(
                  avg_hr       = ("avg_hr", "mean"),
                  max_hr       = ("max_hr", "max"),
                  min_spo2     = ("min_spo2", "min"),
                  avg_spo2     = ("avg_spo2", "mean"),
                  total_steps  = ("total_steps", "sum"),
                  sleep_mins   = ("sleep_mins", "sum"),
                  hr_baseline  = ("hr_baseline", "last"),
              )
              .reset_index()
    )
    daily["avg_hr"]   = daily["avg_hr"].round(1)
    daily["avg_spo2"] = daily["avg_spo2"].round(2)
    daily["sleep_hrs"] = (daily["sleep_mins"] / 60).round(2)
    return daily


def save(df_min, hourly, daily, out_dir="processed"):
    os.makedirs(out_dir, exist_ok=True)
    df_min.to_csv(f"{out_dir}/minute_clean.csv", index=False)
    hourly.to_csv(f"{out_dir}/hourly.csv", index=False)
    daily.to_csv(f"{out_dir}/daily.csv", index=False)
    print(f"Processed files written to '{out_dir}/'")


def run_pipeline(raw_path="raw/all_users.csv"):
    df = load_raw(raw_path)
    print(f"Loaded {len(df):,} raw rows.")
    df = clean(df)
    df, hourly = add_features(df)
    daily = daily_summary(hourly)
    save(df, hourly, daily)
    return df, hourly, daily


if __name__ == "__main__":
    run_pipeline()
