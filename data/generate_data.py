"""
generate_data.py
Simulates smartwatch/wearable sensor data for multiple users.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

np.random.seed(42)


def generate_user_data(user_id: str, days: int = 30) -> pd.DataFrame:
    """Simulate wearable data for a single user over `days` days."""
    records = []
    base_time = datetime(2024, 5, 1, 0, 0, 0)

    # Per-user baseline variation
    hr_base   = np.random.randint(62, 80)
    spo2_base = np.random.uniform(96.0, 99.0)
    step_rate = np.random.uniform(0.08, 0.18)   # steps per minute during waking hours

    for minute in range(days * 24 * 60):
        ts   = base_time + timedelta(minutes=minute)
        hour = ts.hour

        # Sleep window: 22:00–06:00
        is_sleep = hour >= 22 or hour < 6

        # Heart rate
        if is_sleep:
            hr = hr_base - 12 + np.random.normal(0, 2)
        else:
            activity_boost = 10 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0
            hr = hr_base + activity_boost + np.random.normal(0, 4)

        # Occasional spikes (1 % chance)
        if np.random.random() < 0.01:
            hr += np.random.uniform(30, 60)

        hr = max(40, min(200, hr))

        # SpO2
        spo2 = spo2_base + np.random.normal(0, 0.4)
        if np.random.random() < 0.005:      # rare drops
            spo2 -= np.random.uniform(3, 8)
        spo2 = max(85.0, min(100.0, spo2))

        # Steps (0 during sleep)
        if is_sleep:
            steps = 0
        else:
            steps = int(np.random.poisson(step_rate * 60))

        # Introduce missing values (~2 %)
        if np.random.random() < 0.02:
            hr   = np.nan
        if np.random.random() < 0.02:
            spo2 = np.nan

        records.append({
            "user_id":   user_id,
            "timestamp": ts.isoformat(),
            "heart_rate": round(hr, 1) if not np.isnan(hr) else None,
            "spo2":       round(spo2, 1) if not np.isnan(spo2) else None,
            "steps":      steps,
            "is_sleep":   int(is_sleep),
        })

    return pd.DataFrame(records)


def main():
    users = ["user_001", "user_002", "user_003"]
    os.makedirs("raw", exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    all_frames = []
    for uid in users:
        print(f"Generating data for {uid} …")
        df = generate_user_data(uid, days=30)
        df.to_csv(f"raw/{uid}.csv", index=False)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv("raw/all_users.csv", index=False)
    print(f"Raw data saved — {len(combined):,} rows total.")


if __name__ == "__main__":
    main()
