"""
insights.py
Rule-based + statistical insight / alert engine.

Generates ≥ 5 distinct insight types per user:
  1. Heart-rate spike alert
  2. Low SpO2 alert
  3. Low activity trend
  4. Sleep reduction pattern
  5. Personalised HR deviation (z-score anomaly)
  6. Resting-HR trend (deterioration)
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import List, Dict, Any


# ── thresholds ────────────────────────────────────────────────────────────────

HR_SPIKE_ABS       = 150   # bpm  – absolute threshold
HR_SPIKE_Z         = 3.0   # σ    – z-score above personal baseline
SPO2_LOW           = 93.0  # %
STEPS_LOW_DAILY    = 3000  # steps/day  (WHO: ≥ 8,000 recommended)
SLEEP_LOW_HRS      = 6.0   # hours/night
SLEEP_REDUCTION_D  = 5     # days window for trend check
ACTIVITY_WINDOW_D  = 7     # days for activity trend
RHR_TREND_D        = 7     # days for resting-HR trend


# ── helpers ───────────────────────────────────────────────────────────────────

def _trend_slope(series: pd.Series) -> float:
    """Return OLS slope of the series."""
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0
    x = np.arange(len(clean))
    return float(np.polyfit(x, clean, 1)[0])


def severity(level: str) -> int:
    return {"info": 1, "warning": 2, "critical": 3}.get(level, 1)


# ── insight generators ────────────────────────────────────────────────────────

def hr_spike_alerts(hourly: pd.DataFrame) -> List[Dict]:
    alerts = []
    for uid, grp in hourly.groupby("user_id"):
        spikes = grp[
            (grp["max_hr"] >= HR_SPIKE_ABS) |
            (grp["hr_z"].abs() >= HR_SPIKE_Z)
        ]
        for _, row in spikes.iterrows():
            alerts.append({
                "user_id":   uid,
                "type":      "hr_spike",
                "severity":  "critical" if row["max_hr"] >= 160 else "warning",
                "date":      str(row["date"]),
                "hour":      int(row["hour"]),
                "value":     row["max_hr"],
                "hr_z":      row.get("hr_z"),
                "message":   (
                    f"Heart-rate spike detected: {row['max_hr']:.0f} bpm "
                    f"(z={row.get('hr_z', 'n/a')}) at {row['hour']:02d}:00 on {row['date']}."
                ),
            })
    return alerts


def low_spo2_alerts(daily: pd.DataFrame) -> List[Dict]:
    alerts = []
    low = daily[daily["min_spo2"] < SPO2_LOW]
    for _, row in low.iterrows():
        alerts.append({
            "user_id":  row["user_id"],
            "type":     "low_spo2",
            "severity": "critical" if row["min_spo2"] < 90 else "warning",
            "date":     str(row["date"]),
            "value":    row["min_spo2"],
            "message":  (
                f"Low blood oxygen: {row['min_spo2']:.1f}% on {row['date']}. "
                "Values below 95% may require medical attention."
            ),
        })
    return alerts


def low_activity_alerts(daily: pd.DataFrame) -> List[Dict]:
    alerts = []
    for uid, grp in daily.groupby("user_id"):
        grp = grp.sort_values("date").copy()
        # Rolling 7-day average
        grp["step_7d"] = grp["total_steps"].rolling(ACTIVITY_WINDOW_D, min_periods=3).mean()
        low_days = grp[grp["step_7d"] < STEPS_LOW_DAILY].tail(3)
        for _, row in low_days.iterrows():
            alerts.append({
                "user_id": uid,
                "type":    "low_activity",
                "severity": "warning",
                "date":    str(row["date"]),
                "value":   round(row["step_7d"], 0),
                "message": (
                    f"7-day average steps ({row['step_7d']:.0f}/day) below "
                    f"{STEPS_LOW_DAILY:,}. Consider increasing daily movement."
                ),
            })
    return alerts


def sleep_reduction_alerts(daily: pd.DataFrame) -> List[Dict]:
    alerts = []
    for uid, grp in daily.groupby("user_id"):
        grp = grp.sort_values("date").copy()
        grp["sleep_5d"] = grp["sleep_hrs"].rolling(SLEEP_REDUCTION_D, min_periods=3).mean()
        slope = _trend_slope(grp["sleep_hrs"].tail(10))
        low = grp[grp["sleep_5d"] < SLEEP_LOW_HRS].tail(2)
        for _, row in low.iterrows():
            alerts.append({
                "user_id": uid,
                "type":    "sleep_reduction",
                "severity": "warning",
                "date":    str(row["date"]),
                "value":   round(row["sleep_5d"], 2),
                "trend_slope": round(slope, 4),
                "message": (
                    f"Average sleep over last {SLEEP_REDUCTION_D} days is "
                    f"{row['sleep_5d']:.1f} hrs — below recommended 6–8 hrs. "
                    f"Trend slope: {slope:+.3f} hrs/day."
                ),
            })
    return alerts


def resting_hr_trend_alerts(daily: pd.DataFrame) -> List[Dict]:
    """Flag if resting HR is rising steadily (≥ 1 bpm/day over 7 days)."""
    alerts = []
    for uid, grp in daily.groupby("user_id"):
        grp = grp.sort_values("date")
        recent = grp.tail(RHR_TREND_D)
        slope = _trend_slope(recent["avg_hr"])
        if slope >= 1.0:
            alerts.append({
                "user_id": uid,
                "type":    "rhr_rising_trend",
                "severity": "warning",
                "date":    str(recent["date"].iloc[-1]),
                "trend_slope": round(slope, 3),
                "message": (
                    f"Resting heart rate increasing by {slope:.1f} bpm/day "
                    f"over the past {RHR_TREND_D} days. "
                    "Could indicate stress, overtraining, or illness."
                ),
            })
    return alerts


# ── summary builder ───────────────────────────────────────────────────────────

def build_summary(daily: pd.DataFrame) -> Dict[str, Any]:
    summaries = {}
    for uid, grp in daily.groupby("user_id"):
        latest = grp.sort_values("date").iloc[-1]
        summaries[uid] = {
            "user_id":          uid,
            "last_updated":     str(latest["date"]),
            "avg_hr_today":     round(float(latest["avg_hr"]), 1),
            "avg_spo2_today":   round(float(latest["avg_spo2"]), 2),
            "total_steps_today": int(latest["total_steps"]),
            "sleep_hrs_last_night": float(latest["sleep_hrs"]),
            "hr_baseline":      round(float(latest["hr_baseline"]), 1) if not pd.isna(latest["hr_baseline"]) else None,
            "7d_avg_steps":     int(grp.tail(7)["total_steps"].mean()),
            "7d_avg_sleep_hrs": round(grp.tail(7)["sleep_hrs"].mean(), 2),
        }
    return summaries


# ── public API ─────────────────────────────────────────────────────────────────

def generate_all_insights(hourly: pd.DataFrame, daily: pd.DataFrame) -> Dict:
    alerts = (
        hr_spike_alerts(hourly) +
        low_spo2_alerts(daily) +
        low_activity_alerts(daily) +
        sleep_reduction_alerts(daily) +
        resting_hr_trend_alerts(daily)
    )
    # Sort: critical first, then by date desc
    alerts.sort(key=lambda a: (-severity(a["severity"]), a.get("date", "")), reverse=False)
    alerts.sort(key=lambda a: severity(a["severity"]), reverse=True)

    summary = build_summary(daily)
    return {"summary": summary, "alerts": alerts}


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    hourly = pd.read_csv("processed/hourly.csv", parse_dates=["date"])
    daily  = pd.read_csv("processed/daily.csv",  parse_dates=["date"])
    result = generate_all_insights(hourly, daily)
    print(json.dumps(result["summary"], indent=2, default=str))
    print(f"\nTotal alerts: {len(result['alerts'])}")
    for a in result["alerts"][:5]:
        print(f"  [{a['severity'].upper()}] {a['message']}")
