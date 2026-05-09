"""
main.py  –  Stealthera Health Insights API
Run with:  uvicorn main:app --reload
"""

from __future__ import annotations

import os, sys, json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(ROOT))

from generate_data import generate_user_data   # noqa: E402
from process_data  import run_pipeline         # noqa: E402
from insights      import generate_all_insights  # noqa: E402

# ── bootstrap data if missing ──────────────────────────────────────────────────
DATA_DIR = ROOT
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

def _bootstrap():
    if not (RAW_DIR / "all_users.csv").exists():
        print("Bootstrapping data …")
        RAW_DIR.mkdir(exist_ok=True)
        import pandas as pd
        users = ["user_001", "user_002", "user_003"]
        frames = []
        for uid in users:
            df = generate_user_data(uid, days=30)
            df.to_csv(RAW_DIR / f"{uid}.csv", index=False)
            frames.append(df)
        pd.concat(frames).to_csv(RAW_DIR / "all_users.csv", index=False)
    if not (PROC_DIR / "daily.csv").exists():
        print("Processing data …")
        orig_dir = os.getcwd()
        os.chdir(DATA_DIR)
        run_pipeline()
        os.chdir(orig_dir)

_bootstrap()

# ── load processed data once ────────────────────────────────────────────────────
_hourly = pd.read_csv(PROC_DIR / "hourly.csv", parse_dates=["date"])
_daily  = pd.read_csv(PROC_DIR / "daily.csv",  parse_dates=["date"])
_insights = generate_all_insights(_hourly, _daily)

USERS = sorted(_daily["user_id"].unique().tolist())

# ── app setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stealthera Health Insights API",
    description="Wearable sensor data processing and health alert system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── response models ────────────────────────────────────────────────────────────

class UserSummary(BaseModel):
    user_id: str
    last_updated: str
    avg_hr_today: float
    avg_spo2_today: float
    total_steps_today: int
    sleep_hrs_last_night: float
    hr_baseline: Optional[float]
    seven_day_avg_steps: int
    seven_day_avg_sleep_hrs: float

class Alert(BaseModel):
    user_id: str
    type: str
    severity: str
    date: str
    message: str
    value: Optional[float] = None
    trend_slope: Optional[float] = None

class DailyRow(BaseModel):
    date: str
    avg_hr: float
    max_hr: float
    min_spo2: float
    avg_spo2: float
    total_steps: int
    sleep_hrs: float


# ── helper ─────────────────────────────────────────────────────────────────────

def _require_user(user_id: str):
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found. Available: {USERS}")


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Stealthera Health Insights API",
        "version": "1.0.0",
        "endpoints": ["/summary", "/alerts", "/user/{user_id}", "/user/{user_id}/daily", "/users"],
    }


@app.get("/users", tags=["meta"])
def list_users():
    """List all available user IDs."""
    return {"users": USERS}


@app.get("/summary", tags=["insights"])
def get_summary(user_id: Optional[str] = Query(None, description="Filter by user ID")):
    """
    Overall health summary. Returns the most recent snapshot for each user
    (or a specific user if `user_id` is supplied).
    """
    summaries = _insights["summary"]
    if user_id:
        _require_user(user_id)
        return summaries[user_id]
    return {"users": list(summaries.values())}


@app.get("/alerts", tags=["insights"])
def get_alerts(
    user_id:  Optional[str] = Query(None),
    severity: Optional[str] = Query(None, description="Filter: critical | warning | info"),
    type:     Optional[str] = Query(None, description="Filter by alert type"),
    limit:    int           = Query(50, ge=1, le=500),
):
    """
    All generated health alerts, sorted by severity (critical first).
    Supports optional filters on user_id, severity, and alert type.
    """
    alerts = _insights["alerts"]
    if user_id:
        _require_user(user_id)
        alerts = [a for a in alerts if a["user_id"] == user_id]
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    if type:
        alerts = [a for a in alerts if a["type"] == type]
    return {
        "total": len(alerts),
        "alerts": alerts[:limit],
    }


@app.get("/user/{user_id}", tags=["user"])
def get_user(user_id: str):
    """
    Full profile for a single user: summary + all their alerts.
    """
    _require_user(user_id)
    summary = _insights["summary"].get(user_id, {})
    alerts  = [a for a in _insights["alerts"] if a["user_id"] == user_id]
    return {
        "user_id": user_id,
        "summary": summary,
        "alert_count": len(alerts),
        "alerts": alerts,
    }


@app.get("/user/{user_id}/daily", tags=["user"])
def get_user_daily(
    user_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of recent days to return"),
):
    """
    Daily aggregated metrics for a user (HR, SpO2, steps, sleep).
    """
    _require_user(user_id)
    grp = (
        _daily[_daily["user_id"] == user_id]
              .sort_values("date", ascending=False)
              .head(days)
    )
    rows = []
    for _, r in grp.iterrows():
        rows.append({
            "date":        str(r["date"].date()),
            "avg_hr":      round(float(r["avg_hr"]), 1),
            "max_hr":      round(float(r["max_hr"]), 1),
            "min_spo2":    round(float(r["min_spo2"]), 1),
            "avg_spo2":    round(float(r["avg_spo2"]), 2),
            "total_steps": int(r["total_steps"]),
            "sleep_hrs":   round(float(r["sleep_hrs"]), 2),
        })
    return {"user_id": user_id, "days": len(rows), "data": rows}


@app.get("/user/{user_id}/trend", tags=["user"])
def get_user_trend(user_id: str):
    """
    7-day and 30-day trend analysis for key vitals.
    """
    _require_user(user_id)
    grp = _daily[_daily["user_id"] == user_id].sort_values("date")

    def _slope(col, n):
        import numpy as np
        s = grp[col].tail(n).dropna()
        if len(s) < 2:
            return None
        x = range(len(s))
        return round(float(np.polyfit(list(x), list(s), 1)[0]), 4)

    return {
        "user_id": user_id,
        "trends": {
            "heart_rate": {
                "7d_slope":  _slope("avg_hr", 7),
                "30d_slope": _slope("avg_hr", 30),
            },
            "steps": {
                "7d_slope":  _slope("total_steps", 7),
                "30d_slope": _slope("total_steps", 30),
            },
            "sleep_hrs": {
                "7d_slope":  _slope("sleep_hrs", 7),
                "30d_slope": _slope("sleep_hrs", 30),
            },
            "spo2": {
                "7d_slope":  _slope("avg_spo2", 7),
                "30d_slope": _slope("avg_spo2", 30),
            },
        },
    }
