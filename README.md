# Stealthera — Health Insights Platform

> AI/ML Internship Assignment  
> Wearable sensor data processing, insight generation & REST API

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Part 1 — Data Processing](#part-1--data-processing)
5. [Part 2 — Insight / Alert Logic](#part-2--insight--alert-logic)
6. [Part 3 — API Layer](#part-3--api-layer)
7. [Part 4 — Dashboard](#part-4--dashboard)
8. [Part 5 — Bonus Features](#part-5--bonus-features)
9. [Assumptions](#assumptions)
10. [Scalability & Future Improvements](#scalability--future-improvements)
11. [Sample API Responses](#sample-api-responses)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STEALTHERA SYSTEM                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐   │
│  │ Data Layer   │───▶│ Insight Eng. │───▶│  FastAPI    │   │
│  │              │    │              │    │  REST API   │   │
│  │ generate_    │    │ insights.py  │    │  main.py    │   │
│  │ data.py      │    │              │    │             │   │
│  │ process_     │    │  • HR spike  │    │  /summary   │   │
│  │ data.py      │    │  • Low SpO2  │    │  /alerts    │   │
│  │              │    │  • Activity  │    │  /user/{id} │   │
│  │ raw/ ───────▶│    │  • Sleep     │    │  /trend     │   │
│  │ processed/   │    │  • RHR trend │    │             │   │
│  └──────────────┘    └──────────────┘    └─────────────┘   │
│                                                    │        │
│                                          ┌─────────▼──────┐ │
│                                          │   Dashboard    │ │
│                                          │  index.html    │ │
│                                          │  (SVG charts)  │ │
│                                          └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. `generate_data.py` → simulates 30 days × 1-min intervals × 3 users (~129,600 rows)
2. `process_data.py`  → cleans, smooths, and engineers features → `processed/`
3. `insights.py`      → generates alerts and summaries
4. `main.py`          → serves everything via FastAPI
5. `dashboard/index.html` → visual layer (connects to API, works offline too)

---

## Quick Start

### Prerequisites
- Python **3.13.x** (project tested in Python 3.13 environment)
- `pip` (bundled with Python)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate and process data
```bash
cd data
python generate_data.py    # creates raw/ CSVs
python process_data.py     # creates processed/ CSVs
cd ..
```

### 3. Start the API
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the API docs
```
http://localhost:8000/docs         # Swagger UI
http://localhost:8000/redoc        # ReDoc
```

### 5. Open the dashboard
```
Open dashboard/index.html in a browser
(connects to localhost:8000 automatically; falls back to demo data)
```

---

## Project Structure

```
stealthera/
├── data/
│   ├── generate_data.py      # Simulates wearable sensor data
│   ├── process_data.py       # Cleaning, smoothing, feature engineering
│   ├── insights.py           # Alert & insight generation engine
│   ├── raw/                  # Raw per-user CSVs (auto-generated)
│   └── processed/            # Cleaned minute/hourly/daily CSVs
│
├── api/
│   └── main.py               # FastAPI application
│
├── dashboard/
│   └── index.html            # Single-file visual dashboard
│
├── docs/
│   └── sample_api_responses.json
│
├── requirements.txt
└── README.md
```

---

## Part 1 — Data Processing

### Dataset Schema
| Field       | Type    | Description                      |
|-------------|---------|----------------------------------|
| timestamp   | ISO8601 | Minute-level timestamps          |
| heart_rate  | float   | BPM (may be NaN ~2%)            |
| spo2        | float   | % blood oxygen (may be NaN ~2%) |
| steps       | int     | Steps per minute                 |
| is_sleep    | 0/1     | Derived from hour (22:00–06:00) |

### Processing Pipeline (`process_data.py`)

**Step 1 — Missing value handling**
- Forward-fill up to 5 consecutive missing values (≈ 5-min gap)
- Back-fill to handle leading NaNs
- Rationale: short sensor dropouts are best estimated from adjacent readings

**Step 2 — Noise smoothing**
- 5-minute rolling **median** applied to heart rate
- Median chosen over mean to resist motion-artefact spikes

**Step 3 — Physiological clipping**
- HR: 30–220 bpm
- SpO₂: 70–100 %
- Steps: 0–300 steps/min

**Step 4 — Feature engineering**
- Hourly aggregates (avg HR, max HR, min SpO₂, total steps, sleep minutes)
- **Personalised 7-day rolling baseline** (mean + std) per user
- **Z-score deviation** `hr_z = (avg_hr − baseline_mean) / baseline_std`
- Daily summaries (steps, sleep hours, resting HR)

---

## Part 2 — Insight / Alert Logic

Five distinct alert types are generated in `insights.py`:

### 1. Heart-Rate Spike (`hr_spike`)
- Triggers when hourly `max_hr ≥ 150 bpm` **OR** `hr_z ≥ 3.0 σ`
- Severity: **critical** if ≥ 160 bpm, otherwise **warning**
- Combines absolute threshold with personalised z-score anomaly detection

### 2. Low SpO₂ (`low_spo2`)
- Triggers when daily `min_spo2 < 93 %`
- Severity: **critical** if < 90 %, otherwise **warning**
- Clinically: SpO₂ < 95 % may indicate hypoxemia

### 3. Low Activity (`low_activity`)
- 7-day rolling average steps < 3,000/day
- Severity: **warning**
- WHO recommends ≥ 8,000 steps; 3,000 is a conservative alert floor

### 4. Sleep Reduction (`sleep_reduction`)
- 5-day rolling average sleep < 6 hrs
- Includes **OLS trend slope** to distinguish temporary dip from sustained decline
- Severity: **warning**

### 5. Rising Resting HR (`rhr_rising_trend`)
- OLS slope of 7-day daily avg HR ≥ 1.0 bpm/day
- Severity: **warning**
- Indicates stress, overtraining, or early illness

**Alert prioritisation:** sorted by severity (critical → warning → info), then date descending.

---

## Part 3 — API Layer

### Endpoints

| Method | Endpoint                          | Description                                      |
|--------|-----------------------------------|--------------------------------------------------|
| GET    | `/`                               | Service info + endpoint list                     |
| GET    | `/users`                          | List all user IDs                                |
| GET    | `/summary`                        | Latest health snapshot (all or one user)         |
| GET    | `/alerts`                         | All alerts (filterable by user, severity, type)  |
| GET    | `/user/{user_id}`                 | Full profile: summary + alerts                   |
| GET    | `/user/{user_id}/daily`           | Day-by-day metrics (adjustable window)           |
| GET    | `/user/{user_id}/trend`           | 7d & 30d slope for HR, steps, sleep, SpO₂       |

### Query Parameters

`/summary?user_id=user_001`  
`/alerts?severity=critical&type=hr_spike&limit=20`  
`/user/{id}/daily?days=14`

### Interactive Docs
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Part 4 — Dashboard

`dashboard/index.html` is a zero-dependency single-file dashboard that presents API insights in an at-a-glance clinical style view.

### What the UI Shows

**Top navigation / user switcher**
- Tabs for `user_001`, `user_002`, `user_003`
- Switching tabs reloads all metrics for the selected user

**KPI row (4 cards)**
- Card 1: latest heart rate (`avg_hr_today`) + personal baseline
- Card 2: latest SpO₂ (`avg_spo2_today`)
- Card 3: today's steps + 7-day average steps
- Card 4: last-night sleep hours + 7-day average sleep

**Primary analysis panels**
- **Heart Rate chart**: 7-day trend (SVG line + area)
- **Active Alerts list**: severity-tagged events (`critical`, `warning`, `info`) with date/time/value

**Secondary trend panels**
- **Steps chart**: 7-day movement trend
- **Sleep chart**: 7-day sleep trend

**Detail + interpretation panels**
- **Daily Log table**: recent day-wise values (HR, SpO₂, steps, sleep)
- **Trend boxes**: 7-day and 30-day slope direction for HR, steps, sleep, SpO₂

### Dashboard Data Mapping (API -> UI)

| API Endpoint | Consumed By UI |
|---|---|
| `GET /users` | Builds top user tabs |
| `GET /user/{id}` | KPI cards + Active Alerts |
| `GET /user/{id}/daily?days=14` | HR/Steps/Sleep charts + Daily Log table |
| `GET /user/{id}/trend` | Trend direction boxes |

### User Interaction Flow

1. Page loads -> fetches `/users`
2. Default user is auto-selected
3. For selected user, dashboard fetches profile + daily + trend in parallel
4. UI updates all panels in one render cycle
5. Changing user tab repeats the same flow

### Offline / Resilience Behavior

- If API is unreachable, dashboard switches to embedded demo data
- UI remains fully viewable for demo/video purposes
- This ensures evaluators can still inspect interface behavior even if backend is not running

### Evaluator Checklist (What They Can Verify Quickly)

- User switching updates all cards/charts/alerts
- Alert colours match severity levels
- Trends show directional movement (up/down/flat)
- Daily table values align with chart period
- Dashboard works with live API and also in offline fallback mode

---

## Part 5 — Bonus Features

### Personalised Baselines
- Each user has a **7-day rolling mean + std** for heart rate
- Alerts compare against *that user's own history*, not a population average
- Prevents false positives for athletes (naturally low HR) or elderly (naturally higher)

### Anomaly Detection
- **Z-score method** on HR (threshold: 3σ)
- **OLS linear regression** for trend anomalies (sleep reduction, rising RHR)
- **Rolling window statistics** for activity detection
- Extensible: Isolation Forest or LSTM autoencoder could replace/augment rules

### Dashboard Visualisation
- Single-file HTML/SVG/JS dashboard (no build step)
- Live data from API + graceful offline fallback

---

## Assumptions

1. **Simulated data** — generated via NumPy with realistic distributions; real wearable SDKs (Fitbit, Apple HealthKit, Garmin Connect) would replace `generate_data.py`.
2. **Sleep detection** — inferred from hour of day (22:00–06:00); real devices use accelerometer + HR.
3. **Resting HR** — approximated by daily average; ideally measured during confirmed sleep.
4. **3 users, 30 days** — sufficient to demonstrate all alert types and rolling baselines.
5. **Minute-level data** — realistic for smartwatches; some devices only expose 5-min or hourly.
6. **No authentication** — API is open for demo purposes.

---

## Scalability & Future Improvements

### Data Layer
| Now | At Scale |
|-----|----------|
| CSV files | Apache Parquet on S3 / Delta Lake |
| Pandas | PySpark or DuckDB for TB-scale |
| Local simulation | Kafka / Kinesis real-time ingestion |
| Batch processing | Airflow / Prefect DAGs |

### ML Layer
- Replace z-score with **Isolation Forest** for multivariate anomaly detection
- **LSTM autoencoder** for sequence-level anomaly detection (e.g. unusual HR pattern over 30 min)
- **Federated learning** to train on-device without raw data leaving the phone
- **Time-series forecasting** (Prophet, N-BEATS) for predictive alerts

### API Layer
- Add **JWT authentication** (FastAPI + OAuth2)
- **Rate limiting** (slowapi)
- **PostgreSQL / TimescaleDB** backend instead of in-memory DataFrames
- **Redis cache** for `/summary` and `/alerts`
- **WebSocket endpoint** for real-time alert streaming

### Deployment
```
Docker container → Kubernetes (EKS/GKE) → HPA on request volume
CI/CD: GitHub Actions → ECR → ArgoCD
Monitoring: Prometheus + Grafana
```

---

## Sample API Responses

See `docs/sample_api_responses.json` for all endpoints.

Quick example — `GET /alerts?severity=critical`:
```json
{
  "total": 1,
  "alerts": [
    {
      "user_id": "user_002",
      "type": "hr_spike",
      "severity": "critical",
      "date": "2024-05-18",
      "hour": 14,
      "value": 162.0,
      "hr_z": 4.31,
      "message": "Heart-rate spike detected: 162 bpm (z=4.31) at 14:00 on 2024-05-18."
    }
  ]
}
```
