# Training Pal

A personal AI fitness coach for CrossFit + powerlifting. Ingests data from SugarWOD, Garmin Connect, and powerlifting program PDFs — runs ML analysis, and lets you chat with a Claude agent about your training via a mobile-first Streamlit UI.

---

## What it does

| Component | What it covers |
|-----------|---------------|
| **SugarWOD analysis** | Attendance consistency · sentiment from workout notes · RX rate · strength progression · lift correlations · PR forecasting · anomaly detection |
| **Garmin cross-analysis** | Sleep vs performance · HRV vs RX rate · menstrual cycle vs performance · Body Battery drain · HR zones · weekly training load |
| **Powerlifting programs** | Downloads program PDFs from Facebook Messenger, parses sets/reps/weight, charts strength progression (Bench / Squat / Deadlift) over time |
| **ML models** | KMeans clustering · per-lift rolling z-score anomaly detection · auto-selected regression (linear/log/sqrt) for PR forecasting |
| **AI chat** | Claude (Sonnet 4.6) acts as a reasoning agent — picks which analysis tools to call, interprets results, responds in natural language |
| **Mobile UI** | Streamlit app optimised for phone use (480px, sticky input, tabs: Chat / Charts / Stats) |

---

## Project structure

```
├── data/
│   ├── workouts.csv              # SugarWOD export — 381 CrossFit sessions (gitignored)
│   ├── garmin_activities.csv     # 382 Garmin activities, Nov 2024 → present (gitignored)
│   ├── garmin_daily.csv          # 505 days: sleep, HRV, resting HR, SpO2, menstrual phase (gitignored)
│   └── powerlifting/             # program_YYYY-MM-DD.pdf — 18 programs (gitignored)
├── analysis/
│   ├── attendance.py             # Consistency, gaps, day-of-week patterns
│   ├── sentiment.py              # Keyword scoring on workout notes
│   ├── performance.py            # RX rate, strength progression, lift correlations
│   ├── ml_models.py              # Clustering, anomaly detection, PR forecasting
│   ├── garmin.py                 # Garmin biometrics cross-analysis
│   └── powerlifting.py           # PDF parsing, strength progression, volume charts
├── scripts/
│   ├── sync_sugarwod.py          # Playwright: SugarWOD export → data/workouts.csv
│   ├── sync_garmin.py            # garminconnect: Garmin Connect (SSO rate-limited)
│   ├── parse_garmin_export.py    # Parses manual Garmin export ZIP → garmin_*.csv
│   └── sync_powerlifting.py      # Playwright: Facebook Messenger PDFs → data/powerlifting/
├── loader.py                     # Data loading and cleaning
├── tools.py                      # Agent tool definitions (wraps analysis modules)
├── agent.py                      # Claude agent loop with tool use
├── app.py                        # Streamlit mobile-first UI (17 charts, 3 sync buttons)
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd training-pal

# 2. Create virtual environment (required on Debian/Ubuntu)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 4. Create .env with credentials
ANTHROPIC_API_KEY=sk-...
SUGARWOD_EMAIL=...
SUGARWOD_PASSWORD=...
GMAIL_PASSWORD=...          # Gmail App Password (not your main password)
GARMIN_EMAIL=...
GARMIN_PASSWORD=...
FACEBOOK_EMAIL=...
FACEBOOK_PASSWORD=...

# 5. Run
streamlit run app.py
```

Open **http://localhost:8501** in your browser (or on your phone via your local IP).

---

## Data sync

### SugarWOD
```bash
python scripts/sync_sugarwod.py
```
Or use the **Sync SugarWOD** button in the Stats tab.

### Garmin Connect
Garmin's SSO blocks automated logins. Use the manual export instead:
1. Go to [garmin.com/account/datamanagement/exportdata](https://www.garmin.com/en-US/account/datamanagement/exportdata)
2. Click **Request Data Export** — you'll receive a download link by email
3. Download the ZIP and place it in the project root
4. Run: `python scripts/parse_garmin_export.py`

Or use the **Re-parse Garmin Export** button in the Stats tab.

### Powerlifting PDFs
```bash
# Opens a browser window — log in manually if asked, then press Enter
python scripts/sync_powerlifting.py
```
Downloads all program PDFs from the coach's Messenger chat, extracts dates from PDF metadata, saves as `data/powerlifting/program_YYYY-MM-DD.pdf`.

---

## Example questions to ask the agent

- *"How consistent have I been training?"*
- *"When is my next PR likely for the Snatch?"*
- *"How does my sleep affect my performance?"*
- *"What do my workout notes reveal about how I feel?"*
- *"Are there any unusual sessions worth looking at?"*
- *"How am I progressing on bench, squat and deadlift?"*

---

## Agent tools

The Claude agent has 7 tools — it decides which to call based on your question:

1. `attendance_summary` — sessions/month, gaps, consistency trends
2. `sentiment_summary` — mood scores from workout notes
3. `performance_summary` — RX rate, PR count, strength progression
4. `cluster_workouts` — KMeans workout archetypes
5. `detect_anomalies` — flags unusual strength sessions
6. `forecast_prs` — predicts next PR per lift
7. `garmin_summary` — sleep, HRV, menstrual cycle vs performance

---

## CrossFit vs Powerlifting

CrossFit and powerlifting are tracked as **completely separate sports** with different gear, different PRs, and different standards. SugarWOD data contains CrossFit only. Powerlifting PRs (from the program PDFs) are always higher — e.g. deadlift CrossFit PR ~230–250 lbs vs powerlifting PR 300+ lbs. The agent is instructed never to conflate them.

---

## Roadmap

- ✅ Mobile-first Streamlit UI (Chat / Charts / Stats tabs)
- ✅ SugarWOD auto-sync (Playwright + Gmail IMAP)
- ✅ Analysis improvements (lift correlations, anomaly detection, PR forecasting)
- ✅ Powerlifting program sync + PDF parser + strength progression charts
- ✅ Garmin Connect integration (manual export + parser)
- ✅ Cross-data analysis — sleep / HRV / menstrual cycle vs performance, Body Battery, HR zones
- ⬜ Automate Garmin data refresh (Playwright export request + IMAP download)
- ⬜ Video form analysis (squat / deadlift / bench)
