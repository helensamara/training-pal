"""
scripts/sync_garmin.py
Pulls data from Garmin Connect and saves to data/garmin.csv

Fetches per-activity metrics for all activities since the last sync:
  - Heart rate (avg, max)
  - HRV (if available)
  - Training load / intensity
  - Sleep scores (from daily summaries)
  - Activity type, duration, distance, calories

Session is saved to ~/.garth_tokens after first login so subsequent
runs don't re-authenticate.

Usage:
    python scripts/sync_garmin.py
"""
import os
import json
import time
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
from dotenv import load_dotenv
from garminconnect import Garmin, GarminConnectConnectionError

load_dotenv()

GARMIN_EMAIL    = os.environ['GARMIN_EMAIL']
GARMIN_PASSWORD = os.environ['GARMIN_PASSWORD']
TOKENSTORE      = Path.home() / '.garth_tokens'
DEST_CSV        = Path(__file__).parent.parent / 'data' / 'garmin.csv'
# Pull up to 2 years of history on first run
START_DATE      = date(2023, 10, 1)


def _get_client():
    """Return authenticated Garmin client, reusing saved tokens when possible."""
    client = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
    if TOKENSTORE.exists():
        try:
            client.garth.load(str(TOKENSTORE))
            client.display_name  # quick test the token is valid
            print('  ✓ Session restored from token')
            return client
        except Exception:
            print('  Token expired, re-logging in...')

    client.login()
    client.garth.dump(str(TOKENSTORE))
    print('  ✓ Logged in and session saved')
    return client


def _fetch_activities(client, start: date, end: date):
    """Fetch all activities in date range, returning list of dicts."""
    activities = []
    batch = 100
    offset = 0
    while True:
        batch_data = client.get_activities(offset, batch)
        if not batch_data:
            break
        for a in batch_data:
            act_date = pd.to_datetime(a.get('startTimeLocal', '')).date()
            if act_date < start:
                return activities  # past our window, stop
            if act_date <= end:
                activities.append(a)
        offset += batch
        time.sleep(0.5)  # be polite to Garmin's API
    return activities


def _fetch_daily_hrv(client, start: date, end: date):
    """Fetch daily HRV status summaries. Returns dict keyed by date string."""
    hrv_by_date = {}
    try:
        hrv_data = client.get_hrv_data(start.isoformat(), end.isoformat())
        if hrv_data and 'hrvSummaries' in hrv_data:
            for entry in hrv_data['hrvSummaries']:
                d = entry.get('calendarDate', '')
                hrv_by_date[d] = {
                    'hrv_weekly_avg': entry.get('weeklyAvg'),
                    'hrv_last_night': entry.get('lastNight'),
                    'hrv_status':     entry.get('hrvStatus'),
                }
    except Exception as e:
        print(f'  HRV data not available: {e}')
    return hrv_by_date


def _fetch_daily_sleep(client, start: date, end: date):
    """Fetch daily sleep scores. Returns dict keyed by date string."""
    sleep_by_date = {}
    try:
        current = start
        while current <= end:
            try:
                sleep = client.get_sleep_data(current.isoformat())
                if sleep and 'dailySleepDTO' in sleep:
                    dto = sleep['dailySleepDTO']
                    sleep_by_date[current.isoformat()] = {
                        'sleep_score':         dto.get('sleepScores', {}).get('overall', {}).get('value'),
                        'sleep_duration_hrs':  round(dto.get('sleepTimeSeconds', 0) / 3600, 2),
                        'deep_sleep_hrs':      round(dto.get('deepSleepSeconds', 0) / 3600, 2),
                        'rem_sleep_hrs':       round(dto.get('remSleepSeconds', 0) / 3600, 2),
                        'awake_hrs':           round(dto.get('awakeSleepSeconds', 0) / 3600, 2),
                    }
            except Exception:
                pass
            current += timedelta(days=7)  # weekly to avoid hitting rate limits
            time.sleep(0.3)
    except Exception as e:
        print(f'  Sleep data not available: {e}')
    return sleep_by_date


def sync():
    print('→ Connecting to Garmin Connect...')
    client = _get_client()

    # Determine date range: from last record in existing CSV, or full history
    end_date = date.today()
    if DEST_CSV.exists():
        existing = pd.read_csv(DEST_CSV)
        last_date = pd.to_datetime(existing['date']).max().date()
        start_date = last_date - timedelta(days=7)  # overlap 1 week to catch late uploads
        print(f'  Incremental sync from {start_date}')
    else:
        start_date = START_DATE
        print(f'  Full history sync from {start_date}')

    print(f'→ Fetching activities ({start_date} → {end_date})...')
    activities = _fetch_activities(client, start_date, end_date)
    print(f'  Found {len(activities)} activities')

    if not activities:
        print('  Nothing new to sync.')
        return

    # Flatten activity data
    rows = []
    for a in activities:
        act_type = a.get('activityType', {}).get('typeKey', 'unknown')
        row = {
            'date':                  pd.to_datetime(a.get('startTimeLocal', '')).date(),
            'activity_id':           a.get('activityId'),
            'activity_name':         a.get('activityName', ''),
            'activity_type':         act_type,
            'duration_mins':         round(a.get('duration', 0) / 60, 1),
            'distance_km':           round(a.get('distance', 0) / 1000, 2),
            'calories':              a.get('calories'),
            'avg_hr':                a.get('averageHR'),
            'max_hr':                a.get('maxHR'),
            'avg_hrv':               a.get('hrv'),
            'training_load':         a.get('activityTrainingLoad'),
            'aerobic_effect':        a.get('aerobicTrainingEffect'),
            'anaerobic_effect':      a.get('anaerobicTrainingEffect'),
            'vo2max':                a.get('vO2MaxValue'),
        }
        rows.append(row)

    new_df = pd.DataFrame(rows)
    new_df['date'] = pd.to_datetime(new_df['date'])

    # Fetch HRV and sleep data
    print('→ Fetching HRV data...')
    hrv = _fetch_daily_hrv(client, start_date, end_date)
    print('→ Fetching sleep data...')
    sleep = _fetch_daily_sleep(client, start_date, end_date)

    # Join HRV and sleep onto activities by date
    new_df['date_str'] = new_df['date'].dt.date.astype(str)
    hrv_df    = pd.DataFrame(hrv).T.reset_index().rename(columns={'index': 'date_str'}) if hrv else pd.DataFrame()
    sleep_df  = pd.DataFrame(sleep).T.reset_index().rename(columns={'index': 'date_str'}) if sleep else pd.DataFrame()

    if not hrv_df.empty:
        new_df = new_df.merge(hrv_df, on='date_str', how='left')
    if not sleep_df.empty:
        new_df = new_df.merge(sleep_df, on='date_str', how='left')
    new_df = new_df.drop(columns=['date_str'])

    # Merge with existing data if incremental
    if DEST_CSV.exists() and start_date > START_DATE:
        existing = pd.read_csv(DEST_CSV, parse_dates=['date'])
        combined = pd.concat([existing, new_df]).drop_duplicates('activity_id').sort_values('date')
    else:
        combined = new_df.sort_values('date')

    combined.to_csv(DEST_CSV, index=False)
    print(f'\n✅  Done! {len(combined)} activities saved to {DEST_CSV}')
    print(f'    Columns: {list(combined.columns)}')
    print(f'    Date range: {combined["date"].min().date()} → {combined["date"].max().date()}')


if __name__ == '__main__':
    sync()
