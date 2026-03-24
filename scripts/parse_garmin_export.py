"""
scripts/parse_garmin_export.py
Parses the Garmin data export ZIP into two clean CSVs:

  data/garmin_activities.csv  — one row per Garmin activity
  data/garmin_daily.csv       — one row per calendar day (sleep, HRV, menstrual phase)

Data coverage (from this export):
  Activities : 2024-11-29 → 2026-03-23
  Sleep      : 2024-11-28 → 2026-03-24
  HRV / HR   : 2025-09-18 → 2026-03-24
  Menstrual  : 2024-11-05 → 2026-01-21 (16 cycles, avg 27.6 days)

Note: SugarWOD data starts Oct 2023. The cross-analysis window is Nov 2024 onward.

Usage:
    python scripts/parse_garmin_export.py
"""
import json
import re
from pathlib import Path
from datetime import datetime, date, timedelta

import pandas as pd

ROOT       = Path(__file__).parent.parent
EXPORT_DIR = ROOT / 'data' / 'garmin' / 'c9322ca6-4845-45e4-9633-43ae46d757a8_1' / 'DI_CONNECT'
FITNESS    = EXPORT_DIR / 'DI-Connect-Fitness'
WELLNESS   = EXPORT_DIR / 'DI-Connect-Wellness'
OUT_DIR    = ROOT / 'data'


# ── Activities ─────────────────────────────────────────────────────────────────

def _ms_to_date(ms):
    """Convert millisecond epoch timestamp to date using local time."""
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000).date()
    except Exception:
        return None


def _ms_to_mins(ms):
    """Convert milliseconds to minutes, rounded to 1dp."""
    if ms is None:
        return None
    return round(ms / 60000, 1)


def parse_activities():
    path = FITNESS / 'helensamarasantos@gmail.com_0_summarizedActivities.json'
    with open(path) as f:
        raw = json.load(f)
    activities = raw[0]['summarizedActivitiesExport']

    rows = []
    for a in activities:
        activity_date = _ms_to_date(a.get('startTimeLocal'))
        if not activity_date:
            continue

        # Classify activity into a cleaner category
        name = a.get('name', '')
        atype = a.get('activityType', 'other')
        if 'crossfit' in name.lower() or 'CrossFit' in name:
            category = 'crossfit'
        elif 'powerlifting' in name.lower() or 'Powerlifting' in name:
            category = 'powerlifting'
        elif atype == 'walking':
            category = 'walking'
        elif atype == 'indoor_cycling':
            category = 'cycling'
        elif atype == 'strength_training':
            category = 'strength'
        elif atype == 'hiit':
            category = 'hiit'
        else:
            category = atype

        rows.append({
            'date':                  activity_date,
            'activity_id':           a.get('activityId'),
            'activity_name':         name,
            'activity_category':     category,
            'activity_type_raw':     atype,
            'duration_mins':         round(a.get('duration', 0) / 60, 1) if a.get('duration') else None,
            'calories':              a.get('calories'),
            'avg_hr':                a.get('avgHr'),
            'max_hr':                a.get('maxHr'),
            'min_hr':                a.get('minHr'),
            # Body Battery: negative = drained during activity, positive = gained (rare)
            'body_battery_drain':    a.get('differenceBodyBattery'),
            'training_effect_label': a.get('trainingEffectLabel'),
            'aerobic_effect':        a.get('aerobicTrainingEffectMessage'),
            'anaerobic_effect':      a.get('anaerobicTrainingEffectMessage'),
            # HR time in zones (convert ms → mins); zone 0 = rest, 1-5 = intensity zones
            'hr_zone1_mins':         _ms_to_mins(a.get('hrTimeInZone_1')),
            'hr_zone2_mins':         _ms_to_mins(a.get('hrTimeInZone_2')),
            'hr_zone3_mins':         _ms_to_mins(a.get('hrTimeInZone_3')),
            'hr_zone4_mins':         _ms_to_mins(a.get('hrTimeInZone_4')),
            'hr_zone5_mins':         _ms_to_mins(a.get('hrTimeInZone_5')),
            'moderate_intensity_mins': a.get('moderateIntensityMinutes'),
            'vigorous_intensity_mins': a.get('vigorousIntensityMinutes'),
            'is_pr':                 a.get('pr', False),
        })

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


# ── Sleep ──────────────────────────────────────────────────────────────────────

def parse_sleep():
    """Combine all sleep JSON files, deduplicate by calendarDate."""
    all_records = {}

    sleep_files = sorted(WELLNESS.glob('*sleepData*.json'))
    for fpath in sleep_files:
        with open(fpath) as f:
            records = json.load(f)
        for r in records:
            d = r.get('calendarDate')
            if not d or not r.get('deepSleepSeconds') and not r.get('lightSleepSeconds'):
                continue  # skip empty records (like the Apr 2024 placeholder)

            sleep_secs = (
                r.get('deepSleepSeconds', 0) +
                r.get('lightSleepSeconds', 0) +
                r.get('remSleepSeconds', 0)
            )
            spo2 = r.get('spo2SleepSummary', {})

            scores = r.get('sleepScores', {})
            overall_score = None
            if isinstance(scores, dict):
                # Export format uses 'overallScore' (int directly)
                overall_score = scores.get('overallScore') or scores.get('overall')
                # API format uses nested {'overall': {'value': N}}
                if isinstance(overall_score, dict):
                    overall_score = overall_score.get('value')

            all_records[d] = {
                'date':               d,
                'sleep_score':        overall_score,
                'sleep_total_hrs':    round(sleep_secs / 3600, 2),
                'sleep_deep_hrs':     round(r.get('deepSleepSeconds', 0) / 3600, 2),
                'sleep_light_hrs':    round(r.get('lightSleepSeconds', 0) / 3600, 2),
                'sleep_rem_hrs':      round(r.get('remSleepSeconds', 0) / 3600, 2),
                'sleep_awake_hrs':    round(r.get('awakeSleepSeconds', 0) / 3600, 2),
                'sleep_avg_spo2':     spo2.get('averageSPO2'),
                'sleep_lowest_spo2':  spo2.get('lowestSPO2'),
                'sleep_avg_hr':       spo2.get('averageHR'),
                'sleep_respiration':  r.get('averageRespiration'),
                'sleep_stress':       r.get('avgSleepStress'),
                'awake_count':        r.get('awakeCount'),
            }

    df = pd.DataFrame(list(all_records.values()))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


# ── Health Status (HRV, Resting HR, SpO2, Skin Temp, Respiration) ─────────────

def parse_health_status():
    """Combine both health status files, pivot metrics by type into columns."""
    all_records = {}

    health_files = sorted(WELLNESS.glob('*healthStatusData*.json'))
    for fpath in health_files:
        with open(fpath) as f:
            records = json.load(f)
        for r in records:
            d = r.get('calendarDate')
            if not d:
                continue
            if d not in all_records:
                all_records[d] = {'date': d}
            for m in r.get('metrics', []):
                mtype = m.get('type', '').lower()
                val   = m.get('value')
                status = m.get('status')
                b_lo  = m.get('baselineLowerLimit')
                b_hi  = m.get('baselineUpperLimit')

                if mtype == 'hrv':
                    all_records[d]['hrv_value']          = val
                    all_records[d]['hrv_status']         = status
                    all_records[d]['hrv_baseline_lower'] = b_lo
                    all_records[d]['hrv_baseline_upper'] = b_hi
                elif mtype == 'hr':
                    all_records[d]['resting_hr']         = val
                    all_records[d]['resting_hr_status']  = status
                elif mtype == 'spo2':
                    all_records[d]['daily_spo2']         = val
                elif mtype == 'respiration':
                    all_records[d]['daily_respiration']  = val
                elif mtype == 'skin_temp_c':
                    all_records[d]['skin_temp_c']        = val

    df = pd.DataFrame(list(all_records.values()))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


# ── Menstrual Cycles ───────────────────────────────────────────────────────────

def _compute_menstrual_phase(cycle_day, period_length, fertile_start, fertile_length):
    """
    Returns the phase name for a given day within a cycle.
      menstrual       : days 1 → period_length
      follicular      : days after period, before fertile window
      fertile_window  : days fertile_start → fertile_start + fertile_length - 1
      luteal          : days after fertile window until end of cycle
    """
    if cycle_day <= period_length:
        return 'menstrual'
    elif cycle_day < fertile_start:
        return 'follicular'
    elif cycle_day <= fertile_start + fertile_length - 1:
        return 'fertile_window'
    else:
        return 'luteal'


def parse_menstrual_cycles():
    """
    Expand cycle records into a daily lookup table.
    For each calendar date, compute: cycle_day, cycle_phase, cycle_start.
    """
    path = WELLNESS / '127600156_MenstrualCycles.json'
    with open(path) as f:
        cycles = json.load(f)

    # Sort by start date ascending
    cycles = sorted(cycles, key=lambda c: c['startDate'])

    daily_rows = []
    for i, cycle in enumerate(cycles):
        cycle_start  = date.fromisoformat(cycle['startDate'])
        cycle_length = cycle.get('actualCycleLength') or cycle.get('predictedCycleLength') or 28
        period_len   = cycle.get('actualPeriodLength') or cycle.get('predictedPeriodLength') or 5
        fertile_start  = cycle.get('fertileWindowStart', 12)
        fertile_length = cycle.get('fertileWindowLength', 7)

        # Determine end of this cycle's date range
        # (use next cycle start - 1, or cycle_start + cycle_length - 1)
        if i + 1 < len(cycles):
            next_start = date.fromisoformat(cycles[i + 1]['startDate'])
            cycle_end  = next_start - timedelta(days=1)
        else:
            cycle_end  = cycle_start + timedelta(days=cycle_length - 1)

        current = cycle_start
        while current <= cycle_end:
            cycle_day = (current - cycle_start).days + 1  # 1-indexed
            phase = _compute_menstrual_phase(cycle_day, period_len, fertile_start, fertile_length)
            daily_rows.append({
                'date':          current.isoformat(),
                'cycle_start':   cycle['startDate'],
                'cycle_day':     cycle_day,
                'cycle_phase':   phase,
                'cycle_length':  cycle_length,
                'period_length': period_len,
            })
            current += timedelta(days=1)

    df = pd.DataFrame(daily_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('→ Parsing Garmin activities...')
    activities_df = parse_activities()
    print(f'  {len(activities_df)} activities, {activities_df["date"].min().date()} → {activities_df["date"].max().date()}')
    print(f'  Categories: {activities_df["activity_category"].value_counts().to_dict()}')

    print('→ Parsing sleep data...')
    sleep_df = parse_sleep()
    print(f'  {len(sleep_df)} nights, {sleep_df["date"].min().date()} → {sleep_df["date"].max().date()}')

    print('→ Parsing health status (HRV, resting HR, SpO2)...')
    health_df = parse_health_status()
    print(f'  {len(health_df)} days, {health_df["date"].min().date()} → {health_df["date"].max().date()}')
    print(f'  HRV coverage: {health_df["hrv_value"].notna().sum()} days')

    print('→ Parsing menstrual cycles...')
    menstrual_df = parse_menstrual_cycles()
    print(f'  {len(menstrual_df)} days of cycle data, {menstrual_df["date"].min().date()} → {menstrual_df["date"].max().date()}')

    # ── Save activities CSV ─────────────────────────────────────────
    out_activities = OUT_DIR / 'garmin_activities.csv'
    activities_df.to_csv(out_activities, index=False)
    print(f'\n✅  Saved {len(activities_df)} activities → {out_activities}')

    # ── Build daily CSV: sleep + health + menstrual ─────────────────
    # Start from the full date range covered by any of the three daily sources
    all_dates = pd.DataFrame({
        'date': pd.date_range(
            start=min(sleep_df['date'].min(), health_df['date'].min(), menstrual_df['date'].min()),
            end=max(sleep_df['date'].max(), health_df['date'].max(), menstrual_df['date'].max()),
            freq='D'
        )
    })

    daily_df = (all_dates
        .merge(sleep_df,     on='date', how='left')
        .merge(health_df,    on='date', how='left')
        .merge(menstrual_df, on='date', how='left')
    )

    # Also flag whether there was a workout on that day (joins to activities by date)
    activity_dates = activities_df.groupby('date').agg(
        workout_count=('activity_id', 'count'),
        workout_names=('activity_name', lambda x: ' | '.join(x)),
        workout_categories=('activity_category', lambda x: ' | '.join(x.unique())),
    ).reset_index()
    daily_df = daily_df.merge(activity_dates, on='date', how='left')
    daily_df['workout_count'] = daily_df['workout_count'].fillna(0).astype(int)

    out_daily = OUT_DIR / 'garmin_daily.csv'
    daily_df.to_csv(out_daily, index=False)
    print(f'✅  Saved {len(daily_df)} daily rows → {out_daily}')

    # ── Summary ─────────────────────────────────────────────────────
    print('\n── Coverage summary ──────────────────────────────────')
    print(f'  Activities  : {activities_df["date"].min().date()} → {activities_df["date"].max().date()}')
    print(f'  Sleep       : {sleep_df["date"].min().date()} → {sleep_df["date"].max().date()}')
    print(f'  HRV/HR      : {health_df["date"].min().date()} → {health_df["date"].max().date()}')
    print(f'  Menstrual   : {menstrual_df["date"].min().date()} → {menstrual_df["date"].max().date()}')
    print(f'  Cross-analysis window (all sources): Nov 2024 → Mar 2026')
    print(f'  SugarWOD overlap: ~16 months of the 29-month SugarWOD history')
    print('\n  Columns in garmin_activities.csv:')
    print(f'  {list(activities_df.columns)}')
    print('\n  Columns in garmin_daily.csv:')
    print(f'  {list(daily_df.columns)}')


if __name__ == '__main__':
    main()
