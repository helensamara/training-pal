"""
analysis/powerlifting.py
Parses powerlifting program PDFs and matches them to Garmin sessions.

Program structure (8 pages per PDF, sent every ~2 weeks by coach Tom Kean):
    Page 1: Week 1 - Day 1  (Bench + Deadlift)
    Page 2: Week 1 - Day 2  (Squat + Bench)
    Page 3: Week 1 - Day 3  (Squat + Deadlift)
    Page 4: Week 1 - Day 4  (Squat + Bench)
    Page 5: Week 2 - Day 1  (Bench + Deadlift)
    Page 6: Week 2 - Day 2  (Squat + Bench)
    Page 7: Week 2 - Day 3  (Squat + Deadlift)
    Page 8: Week 2 - Day 4  (Squat + Bench)

Helen's actual schedule (from Garmin weekday + time):
    Wednesday evening  → Day 3 Squats  + Day 1 Deadlifts
    Saturday morning   → Day 1 Bench   + Day 3 Deadlifts
    Monday afternoon   → Day 4 Squat   + Day 4 Bench
    Thursday           → Day 2 Bench   (+ Day 2 Squat if session > 30 min)
"""
import re
import os
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BLUE   = '#4a9eed'
GREEN  = '#22c55e'
AMBER  = '#f59e0b'
RED    = '#ef4444'
PURPLE = '#8b5cf6'

DATA_DIR = Path(__file__).parent.parent / 'data' / 'powerlifting'

# Day content map
DAY_MOVEMENTS = {
    1: ['bench', 'deadlift'],
    2: ['squat', 'bench'],
    3: ['squat', 'deadlift'],
    4: ['squat', 'bench'],
}

# Helen's schedule: weekday (0=Mon) → (day_numbers_done, full_if_duration_over_mins)
SCHEDULE = {
    0: {'days': [4],    'full_threshold': None},   # Monday → Day 4
    2: {'days': [3, 1], 'full_threshold': None},   # Wednesday → Day 3 + Day 1
    3: {'days': [2],    'full_threshold': 30},     # Thursday → Day 2 (Bench always, Squat if >30min)
    5: {'days': [1, 3], 'full_threshold': None},   # Saturday → Day 1 + Day 3
}


# ── PDF Parsing ────────────────────────────────────────────────────

def _parse_sets(text):
    """
    Parse sets from a PDF page.

    Tom's programs use an Excel-exported spreadsheet format:
        LiftName  Reps  Sets  Volume  Details
        30        5     1     150     Paused
        55        2     3     330
        1377.5    12    Sets          ← total row, skip

    Verified: volume == weight * reps * sets (within 1% rounding tolerance).
    """
    exercises = []
    current   = None

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Skip page header "Week X - Day Y"
        if re.match(r'^Week\s+\d', line, re.IGNORECASE):
            continue

        # Lift header: "Bench Reps Sets Volume Details"
        # The lift name is everything before "Reps Sets Volume"
        header = re.match(r'^([A-Za-z][A-Za-z\s\-/]+?)\s+Reps\s+Sets\s+Volume', line, re.IGNORECASE)
        if header:
            if current:
                exercises.append(current)
            lift_name = header.group(1).strip()
            # Normalise common variants
            name_lower = lift_name.lower()
            if 'deadlift' in name_lower:
                lift_name = 'Deadlift'
            elif 'squat' in name_lower:
                lift_name = 'Squat'
            elif 'bench' in name_lower:
                lift_name = 'Bench'
            current = {'name': lift_name, 'sets': []}
            continue

        if current is None:
            continue

        # Total/summary row: ends with "Sets" (e.g. "1377.5  12  Sets")
        if re.search(r'\bSets\s*$', line, re.IGNORECASE):
            continue

        # Data row: weight  reps  sets  volume  [details]
        parts = line.split()
        if len(parts) >= 4:
            try:
                weight = float(parts[0])
                reps   = int(parts[1])
                sets   = int(parts[2])
                volume = float(parts[3])
                details = ' '.join(parts[4:]) if len(parts) > 4 else ''

                # Sanity: volume ≈ weight × reps × sets (allow small rounding)
                expected = weight * reps * sets
                if (expected > 0
                        and abs(volume - expected) / expected < 0.02
                        and 5 < weight < 500
                        and 1 <= reps <= 30
                        and 1 <= sets <= 15):
                    current['sets'].append({
                        'weight':  weight,
                        'reps':    reps,
                        'sets':    sets,
                        'volume':  volume,
                        'details': details,
                    })
            except (ValueError, ZeroDivisionError):
                pass

    if current:
        exercises.append(current)
    return exercises


def parse_pdf(pdf_path):
    """
    Parse a program PDF. Returns a dict:
    {
      'program_date': '2024-03-01',   # date from filename
      'weeks': {
        1: { 1: [exercises], 2: [exercises], ... },
        2: { 1: [exercises], ... }
      }
    }
    """
    path = Path(pdf_path)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
    program_date = date_match.group(1) if date_match else None

    result = {'program_date': program_date, 'weeks': {1: {}, 2: {}}}

    try:
        with pdfplumber.open(path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                week = 1 if page_idx < 4 else 2
                day  = (page_idx % 4) + 1
                text = page.extract_text() or ''
                result['weeks'][week][day] = _parse_sets(text)
    except Exception as e:
        print(f'  Warning: could not parse {path.name}: {e}')

    return result


def load_all_programs():
    """Load and parse all PDFs in data/powerlifting/. Returns sorted list of program dicts."""
    pdfs = sorted(DATA_DIR.glob('program_*.pdf'))
    if not pdfs:
        return []
    programs = [parse_pdf(p) for p in pdfs]
    return [p for p in programs if p['program_date']]


# ── Session Matching ───────────────────────────────────────────────

def match_garmin_to_program(garmin_df, programs):
    """
    For each Garmin activity that looks like a powerlifting session,
    determine which program week/day it corresponds to.

    Returns garmin_df with added columns:
        pl_program_date, pl_week, pl_day, pl_movements
    """
    if garmin_df is None or garmin_df.empty or not programs:
        return garmin_df

    df = garmin_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    pl_cols = {'pl_program_date': None, 'pl_week': None, 'pl_day': None, 'pl_movements': None}
    for col, val in pl_cols.items():
        df[col] = val

    # Build program timeline: for each calendar date, which program is active?
    # A program is active from its send date until the next program arrives
    prog_dates = sorted([(pd.to_datetime(p['program_date']), p) for p in programs])

    def _active_program(session_date):
        prog = None
        for pdate, p in prog_dates:
            if pdate <= session_date:
                prog = p
            else:
                break
        return prog

    def _which_week(session_date, program_date):
        """Determine if session falls in week 1 or week 2 of the program."""
        days_since = (session_date - pd.to_datetime(program_date)).days
        return 1 if days_since < 7 else 2

    for idx, row in df.iterrows():
        weekday = row['date'].weekday()
        if weekday not in SCHEDULE:
            continue

        # Check it's a strength/gym activity
        activity_type = str(row.get('activity_type', '')).lower()
        if not any(k in activity_type for k in ['strength', 'gym', 'training', 'indoor', 'crossfit']):
            continue

        sched  = SCHEDULE[weekday]
        duration = float(row.get('duration_mins', 0) or 0)

        # Determine which days were done
        days_done = sched['days'][:]
        if sched['full_threshold'] and duration <= sched['full_threshold']:
            days_done = [sched['days'][0]]  # only first movement if short session

        prog = _active_program(row['date'])
        if not prog:
            continue

        week = _which_week(row['date'], prog['program_date'])
        movements = list({m for d in days_done for m in DAY_MOVEMENTS.get(d, [])})

        df.at[idx, 'pl_program_date'] = prog['program_date']
        df.at[idx, 'pl_week']         = week
        df.at[idx, 'pl_day']          = str(days_done)
        df.at[idx, 'pl_movements']    = ', '.join(movements)

    return df


# ── Summary for agent ──────────────────────────────────────────────

def summary(garmin_df=None):
    """Return dict of program stats for agent use."""
    programs = load_all_programs()
    if not programs:
        return {'error': 'No program PDFs found in data/powerlifting/'}

    out = {
        'total_programs': len(programs),
        'date_range': f"{programs[0]['program_date']} → {programs[-1]['program_date']}",
    }

    # Max weight and volume progression per lift
    maxes = _extract_program_maxes()
    if not maxes.empty:
        progression = {}
        for lift in ['Bench', 'Squat', 'Deadlift']:
            ldf = maxes[maxes['lift'] == lift].sort_values('program_date')
            if ldf.empty:
                continue
            first = ldf.iloc[0]
            last  = ldf.iloc[-1]
            progression[lift] = {
                'first_program_date':    str(first['program_date'].date()),
                'first_max_kg':          round(first['max_weight'], 1),
                'latest_program_date':   str(last['program_date'].date()),
                'latest_max_kg':         round(last['max_weight'], 1),
                'total_gain_kg':         round(last['max_weight'] - first['max_weight'], 1),
                'programs_with_data':    len(ldf),
                'latest_total_volume_kg': round(float(last['total_volume']), 1),
                'history': [
                    {'date': str(r['program_date'].date()), 'max_kg': round(r['max_weight'], 1)}
                    for _, r in ldf.iterrows()
                ],
            }
        out['lift_progression'] = progression

    if garmin_df is not None and not garmin_df.empty:
        matched = match_garmin_to_program(garmin_df, programs)
        pl_sessions = matched[matched['pl_program_date'].notna()]
        out['pl_sessions_tracked'] = int(len(pl_sessions))

    return out


# ── Charts ─────────────────────────────────────────────────────────

def program_adherence_chart(garmin_df):
    """Bar chart: planned sessions per 2-week block vs actual done."""
    programs   = load_all_programs()
    if not programs or garmin_df is None:
        return None

    matched = match_garmin_to_program(garmin_df, programs)
    pl      = matched[matched['pl_program_date'].notna()].copy()
    pl['pl_program_date'] = pd.to_datetime(pl['pl_program_date'])

    counts = pl.groupby('pl_program_date').size().reset_index(name='actual')
    counts['planned'] = 8  # 4 days × 2 weeks (theoretical max)
    counts['adherence_pct'] = (counts['actual'] / counts['planned'] * 100).round(1)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(counts))
    ax.bar(x, counts['planned'], color='#e5e7eb', label='Planned (8 sessions)')
    ax.bar(x, counts['actual'],  color=BLUE, alpha=0.85, label='Completed')
    ax.set_xticks(list(x))
    ax.set_xticklabels(counts['pl_program_date'].dt.strftime('%b %d'), rotation=45, ha='right')
    ax.axhline(counts['actual'].mean(), color=AMBER, linestyle='--', linewidth=1.5,
               label=f"Avg: {counts['actual'].mean():.1f} sessions")
    ax.set_title('Powerlifting Program Adherence', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Sessions per 2-week block')
    ax.legend()
    plt.tight_layout()
    return fig


def _extract_program_maxes():
    """
    For each program, extract the max prescribed weight per lift (Bench, Squat, Deadlift).
    Returns a DataFrame with columns: program_date, lift, max_weight, total_volume.
    """
    programs = load_all_programs()
    rows = []
    for p in programs:
        prog_date = pd.to_datetime(p['program_date'])
        lift_maxes  = {'Bench': 0, 'Squat': 0, 'Deadlift': 0}
        lift_volume = {'Bench': 0, 'Squat': 0, 'Deadlift': 0}
        for week, days in p['weeks'].items():
            for day, exercises in days.items():
                for ex in exercises:
                    name = ex['name']
                    if name not in lift_maxes or not ex['sets']:
                        continue
                    for s in ex['sets']:
                        lift_maxes[name]  = max(lift_maxes[name], s['weight'])
                        lift_volume[name] += s['volume']
        for lift in ['Bench', 'Squat', 'Deadlift']:
            if lift_maxes[lift] > 0:
                rows.append({
                    'program_date': prog_date,
                    'lift':         lift,
                    'max_weight':   lift_maxes[lift],
                    'total_volume': lift_volume[lift],
                })
    return pd.DataFrame(rows)


def strength_progression_chart(sugarwod_df=None):
    """
    Three-panel chart showing programmed max weight progression for
    Bench, Squat, and Deadlift across all 18 programs.
    Overlays actual PRs logged in SugarWOD where available.
    """
    program_maxes = _extract_program_maxes()
    if program_maxes.empty:
        return None

    lift_colors = {'Bench': BLUE, 'Squat': GREEN, 'Deadlift': RED}
    lift_sugarwod_names = {
        'Bench':    ['Bench Press', 'Bench'],
        'Squat':    ['Back Squat', 'Squat'],
        'Deadlift': ['Deadlift'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Powerlifting Strength Progression — Programmed Weights', fontsize=15, fontweight='bold')

    for ax, lift in zip(axes, ['Bench', 'Squat', 'Deadlift']):
        color = lift_colors[lift]
        ldf   = program_maxes[program_maxes['lift'] == lift].sort_values('program_date')

        # Programmed max line
        ax.plot(ldf['program_date'], ldf['max_weight'],
                color=color, linewidth=2.5, marker='o', markersize=6,
                label='Programmed max')
        ax.fill_between(ldf['program_date'], ldf['max_weight'],
                        alpha=0.1, color=color)

        # SugarWOD CrossFit overlay (separate sport — different PRs, lighter loads, different gear)
        # SugarWOD stores loads in lbs — convert to kg for comparison with PDF weights
        # NOTE: SugarWOD = CrossFit PRs only. Powerlifting PRs are higher (different gear, context, rules).
        LBS_TO_KG = 0.453592
        if sugarwod_df is not None and not sugarwod_df.empty:
            sw = sugarwod_df.copy()
            sw['date'] = pd.to_datetime(sw['date'])
            sw_names = lift_sugarwod_names[lift]
            sw_lift  = sw[sw['barbell_lift'].isin(sw_names) & sw['score_load'].notna()].copy()
            if not sw_lift.empty:
                sw_lift['score_kg'] = sw_lift['score_load'] * LBS_TO_KG
                ax.scatter(sw_lift['date'], sw_lift['score_kg'],
                           color=color, alpha=0.35, s=30, zorder=3,
                           label='CrossFit sessions (lbs→kg)')
                prs = sw_lift[sw_lift['is_pr']]
                if not prs.empty:
                    ax.scatter(prs['date'], prs['score_kg'],
                               color='gold', edgecolors=color, linewidths=1.5,
                               s=100, zorder=5, marker='D', label='CrossFit PR')
                    for _, row in prs.iterrows():
                        ax.annotate(f"{row['score_kg']:.0f}kg",
                                    (row['date'], row['score_kg']),
                                    textcoords='offset points', xytext=(4, 6),
                                    fontsize=7.5, color=color, fontweight='bold')

        ax.set_title(lift, fontsize=13, fontweight='bold', color=color)
        ax.set_ylabel('Weight (kg)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.legend(fontsize=7.5)

    plt.tight_layout()
    return fig


def training_frequency_chart(garmin_activities_df):
    """
    Weekly CrossFit + Powerlifting session count from Garmin data.
    Shows the combined training load over time.
    """
    if garmin_activities_df is None or garmin_activities_df.empty:
        return None

    df = garmin_activities_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['activity_category'].isin(['crossfit', 'powerlifting'])]
    df['week'] = df['date'].dt.to_period('W').dt.start_time

    weekly = (df.groupby(['week', 'activity_category'])
                .size()
                .unstack(fill_value=0)
                .reset_index())

    if 'crossfit' not in weekly.columns:
        weekly['crossfit'] = 0
    if 'powerlifting' not in weekly.columns:
        weekly['powerlifting'] = 0

    weekly['total'] = weekly['crossfit'] + weekly['powerlifting']
    weekly['rolling_total'] = weekly['total'].rolling(4, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(weekly['week'], weekly['crossfit'],    width=5, color=RED,    alpha=0.8, label='CrossFit')
    ax.bar(weekly['week'], weekly['powerlifting'], width=5, color=PURPLE, alpha=0.8,
           bottom=weekly['crossfit'], label='Powerlifting')
    ax.plot(weekly['week'], weekly['rolling_total'], color=AMBER, linewidth=2,
            label=f"4-week avg: {weekly['total'].mean():.1f} sessions/week")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Sessions per week')
    ax.set_title('Weekly Training Frequency — CrossFit + Powerlifting', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig


def volume_progression_chart():
    """
    Total prescribed training volume per lift per program.
    Shows how training load has been periodised over time.
    """
    program_maxes = _extract_program_maxes()
    if program_maxes.empty:
        return None

    lift_colors = {'Bench': BLUE, 'Squat': GREEN, 'Deadlift': RED}
    fig, ax = plt.subplots(figsize=(14, 5))

    for lift, color in lift_colors.items():
        ldf = program_maxes[program_maxes['lift'] == lift].sort_values('program_date')
        ax.plot(ldf['program_date'], ldf['total_volume'] / 1000,
                color=color, linewidth=2, marker='o', markersize=5, label=lift)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Total Prescribed Volume (tonnes)')
    ax.set_title('Programmed Volume per Lift Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig
