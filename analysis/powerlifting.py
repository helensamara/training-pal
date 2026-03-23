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
    """Extract sets/reps/weight/percentage lines from page text."""
    exercises = []
    current = None

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # New exercise header (e.g. "Squat", "Bench Press", "Deadlift", "RDL")
        if re.match(r'^[A-Z][a-zA-Z\s]+$', line) and len(line) < 40 and not re.search(r'\d', line):
            if current:
                exercises.append(current)
            current = {'name': line, 'sets': []}
            continue

        if current is None:
            continue

        # Set line patterns: "3x5 @ 80%", "4 sets x 3 reps @ 85%", "5 reps @ 100kg", "3x5 @80% (185lbs)"
        set_match = re.search(
            r'(\d+)\s*[xX×]\s*(\d+)'       # sets x reps
            r'(?:\s*[@at]+\s*(\d+(?:\.\d+)?)\s*(%|kg|lbs?))?'  # optional weight
            r'(?:.*?(\d+(?:\.\d+)?)\s*(lbs?|kg))?',  # optional explicit weight
            line, re.IGNORECASE
        )
        if set_match:
            sets, reps = int(set_match.group(1)), int(set_match.group(2))
            pct   = float(set_match.group(3)) if set_match.group(3) and set_match.group(4) == '%' else None
            wt    = float(set_match.group(3)) if set_match.group(3) and set_match.group(4) in ('kg','lbs','lbs') else None
            if not wt and set_match.group(5):
                wt = float(set_match.group(5))
            current['sets'].append({'sets': sets, 'reps': reps, 'pct': pct, 'weight': wt, 'raw': line})

        # RPE lines: "3x3 @RPE 8"
        rpe_match = re.search(r'(\d+)\s*[xX×]\s*(\d+)\s*@?\s*RPE\s*(\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if rpe_match:
            current['sets'].append({
                'sets': int(rpe_match.group(1)), 'reps': int(rpe_match.group(2)),
                'rpe': float(rpe_match.group(3)), 'raw': line
            })

        # Percentage-only lines: "@80%"
        pct_only = re.search(r'@\s*(\d+(?:\.\d+)?)\s*%', line)
        if pct_only and current['sets']:
            current['sets'][-1]['pct'] = float(pct_only.group(1))

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
        'programs': [p['program_date'] for p in programs],
    }

    if garmin_df is not None and not garmin_df.empty:
        matched = match_garmin_to_program(garmin_df, programs)
        pl_sessions = matched[matched['pl_program_date'].notna()]
        out['pl_sessions_tracked'] = len(pl_sessions)

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
