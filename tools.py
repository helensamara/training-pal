"""
tools.py
Agent-facing tools. Each function wraps an analysis module and returns
a JSON-serialisable dict that Claude can reason over.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loader import load_sugarwod, load_garmin_activities, load_garmin_daily
from analysis import attendance, sentiment, performance, ml_models, garmin, powerlifting

_df              = None
_garmin_daily    = None
_garmin_activities = None


def _get_df():
    global _df
    if _df is None:
        _df = load_sugarwod()
        _df = sentiment.enrich(_df)
    return _df


def _get_garmin_daily():
    global _garmin_daily
    if _garmin_daily is None:
        _garmin_daily = load_garmin_daily()
    return _garmin_daily


def _get_garmin_activities():
    global _garmin_activities
    if _garmin_activities is None:
        _garmin_activities = load_garmin_activities()
    return _garmin_activities


def tool_attendance_summary():
    """Attendance stats: sessions/month, gaps, best/worst periods."""
    return attendance.summary(_get_df())


def tool_sentiment_summary():
    """Sentiment stats: avg score, correlation with RX, top positive/negative sessions."""
    return sentiment.summary(_get_df())


def tool_performance_summary():
    """Performance stats: RX rate, PR count, scaling ratio trend."""
    return performance.summary(_get_df())


def tool_cluster_workouts():
    """KMeans cluster profiles — workout archetypes with counts and RX/PR rates."""
    _, profiles, _ = ml_models.cluster_workouts(_get_df())
    plt.close('all')
    return profiles[['name','count','rx_rate','pr_rate']].to_dict('index')


def tool_detect_anomalies():
    """Isolation Forest — list of flagged unusual strength sessions."""
    anom, _ = ml_models.detect_anomalies(_get_df())
    plt.close('all')
    return anom.assign(date=anom['date'].astype(str)).to_dict('records')


def tool_forecast_prs():
    """Linear Regression — forecasted days to next PR per lift."""
    forecasts, _ = ml_models.forecast_prs(_get_df())
    plt.close('all')
    return forecasts


# ── Registry ──────────────────────────────────────────────────────

TOOLS = {
    'attendance_summary':  tool_attendance_summary,
    'sentiment_summary':   tool_sentiment_summary,
    'performance_summary': tool_performance_summary,
    'cluster_workouts':    tool_cluster_workouts,
    'detect_anomalies':    tool_detect_anomalies,
    'forecast_prs':        tool_forecast_prs,
}

TOOL_SCHEMAS = [
    {'name': 'attendance_summary',
     'description': 'Get attendance stats: sessions per month, rest gaps over 7 days, best/worst training periods.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'sentiment_summary',
     'description': 'Analyse sentiment in workout notes. Returns avg score, correlation with RX rate, most positive/negative sessions.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'performance_summary',
     'description': 'Get performance stats: overall RX rate, PR count, scaling ratio (actual vs prescribed weight) trend.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'cluster_workouts',
     'description': 'Use KMeans to group workouts into archetypes: Strength, Timed Metcon, AMRAP, Accessory.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'detect_anomalies',
     'description': 'Use Isolation Forest to flag statistically unusual strength sessions worth investigating.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'forecast_prs',
     'description': 'Use Linear Regression on lift progression curves to forecast when the next PR is likely per lift.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
]


def tool_garmin_summary():
    """Cross-analysis of Garmin biometrics with SugarWOD performance."""
    gd = _get_garmin_daily()
    ga = _get_garmin_activities()
    if gd.empty or ga.empty:
        return {'error': 'Garmin data not found. Run scripts/parse_garmin_export.py first.'}
    return garmin.summary(_get_df(), gd, ga)


TOOLS['garmin_summary'] = tool_garmin_summary
TOOL_SCHEMAS.append({
    'name': 'garmin_summary',
    'description': (
        'Cross-analysis of Garmin biometric data with CrossFit performance. '
        'Returns correlations between sleep score/duration and RX rate, '
        'HRV vs performance, Body Battery drain per activity type, '
        'and RX rate broken down by menstrual cycle phase.'
    ),
    'input_schema': {'type': 'object', 'properties': {}, 'required': []},
})


def tool_powerlifting_summary():
    """Powerlifting program stats: lift progression (Bench/Squat/Deadlift) across all programs."""
    ga = _get_garmin_activities()
    return powerlifting.summary(garmin_df=ga if not ga.empty else None)


TOOLS['powerlifting_summary'] = tool_powerlifting_summary
TOOL_SCHEMAS.append({
    'name': 'powerlifting_summary',
    'description': (
        'Powerlifting program analysis. Returns max prescribed weight progression for '
        'Bench, Squat, and Deadlift across all 18 programs from coach Tom Kean '
        '(Sep 2024 – Mar 2026), including total gains in kg, volume per program, '
        'and full history per lift. Use this for ANY question about powerlifting '
        'strength, programs, or progress — completely separate from CrossFit data.'
    ),
    'input_schema': {'type': 'object', 'properties': {}, 'required': []},
})


def run_tool(name, _input=None):
    if name not in TOOLS:
        return {'error': f'Unknown tool: {name}'}
    return TOOLS[name]()
