"""
analysis/garmin.py
Cross-analysis of Garmin biometric data with SugarWOD performance.

Data coverage:
  Activities + Sleep + Menstrual : Nov 2024 → Mar 2026
  HRV / Resting HR               : Sep 2025 → Mar 2026
  SugarWOD overlap               : Nov 2024 → Mar 2026 (~16 months)

All join keys are calendar date (date of workout = date you woke up that morning).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

BLUE   = '#4a9eed'
GREEN  = '#22c55e'
AMBER  = '#f59e0b'
RED    = '#ef4444'
PURPLE = '#8b5cf6'
PINK   = '#ec4899'
TEAL   = '#14b8a6'

PHASE_COLORS = {
    'menstrual':     RED,
    'follicular':    GREEN,
    'fertile_window': AMBER,
    'luteal':        PURPLE,
}
PHASE_LABELS = {
    'menstrual':     'Menstrual',
    'follicular':    'Follicular',
    'fertile_window': 'Fertile Window',
    'luteal':        'Luteal',
}


def _merge_sugarwod_garmin(sugarwod_df, garmin_daily_df, garmin_activities_df=None):
    """
    Inner join SugarWOD with Garmin daily metrics by date.
    Only returns rows in the overlap window (Nov 2024 onward).
    Optionally also joins the CrossFit Garmin activity for that day.
    """
    sw = sugarwod_df.copy()
    sw['date'] = pd.to_datetime(sw['date']).dt.normalize()

    gd = garmin_daily_df.copy()
    gd['date'] = pd.to_datetime(gd['date']).dt.normalize()

    # Restrict to overlap window
    overlap_start = pd.Timestamp('2024-11-01')
    sw = sw[sw['date'] >= overlap_start]

    merged = sw.merge(gd, on='date', how='left')

    if garmin_activities_df is not None:
        ga = garmin_activities_df.copy()
        ga['date'] = pd.to_datetime(ga['date']).dt.normalize()
        # For days with multiple activities, keep the CrossFit one first, then any
        cf = ga[ga['activity_category'] == 'crossfit'].copy()
        cf_agg = cf.groupby('date').agg(
            cf_avg_hr=('avg_hr', 'mean'),
            cf_max_hr=('max_hr', 'mean'),
            cf_duration_mins=('duration_mins', 'sum'),
            cf_body_battery_drain=('body_battery_drain', 'sum'),
            cf_hr_zone4_mins=('hr_zone4_mins', 'sum'),
            cf_hr_zone5_mins=('hr_zone5_mins', 'sum'),
            cf_training_effect=('training_effect_label', 'first'),
        ).reset_index()
        merged = merged.merge(cf_agg, on='date', how='left')

    return merged


# ── Sleep vs Performance ───────────────────────────────────────────────────────

def sleep_vs_performance(sugarwod_df, garmin_daily_df):
    """
    Does sleep quality predict next-day CrossFit performance?
    Shows RX rate by sleep score bucket and sleep duration vs RX rate.
    """
    df = _merge_sugarwod_garmin(sugarwod_df, garmin_daily_df)
    df = df[df['sleep_score'].notna() & df['rx_or_scaled'].notna()].copy()

    if len(df) < 10:
        return None

    df['rx_binary'] = (df['rx_or_scaled'].str.lower() == 'rx').astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sleep vs CrossFit Performance', fontsize=15, fontweight='bold', y=1.01)

    # ── Chart 1: RX rate by sleep score bucket ──────────────────────
    ax = axes[0]
    bins   = [0, 60, 70, 80, 90, 100]
    labels = ['<60', '60–70', '70–80', '80–90', '90+']
    df['sleep_bucket'] = pd.cut(df['sleep_score'], bins=bins, labels=labels, right=False)
    bucket_stats = df.groupby('sleep_bucket', observed=True).agg(
        rx_rate=('rx_binary', 'mean'),
        count=('rx_binary', 'count'),
    ).reset_index()
    bucket_stats['rx_pct'] = bucket_stats['rx_rate'] * 100

    colors = [GREEN if r >= 0.6 else AMBER if r >= 0.45 else RED
              for r in bucket_stats['rx_rate']]
    bars = ax.bar(bucket_stats['sleep_bucket'].astype(str), bucket_stats['rx_pct'],
                  color=colors, alpha=0.85, edgecolor='white')
    for bar, (_, row) in zip(bars, bucket_stats.iterrows()):
        if row['count'] > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{row['rx_pct']:.0f}%\n(n={row['count']})",
                    ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Sleep Score')
    ax.set_ylabel('RX Rate (%)')
    ax.set_title('RX Rate by Sleep Score')
    ax.set_ylim(0, 100)

    # ── Chart 2: Sleep duration vs RX ──────────────────────────────
    ax = axes[1]
    rx_mask  = df['rx_binary'] == 1
    sc_mask  = df['rx_binary'] == 0
    ax.scatter(df.loc[sc_mask, 'sleep_total_hrs'], df.loc[sc_mask, 'rx_binary'] + np.random.uniform(-0.05, 0.05, sc_mask.sum()),
               alpha=0.4, color=RED, s=20, label='Scaled')
    ax.scatter(df.loc[rx_mask, 'sleep_total_hrs'], df.loc[rx_mask, 'rx_binary'] + np.random.uniform(-0.05, 0.05, rx_mask.sum()),
               alpha=0.4, color=GREEN, s=20, label='RX')

    # Binned mean line
    df['sleep_hrs_bin'] = pd.cut(df['sleep_total_hrs'], bins=8)
    bin_mean = df.groupby('sleep_hrs_bin', observed=True)['rx_binary'].mean()
    bin_mid  = [(b.left + b.right) / 2 for b in bin_mean.index]
    ax.plot(bin_mid, bin_mean.values, color=BLUE, linewidth=2.5, marker='o', markersize=5, label='Avg RX rate')

    corr = df['sleep_total_hrs'].corr(df['rx_binary'])
    ax.set_xlabel('Sleep Duration (hours)')
    ax.set_ylabel('RX (1) / Scaled (0)')
    ax.set_title(f'Sleep Duration vs Performance  (r={corr:.2f})')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Scaled', 'RX'])
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ── HRV vs Performance ─────────────────────────────────────────────────────────

def hrv_vs_performance(sugarwod_df, garmin_daily_df):
    """
    Morning HRV status vs same-day CrossFit RX rate.
    Only meaningful from Sep 2025 onward when HRV data exists.
    """
    df = _merge_sugarwod_garmin(sugarwod_df, garmin_daily_df)
    df = df[df['hrv_value'].notna() & df['rx_or_scaled'].notna()].copy()

    if len(df) < 5:
        return None

    df['rx_binary'] = (df['rx_or_scaled'].str.lower() == 'rx').astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('HRV vs CrossFit Performance', fontsize=15, fontweight='bold', y=1.01)

    # ── Chart 1: RX rate by HRV status ──────────────────────────────
    ax = axes[0]
    status_order  = ['POOR', 'UNBALANCED', 'IN_RANGE', 'BALANCED']
    status_colors = [RED, AMBER, BLUE, GREEN]
    status_stats  = (
        df[df['hrv_status'].isin(status_order)]
        .groupby('hrv_status')
        .agg(rx_rate=('rx_binary', 'mean'), count=('rx_binary', 'count'))
        .reindex(status_order).dropna()
    )
    status_stats['rx_pct'] = status_stats['rx_rate'] * 100
    visible = [s for s in status_order if s in status_stats.index]
    colors  = [status_colors[status_order.index(s)] for s in visible]
    bars = ax.bar(visible, status_stats.loc[visible, 'rx_pct'], color=colors, alpha=0.85, edgecolor='white')
    for bar, s in zip(bars, visible):
        n = status_stats.loc[s, 'count']
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{status_stats.loc[s, 'rx_pct']:.0f}%\n(n={n:.0f})",
                ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Morning HRV Status')
    ax.set_ylabel('RX Rate (%)')
    ax.set_title('RX Rate by HRV Status')
    ax.set_ylim(0, 100)

    # ── Chart 2: HRV value vs RX scatter ────────────────────────────
    ax = axes[1]
    rx_mask = df['rx_binary'] == 1
    sc_mask = df['rx_binary'] == 0
    ax.scatter(df.loc[sc_mask, 'hrv_value'], df.loc[sc_mask, 'rx_binary'] + np.random.uniform(-0.05, 0.05, sc_mask.sum()),
               alpha=0.5, color=RED, s=25, label='Scaled')
    ax.scatter(df.loc[rx_mask, 'hrv_value'], df.loc[rx_mask, 'rx_binary'] + np.random.uniform(-0.05, 0.05, rx_mask.sum()),
               alpha=0.5, color=GREEN, s=25, label='RX')

    # Baseline band
    baseline_lo = df['hrv_baseline_lower'].median()
    baseline_hi = df['hrv_baseline_upper'].median()
    if pd.notna(baseline_lo) and pd.notna(baseline_hi):
        ax.axvspan(baseline_lo, baseline_hi, alpha=0.1, color=BLUE, label=f'Your baseline ({baseline_lo:.0f}–{baseline_hi:.0f})')

    corr = df['hrv_value'].corr(df['rx_binary'])
    ax.set_xlabel('HRV (ms)')
    ax.set_ylabel('RX (1) / Scaled (0)')
    ax.set_title(f'HRV Value vs Performance  (r={corr:.2f})')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Scaled', 'RX'])
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ── Menstrual Cycle vs Performance ────────────────────────────────────────────

def menstrual_vs_performance(sugarwod_df, garmin_daily_df):
    """
    RX rate, sleep quality, and training load by menstrual cycle phase.
    """
    df = _merge_sugarwod_garmin(sugarwod_df, garmin_daily_df)
    df = df[df['cycle_phase'].notna()].copy()

    if len(df) < 20:
        return None

    df['rx_binary'] = (df['rx_or_scaled'].str.lower() == 'rx').astype(int)

    phase_order = ['menstrual', 'follicular', 'fertile_window', 'luteal']

    phase_stats = df.groupby('cycle_phase').agg(
        rx_rate=('rx_binary', 'mean'),
        count=('rx_binary', 'count'),
        avg_sleep_score=('sleep_score', 'mean'),
        avg_sleep_hrs=('sleep_total_hrs', 'mean'),
        avg_sentiment=('sentiment', 'mean') if 'sentiment' in df.columns else ('rx_binary', 'count'),
    ).reindex(phase_order).dropna(how='all')
    phase_stats['rx_pct'] = phase_stats['rx_rate'] * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Menstrual Cycle Phase vs Performance & Recovery', fontsize=15, fontweight='bold', y=1.01)

    visible_phases = [p for p in phase_order if p in phase_stats.index]
    colors = [PHASE_COLORS[p] for p in visible_phases]
    labels = [PHASE_LABELS[p] for p in visible_phases]

    # ── Chart 1: RX rate by phase ───────────────────────────────────
    ax = axes[0]
    bars = ax.bar(labels, phase_stats.loc[visible_phases, 'rx_pct'],
                  color=colors, alpha=0.85, edgecolor='white')
    for bar, phase in zip(bars, visible_phases):
        n   = phase_stats.loc[phase, 'count']
        pct = phase_stats.loc[phase, 'rx_pct']
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%\n(n={n:.0f})", ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('RX Rate (%)')
    ax.set_title('RX Rate by Cycle Phase')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=15)

    # ── Chart 2: Sleep score by phase ──────────────────────────────
    ax = axes[1]
    sleep_vals = phase_stats.loc[visible_phases, 'avg_sleep_score']
    bars = ax.bar(labels, sleep_vals, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, sleep_vals):
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Average Sleep Score')
    ax.set_title('Sleep Quality by Cycle Phase')
    ax.tick_params(axis='x', rotation=15)

    # ── Chart 3: Sleep duration by phase ───────────────────────────
    ax = axes[2]
    sleep_hrs = phase_stats.loc[visible_phases, 'avg_sleep_hrs']
    bars = ax.bar(labels, sleep_hrs, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, sleep_hrs):
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.1f}h", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Avg Sleep Duration (hours)')
    ax.set_title('Sleep Duration by Cycle Phase')
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    return fig


# ── Body Battery & HR Zones ───────────────────────────────────────────────────

def body_battery_chart(garmin_activities_df):
    """
    Body Battery drain per activity category over time.
    Shows how much each type of workout costs energetically.
    """
    df = garmin_activities_df.copy()
    df = df[df['body_battery_drain'].notna()].copy()
    df['date'] = pd.to_datetime(df['date'])
    # Body Battery drain is negative — flip sign for readability
    df['drain'] = df['body_battery_drain'].abs()

    if len(df) < 5:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Body Battery Drain by Activity', fontsize=15, fontweight='bold', y=1.01)

    # ── Chart 1: Avg drain by category ──────────────────────────────
    ax = axes[0]
    cat_stats = (df.groupby('activity_category')
                   .agg(avg_drain=('drain', 'mean'), count=('drain', 'count'))
                   .sort_values('avg_drain', ascending=True))
    cat_colors = {
        'crossfit':    RED,
        'powerlifting': PURPLE,
        'cycling':     BLUE,
        'strength':    AMBER,
        'hiit':        PINK,
        'walking':     GREEN,
    }
    colors = [cat_colors.get(c, TEAL) for c in cat_stats.index]
    bars = ax.barh(cat_stats.index, cat_stats['avg_drain'], color=colors, alpha=0.85)
    for bar, (idx, row) in zip(bars, cat_stats.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{row['avg_drain']:.0f} pts  (n={row['count']:.0f})",
                va='center', fontsize=8)
    ax.set_xlabel('Avg Body Battery Drain (points)')
    ax.set_title('Average Drain by Activity Type')
    ax.set_xlim(0, cat_stats['avg_drain'].max() * 1.3)

    # ── Chart 2: Body Battery drain over time (CrossFit only) ───────
    ax = axes[1]
    cf = df[df['activity_category'] == 'crossfit'].copy()
    cf = cf.sort_values('date')
    cf['rolling_drain'] = cf['drain'].rolling(8, min_periods=3).mean()
    ax.scatter(cf['date'], cf['drain'], alpha=0.35, color=RED, s=15, label='Per session')
    ax.plot(cf['date'], cf['rolling_drain'], color=RED, linewidth=2, label='8-session avg')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Body Battery Drain (points)')
    ax.set_title('CrossFit Energy Cost Over Time')
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def hr_zones_chart(garmin_activities_df):
    """
    HR zone distribution for CrossFit sessions — shows training intensity profile.
    """
    df = garmin_activities_df.copy()
    cf = df[df['activity_category'] == 'crossfit'].copy()

    zone_cols = ['hr_zone1_mins', 'hr_zone2_mins', 'hr_zone3_mins', 'hr_zone4_mins', 'hr_zone5_mins']
    cf = cf.dropna(subset=zone_cols, how='all')

    if len(cf) < 5:
        return None

    zone_means = cf[zone_cols].mean()
    zone_labels = ['Zone 1\n(Recovery)', 'Zone 2\n(Aerobic Base)', 'Zone 3\n(Tempo)',
                   'Zone 4\n(Threshold)', 'Zone 5\n(VO2 Max)']
    zone_colors = [GREEN, TEAL, AMBER, RED, PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Heart Rate Zones in CrossFit Sessions', fontsize=15, fontweight='bold', y=1.01)

    # ── Chart 1: Avg minutes per zone ───────────────────────────────
    ax = axes[0]
    bars = ax.bar(zone_labels, zone_means.values, color=zone_colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, zone_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f} min", ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Average Minutes per Session')
    ax.set_title('Avg Time in Each HR Zone (CrossFit)')

    # ── Chart 2: Zone 4+5 (high intensity) over time ────────────────
    ax = axes[1]
    cf = cf.sort_values('date')
    cf['date'] = pd.to_datetime(cf['date'])
    cf['high_intensity_mins'] = cf['hr_zone4_mins'].fillna(0) + cf['hr_zone5_mins'].fillna(0)
    cf['rolling_hi'] = cf['high_intensity_mins'].rolling(8, min_periods=3).mean()
    ax.scatter(cf['date'], cf['high_intensity_mins'], alpha=0.3, color=RED, s=15)
    ax.plot(cf['date'], cf['rolling_hi'], color=RED, linewidth=2, label='8-session avg')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Minutes in Zone 4+5')
    ax.set_title('High-Intensity Training Load Over Time')
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ── Weekly Training Load ───────────────────────────────────────────────────────

def weekly_load_chart(garmin_activities_df):
    """
    Weekly training load breakdown: CrossFit vs Powerlifting sessions and Body Battery cost.
    """
    df = garmin_activities_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W').dt.start_time
    df['drain'] = df['body_battery_drain'].abs().fillna(0)

    weekly = df.groupby(['week', 'activity_category']).agg(
        sessions=('activity_id', 'count'),
        total_drain=('drain', 'sum'),
    ).reset_index()

    cf_weekly = weekly[weekly['activity_category'] == 'crossfit'].set_index('week')
    pl_weekly = weekly[weekly['activity_category'] == 'powerlifting'].set_index('week')
    all_weeks = pd.date_range(df['week'].min(), df['week'].max(), freq='W-MON')

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Weekly Training Load', fontsize=15, fontweight='bold')

    # ── Chart 1: Sessions per week ───────────────────────────────────
    ax = axes[0]
    cf_sessions = cf_weekly['sessions'].reindex(all_weeks, fill_value=0)
    pl_sessions = pl_weekly['sessions'].reindex(all_weeks, fill_value=0)
    ax.bar(all_weeks, cf_sessions, width=5, color=RED, alpha=0.75, label='CrossFit')
    ax.bar(all_weeks, pl_sessions, width=5, bottom=cf_sessions, color=PURPLE, alpha=0.75, label='Powerlifting')
    ax.axhline(cf_sessions.mean() + pl_sessions.mean(), color=AMBER, linestyle='--',
               linewidth=1.5, label=f'Avg total: {(cf_sessions+pl_sessions).mean():.1f}/week')
    ax.set_ylabel('Sessions')
    ax.set_title('Weekly Session Count')
    ax.legend(fontsize=8)

    # ── Chart 2: Body Battery drain per week ────────────────────────
    ax = axes[1]
    cf_drain = cf_weekly['total_drain'].reindex(all_weeks, fill_value=0)
    pl_drain = pl_weekly['total_drain'].reindex(all_weeks, fill_value=0)
    ax.bar(all_weeks, cf_drain, width=5, color=RED, alpha=0.75, label='CrossFit')
    ax.bar(all_weeks, pl_drain, width=5, bottom=cf_drain, color=PURPLE, alpha=0.75, label='Powerlifting')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Total Body Battery Drain')
    ax.set_title('Weekly Energy Cost')
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ── Summary for agent ──────────────────────────────────────────────────────────

def summary(sugarwod_df, garmin_daily_df, garmin_activities_df):
    """Return key cross-analysis stats for the agent to reason over."""
    df = _merge_sugarwod_garmin(sugarwod_df, garmin_daily_df, garmin_activities_df)
    df['rx_binary'] = (df['rx_or_scaled'].str.lower() == 'rx').astype(int)

    out = {
        'overlap_window': 'Nov 2024 → Mar 2026',
        'sugarwod_workouts_in_window': int(len(df)),
        'workouts_with_sleep_data': int(df['sleep_score'].notna().sum()),
        'workouts_with_hrv_data': int(df['hrv_value'].notna().sum()),
        'workouts_with_cycle_data': int(df['cycle_phase'].notna().sum()),
    }

    # Sleep correlations
    sleep_sub = df[df['sleep_score'].notna()]
    if len(sleep_sub) >= 10:
        out['sleep_score_vs_rx_corr'] = round(sleep_sub['sleep_score'].corr(sleep_sub['rx_binary']), 3)
        out['sleep_hrs_vs_rx_corr']   = round(sleep_sub['sleep_total_hrs'].corr(sleep_sub['rx_binary']), 3)
        out['avg_sleep_score_on_rx_days']     = round(sleep_sub[sleep_sub['rx_binary'] == 1]['sleep_score'].mean(), 1)
        out['avg_sleep_score_on_scaled_days'] = round(sleep_sub[sleep_sub['rx_binary'] == 0]['sleep_score'].mean(), 1)

    # HRV correlations
    hrv_sub = df[df['hrv_value'].notna()]
    if len(hrv_sub) >= 5:
        out['hrv_vs_rx_corr'] = round(hrv_sub['hrv_value'].corr(hrv_sub['rx_binary']), 3)
        out['avg_hrv_on_rx_days']     = round(hrv_sub[hrv_sub['rx_binary'] == 1]['hrv_value'].mean(), 1)
        out['avg_hrv_on_scaled_days'] = round(hrv_sub[hrv_sub['rx_binary'] == 0]['hrv_value'].mean(), 1)

    # Menstrual phase stats
    cycle_sub = df[df['cycle_phase'].notna()]
    if len(cycle_sub) >= 10:
        phase_rx = cycle_sub.groupby('cycle_phase')['rx_binary'].mean().round(3)
        out['rx_rate_by_cycle_phase'] = phase_rx.to_dict()

    # Body Battery
    ga = garmin_activities_df.copy()
    cf = ga[ga['activity_category'] == 'crossfit']
    if len(cf) > 0 and cf['body_battery_drain'].notna().sum() > 5:
        out['avg_body_battery_drain_crossfit']      = round(cf['body_battery_drain'].abs().mean(), 1)
        out['avg_body_battery_drain_powerlifting']  = round(
            ga[ga['activity_category'] == 'powerlifting']['body_battery_drain'].abs().mean(), 1
        )

    return out
