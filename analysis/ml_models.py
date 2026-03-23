"""
analysis/ml_models.py
Machine learning layer — clustering, anomaly detection, PR forecasting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BLUE   = '#4a9eed'
GREEN  = '#22c55e'
AMBER  = '#f59e0b'
RED    = '#ef4444'
PURPLE = '#8b5cf6'


# ── Clustering ────────────────────────────────────────────────────

def _build_features(df):
    c = df.copy()
    c['is_rx_num']        = (c['rx_or_scaled'] == 'RX').astype(int)
    c['is_pr_num']        = c['is_pr'].astype(int)
    c['is_strength']      = (c['score_type'] == 'Load').astype(int)
    c['is_timed']         = c['best_result_display'].str.contains(r'\d+:\d+', na=False).astype(int)
    c['is_amrap']         = (c['score_type'] == 'Rounds + Reps').astype(int)
    c['has_barbell']      = (c['barbell_lift'] != '').astype(int)
    c['sentiment_filled'] = c['sentiment'].fillna(0) if 'sentiment' in c.columns else 0
    return c


def _name_cluster(row):
    if row['strength_rate'] > 0.5: return 'Strength / Lifting'
    if row['amrap_rate']    > 0.3: return 'AMRAP Metcon'
    if row['timed_rate']    > 0.3: return 'Timed Metcon'
    return 'Accessory / Mixed'


def cluster_workouts(df, k=4):
    """
    KMeans clustering of workouts into archetypes.
    Returns (df_with_clusters, profiles_df, fig).
    """
    df = _build_features(df)
    features = ['is_rx_num', 'is_strength', 'is_timed', 'is_amrap',
                'has_barbell', 'sentiment_filled', 'is_pr_num']
    X = StandardScaler().fit_transform(df[features].fillna(0))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X)

    profiles = df.groupby('cluster').agg(
        count=('cluster', 'size'),
        rx_rate=('is_rx_num', 'mean'),
        strength_rate=('is_strength', 'mean'),
        timed_rate=('is_timed', 'mean'),
        amrap_rate=('is_amrap', 'mean'),
        pr_rate=('is_pr_num', 'mean'),
    ).round(2)
    profiles['name'] = profiles.apply(_name_cluster, axis=1)
    # Deduplicate names — if two clusters get the same label, append cluster ID
    seen = {}
    deduped = []
    for cid, name in profiles['name'].items():
        if name in seen:
            deduped.append(f'{name} ({cid})')
        else:
            seen[name] = cid
            deduped.append(name)
    profiles['name'] = deduped
    df['cluster_name'] = df['cluster'].map(profiles['name'])

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df['cluster_name'].value_counts()
    colors = [BLUE, GREEN, AMBER, PURPLE][:len(counts)]
    bars   = ax.bar(counts.index, counts.values, color=colors, alpha=0.85)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha='center', fontsize=12, fontweight='bold')
    ax.set_title('Workout Archetypes — Auto-Clustered (KMeans)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Number of sessions')
    plt.tight_layout()

    return df, profiles, fig


# ── Anomaly Detection ─────────────────────────────────────────────

def detect_anomalies(df, window=5, z_low=-1.5, z_high=1.5, min_sessions=3):
    """
    Per-lift rolling z-score anomaly detection.

    For each lift independently, computes a rolling mean and std over the
    last `window` sessions.  A session is flagged when its load deviates
    more than `z_low` (unusually light — fatigue, deload, injury) or
    `z_high` (unusually heavy — big PR jump or data error) standard
    deviations from that lift's recent average.

    This is far more meaningful than a global Isolation Forest because
    95 lbs is normal for a Snatch but anomalous for a Back Squat.

    Returns (anomalies_df, fig).
    """
    strength = df[(df['score_type'] == 'Load') & (df['barbell_lift'] != '')].copy()
    strength['load'] = pd.to_numeric(strength['best_result_raw'], errors='coerce')
    strength = strength[strength['load'].notna()].sort_values('date').reset_index(drop=True)

    segments = []
    for lift, grp in strength.groupby('barbell_lift'):
        g = grp.sort_values('date').copy()
        if len(g) < min_sessions + 1:
            continue
        g['rolling_mean']  = g['load'].rolling(window, min_periods=min_sessions).mean()
        g['rolling_std']   = g['load'].rolling(window, min_periods=min_sessions).std()
        g['z_score']       = (g['load'] - g['rolling_mean']) / g['rolling_std'].replace(0, np.nan)
        g['is_anomaly']    = (g['z_score'] < z_low) | (g['z_score'] > z_high)
        g['deviation_lbs'] = (g['load'] - g['rolling_mean']).round(1)
        segments.append(g)

    if not segments:
        return pd.DataFrame(), plt.figure()

    all_data  = pd.concat(segments).sort_values('date')
    anomalies = all_data[all_data['is_anomaly'] == True].copy()
    anomalies['deviation_lbs'] = (anomalies['load'] - anomalies['rolling_mean']).round(1)
    anomalies['direction']     = anomalies['z_score'].apply(
        lambda z: 'unusually light' if z < 0 else 'unusually heavy'
    )

    # ── Chart ─────────────────────────────────────────────────────
    # One subplot per lift that has enough data, arranged in a grid
    lifts_with_data = [g['barbell_lift'].iloc[0] for g in segments]
    n = len(lifts_with_data)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (lift, grp) in enumerate(all_data.groupby('barbell_lift')):
        ax  = axes[idx]
        anom = grp[grp['is_anomaly'] == True]
        norm = grp[grp['is_anomaly'] != True]

        ax.plot(grp['date'], grp['rolling_mean'], color=AMBER,
                linewidth=1.5, linestyle='--', label='Rolling avg', zorder=1)
        ax.fill_between(grp['date'],
                        grp['rolling_mean'] - grp['rolling_std'].abs(),
                        grp['rolling_mean'] + grp['rolling_std'].abs(),
                        color=AMBER, alpha=0.10, zorder=0)
        ax.scatter(norm['date'],  norm['load'],  color=BLUE, s=30, alpha=0.6,
                   label='Normal', zorder=2)
        if len(anom):
            ax.scatter(anom['date'], anom['load'], color=RED, s=90,
                       marker='X', zorder=5, label=f'Flagged ({len(anom)})')
            for _, row in anom.iterrows():
                sign = '+' if row['deviation_lbs'] > 0 else ''
                ax.annotate(
                    f"{sign}{row['deviation_lbs']:.0f} lbs",
                    (row['date'], row['load']),
                    xytext=(0, 10 if row['z_score'] > 0 else -14),
                    textcoords='offset points',
                    ha='center', fontsize=7.5, color=RED, fontweight='bold',
                )
        ax.set_title(lift, fontsize=11, fontweight='bold')
        ax.set_ylabel('Load (lbs)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.legend(fontsize=8)

    for j in range(len(lifts_with_data), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Unusual Strength Sessions — Per-Lift Z-Score Analysis',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    return (
        anomalies[['date', 'barbell_lift', 'load', 'rolling_mean',
                   'z_score', 'deviation_lbs', 'direction']],
        fig,
    )


# ── PR Forecasting ────────────────────────────────────────────────

# Config per score type: (score_col, group_col, higher_is_better, ylabel, pr_delta)
_FORECAST_CFG = {
    'Load':          ('score_load',      'barbell_lift', True,  'lbs',         2.5),
    'Rounds + Reps': ('best_result_raw', 'title',        True,  'rounds+reps', 0.05),
    'Reps':          ('score_reps',      'title',        True,  'reps',        1.0),
    'Time':          ('score_seconds',   'title',        False, 'seconds',     10.0),
}

_BLOCK_HALF_LIFE   = 180   # days — recency decay for weighted regression
_MIN_STAGNANT      = 6     # sessions without improvement → new block boundary
_MIN_BLOCK_SESSIONS = 4    # fall back to full history if block is smaller


def _detect_block_start(running_best, min_stagnant=_MIN_STAGNANT):
    """Return (block_start_iloc, is_plateau) for the most recent training block.

    A block boundary is a gap of >= min_stagnant consecutive sessions where the
    running best did not improve.  is_plateau is True when the tail of the series
    (last min_stagnant entries) shows no improvement.
    """
    rb = np.asarray(running_best, dtype=float)
    n  = len(rb)
    if n <= min_stagnant:
        return 0, False

    # Indices where the running best actually improved
    improving = np.where(np.diff(rb) > 0)[0] + 1

    # Plateau: no improvement in the last min_stagnant sessions
    tail_gap   = (n - 1 - improving[-1]) if len(improving) else n
    is_plateau = len(improving) == 0 or tail_gap >= min_stagnant

    if len(improving) == 0:
        return 0, is_plateau

    if is_plateau:
        # Current block = everything from the last improvement onward
        return int(improving[-1]), True

    # Improving: scan backwards for the gap that opened this block
    for i in range(len(improving) - 1, 0, -1):
        if improving[i] - improving[i - 1] >= min_stagnant:
            return int(improving[i]), False

    return 0, False  # whole history is one improvement block


def _best_fit(days, values, weights):
    """Fit linear, log, and sqrt models; return the one with highest weighted R².

    Returns (model, transform_fn, label, r2).
    """
    candidates = [
        ('linear', lambda x: x.reshape(-1, 1),        lambda x: x),
        ('log',    lambda x: np.log1p(x).reshape(-1,1), lambda x: np.log1p(x)),
        ('sqrt',   lambda x: np.sqrt(x).reshape(-1, 1), lambda x: np.sqrt(x)),
    ]
    best = None
    for label, fit_transform, pred_transform in candidates:
        X = fit_transform(days)
        m = LinearRegression().fit(X, values, sample_weight=weights)
        r2 = r2_score(values, m.predict(X), sample_weight=weights)
        if best is None or r2 > best[3]:
            best = (m, pred_transform, label, r2)
    return best


def forecast_prs(df, top_n=6, horizon_days=90, score_type='Load'):
    """
    Block-aware PR forecasting with recency-weighted regression.

    For each workout, the model:
      1. Detects the current training block (improvement run or plateau phase).
      2. Fits a recency-weighted linear regression on that block only,
         so early history doesn't drag the slope down during active cycles
         and a plateau naturally produces a near-flat forecast.
      3. Falls back to full-history recency-weighted fit if the detected block
         is too small (< _MIN_BLOCK_SESSIONS).

    score_type: 'Load' (lbs, higher=better) | 'Rounds + Reps' (decimal, higher=better)
                'Reps' (count, higher=better) | 'Time' (seconds, lower=better)

    Returns (forecasts_dict, fig).
    """
    if score_type not in _FORECAST_CFG:
        raise ValueError(f'score_type must be one of {list(_FORECAST_CFG)}')

    score_col, group_col, higher_is_better, ylabel, pr_delta = _FORECAST_CFG[score_type]

    sessions = df[df['score_type'] == score_type].copy()
    if score_type == 'Load':
        sessions = sessions[sessions['barbell_lift'] != '']
    sessions['score'] = pd.to_numeric(sessions[score_col], errors='coerce')
    sessions = sessions[sessions['score'].notna()]

    top_groups = sessions[group_col].value_counts().head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    forecasts = {}

    for i, name in enumerate(top_groups):
        ax   = axes[i]
        data = sessions[sessions[group_col] == name].sort_values('date').copy()
        data['days']         = (data['date'] - data['date'].min()).dt.days
        data['running_best'] = (data['score'].cummax()
                                if higher_is_better else data['score'].cummin())

        # ── Block detection ──────────────────────────────────────────
        block_start, is_plateau = _detect_block_start(data['running_best'].values)
        block = (data.iloc[block_start:]
                 if len(data) - block_start >= _MIN_BLOCK_SESSIONS
                 else data)

        # ── Recency-weighted regression on current block ─────────────
        last_day  = int(data['days'].max())
        fit_days  = block['days'].values
        fit_rb    = block['running_best'].values
        weights   = np.exp(-np.log(2) * (last_day - fit_days) / _BLOCK_HALF_LIFE)

        reg, pred_transform, model_label, r2 = _best_fit(fit_days, fit_rb, weights)

        future_days_arr = np.arange(last_day, last_day + horizon_days + 1)
        future_dates    = pd.date_range(data['date'].max(), periods=horizon_days + 1, freq='D')
        y_future        = reg.predict(pred_transform(future_days_arr).reshape(-1, 1))

        # Residual std for a ±1σ confidence band on the forecast
        fitted_rb = reg.predict(pred_transform(fit_days).reshape(-1, 1))
        residual_std = float(np.std(fit_rb - fitted_rb))

        # ── PR prediction ────────────────────────────────────────────
        current_best = float(data['running_best'].iloc[-1])
        if higher_is_better:
            next_pr_day = next(
                (int(fd - last_day) for fd, fv in zip(future_days_arr, y_future)
                 if fv > current_best + pr_delta),
                None,
            )
        else:
            next_pr_day = next(
                (int(fd - last_day) for fd, fv in zip(future_days_arr, y_future)
                 if fv < current_best - pr_delta),
                None,
            )

        forecasts[name] = {
            f'current_best_{ylabel}':          round(current_best, 2),
            f'predicted_{horizon_days}d_{ylabel}': round(float(y_future[-1]), 2),
            'days_to_next_pr': next_pr_day,
            'phase':           'plateau' if is_plateau else 'improving',
            'model':           model_label,
            'r2':              round(r2, 3),
        }

        # ── Chart ────────────────────────────────────────────────────
        today          = data['date'].max()
        forecast_color = RED if is_plateau else AMBER

        # History
        ax.scatter(data['date'], data['score'], color=BLUE, alpha=0.35,
                   s=25, zorder=2, label='Each session')
        ax.plot(data['date'], data['running_best'], color=GREEN,
                linewidth=2, zorder=3, label='Your PR so far')

        # PR stars
        prs_l = data[data['is_pr']]
        if len(prs_l):
            ax.scatter(prs_l['date'], prs_l['score'], color=AMBER,
                       s=100, zorder=5, marker='*', label='Marked PR')

        # Block boundary
        if not is_plateau and block_start > 0:
            ax.axvline(data['date'].iloc[block_start], color='#bbb',
                       linestyle=':', linewidth=1.2, label='New training block')

        # "Today" divider
        ax.axvline(today, color='#555', linewidth=1.2, linestyle='-')
        ax.text(today, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(data['score']) * 1.02,
                ' Today', fontsize=7.5, color='#555', va='top')

        # Forecast line + confidence band
        phase_tag = '(plateau — unlikely to PR)' if is_plateau else '(improving)'
        ax.plot(future_dates, y_future, color=forecast_color,
                linewidth=2, linestyle='--',
                label=f'Forecast {phase_tag}\n{model_label} fit, R²={r2:.2f}')
        ax.fill_between(future_dates,
                        y_future - residual_std,
                        y_future + residual_std,
                        color=forecast_color, alpha=0.12,
                        label='±1σ uncertainty range')

        # Summary box in corner
        pr_text = f'in ~{next_pr_day}d' if next_pr_day else 'not within 90d'
        box_lines = [
            f'Current best: {current_best:.1f} {ylabel}',
            f'In 90 days:   {float(y_future[-1]):.1f} {ylabel}',
            f'Next PR:      {pr_text}',
        ]
        ax.text(0.02, 0.97, '\n'.join(box_lines),
                transform=ax.transAxes, fontsize=7.5,
                va='top', ha='left', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#ccc', alpha=0.85))

        ax.set_title(str(name)[:28], fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.legend(fontsize=7.5, loc='lower right')

    for j in range(len(top_groups), len(axes)):
        axes[j].set_visible(False)

    type_labels = {'Load': 'Lift', 'Rounds + Reps': 'AMRAP', 'Reps': 'Reps', 'Time': 'Timed Workout'}
    plt.suptitle(f'PR Forecasting — {type_labels.get(score_type, score_type)} Progressions',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    return forecasts, fig
