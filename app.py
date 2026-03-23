"""
app.py — Training Pal · mobile-first Streamlit UI
Run with: streamlit run app.py
"""
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loader import load_sugarwod
from analysis import attendance, sentiment, performance, ml_models
from agent import chat

st.set_page_config(
    page_title='Training Pal',
    page_icon='🏋️',
    layout='centered',
    initial_sidebar_state='collapsed',
)

# ── Mobile-first CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

/* Constrain width like a phone app */
.block-container {
    max-width: 480px !important;
    padding: 0.75rem 1rem 5rem !important;
    margin: 0 auto !important;
}

/* Tabs — full width, large touch targets */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #e0e0e0;
}
.stTabs [data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
    font-size: 15px;
    font-weight: 600;
    padding: 12px 4px;
    color: #888;
}
.stTabs [aria-selected="true"] {
    color: #111 !important;
    border-bottom: 3px solid #111 !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 8px 12px;
    margin-bottom: 4px;
}

/* Sticky chat input at bottom */
[data-testid="stChatInputContainer"] {
    position: fixed !important;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: 480px;
    background: white;
    padding: 10px 12px 14px;
    border-top: 1px solid #e0e0e0;
    z-index: 999;
}

/* Suggestion + chart buttons — full width, rounded, tappable */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 14px;
    text-align: left;
    border: 1px solid #e0e0e0 !important;
    background: #fafafa !important;
    color: #333 !important;
    margin-bottom: 4px;
}
.stButton > button:hover {
    background: #f0f0f0 !important;
    border-color: #bbb !important;
}

/* Metrics on stats tab */
[data-testid="stMetric"] {
    background: #f7f7f7;
    border-radius: 12px;
    padding: 12px;
}

/* Charts fill width */
.stPlotlyChart, .stImage, [data-testid="stPyplot"] {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load data (cached) ─────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_sugarwod()
    df = sentiment.enrich(df)
    return df

try:
    df = get_data()
except Exception as e:
    st.error(f'Could not load data/workouts.csv: {e}')
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────
tab_chat, tab_charts, tab_stats = st.tabs(['💬  Chat', '📊  Charts', '📋  Stats'])

# ══════════════════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════════════════
with tab_chat:
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Render message history
    for msg in st.session_state['messages']:
        if isinstance(msg, dict) and isinstance(msg.get('content'), str):
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    # Suggested questions when chat is empty
    if not st.session_state['messages']:
        st.markdown('**Ask me anything about your training:**')
        suggestions = [
            '🗓 How consistent have I been training?',
            '🏆 When is my next PR likely?',
            '😊 What do my notes say about how I feel?',
            '⚠️ Am I showing signs of overtraining?',
            '🤖 What are my workout archetypes?',
            '📅 Which months had the most PRs?',
        ]
        for q in suggestions:
            if st.button(q, key=f'sug_{q}'):
                st.session_state['pending'] = q.split(' ', 1)[1]  # strip emoji prefix
                st.rerun()

    user_input = st.session_state.pop('pending', None) or st.chat_input('Ask your agent...')

    if user_input:
        with st.chat_message('user'):
            st.markdown(user_input)

        history = list(st.session_state['messages'])
        history.append({'role': 'user', 'content': user_input})

        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                try:
                    reply, updated = chat(history)
                    st.markdown(reply)
                except Exception as e:
                    reply = f'Error: {e}'
                    updated = history + [{'role': 'assistant', 'content': reply}]
                    st.error(reply)

        st.session_state['messages'] = updated

    # Clear chat — only show when there's history
    if st.session_state.get('messages'):
        st.markdown('---')
        if st.button('🗑️  Clear chat history'):
            st.session_state['messages'] = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# CHARTS TAB
# ══════════════════════════════════════════════════════════════════
with tab_charts:
    chart_options = [
        ('📅  Attendance',        'attendance'),
        ('😊  Sentiment',         'sentiment'),
        ('📈  RX Rate',           'rx_rate'),
        ('💪  Strength Progress', 'strength'),
        ('🔗  Lift Correlation',  'correlation'),
        ('🤖  Workout Clusters',  'clusters'),
        ('🔮  PR Forecast',       'forecast'),
        ('⚠️  Anomalies',         'anomalies'),
    ]

    for label, key in chart_options:
        if st.button(label, key=f'chart_{key}'):
            st.session_state['chart'] = key
            st.session_state['chart_label'] = label.split('  ', 1)[1]

    chart = st.session_state.get('chart')
    if chart:
        st.markdown(f"#### {st.session_state.get('chart_label', chart)}")
        try:
            if chart == 'attendance':
                fig, _ = attendance.monthly_chart(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                fig = attendance.weekly_trend(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                fig = attendance.day_of_week(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            elif chart == 'sentiment':
                fig = sentiment.sentiment_over_time(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                fig = sentiment.word_frequency(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                fig, r = sentiment.sentiment_vs_performance(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.caption(f'Mood vs RX correlation: r = {r:.3f}')
            elif chart == 'rx_rate':
                fig = performance.rx_rate_over_time(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            elif chart == 'strength':
                fig = performance.strength_progression(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            elif chart == 'correlation':
                fig, _ = performance.lift_correlation(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            elif chart == 'clusters':
                _, _, fig = ml_models.cluster_workouts(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            elif chart == 'forecast':
                _, fig = ml_models.forecast_prs(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            elif chart == 'anomalies':
                _, fig = ml_models.detect_anomalies(df)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
        except Exception as e:
            st.error(f'Chart error: {e}')

# ══════════════════════════════════════════════════════════════════
# STATS TAB
# ══════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown('#### Your Dataset')
    st.caption(f"📅 {df['date'].min().date()}  →  {df['date'].max().date()}")

    col1, col2 = st.columns(2)
    col1.metric('Workouts', len(df))
    col2.metric('PRs', int(df['is_pr'].sum()))

    rx_pct  = (df['rx_or_scaled'] == 'RX').mean() * 100
    sc_pct  = (df['rx_or_scaled'] == 'SCALED').mean() * 100
    col1.metric('RX Rate', f'{rx_pct:.0f}%')
    col2.metric('Scaled',  f'{sc_pct:.0f}%')

    st.markdown('---')
    st.markdown('#### Attendance snapshot')
    att = attendance.summary(df)
    st.markdown(f"Avg sessions / month: **{att['avg_per_month']:.1f}**")
    st.markdown(f"Best month: **{att['best_month']}** ({att['best_count']} sessions)")
    st.markdown(f"Worst month: **{att['worst_month']}** ({att['worst_count']} sessions)")
    st.markdown(f"Training gaps (>7 days): **{len(att['gaps_over_7d'])}**")

    st.markdown('---')
    st.markdown('#### Performance snapshot')
    perf = performance.summary(df)
    st.markdown(f"Total PRs logged: **{perf['total_prs']}**")
    st.markdown(f"RX sessions: **{perf['rx_sessions']}**  ·  Scaled: **{perf['scaled_sessions']}**")
