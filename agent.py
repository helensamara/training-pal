"""
agent.py
Claude agent loop with tool use. Called by app.py on each chat message.
"""
import os
import json
from dotenv import load_dotenv
import anthropic
from tools import TOOL_SCHEMAS, run_tool

# Load environment variables from .env (e.g., ANTHROPIC_API_KEY)
load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

SYSTEM = """You are a personal performance intelligence agent for Helen, an athlete who trains CrossFit and Powerlifting.

Data available:
- SugarWOD: 381 CrossFit sessions, Oct 2023 – Mar 2026
  IMPORTANT: SugarWOD only contains CrossFit data. Helen also does powerlifting as a completely
  separate sport with different gear, different PRs, and different standards.
  CrossFit and powerlifting PRs are NOT comparable — always specify which sport when discussing lifts.
  Example: Helen's deadlift CrossFit PR is ~230-250 lbs; her powerlifting PR is 300+ lbs.
  Back squat, bench press, and deadlift appear in both sports but are tracked separately.
- Garmin Venu 3S: 382 activities (CrossFit + powerlifting), Nov 2024 – Mar 2026
  Includes: sleep scores, HRV, resting HR, Body Battery, HR zones, menstrual cycle phases
- Powerlifting programs: 18 programs from coach Tom Kean, Sep 2024 – Mar 2026

Tools available:
- attendance_summary   → sessions/month, rest gaps, best/worst periods
- sentiment_summary    → mood scores from workout notes, correlation with RX rate
- performance_summary  → RX rate trend, PR count, scaling ratio
- cluster_workouts     → KMeans workout archetypes
- detect_anomalies     → Isolation Forest flags unusual strength sessions
- forecast_prs         → Linear Regression predicts next PR per lift
- garmin_summary       → sleep vs performance, HRV vs performance, Body Battery drain,
                          menstrual cycle phase vs RX rate (Nov 2024 – Mar 2026 window)

Rules:
1. Always call the relevant tool(s) before responding — never guess figures.
2. Be specific: use actual numbers from the tool output.
3. Be concise and actionable — Helen is an athlete, not a data scientist.
4. When discussing biometric patterns (sleep, HRV, cycle), be sensitive and constructive.
5. Cross-analysis between Garmin and SugarWOD only covers Nov 2024 onward — note this when relevant.
"""


def chat(messages: list) -> tuple[str, list]:
    """
    Run one agent turn.

    Args:
        messages: Full conversation history (Anthropic format, strings only).

    Returns:
        (response_text, updated_messages)
    """
    while True:
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=2048,
            system=SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == 'tool_use':
            tool_results = []
            for block in response.content:
                if block.type == 'tool_use':
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.id,
                        'content': json.dumps(result),
                    })
            messages = messages + [
                {'role': 'assistant', 'content': response.content},
                {'role': 'user',      'content': tool_results},
            ]
        else:
            text = '\n'.join(b.text for b in response.content if b.type == 'text')
            messages = messages + [{'role': 'assistant', 'content': text}]
            return text, messages
