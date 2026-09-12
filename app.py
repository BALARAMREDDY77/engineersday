"""Streamlit dashboard for THE AGENT local-only competition prototype."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from agent.core import AgentEvent, DemoAgent
from agent.tools import ToolResult


st.set_page_config(page_title="THE AGENT", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2.3rem;}
      .hero {padding: 1.6rem 1.8rem; border-radius: 18px; color: #f8fafc;
             background: linear-gradient(120deg, #0b1220, #16376b); margin-bottom: 1.1rem;}
      .hero h1 {margin: 0; font-size: 2.4rem; letter-spacing: .05em;}
      .hero p {margin: .5rem 0 0; color: #cbd5e1; font-size: 1.05rem;}
      .eyebrow {color: #60a5fa; font-weight: 700; letter-spacing: .12em; font-size: .74rem;}
      .event {border-left: 4px solid #64748b; padding: .55rem .8rem; margin: .48rem 0;
              border-radius: 0 8px 8px 0; background: #f8fafc; color: #0f172a;}
      .event.error {border-color: #f59e0b; background: #fffbeb;}
      .event.recovery {border-color: #2563eb; background: #eff6ff;}
      .event.completed {border-color: #16a34a; background: #f0fdf4;}
      .result {padding: 1.1rem 1.25rem; border: 1px solid #86efac; border-radius: 12px;
               background: #f0fdf4; color: #14532d;}
      .warning {padding: 1.1rem 1.25rem; border: 1px solid #fcd34d; border-radius: 12px;
                background: #fffbeb; color: #78350f;}
    </style>
    """,
    unsafe_allow_html=True,
)


DEMO_TASKS = {
    "Student Performance Analyst": "Analyze the student dataset and identify students who may need attention.",
    "Environmental Research": "Compare electric vehicles and petrol vehicles using the local knowledge base.",
    "Error Recovery": "Calculate the average performance from a dataset containing an intentional invalid/missing value.",
}


def choose_workflow(task: str) -> str | None:
    """Deterministic, safe routing to approved local workflows only."""
    normalized = task.lower()
    if any(word in normalized for word in ("invalid", "missing", "average performance", "error recovery")):
        return "recovery"
    if any(word in normalized for word in ("student", "performance", "attention")):
        return "students"
    if any(word in normalized for word in ("electric", "petrol", "vehicle", "environment")):
        return "environment"
    return None


def execute_workflow(workflow: str) -> tuple[list[AgentEvent], ToolResult | None, str]:
    agent = DemoAgent()
    if workflow == "recovery":
        run = agent.run_average_performance_demo()
        return run.events, run.result, "error_recovery"
    if workflow == "students":
        run = agent.execute_plan([
            {"tool": "data_analyzer", "arguments": {"dataset": "students.csv", "score_column": "score"}}
        ])
        return run.events, run.result, "students"
    run = agent.execute_plan([
        {"tool": "knowledge_base", "arguments": {"query": "electric vehicles petrol vehicles comparison"}}
    ])
    return run.events, run.result, "environment"


def event_html(event: AgentEvent) -> str:
    symbols = {
        "planning": "✓ Planning",
        "tool_selected": "✓ Tool selected",
        "executing": "● Executing",
        "error": "⚠ Error detected",
        "replanning": "↻ Replanning",
        "retrying": "↻ Retrying",
        "completed": "✓ Completed",
    }
    style = ""
    if event.status == "error":
        style = "error"
    elif event.status in {"replanning", "retrying"}:
        style = "recovery"
    elif event.status == "completed":
        style = "completed"
    tool_label = f" <small>• {event.tool}</small>" if event.tool else ""
    return f'<div class="event {style}"><b>{symbols[event.status]}</b><br>{event.message}{tool_label}</div>'


def result_html(result: ToolResult, workflow: str) -> str:
    if not result.success:
        return f'<div class="warning"><b>Task needs attention</b><br>{result.message}</div>'

    data: dict[str, Any] = result.data or {}
    if workflow == "students":
        students = data.get("students_needing_attention", [])
        names = ", ".join(students) if students else "No students"
        body = (
            f"Analyzed <b>{data['records_analyzed']}</b> student records. "
            f"Average score: <b>{data['average_score']}</b>. "
            f"Students who may need attention: <b>{names}</b>."
        )
    elif workflow == "error_recovery":
        body = (
            f"Average performance: <b>{data['average_score']}</b>. "
            f"The agent safely excluded <b>{data['invalid_values_handled']}</b> invalid or missing value(s) "
            f"and completed the retry using <b>{data['records_analyzed']}</b> valid records."
        )
    else:
        matches = data.get("matches", [])
        summaries = " ".join(item["summary"] for item in matches)
        body = f"<b>Local comparison complete.</b> {summaries}"
    return f'<div class="result"><b>FINAL RESULT</b><br>{body}</div>'


st.markdown("""<div class="hero"><div class="eyebrow">LOCAL-ONLY AGENTIC AI PROTOTYPE</div><h1>THE AGENT</h1><p>Watch AI do a job, not just chat.</p></div>""", unsafe_allow_html=True)

left, right = st.columns([2, 1])
with left:
    st.subheader("Give the agent a task")
    selected_demo = st.selectbox("Reliable demo scenario", ["Custom task", *DEMO_TASKS], label_visibility="collapsed")
    initial_task = DEMO_TASKS.get(selected_demo, "")
    task = st.text_area("Task", value=initial_task, height=95, placeholder="Describe a local analysis task...")
    run_clicked = st.button("Run THE AGENT", type="primary", use_container_width=True)
with right:
    st.subheader("Safety boundary")
    st.info("Local only. The agent can choose only Data Analyzer, Calculator, or Knowledge Base. No shell or system access.")
    st.caption("Model: local Ollama/Qwen • Data: local CSV/JSON")

st.divider()
status_col, timeline_col = st.columns([1, 2])
with status_col:
    st.subheader("Agent status")
    status_box = st.empty()
    action_box = st.empty()
    tool_box = st.empty()
with timeline_col:
    st.subheader("Execution timeline")
    timeline_box = st.empty()

if run_clicked:
    workflow = choose_workflow(task)
    if workflow is None:
        status_box.error("No safe workflow matched this task.")
        action_box.write("Try one of the three provided demo scenarios.")
    else:
        events, result, result_type = execute_workflow(workflow)
        rendered_events: list[str] = []
        for event in events:
            rendered_events.append(event_html(event))
            timeline_box.markdown("".join(rendered_events), unsafe_allow_html=True)
            status_box.markdown(f"**{event.status.replace('_', ' ').title()}**")
            action_box.write(event.message)
            tool_box.caption(f"Tool: {event.tool or 'Planning / recovery'}")
            time.sleep(0.35)
        st.divider()
        st.markdown(result_html(result, result_type), unsafe_allow_html=True)
else:
    status_box.markdown("**Ready**")
    action_box.write("Choose a demo and run the agent.")
    tool_box.caption("Tool: waiting")
    timeline_box.caption("The verified execution events will appear here.")
