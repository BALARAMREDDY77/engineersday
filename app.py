"""Streamlit dashboard for THE AGENT local-only competition prototype."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from agent.core import AgentEvent, DemoAgent
from agent.planner import DatasetPlanner, LocalPlanner
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


def execute_workflow(workflow: str, goal: str) -> tuple[list[AgentEvent], ToolResult | None, str]:
    agent = DemoAgent()
    if workflow == "recovery":
        run = agent.run_average_performance_demo()
        return run.events, run.result, "error_recovery"
    if workflow == "students":
        run = agent.execute_plan([
            {"tool": "data_analyzer", "arguments": {
                "dataset": "students.csv", "score_column": "score", "analysis_mode": goal
            }}
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
        "observing": "✓ Observed",
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


def result_html(result: ToolResult, workflow: str, goal: str) -> str:
    if not result.success:
        return f'<div class="warning"><b>Task needs attention</b><br>{result.message}</div>'

    data: dict[str, Any] = result.data or {}
    if workflow == "students":
        if goal == "top_performers":
            top = ", ".join(f"{item['student']} ({item['score']})" for item in data["top_performers"])
            body = f"Top performers from <b>{data['records_analyzed']}</b> records: <b>{top}</b>."
        elif goal == "attendance":
            names = ", ".join(data.get("students_with_low_attendance", [])) or "No students"
            body = f"Students below 75% attendance: <b>{names}</b>."
        elif goal == "summary":
            body = f"Score summary: average <b>{data['average_score']}</b>, highest <b>{data['highest_score']}</b>, lowest <b>{data['lowest_score']}</b>."
        else:
            names = ", ".join(data.get("students_needing_attention", [])) or "No students"
            body = f"Analyzed <b>{data['records_analyzed']}</b> records. Students who may need attention: <b>{names}</b>."
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


def render_uploaded_results(results: list[ToolResult]) -> None:
    st.markdown('<div class="result"><b>FINAL RESULT — UPLOADED DATASET</b><br>The agent completed these verified operations.</div>', unsafe_allow_html=True)
    for result in results:
        data = result.data or {}
        operation = data.get("operation")
        if operation == "profile":
            st.write(f"**Dataset profile:** {data['rows']} rows • columns: {', '.join(data['columns'])}")
            if data["missing_values"]:
                st.warning(f"Missing values found: {data['missing_values']}")
        elif operation == "missing_values":
            st.write(f"**Missing-value scan:** {data['missing_values'] or 'No missing values found.'}")
        elif operation == "average":
            st.metric(f"Average {data['column']}", data["average"])
            if data["invalid_values_handled"]:
                st.caption(f"Recovered by excluding {data['invalid_values_handled']} invalid/missing value(s).")
        elif operation == "top_n":
            st.write(f"**Top values by {data['column']}:**")
            st.dataframe(pd.DataFrame(data["top_records"]), use_container_width=True, hide_index=True)
        elif operation == "group_count_below":
            st.write(
                f"**Count by {data['category_column']} where {data['numeric_column']} is below {data['threshold']}:**"
            )
            st.dataframe(
                pd.DataFrame(data["counts"].items(), columns=[data["category_column"], "count"]),
                use_container_width=True,
                hide_index=True,
            )
        elif operation in {"group_count", "filtered_group_count"}:
            label = f"Count by {data['category_column']}"
            if operation == "filtered_group_count":
                label += f" where {data['filter_column']} {data['comparison']} {data['filter_value']}"
            st.write(f"**{label}:**")
            st.dataframe(pd.DataFrame(data["counts"].items(), columns=[data["category_column"], "count"]), use_container_width=True, hide_index=True)
        elif operation == "group_aggregate":
            st.write(f"**{data['aggregation']} {data['numeric_column']} by {data['category_column']}:**")
            st.dataframe(pd.DataFrame(data["values"].items(), columns=[data["category_column"], data["aggregation"]]), use_container_width=True, hide_index=True)
        elif operation == "summary":
            st.dataframe(pd.DataFrame(data["summary"]), use_container_width=True)


st.markdown("""<div class="hero"><div class="eyebrow">LOCAL-ONLY AGENTIC AI PROTOTYPE</div><h1>THE AGENT</h1><p>Watch AI do a job, not just chat.</p></div>""", unsafe_allow_html=True)

left, right = st.columns([2, 1])
with left:
    st.subheader("Give the agent a task")
    selected_demo = st.selectbox("Reliable demo scenario", ["Custom task", *DEMO_TASKS], label_visibility="collapsed")
    initial_task = DEMO_TASKS.get(selected_demo, "")
    task = st.text_area("Task", value=initial_task, height=95, placeholder="Describe a local analysis task...")
    uploaded_csv = st.file_uploader("Optional: upload a CSV for a live custom analysis", type=["csv"])
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
    planner_box = st.empty()
    tool_box = st.empty()
with timeline_col:
    st.subheader("Execution timeline")
    timeline_box = st.empty()

if run_clicked:
    if uploaded_csv is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded_csv)
        except (UnicodeDecodeError, pd.errors.ParserError):
            status_box.error("This file could not be read as a CSV.")
            st.stop()
        dataset_plan = DatasetPlanner().create_plan(task, list(uploaded_frame.columns))
        uploaded_run = DemoAgent().run_uploaded_data_plan(uploaded_frame, dataset_plan.actions, dataset_plan.message)
        events, result, result_type = uploaded_run.events, uploaded_run.result, "uploaded"
        planner_source = dataset_plan.source
    else:
        plan = LocalPlanner().create_plan(task)
        if plan.workflow == "unsupported":
            status_box.error("No safe workflow matched this task. Upload a CSV for a live analysis, or try a prepared demo.")
            action_box.write("Use an approved demo or a CSV with a clear analysis question.")
            st.stop()
        events, result, result_type = execute_workflow(plan.workflow, plan.goal)
        events[0] = AgentEvent("planning", plan.message)
        planner_source = plan.source

    planner_box.caption(f"Planner: {planner_source}")
    rendered_events: list[str] = []
    for event in events:
        rendered_events.append(event_html(event))
        timeline_box.markdown("".join(rendered_events), unsafe_allow_html=True)
        status_box.markdown(f"**{event.status.replace('_', ' ').title()}**")
        action_box.write(event.message)
        tool_box.caption(f"Tool: {event.tool or 'Planning / recovery'}")
        time.sleep(0.15)
    st.divider()
    if result_type == "uploaded":
        render_uploaded_results(uploaded_run.results)
    else:
        st.markdown(result_html(result, result_type, plan.goal), unsafe_allow_html=True)
else:
    status_box.markdown("**Ready**")
    action_box.write("Choose a demo and run the agent.")
    planner_box.caption("Planner: local Qwen when available; safe local fallback otherwise")
    tool_box.caption("Tool: waiting")
    timeline_box.caption("The verified execution events will appear here.")
