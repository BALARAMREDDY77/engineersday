"""A transparent, bounded execution loop for THE AGENT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tools import SafeToolRegistry, ToolResult


@dataclass
class AgentEvent:
    status: str
    message: str
    tool: str | None = None


@dataclass
class AgentRun:
    events: list[AgentEvent] = field(default_factory=list)
    result: ToolResult | None = None
    results: list[ToolResult] = field(default_factory=list)


class DemoAgent:
    """Executes a supplied, approved action plan; no untrusted code is run."""

    def __init__(self, registry: SafeToolRegistry | None = None) -> None:
        self.registry = registry or SafeToolRegistry()

    def execute_plan(self, actions: list[dict[str, Any]]) -> AgentRun:
        run = AgentRun([AgentEvent("planning", "Creating a short execution plan.")])
        for action in actions:
            tool_name = action["tool"]
            arguments = action.get("arguments", {})
            run.events.append(AgentEvent("tool_selected", f"Selecting {tool_name.replace('_', ' ')}.", tool_name))
            run.events.append(AgentEvent("executing", "Executing the approved tool.", tool_name))
            result = self.registry.execute(tool_name, **arguments)
            run.result = result
            if result.success:
                run.events.append(AgentEvent("completed", "Task step completed.", tool_name))
                continue
            run.events.append(AgentEvent("error", "Tool returned an error.", tool_name))
            return run
        return run

    def run_average_performance_demo(self) -> AgentRun:
        """Run the intentional failure scenario with an actual recovery and retry.

        The first call deliberately uses strict validation.  The returned
        INVALID_SCORE_DATA error is examined before the recovery call is made.
        """
        run = AgentRun([AgentEvent("planning", "Planning average-performance analysis.")])
        initial_arguments = {
            "dataset": "performance_with_invalid.csv",
            "score_column": "score",
            "allow_cleaning": False,
        }
        run.events.append(AgentEvent("tool_selected", "Selecting Data Analyzer.", "data_analyzer"))
        run.events.append(AgentEvent("executing", "Analyzing the dataset.", "data_analyzer"))
        first_result = self.registry.execute("data_analyzer", **initial_arguments)

        if first_result.success:
            run.result = first_result
            run.events.append(AgentEvent("completed", "Task completed.", "data_analyzer"))
            return run

        run.events.append(AgentEvent("error", "Tool returned an error: invalid data detected.", "data_analyzer"))
        if first_result.error_code != "INVALID_SCORE_DATA":
            run.result = first_result
            return run

        run.events.append(AgentEvent("replanning", "Handling invalid data by excluding invalid scores."))
        run.events.append(AgentEvent("retrying", "Retrying analysis with safe data cleaning.", "data_analyzer"))
        retry_arguments = {**initial_arguments, "allow_cleaning": True}
        run.events.append(AgentEvent("executing", "Re-analyzing valid scores.", "data_analyzer"))
        retry_result = self.registry.execute("data_analyzer", **retry_arguments)
        run.result = retry_result
        if retry_result.success:
            run.events.append(AgentEvent("completed", "Task completed after recovery.", "data_analyzer"))
        else:
            run.events.append(AgentEvent("error", "Retry did not complete.", "data_analyzer"))
        return run

    def run_uploaded_data_plan(self, frame: Any, actions: list[dict[str, Any]], plan_message: str) -> AgentRun:
        """Execute validated, in-memory CSV actions and retry invalid numeric data."""
        run = AgentRun([AgentEvent("planning", plan_message)])
        for action in actions:
            operation = action["operation"]
            arguments = {key: value for key, value in action.items() if key != "operation"}
            run.events.append(AgentEvent("tool_selected", f"Selecting Data Analyzer: {operation.replace('_', ' ')}.", "data_analyzer"))
            run.events.append(AgentEvent("executing", f"Executing {operation.replace('_', ' ')} on the uploaded dataset.", "data_analyzer"))
            result = self.registry.execute_uploaded_data(frame, operation, **arguments)
            if not result.success and result.error_code == "INVALID_NUMERIC_DATA":
                column = arguments.get("column", "the selected column")
                run.events.append(AgentEvent("error", result.message, "data_analyzer"))
                run.events.append(AgentEvent("replanning", f"Handling invalid values in '{column}' safely."))
                run.events.append(AgentEvent("retrying", f"Retrying {operation.replace('_', ' ')} using valid numeric values.", "data_analyzer"))
                result = self.registry.execute_uploaded_data(frame, operation, **arguments, allow_cleaning=True)
            run.results.append(result)
            run.result = result
            if not result.success:
                run.events.append(AgentEvent("error", "The task could not complete safely.", "data_analyzer"))
                return run
            run.events.append(AgentEvent("observing", result.message, "data_analyzer"))
        run.events.append(AgentEvent("completed", "Task completed using the uploaded dataset.", "data_analyzer"))
        return run
