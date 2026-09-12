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
