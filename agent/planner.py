"""Constrained local-LLM task understanding for THE AGENT.

The model may choose only a named demo workflow.  It never produces executable
commands, file paths, Python, or tool arguments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from ollama import chat


WORKFLOW_TO_TOOL = {
    "students": "data_analyzer",
    "environment": "knowledge_base",
    "recovery": "data_analyzer",
    "unsupported": None,
}

WORKFLOW_MESSAGES = {
    "students": "Local model selected Student Performance analysis.",
    "environment": "Local model selected Environmental Research.",
    "recovery": "Local model selected Error Recovery analysis.",
    "unsupported": "This task does not match an approved local workflow.",
}


@dataclass(frozen=True)
class TaskPlan:
    workflow: str
    tool: str | None
    source: str
    message: str


def fallback_workflow(task: str) -> str:
    """Reliable keyword fallback if Ollama is offline or returns invalid output."""
    normalized = task.lower()
    if any(word in normalized for word in ("invalid", "missing", "average performance", "error recovery")):
        return "recovery"
    if any(word in normalized for word in ("student", "performance", "attention")):
        return "students"
    if any(word in normalized for word in ("electric", "petrol", "vehicle", "environment")):
        return "environment"
    return "unsupported"


def parse_workflow(response_text: str) -> str | None:
    """Accept only a valid JSON workflow value from the model response."""
    match = re.search(r"\{\s*\"workflow\"\s*:\s*\"([^\"]+)\"\s*\}", response_text)
    if not match:
        return None
    workflow = match.group(1)
    return workflow if workflow in WORKFLOW_TO_TOOL else None


class LocalPlanner:
    def __init__(self, model: str = "qwen3:1.7b", chat_client: Callable = chat) -> None:
        self.model = model
        self.chat_client = chat_client

    def create_plan(self, task: str) -> TaskPlan:
        prompt = (
            "Classify the user's task into exactly one approved local workflow. "
            "Allowed workflows: students (student performance / students needing attention), "
            "environment (electric versus petrol vehicle comparison), "
            "recovery (average performance with missing or invalid values), "
            "unsupported (anything else). "
            "Return only JSON with this exact schema: {\"workflow\":\"one_allowed_value\"}. "
            f"User task: {task}"
        )
        try:
            response = self.chat_client(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a constrained local task classifier. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0},
            )
            workflow = parse_workflow(response.message.content)
            if workflow is not None:
                return TaskPlan(workflow, WORKFLOW_TO_TOOL[workflow], "local Qwen model", WORKFLOW_MESSAGES[workflow])
        except Exception:
            pass

        workflow = fallback_workflow(task)
        return TaskPlan(workflow, WORKFLOW_TO_TOOL[workflow], "safe local fallback", WORKFLOW_MESSAGES[workflow])
