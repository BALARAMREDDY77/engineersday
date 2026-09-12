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

VALID_GOALS = {
    "students": {"attention", "top_performers", "attendance", "summary"},
    "environment": {"comparison"},
    "recovery": {"average"},
    "unsupported": {"unsupported"},
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
    goal: str
    tool: str | None
    source: str
    message: str


def fallback_plan(task: str) -> tuple[str, str]:
    """Reliable keyword fallback if Ollama is offline or returns invalid output."""
    normalized = task.lower()
    if any(word in normalized for word in ("invalid", "missing", "average performance", "error recovery")):
        return "recovery", "average"
    if any(word in normalized for word in ("student", "performance", "attention")):
        if any(word in normalized for word in ("top", "highest", "best", "rank")):
            return "students", "top_performers"
        if any(word in normalized for word in ("attendance", "absent", "present")):
            return "students", "attendance"
        if any(word in normalized for word in ("average", "summary", "overall", "trend")):
            return "students", "summary"
        return "students", "attention"
    if any(word in normalized for word in ("electric", "petrol", "vehicle", "environment")):
        return "environment", "comparison"
    return "unsupported", "unsupported"


def parse_plan(response_text: str) -> tuple[str, str] | None:
    """Accept only an allowlisted workflow/goal JSON object from the model."""
    for match in re.finditer(r"\{[^{}]*\}", response_text):
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        workflow, goal = payload.get("workflow"), payload.get("goal")
        if workflow in WORKFLOW_TO_TOOL and goal in VALID_GOALS.get(workflow, set()):
            return workflow, goal
    return None


class LocalPlanner:
    def __init__(self, model: str = "qwen3:1.7b", chat_client: Callable = chat) -> None:
        self.model = model
        self.chat_client = chat_client

    def create_plan(self, task: str) -> TaskPlan:
        prompt = (
            "Classify the user's task into one approved local workflow and one approved goal. "
            "Allowed workflows: students (student performance requests), "
            "environment (electric versus petrol vehicle comparison), "
            "recovery (average performance with missing or invalid values), "
            "unsupported (anything else). Allowed goals: students=attention, top_performers, attendance, summary; "
            "environment=comparison; recovery=average; unsupported=unsupported. "
            "Return only JSON with this exact schema: {\"workflow\":\"students\",\"goal\":\"attention\"}. "
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
            parsed = parse_plan(response.message.content)
            if parsed is not None:
                workflow, goal = parsed
                return TaskPlan(workflow, goal, WORKFLOW_TO_TOOL[workflow], "local Qwen model", WORKFLOW_MESSAGES[workflow])
        except Exception:
            pass

        workflow, goal = fallback_plan(task)
        return TaskPlan(workflow, goal, WORKFLOW_TO_TOOL[workflow], "safe local fallback", WORKFLOW_MESSAGES[workflow])
