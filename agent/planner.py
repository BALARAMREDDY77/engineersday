"""Constrained local-LLM task understanding for THE AGENT.

The model may choose only a named demo workflow.  It never produces executable
commands, file paths, Python, or tool arguments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(frozen=True)
class DatasetPlan:
    actions: list[dict[str, Any]]
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
                options={"temperature": 0, "num_predict": 60},
                think=False,
            )
            parsed = parse_plan(response.message.content)
            if parsed is not None:
                workflow, goal = parsed
                return TaskPlan(workflow, goal, WORKFLOW_TO_TOOL[workflow], "local Qwen model", WORKFLOW_MESSAGES[workflow])
        except Exception:
            pass

        workflow, goal = fallback_plan(task)
        return TaskPlan(workflow, goal, WORKFLOW_TO_TOOL[workflow], "safe local fallback", WORKFLOW_MESSAGES[workflow])


def _validate_dataset_actions(payload: object, columns: list[str]) -> list[dict[str, Any]] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        return None
    actions: list[dict[str, Any]] = []
    for action in payload["actions"][:3]:
        if not isinstance(action, dict):
            return None
        operation = action.get("operation")
        if operation not in {"profile", "missing_values", "average", "top_n", "summary", "group_count_below"}:
            return None
        item: dict[str, str] = {"operation": operation}
        if operation in {"average", "top_n"}:
            column = action.get("column")
            if column not in columns:
                return None
            item["column"] = column
        if operation == "group_count_below":
            category_column, numeric_column, threshold = action.get("category_column"), action.get("numeric_column"), action.get("threshold")
            if category_column not in columns or numeric_column not in columns or not isinstance(threshold, (int, float)):
                return None
            item.update({"category_column": category_column, "numeric_column": numeric_column, "threshold": float(threshold)})
        actions.append(item)
    return actions or None


def _fallback_dataset_actions(task: str, columns: list[str]) -> list[dict[str, Any]]:
    normalized = task.lower()
    numeric_candidates = [name for name in columns if any(key in name.lower() for key in ("score", "mark", "amount", "sales", "value", "price"))]
    numeric_column = numeric_candidates[0] if numeric_candidates else None
    actions: list[dict[str, Any]] = [{"operation": "profile"}]
    threshold_match = re.search(r"(?:less than|below|under|lower than)\s*(\d+(?:\.\d+)?)", normalized)
    category_column = next((name for name in columns if any(key in name.lower() for key in ("gender", "sex"))), None)
    requested_numeric = next(
        (name for name in columns if all(word in normalized for word in name.lower().replace(".", " ").split())), None
    )
    if threshold_match and category_column and requested_numeric:
        return actions + [{
            "operation": "group_count_below", "category_column": category_column,
            "numeric_column": requested_numeric, "threshold": float(threshold_match.group(1)),
        }]
    if any(word in normalized for word in ("missing", "invalid", "empty", "quality")):
        actions.append({"operation": "missing_values"})
    if numeric_column and any(word in normalized for word in ("top", "highest", "best", "rank")):
        actions.append({"operation": "top_n", "column": numeric_column})
    elif numeric_column and any(word in normalized for word in ("average", "mean", "overall", "calculate")):
        actions.append({"operation": "average", "column": numeric_column})
    elif len(actions) == 1:
        actions.append({"operation": "summary"})
    return actions


class DatasetPlanner:
    """Uses Qwen to choose a tiny allowlisted plan for an observed CSV schema."""

    def __init__(self, model: str = "qwen3:1.7b", chat_client: Callable = chat) -> None:
        self.model = model
        self.chat_client = chat_client

    def create_plan(self, task: str, columns: list[str]) -> DatasetPlan:
        fast_actions = _fallback_dataset_actions(task, columns)
        if any(action["operation"] == "group_count_below" for action in fast_actions):
            return DatasetPlan(fast_actions, "validated local rule", "Planning a grouped threshold analysis from the observed dataset schema.")
        prompt = (
            "Create a safe plan for a local CSV. Available columns are: " + ", ".join(columns) + ". "
            "Allowed operations only: profile, missing_values, summary, average, top_n, group_count_below. "
            "group_count_below requires category_column, numeric_column, and numeric threshold. "
            "average and top_n require a column exactly from the available list. Use at most three actions. "
            "Return JSON only: {\"actions\":[{\"operation\":\"profile\"},{\"operation\":\"average\",\"column\":\"score\"}]}. "
            f"User task: {task}"
        )
        try:
            response = self.chat_client(
                model=self.model,
                messages=[{"role": "system", "content": "Return JSON only. Never produce code or commands."}, {"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": 80},
                think=False,
            )
            for match in re.finditer(r"\{[\s\S]*?\}\s*\}", response.message.content):
                try:
                    actions = _validate_dataset_actions(json.loads(match.group()), columns)
                except json.JSONDecodeError:
                    continue
                if actions:
                    return DatasetPlan(actions, "local Qwen model", "Creating a safe plan from the observed dataset schema.")
        except Exception:
            pass
        return DatasetPlan(
            _fallback_dataset_actions(task, columns),
            "safe local fallback",
            "Creating a safe plan from the observed dataset schema.",
        )
