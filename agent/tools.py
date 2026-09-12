"""Predefined, local-only tool registry.

This module intentionally exposes no shell, filesystem mutation, or arbitrary
Python execution capability.  The agent can call only these three tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ALLOWED_DATASETS = {"students.csv", "performance_with_invalid.csv"}


@dataclass
class ToolResult:
    success: bool
    message: str
    data: dict[str, Any] | None = None
    error_code: str | None = None


class SafeToolRegistry:
    """Maps fixed tool names to fixed Python functions."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., ToolResult]] = {
            "data_analyzer": analyze_dataset,
            "calculator": calculate,
            "knowledge_base": search_knowledge_base,
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def execute(self, tool_name: str, **arguments: Any) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(False, "That tool is not available.", error_code="TOOL_NOT_ALLOWED")
        try:
            return tool(**arguments)
        except (TypeError, ValueError) as error:
            return ToolResult(False, str(error), error_code="INVALID_ARGUMENTS")
        except Exception:
            return ToolResult(False, "The tool could not complete safely.", error_code="TOOL_FAILURE")


def analyze_dataset(dataset: str, score_column: str = "score", allow_cleaning: bool = False) -> ToolResult:
    """Return basic score statistics from one of the demo datasets only."""
    if dataset not in ALLOWED_DATASETS:
        return ToolResult(False, "Dataset is not in the approved demo set.", error_code="DATASET_NOT_ALLOWED")

    file_path = DATA_DIR / dataset
    if not file_path.is_file():
        return ToolResult(False, f"Demo dataset '{dataset}' was not found.", error_code="DATASET_NOT_FOUND")

    frame = pd.read_csv(file_path)
    if score_column not in frame.columns:
        return ToolResult(False, f"Column '{score_column}' was not found.", error_code="COLUMN_NOT_FOUND")

    numeric_scores = pd.to_numeric(frame[score_column], errors="coerce")
    invalid_count = int(numeric_scores.isna().sum())
    if invalid_count and not allow_cleaning:
        return ToolResult(
            False,
            f"Found {invalid_count} missing or invalid score value(s).",
            error_code="INVALID_SCORE_DATA",
        )

    valid_scores = numeric_scores.dropna()
    if valid_scores.empty:
        return ToolResult(False, "No valid numeric scores are available.", error_code="NO_VALID_SCORES")

    findings: dict[str, Any] = {
        "records_analyzed": int(len(valid_scores)),
        "invalid_values_handled": invalid_count,
        "average_score": round(float(valid_scores.mean()), 2),
        "highest_score": round(float(valid_scores.max()), 2),
        "lowest_score": round(float(valid_scores.min()), 2),
    }
    if "student" in frame.columns:
        attention_rows = frame.loc[(numeric_scores < 50).fillna(False), "student"].tolist()
        findings["students_needing_attention"] = attention_rows

    return ToolResult(True, "Dataset analysis completed.", findings)


def calculate(operation: str, values: list[float]) -> ToolResult:
    """Perform a small, explicit set of mathematical operations."""
    if not values:
        return ToolResult(False, "Provide at least one value.", error_code="NO_VALUES")
    numbers = [float(value) for value in values]
    operations: dict[str, Callable[[], float]] = {
        "total": lambda: sum(numbers),
        "average": lambda: sum(numbers) / len(numbers),
        "maximum": lambda: max(numbers),
        "minimum": lambda: min(numbers),
    }
    if operation not in operations:
        return ToolResult(False, "Operation is not allowed.", error_code="OPERATION_NOT_ALLOWED")
    result = round(operations[operation](), 2)
    return ToolResult(True, "Calculation completed.", {"operation": operation, "result": result})


def search_knowledge_base(query: str) -> ToolResult:
    """Search the fixed local JSON knowledge base with simple keyword matching."""
    knowledge_path = DATA_DIR / "knowledge_base.json"
    if not knowledge_path.is_file():
        return ToolResult(False, "Local knowledge base was not found.", error_code="KNOWLEDGE_BASE_NOT_FOUND")

    entries = json.loads(knowledge_path.read_text(encoding="utf-8"))
    terms = {term.lower() for term in query.split() if len(term) > 2}
    matches = [
        entry for entry in entries
        if terms.intersection(str(entry).lower().replace("/", " ").split())
    ]
    return ToolResult(True, f"Found {len(matches)} relevant knowledge-base entry(s).", {"matches": matches})
