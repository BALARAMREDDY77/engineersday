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

    def execute_uploaded_data(self, frame: pd.DataFrame, operation: str, **arguments: Any) -> ToolResult:
        """Run a fixed analysis operation on an in-memory user-uploaded CSV.

        The upload never becomes a user-supplied path or executable instruction.
        """
        try:
            return analyze_uploaded_dataframe(frame, operation, **arguments)
        except (TypeError, ValueError) as error:
            return ToolResult(False, str(error), error_code="INVALID_ARGUMENTS")
        except Exception:
            return ToolResult(False, "The uploaded-data analysis could not complete safely.", error_code="TOOL_FAILURE")


def analyze_dataset(
    dataset: str,
    score_column: str = "score",
    allow_cleaning: bool = False,
    analysis_mode: str = "attention",
) -> ToolResult:
    """Return basic score statistics from one of the demo datasets only."""
    if dataset not in ALLOWED_DATASETS:
        return ToolResult(False, "Dataset is not in the approved demo set.", error_code="DATASET_NOT_ALLOWED")
    allowed_modes = {"attention", "top_performers", "attendance", "summary"}
    if analysis_mode not in allowed_modes:
        return ToolResult(False, "Analysis mode is not allowed.", error_code="ANALYSIS_MODE_NOT_ALLOWED")

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
        "analysis_mode": analysis_mode,
    }
    if "student" in frame.columns:
        findings["students_needing_attention"] = frame.loc[(numeric_scores < 50).fillna(False), "student"].tolist()
        ranked = frame.assign(_score=numeric_scores).dropna(subset=["_score"]).sort_values("_score", ascending=False)
        findings["top_performers"] = [
            {"student": row["student"], "score": round(float(row["_score"]), 2)}
            for _, row in ranked.head(3).iterrows()
        ]
    if "attendance_percent" in frame.columns:
        findings["students_with_low_attendance"] = frame.loc[
            (pd.to_numeric(frame["attendance_percent"], errors="coerce") < 75).fillna(False), "student"
        ].tolist()

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


def analyze_uploaded_dataframe(
    frame: pd.DataFrame,
    operation: str,
    column: str | None = None,
    category_column: str | None = None,
    numeric_column: str | None = None,
    threshold: float | None = None,
    top_n: int = 3,
    allow_cleaning: bool = False,
) -> ToolResult:
    """Perform allowlisted analytics on an already-loaded CSV dataframe."""
    allowed_operations = {"profile", "missing_values", "average", "top_n", "summary", "group_count_below"}
    if operation not in allowed_operations:
        return ToolResult(False, "Analysis operation is not allowed.", error_code="OPERATION_NOT_ALLOWED")
    if frame.empty:
        return ToolResult(False, "The uploaded CSV has no data rows.", error_code="EMPTY_DATASET")

    if operation == "profile":
        return ToolResult(True, "Dataset profile completed.", {
            "operation": operation,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "missing_values": {name: int(count) for name, count in frame.isna().sum().items() if count},
        })
    if operation == "missing_values":
        missing = {name: int(count) for name, count in frame.isna().sum().items() if count}
        return ToolResult(True, "Missing-value scan completed.", {"operation": operation, "missing_values": missing})
    if operation == "summary":
        numeric = frame.select_dtypes(include="number")
        if numeric.empty:
            return ToolResult(False, "No numeric columns are available for a summary.", error_code="NO_NUMERIC_COLUMNS")
        return ToolResult(True, "Numeric summary completed.", {
            "operation": operation,
            "summary": numeric.describe().round(2).to_dict(),
        })

    if operation == "group_count_below":
        if category_column not in frame.columns or numeric_column not in frame.columns:
            return ToolResult(False, "Requested columns are not available in this dataset.", error_code="COLUMN_NOT_FOUND")
        if not isinstance(threshold, (int, float)) or not -1_000_000 <= threshold <= 1_000_000:
            return ToolResult(False, "Threshold must be a safe numeric value.", error_code="INVALID_THRESHOLD")
        numeric_values = pd.to_numeric(frame[numeric_column], errors="coerce")
        invalid_count = int(numeric_values.isna().sum())
        if invalid_count and not allow_cleaning:
            return ToolResult(False, f"Found {invalid_count} missing or invalid value(s) in '{numeric_column}'.", error_code="INVALID_NUMERIC_DATA")
        filtered = frame.assign(_value=numeric_values).dropna(subset=["_value"])
        filtered = filtered[filtered["_value"] < float(threshold)]
        counts = filtered[category_column].fillna("Unknown").astype(str).value_counts().to_dict()
        return ToolResult(True, "Grouped threshold count completed.", {
            "operation": operation, "category_column": category_column, "numeric_column": numeric_column,
            "threshold": float(threshold), "counts": {str(key): int(value) for key, value in counts.items()},
            "invalid_values_handled": invalid_count,
        })

    if column not in frame.columns:
        return ToolResult(False, "Requested column is not available in this dataset.", error_code="COLUMN_NOT_FOUND")
    numeric_values = pd.to_numeric(frame[column], errors="coerce")
    invalid_count = int(numeric_values.isna().sum())
    if invalid_count and not allow_cleaning:
        return ToolResult(
            False,
            f"Found {invalid_count} missing or invalid value(s) in '{column}'.",
            error_code="INVALID_NUMERIC_DATA",
        )
    cleaned = frame.assign(_value=numeric_values).dropna(subset=["_value"])
    if cleaned.empty:
        return ToolResult(False, "No valid numeric values remain after cleaning.", error_code="NO_VALID_VALUES")
    if operation == "average":
        return ToolResult(True, "Average calculation completed.", {
            "operation": operation, "column": column,
            "average": round(float(cleaned["_value"].mean()), 2), "invalid_values_handled": invalid_count,
        })
    if not isinstance(top_n, int) or not 1 <= top_n <= 10:
        return ToolResult(False, "top_n must be between 1 and 10.", error_code="INVALID_TOP_N")
    label_column = next((name for name in frame.columns if name != column and frame[name].dtype == object), None)
    ranked = cleaned.sort_values("_value", ascending=False).head(top_n)
    records = [
        {"label": str(row[label_column]) if label_column else f"Row {index + 1}", "value": round(float(row["_value"]), 2)}
        for index, row in ranked.iterrows()
    ]
    return ToolResult(True, "Top-value analysis completed.", {
        "operation": operation, "column": column, "top_records": records, "invalid_values_handled": invalid_count,
    })


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
