import unittest

import pandas as pd

from agent.core import DemoAgent
from agent.planner import DatasetPlanner, LocalPlanner, parse_plan
from agent.tools import SafeToolRegistry


class SafeToolsTest(unittest.TestCase):
    def test_calculator_is_available(self):
        result = SafeToolRegistry().execute("calculator", operation="average", values=[70, 80, 90])
        self.assertTrue(result.success)
        self.assertEqual(result.data["result"], 80.0)

    def test_arbitrary_tool_is_rejected(self):
        result = SafeToolRegistry().execute("shell", command="dir")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "TOOL_NOT_ALLOWED")

    def test_agent_records_execution_events(self):
        run = DemoAgent().execute_plan([{"tool": "calculator", "arguments": {"operation": "total", "values": [2, 3]}}])
        self.assertTrue(run.result.success)
        self.assertEqual([event.status for event in run.events], ["planning", "tool_selected", "executing", "completed"])

    def test_invalid_data_triggers_real_recovery_and_retry(self):
        run = DemoAgent().run_average_performance_demo()
        self.assertTrue(run.result.success)
        self.assertEqual(run.result.data["average_score"], 83.33)
        self.assertEqual(run.result.data["invalid_values_handled"], 2)
        self.assertEqual(
            [event.status for event in run.events],
            ["planning", "tool_selected", "executing", "error", "replanning", "retrying", "executing", "completed"],
        )

    def test_planner_accepts_only_allowed_json_plan(self):
        self.assertEqual(parse_plan('{"workflow":"recovery", "goal":"average"}'), ("recovery", "average"))
        self.assertIsNone(parse_plan('{"workflow":"shell", "goal":"anything"}'))

    def test_planner_keeps_model_choice_inside_approved_boundary(self):
        class FakeResponse:
            class message:
                content = '{"workflow":"environment", "goal":"comparison"}'

        plan = LocalPlanner(chat_client=lambda **_: FakeResponse()).create_plan("Compare vehicles")
        self.assertEqual(plan.workflow, "environment")
        self.assertEqual(plan.goal, "comparison")
        self.assertEqual(plan.tool, "knowledge_base")
        self.assertEqual(plan.source, "local Qwen model")

    def test_student_attention_demo_data(self):
        run = DemoAgent().execute_plan([
            {"tool": "data_analyzer", "arguments": {"dataset": "students.csv", "score_column": "score"}}
        ])
        self.assertTrue(run.result.success)
        self.assertEqual(run.result.data["students_needing_attention"], ["Ishaan", "Rohan"])

    def test_student_top_performer_analysis(self):
        result = SafeToolRegistry().execute(
            "data_analyzer", dataset="students.csv", analysis_mode="top_performers"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data["top_performers"][0], {"student": "Aarav", "score": 92.0})

    def test_environment_knowledge_base_demo_data(self):
        result = SafeToolRegistry().execute(
            "knowledge_base", query="electric vehicles petrol vehicles comparison"
        )
        self.assertTrue(result.success)
        self.assertGreaterEqual(len(result.data["matches"]), 2)

    def test_uploaded_data_average_recovers_from_invalid_value(self):
        frame = pd.DataFrame({"student": ["Aarav", "Diya", "Ishaan"], "score": [80, "invalid", 100]})
        run = DemoAgent().run_uploaded_data_plan(
            frame,
            [{"operation": "profile"}, {"operation": "average", "column": "score"}],
            "Planning a safe uploaded-data analysis.",
        )
        self.assertTrue(run.result.success)
        self.assertEqual(run.result.data["average"], 90.0)
        self.assertIn("replanning", [event.status for event in run.events])

    def test_uploaded_data_plan_rejects_unknown_operation(self):
        frame = pd.DataFrame({"score": [80, 90]})
        run = DemoAgent().run_uploaded_data_plan(frame, [{"operation": "shell"}], "Test plan")
        self.assertFalse(run.result.success)
        self.assertEqual(run.result.error_code, "OPERATION_NOT_ALLOWED")

    def test_dataset_planner_validates_model_plan_against_schema(self):
        class FakeResponse:
            class message:
                content = '{"actions":[{"operation":"profile"},{"operation":"top_n","column":"score"}]}'

        plan = DatasetPlanner(chat_client=lambda **_: FakeResponse()).create_plan("Find top students", ["student", "score"])
        self.assertEqual(plan.actions[1], {"operation": "top_n", "column": "score"})

    def test_group_count_below_answers_gender_threshold_question(self):
        frame = pd.DataFrame({
            "gender": ["female", "male", "female", "male"],
            "english.grade": [3.0, 3.2, 3.8, 2.9],
        })
        plan = DatasetPlanner(chat_client=lambda **_: None).create_plan(
            "How many females and males have english grade less than 3.4?", list(frame.columns)
        )
        self.assertEqual(plan.actions[1]["operation"], "group_count_below")
        run = DemoAgent().run_uploaded_data_plan(frame, plan.actions, plan.message)
        self.assertTrue(run.result.success)
        self.assertEqual(run.result.data["counts"], {"male": 2, "female": 1})


if __name__ == "__main__":
    unittest.main()
