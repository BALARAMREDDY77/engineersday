import unittest

from agent.core import DemoAgent
from agent.planner import LocalPlanner, parse_workflow
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

    def test_planner_accepts_only_allowed_json_workflow(self):
        self.assertEqual(parse_workflow('{"workflow":"recovery"}'), "recovery")
        self.assertIsNone(parse_workflow('{"workflow":"shell"}'))

    def test_planner_keeps_model_choice_inside_approved_boundary(self):
        class FakeResponse:
            class message:
                content = '{"workflow":"environment"}'

        plan = LocalPlanner(chat_client=lambda **_: FakeResponse()).create_plan("Compare vehicles")
        self.assertEqual(plan.workflow, "environment")
        self.assertEqual(plan.tool, "knowledge_base")
        self.assertEqual(plan.source, "local Qwen model")


if __name__ == "__main__":
    unittest.main()
