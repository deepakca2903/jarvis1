import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from mongomock_motor import AsyncMongoMockClient

from app.database import set_client, errors_collection, tool_steps_collection
from app.agent import execute_agent_command
from app.tools import create_task
from app.models import ToolStepStatus, RunStatus


@pytest.fixture(autouse=True)
def setup_mock_db():
    mock_client = AsyncMongoMockClient()
    set_client(mock_client)
    yield
    set_client(None)


@pytest.mark.asyncio
async def test_multi_step_dependent_workflow():
    """
    Test multi-step execution:
    Turn 1: Model calls list_tasks
    Turn 2: Model calls prioritize_tasks with results from step 1
    Turn 3: Model calls generate_report
    Turn 4: Model finishes with final summary
    """
    # Seed sample tasks in DB
    await create_task(title="Fix Vulnerability", priority="critical", deadline="2026-10-16")
    await create_task(title="Review Documentation", priority="medium", deadline="2026-10-25")

    # Mock tool calls from Groq
    tc1 = MagicMock()
    tc1.id = "call_1"
    tc1.function.name = "list_tasks"
    tc1.function.arguments = json.dumps({"status": "pending"})

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message.tool_calls = [tc1]
    resp1.choices[0].message.content = None

    tc2 = MagicMock()
    tc2.id = "call_2"
    tc2.function.name = "prioritize_tasks"
    tc2.function.arguments = json.dumps({"sort_by": "deadline"})

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message.tool_calls = [tc2]
    resp2.choices[0].message.content = None

    tc3 = MagicMock()
    tc3.id = "call_3"
    tc3.function.name = "generate_report"
    tc3.function.arguments = json.dumps({"title": "Prioritized Pending Tasks Report"})

    resp3 = MagicMock()
    resp3.choices = [MagicMock()]
    resp3.choices[0].message.tool_calls = [tc3]
    resp3.choices[0].message.content = None

    resp4 = MagicMock()
    resp4.choices = [MagicMock()]
    resp4.choices[0].message.tool_calls = None
    resp4.choices[0].message.content = "I have listed your pending tasks, prioritized them by deadline, and generated the verified report."

    with patch("app.agent.get_groq_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [resp1, resp2, resp3, resp4]
        mock_get_client.return_value = mock_client

        run = await execute_agent_command("List my pending tasks, prioritize them by deadline, and generate a report.")

        assert run.status == RunStatus.COMPLETED
        assert len(run.steps) == 3
        assert run.steps[0].tool_name == "list_tasks"
        assert run.steps[0].status == ToolStepStatus.SUCCESS
        assert run.steps[1].tool_name == "prioritize_tasks"
        assert run.steps[1].status == ToolStepStatus.SUCCESS
        assert run.steps[2].tool_name == "generate_report"
        assert run.steps[2].status == ToolStepStatus.SUCCESS
        assert run.report_url is not None
        assert "/api/reports/" in run.report_url
        assert "prioritized them by deadline" in run.output


@pytest.mark.asyncio
async def test_tool_allowlist_rejection():
    """Verify that attempting to call an unpermitted tool is rejected safely."""
    bad_tc = MagicMock()
    bad_tc.id = "call_bad"
    bad_tc.function.name = "execute_arbitrary_shell_command"
    bad_tc.function.arguments = json.dumps({"cmd": "rm -rf /"})

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message.tool_calls = [bad_tc]
    resp1.choices[0].message.content = None

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message.tool_calls = None
    resp2.choices[0].message.content = "Action disallowed."

    with patch("app.agent.get_groq_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [resp1, resp2]
        mock_get_client.return_value = mock_client

        run = await execute_agent_command("Execute arbitrary shell command")
        assert len(run.steps) == 1
        assert run.steps[0].status == ToolStepStatus.FAILED
        assert "not in the allowlist" in run.steps[0].error

        # Verify error recorded in database
        errors = await errors_collection().find().to_list(10)
        assert len(errors) == 1
        assert "execute_arbitrary_shell_command" in errors[0]["tool_name"]


@pytest.mark.asyncio
async def test_retry_on_temporary_failure():
    """Verify that a temporary failure is retried at most once."""
    tc = MagicMock()
    tc.id = "call_flaky"
    tc.function.name = "list_tasks"
    tc.function.arguments = json.dumps({})

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message.tool_calls = [tc]
    resp1.choices[0].message.content = None

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message.tool_calls = None
    resp2.choices[0].message.content = "Tasks retrieved on retry."

    attempt = 0

    async def flaky_list_tasks(**kwargs):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise ConnectionError("Temporary network blip")
        return {"success": True, "tasks": [], "count": 0}

    with patch("app.agent.get_groq_client") as mock_get_client, \
         patch.dict("app.agent.TOOL_REGISTRY", {"list_tasks": flaky_list_tasks}):

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [resp1, resp2]
        mock_get_client.return_value = mock_client

        run = await execute_agent_command("List tasks please")
        assert len(run.steps) == 1
        assert run.steps[0].retry_count == 1
        assert run.steps[0].status == ToolStepStatus.RETRIED
        assert attempt == 2
