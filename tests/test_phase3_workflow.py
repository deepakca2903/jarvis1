import os
import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock
from mongomock_motor import AsyncMongoMockClient

from app.main import app
from app.database import set_client, get_db
from app.tools import create_task, generate_report, verify_action, workflow_status
from app.agent import execute_agent_command, _clean_spoken_summary
from app.models import ToolStepStatus, RunStatus


@pytest.fixture(autouse=True)
def setup_mock_db():
    mock_client = AsyncMongoMockClient()
    set_client(mock_client)
    yield
    set_client(None)


@pytest.mark.asyncio
async def test_verify_action_file_success_and_failure():
    """Test verify_action with existing report and missing file."""
    # 1. Generate a real report
    await create_task(title="Sample Task for Report", priority="high")
    rep_res = await generate_report(title="Phase3 Verification Report")
    filename = rep_res["filename"]

    # Verify existing file
    v_res = await verify_action(target_type="file", target=filename)
    assert v_res["success"] is True
    assert v_res["verified"] is True
    assert v_res["filename"] == filename
    assert v_res["size_bytes"] > 0
    assert "/api/reports/" in v_res["download_url"]

    # Verify non-existent file
    v_missing = await verify_action(target_type="file", target="non_existent_artifact_xyz.md")
    assert v_missing["success"] is False
    assert v_missing["verified"] is False
    assert "does not exist on disk" in v_missing["error"]


@pytest.mark.asyncio
async def test_verify_action_task_success_and_failure():
    """Test verify_action for a persisted task in MongoDB."""
    task_res = await create_task(title="Security Audit Check", priority="critical")
    task_id = task_res["task"]["id"]

    # Verify existing task
    v_task = await verify_action(target_type="task", target=task_id, expected_state="pending")
    assert v_task["success"] is True
    assert v_task["verified"] is True
    assert v_task["task"]["id"] == task_id

    # Verify state mismatch
    v_mismatch = await verify_action(target_type="task", target=task_id, expected_state="completed")
    assert v_mismatch["success"] is False
    assert v_mismatch["verified"] is False
    assert "expected 'completed'" in v_mismatch["error"]

    # Verify missing task
    v_fake = await verify_action(target_type="task", target="fake_task_9999")
    assert v_fake["success"] is False
    assert v_fake["verified"] is False
    assert "not found in MongoDB" in v_fake["error"]


@pytest.mark.asyncio
async def test_verify_action_invalid_target_type():
    """Test verify_action rejects unsupported target types."""
    with pytest.raises(ValueError, match="Unsupported target_type"):
        await verify_action(target_type="unsupported_type", target="xyz")


@pytest.mark.asyncio
async def test_workflow_status_tool_and_endpoint():
    """Test workflow_status tool and GET /api/agent/workflow/{run_id} endpoint."""
    db = get_db()
    test_run_id = "run_test_phase3_123"
    await db.execution_runs.insert_one({
        "id": test_run_id,
        "command": "Verify workflow status",
        "status": RunStatus.COMPLETED.value,
        "model": "openai/gpt-oss-20b",
        "steps": [],
        "output": "All operations verified.",
        "goal_achieved": True,
    })

    # Test tool function directly
    status_tool_res = await workflow_status(run_id=test_run_id)
    assert status_tool_res["success"] is True
    assert status_tool_res["run_id"] == test_run_id
    assert status_tool_res["status"] == "completed"

    # Test HTTP endpoint
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(f"/api/agent/workflow/{test_run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["run_id"] == test_run_id


@pytest.mark.asyncio
async def test_integrations_status_endpoint():
    """Test GET /api/integrations/status returns readiness for Gmail, Drive, Browser."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/integrations/status")
    assert response.status_code == 200
    data = response.json()
    assert "gmail" in data
    assert "drive" in data
    assert "browser" in data
    assert data["gmail"]["status"] in ["ready", "pending_setup"]
    assert data["drive"]["status"] in ["ready", "pending_setup"]
    assert data["browser"]["status"] in ["ready", "pending_setup"]


def test_clean_spoken_summary_formatting():
    """Test that _clean_spoken_summary strips markdown tables and code blocks for speech."""
    markdown_sample = """### Prioritized Tasks Report
Here is the prioritized list of tasks for your project.

| Rank | Task ID | Title |
| :--- | :--- | :--- |
| 1 | `task_1` | Fix login bug |

```python
import os
print("code block")
```

The report has been saved to disk and verified. Please check the reports tab."""

    spoken = _clean_spoken_summary(markdown_sample, RunStatus.COMPLETED, [])
    assert "|" not in spoken
    assert "```" not in spoken
    assert "###" not in spoken
    assert "Here is the prioritized list of tasks for your project." in spoken


@pytest.mark.asyncio
async def test_agent_partially_completed_on_step_failure():
    """Test agent assigns PARTIALLY_COMPLETED when one step succeeds and a subsequent step fails."""
    # Seed task
    await create_task(title="Deploy Release", priority="high")

    # Step 1: list_tasks (succeeds)
    tc1 = MagicMock()
    tc1.id = "call_1"
    tc1.function.name = "list_tasks"
    tc1.function.arguments = json.dumps({"status": "pending"})

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message.tool_calls = [tc1]
    resp1.choices[0].message.content = None

    # Step 2: unpermitted or failing tool
    tc2 = MagicMock()
    tc2.id = "call_2"
    tc2.function.name = "unauthorized_remote_action"
    tc2.function.arguments = json.dumps({})

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message.tool_calls = [tc2]
    resp2.choices[0].message.content = None

    # Step 3: Assistant concludes
    resp3 = MagicMock()
    resp3.choices = [MagicMock()]
    resp3.choices[0].message.tool_calls = None
    resp3.choices[0].message.content = "Step 1 succeeded, but step 2 failed."

    with patch("app.agent.get_groq_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [resp1, resp2, resp3]
        mock_get_client.return_value = mock_client

        run = await execute_agent_command("Execute partially failing workflow")

        # Must be marked PARTIALLY_COMPLETED because list_tasks succeeded and unauthorized failed
        assert run.status == RunStatus.PARTIALLY_COMPLETED
        assert run.goal_achieved is False
        assert len(run.steps) == 2
        assert run.steps[0].status == ToolStepStatus.SUCCESS
        assert run.steps[1].status == ToolStepStatus.FAILED
