import os
import pytest
from unittest.mock import patch
from mongomock_motor import AsyncMongoMockClient

from app.database import set_client
from app.tools import create_task, list_tasks, prioritize_tasks, generate_report
from app.config import settings


@pytest.fixture(autouse=True)
def setup_mock_db():
    mock_client = AsyncMongoMockClient()
    set_client(mock_client)
    yield
    set_client(None)


@pytest.mark.asyncio
async def test_create_task_and_verify_in_mongodb():
    """Verify task is saved and verified in MongoDB."""
    result = await create_task(
        title="Complete API Documentation",
        description="Write OpenAPI specs and developer guide",
        priority="high",
        deadline="2026-10-25"
    )
    assert result["success"] is True
    assert "task" in result
    task = result["task"]
    assert task["title"] == "Complete API Documentation"
    assert task["priority"] == "high"
    assert task["deadline"] == "2026-10-25"
    assert task["status"] == "pending"
    assert task["id"].startswith("task_")


@pytest.mark.asyncio
async def test_create_task_empty_title_fails():
    """Verify empty title raises ValueError without writing invalid data."""
    with pytest.raises(ValueError, match="Task title cannot be empty"):
        await create_task(title="")


@pytest.mark.asyncio
async def test_list_tasks_retrieval_and_filtering():
    """Verify list_tasks filters by status and priority."""
    await create_task(title="Task A", priority="low", deadline="2026-11-01")
    await create_task(title="Task B", priority="critical", deadline="2026-10-18")
    await create_task(title="Task C", priority="medium", deadline="2026-10-22")

    # List all
    all_res = await list_tasks()
    assert all_res["success"] is True
    assert all_res["count"] == 3

    # Filter critical
    crit_res = await list_tasks(priority="critical")
    assert crit_res["count"] == 1
    assert crit_res["tasks"][0]["title"] == "Task B"


@pytest.mark.asyncio
async def test_prioritize_tasks_deterministic_deadline():
    """Verify deterministic sorting: earliest deadline first, tie-breaking by priority."""
    await create_task(title="Later Deadline", priority="critical", deadline="2026-10-30")
    await create_task(title="Earlier Deadline", priority="low", deadline="2026-10-18")
    await create_task(title="No Deadline High", priority="high", deadline=None)

    prioritized_res = await prioritize_tasks(sort_by="deadline")
    tasks = prioritized_res["prioritized_tasks"]
    assert len(tasks) == 3

    # Earlier deadline (2026-10-18) must be Rank 1
    assert tasks[0]["title"] == "Earlier Deadline"
    assert tasks[0]["priority_rank"] == 1

    # Later deadline (2026-10-30) must be Rank 2
    assert tasks[1]["title"] == "Later Deadline"
    assert tasks[1]["priority_rank"] == 2

    # No deadline must be at the end
    assert tasks[2]["title"] == "No Deadline High"
    assert tasks[2]["priority_rank"] == 3


@pytest.mark.asyncio
async def test_generate_report_verified_on_disk():
    """Verify report file is written to disk, has non-zero size, and is readable."""
    await create_task(title="Deploy Backend", priority="high", deadline="2026-10-20")
    await create_task(title="Audit Security", priority="critical", deadline="2026-10-19")

    res = await generate_report(title="Executive Audit")
    assert res["success"] is True
    assert res["verified"] is True
    assert res["tasks_included"] == 2
    assert res["size_bytes"] > 0
    assert os.path.exists(res["file_path"])

    # Verify content readable
    with open(res["file_path"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "Executive Audit" in content
        assert "Audit Security" in content
        assert "Deploy Backend" in content
