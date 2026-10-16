import os
import pytest
from app.config import settings
from app.llm import check_groq_health
from app.tools import create_task
from app.agent import execute_agent_command
from app.models import RunStatus

# Only run live integration tests if GROQ_API_KEY is configured
pytestmark = pytest.mark.skipif(
    not settings.GROQ_API_KEY or settings.GROQ_API_KEY.startswith("your_"),
    reason="Live integration tests require a valid GROQ_API_KEY in backend/.env"
)


@pytest.mark.asyncio
async def test_live_groq_health():
    """Live test: verifies real Groq API request succeeds and returns low latency."""
    res = await check_groq_health()
    assert res["status"] == "connected", f"Expected connected but got: {res}"
    assert res["model"] == settings.GROQ_MODEL
    assert res["latency_ms"] is not None
    assert res["latency_ms"] > 0
    assert res["error"] is None


@pytest.mark.asyncio
async def test_live_groq_invalid_key_safe_error():
    """Live test: verifies invalid Groq API key returns safe error without leakage."""
    res = await check_groq_health(api_key="gsk_invalid_test_key_12345")
    assert res["status"] == "disconnected"
    assert "Authentication failed" in res["error"]
    assert "gsk_" not in res["error"]


@pytest.mark.asyncio
async def test_live_agent_multi_step_workflow():
    """
    Live test: executes the multi-step request with Groq LLM (openai/gpt-oss-20b):
    'List my pending tasks, prioritize them by deadline, and generate a report.'
    Verifies that real tool calling occurs, at least two dependent tools are executed,
    and a report file is generated and verified on disk.
    """
    # Clean up any leftover test tasks to keep context small and clean
    from app.database import tasks_collection
    await tasks_collection().delete_many({"title": {"$regex": "^Live Test:"}})

    # Seed sample tasks into the real database
    t1 = await create_task(
        title="Live Test: Security Patch",
        description="Apply security hotfix for CVE",
        priority="critical",
        deadline="2026-10-18"
    )
    t2 = await create_task(
        title="Live Test: UI Polish",
        description="Verify responsive layout on phone and tablet",
        priority="medium",
        deadline="2026-10-25"
    )
    assert t1["success"] is True
    assert t2["success"] is True

    # Execute natural-language multi-step command
    command = "List my pending tasks, prioritize them by deadline, and generate a report."
    run = await execute_agent_command(command)

    # Verifications
    assert run.status == RunStatus.COMPLETED
    assert len(run.steps) >= 2, f"Expected at least 2 dependent tool calls, got {len(run.steps)}: {[s.tool_name for s in run.steps]}"
    assert run.report_url is not None or any(s.tool_name == "generate_report" for s in run.steps)

    # Check verified report file on disk
    if run.report_url:
        filename = os.path.basename(run.report_url)
        filepath = settings.reports_path / filename
        assert filepath.exists(), f"Report file {filepath} was not found on disk"
        assert filepath.stat().st_size > 0, "Report file is empty"
