import os
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from mongomock_motor import AsyncMongoMockClient

from app.main import app
from app.database import set_client, get_db
from app.config import settings
from app.tools import (
    search_drive_files,
    read_drive_file,
    upload_report_to_drive,
    search_gmail_messages,
    draft_email,
    browse_web_page,
    generate_report,
    create_task,
)
from app.integrations.google_auth import (
    get_google_auth_status,
    revoke_google_auth,
    create_authorization_url,
)
from app.models import ToolStepStatus, RunStatus
from app.agent import execute_agent_command


@pytest.fixture(autouse=True)
def setup_mock_db():
    mock_client = AsyncMongoMockClient()
    set_client(mock_client)
    yield
    set_client(None)


# ==============================================================================
# Google Drive Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_drive_tools_unauthorized_state():
    """Verify Drive tools return authorization_required when no OAuth token exists."""
    with patch("app.integrations.drive.get_google_credentials", new_callable=AsyncMock) as mock_creds:
        mock_creds.return_value = None

        search_res = await search_drive_files(query="Quarterly Roadmap")
        assert search_res["success"] is False
        assert search_res["status"] == "authorization_required"
        assert "authorization" in search_res["message"].lower()

        read_res = await read_drive_file(file_id="doc_xyz_123")
        assert read_res["success"] is False
        assert read_res["status"] == "authorization_required"

        upload_res = await upload_report_to_drive(file_path="non_existent.md")
        assert upload_res["success"] is False
        assert upload_res["status"] == "authorization_required"


@pytest.mark.asyncio
async def test_drive_search_files_mock_success():
    """Verify search_drive_files queries Drive v3 and parses file metadata."""
    mock_drive_service = MagicMock()
    mock_files_resource = MagicMock()
    mock_list_call = MagicMock()

    fake_files = [
        {
            "id": "file_123",
            "name": "Q3_Strategic_Goals.md",
            "mimeType": "text/markdown",
            "webViewLink": "https://drive.google.com/file/d/file_123/view",
            "modifiedTime": "2026-10-15T10:00:00Z",
            "size": "4096",
        }
    ]
    mock_list_call.execute.return_value = {"files": fake_files}
    mock_files_resource.list.return_value = mock_list_call
    mock_drive_service.files.return_value = mock_files_resource

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.drive.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.drive.build", return_value=mock_drive_service):
        mock_creds.return_value = fake_creds

        res = await search_drive_files(query="Strategic Goals", page_size=5)
        assert res["success"] is True
        assert res["count"] == 1
        assert res["files"][0]["id"] == "file_123"
        assert res["files"][0]["name"] == "Q3_Strategic_Goals.md"
        assert "https://drive.google.com" in res["files"][0]["webViewLink"]


@pytest.mark.asyncio
async def test_drive_read_file_mock_success():
    """Verify read_drive_file retrieves text content from Drive document."""
    mock_drive_service = MagicMock()
    mock_files_resource = MagicMock()

    # Mock metadata call
    mock_meta_call = MagicMock()
    mock_meta_call.execute.return_value = {
        "id": "doc_456",
        "name": "Project_Spec.txt",
        "mimeType": "text/plain",
        "webViewLink": "https://drive.google.com/file/d/doc_456/view",
    }
    mock_files_resource.get.return_value = mock_meta_call

    # Mock media download call
    mock_media_call = MagicMock()
    mock_media_call.execute.return_value = b"# Project Specification\nComplete Phase 3 with Drive and Gmail."
    mock_files_resource.get_media.return_value = mock_media_call
    mock_drive_service.files.return_value = mock_files_resource

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.drive.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.drive.build", return_value=mock_drive_service):
        mock_creds.return_value = fake_creds

        res = await read_drive_file(file_id="doc_456")
        assert res["success"] is True
        assert res["name"] == "Project_Spec.txt"
        assert "Complete Phase 3 with Drive and Gmail" in res["content"]


@pytest.mark.asyncio
async def test_drive_upload_report_and_verify_link():
    """Verify upload_report_to_drive uploads local report and validates verified link."""
    # 1. Generate local report first
    await create_task(title="Prepare Demo", priority="high")
    rep_res = await generate_report(title="Drive Upload Test")
    report_filename = rep_res["filename"]

    # 2. Mock Drive API upload and verification
    mock_drive_service = MagicMock()
    mock_files_resource = MagicMock()

    # Upload create call
    mock_create_call = MagicMock()
    mock_create_call.execute.return_value = {
        "id": "uploaded_drive_file_789",
        "name": report_filename,
    }
    mock_files_resource.create.return_value = mock_create_call

    # Verification get call
    mock_get_call = MagicMock()
    mock_get_call.execute.return_value = {
        "id": "uploaded_drive_file_789",
        "name": report_filename,
        "webViewLink": "https://drive.google.com/file/d/uploaded_drive_file_789/view?usp=sharing",
        "size": "1280",
    }
    mock_files_resource.get.return_value = mock_get_call
    mock_drive_service.files.return_value = mock_files_resource

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.drive.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.drive.build", return_value=mock_drive_service), \
         patch("app.integrations.drive.MediaFileUpload"):
        mock_creds.return_value = fake_creds

        res = await upload_report_to_drive(file_path=report_filename)
        assert res["success"] is True
        assert res["verified"] is True
        assert res["drive_file_id"] == "uploaded_drive_file_789"
        assert "https://drive.google.com/file/d/" in res["web_view_link"]


# ==============================================================================
# Gmail Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_gmail_tools_unauthorized_state():
    """Verify Gmail tools report authorization_required when credentials are absent."""
    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds:
        mock_creds.return_value = None

        search_res = await search_gmail_messages(query="urgent update")
        assert search_res["success"] is False
        assert search_res["status"] == "authorization_required"

        draft_res = await draft_email(to="test@example.com", subject="Test", body="Hello")
        assert draft_res["success"] is False
        assert draft_res["status"] == "authorization_required"


@pytest.mark.asyncio
async def test_gmail_search_mock_success():
    """Verify search_gmail_messages queries Gmail API and parses message headers."""
    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()

    mock_list_call = MagicMock()
    mock_list_call.execute.return_value = {"messages": [{"id": "msg_001", "threadId": "th_001"}]}
    mock_messages.list.return_value = mock_list_call

    mock_get_call = MagicMock()
    mock_get_call.execute.return_value = {
        "id": "msg_001",
        "snippet": "Here is the release announcement for v0.3.0.",
        "payload": {
            "headers": [
                {"name": "From", "value": "lead@company.com"},
                {"name": "Subject", "value": "Release Announcement"},
                {"name": "Date", "value": "Thu, 15 Oct 2026 12:00:00 GMT"},
            ]
        },
    }
    mock_messages.get.return_value = mock_get_call

    mock_users.messages.return_value = mock_messages
    mock_service.users.return_value = mock_users

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.gmail.build", return_value=mock_service):
        mock_creds.return_value = fake_creds

        res = await search_gmail_messages(query="Release Announcement")
        assert res["success"] is True
        assert res["count"] == 1
        assert res["messages"][0]["subject"] == "Release Announcement"
        assert res["messages"][0]["sender"] == "lead@company.com"


@pytest.mark.asyncio
async def test_gmail_draft_email_staged_confirmation():
    """Verify draft_email creates a draft without auto-sending."""
    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_drafts = MagicMock()

    mock_create_call = MagicMock()
    mock_create_call.execute.return_value = {
        "id": "draft_999",
        "message": {"id": "msg_staged_999"},
    }
    mock_drafts.create.return_value = mock_create_call
    mock_users.drafts.return_value = mock_drafts
    mock_service.users.return_value = mock_users

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.gmail.build", return_value=mock_service):
        mock_creds.return_value = fake_creds

        res = await draft_email(
            to="stakeholder@company.com",
            subject="Action Plan Draft",
            body="Please review the attached project plan."
        )
        assert res["success"] is True
        assert res["status"] == "draft_created"
        assert res["draft_id"] == "draft_999"
        assert "Ready for your review" in res["message"]


# ==============================================================================
# Playwright Browser Automation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_browse_web_page_invalid_url():
    """Verify browse_web_page rejects unsupported protocols."""
    res = await browse_web_page(url="ftp://files.example.com/data")
    assert res["success"] is False
    assert "Invalid URL protocol" in res["error"]


@pytest.mark.asyncio
async def test_browse_web_page_content_extraction():
    """Verify browse_web_page successfully parses webpage content."""
    fake_html = """
    <html>
      <head><title>Antigravity Autonomous Agent</title></head>
      <body>
        <h1>Welcome to Autonomous Agent Hackathon</h1>
        <p>This agent completes multi-step workflows across applications.</p>
      </body>
    </html>
    """
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.text = fake_html

    with patch("httpx.AsyncClient.get", return_value=mock_http_response):
        # Force fallback to HTTP parser path to test resilient text extraction
        with patch.dict(os.environ, {"PLAYWRIGHT_FORCE_FALLBACK": "1"}):
            res = await browse_web_page("https://example.com/agent-docs")
            assert res["success"] is True
            assert "Antigravity Autonomous Agent" in res["title"]
            assert "multi-step workflows" in res["content"]


# ==============================================================================
# Google OAuth Endpoints Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_google_auth_status_and_disconnect_endpoints():
    """Test /api/auth/google/status and disconnect endpoints."""
    # Seed token in mock DB
    db = get_db()
    await db.auth_tokens.insert_one({
        "provider": "google",
        "token": "fake_access_token",
        "refresh_token": "fake_refresh_token",
        "user_email": "developer@jarvis.ai",
        "scopes": ["https://www.googleapis.com/auth/drive.file"],
    })

    # Test status endpoint
    with patch("app.integrations.google_auth.is_google_configured", return_value=True), \
         patch("app.integrations.google_auth.get_google_credentials") as mock_creds:
        fake_c = MagicMock()
        fake_c.valid = True
        mock_creds.return_value = fake_c

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.get("/api/auth/google/status")
        assert res.status_code == 200
        data = res.json()
        assert data["configured"] is True
        assert data["authenticated"] is True
        assert data["user_email"] == "developer@jarvis.ai"

        # Test disconnect
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            disc_res = await ac.post("/api/auth/google/disconnect")
        assert disc_res.status_code == 200
        assert disc_res.json()["success"] is True

        # Verify token was purged from DB
        doc = await db.auth_tokens.find_one({"provider": "google"})
        assert doc is None


# ==============================================================================
# Multi-Step Cross-App Dependent Workflow Test
# ==============================================================================

@pytest.mark.asyncio
async def test_cross_app_multi_step_workflow():
    """
    Test dependent multi-step execution across Drive, Tasks, Reports, and Email:
    Turn 1: search_drive_files finds project document
    Turn 2: read_drive_file extracts content
    Turn 3: create_task records action item in MongoDB
    Turn 4: draft_email stages email draft for review
    Turn 5: Concluding agent summary
    """
    tc1 = MagicMock()
    tc1.id = "call_drive_search"
    tc1.function.name = "search_drive_files"
    tc1.function.arguments = json.dumps({"query": "Hackathon Specs"})

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message.tool_calls = [tc1]
    resp1.choices[0].message.content = None

    tc2 = MagicMock()
    tc2.id = "call_drive_read"
    tc2.function.name = "read_drive_file"
    tc2.function.arguments = json.dumps({"file_id": "file_hackathon_99"})

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message.tool_calls = [tc2]
    resp2.choices[0].message.content = None

    tc3 = MagicMock()
    tc3.id = "call_task_create"
    tc3.function.name = "create_task"
    tc3.function.arguments = json.dumps({
        "title": "Deliver Cross-App Demo",
        "priority": "critical",
        "deadline": "2026-10-18"
    })

    resp3 = MagicMock()
    resp3.choices = [MagicMock()]
    resp3.choices[0].message.tool_calls = [tc3]
    resp3.choices[0].message.content = None

    tc4 = MagicMock()
    tc4.id = "call_email_draft"
    tc4.function.name = "draft_email"
    tc4.function.arguments = json.dumps({
        "to": "judges@hackathon.org",
        "subject": "Jarvis Multi-Step Workflow Ready",
        "body": "Cross-app workflow completed across Drive, Tasks, and Gmail."
    })

    resp4 = MagicMock()
    resp4.choices = [MagicMock()]
    resp4.choices[0].message.tool_calls = [tc4]
    resp4.choices[0].message.content = None

    resp5 = MagicMock()
    resp5.choices = [MagicMock()]
    resp5.choices[0].message.tool_calls = None
    resp5.choices[0].message.content = "Multi-step cross-app operations executed successfully."

    # Mock tool functions for isolation
    async def mock_search_drive(**kwargs):
        return {"success": True, "files": [{"id": "file_hackathon_99", "name": "Hackathon_Specs.md"}]}

    async def mock_read_drive(**kwargs):
        return {"success": True, "content": "Deliver multi-step task across apps."}

    async def mock_draft_mail(**kwargs):
        return {"success": True, "draft_id": "draft_777", "status": "draft_created"}

    with patch("app.agent.get_groq_client") as mock_get_client, \
         patch.dict("app.agent.TOOL_REGISTRY", {
             "search_drive_files": mock_search_drive,
             "read_drive_file": mock_read_drive,
             "draft_email": mock_draft_mail,
         }):

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [resp1, resp2, resp3, resp4, resp5]
        mock_get_client.return_value = mock_client

        run = await execute_agent_command(
            "Search Drive for Hackathon Specs, read it, create a task in MongoDB, and draft an email."
        )

        assert run.status == RunStatus.COMPLETED
        assert run.goal_achieved is True
        assert len(run.steps) == 4
        assert run.steps[0].tool_name == "search_drive_files"
        assert run.steps[1].tool_name == "read_drive_file"
        assert run.steps[2].tool_name == "create_task"
        assert run.steps[3].tool_name == "draft_email"
        assert run.steps[0].status == ToolStepStatus.SUCCESS
        assert run.steps[1].status == ToolStepStatus.SUCCESS
        assert run.steps[2].status == ToolStepStatus.SUCCESS
        assert run.steps[3].status == ToolStepStatus.SUCCESS
