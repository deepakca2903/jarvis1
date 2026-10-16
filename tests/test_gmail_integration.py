import os
import base64
import json
import socket
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mongomock_motor import AsyncMongoMockClient
from googleapiclient.errors import HttpError
from google.auth.exceptions import RefreshError
import httplib2

from app.database import set_client, get_db
from app.tools import (
    search_gmail_messages,
    read_gmail_message,
    summarize_email,
    draft_email,
)
from app.agent import execute_agent_command
from app.models import ToolStepStatus, RunStatus


@pytest.fixture(autouse=True)
def setup_mock_db():
    mock_client = AsyncMongoMockClient()
    set_client(mock_client)
    yield
    set_client(None)


# ==============================================================================
# 1. Unauthorized & Authorization Lifecycle Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_gmail_unauthorized_state():
    """Verify all Gmail tools report authorization_required when credentials are absent."""
    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds:
        mock_creds.return_value = None

        search_res = await search_gmail_messages(query="Quarterly Report")
        assert search_res["success"] is False
        assert search_res["status"] == "authorization_required"

        read_res = await read_gmail_message(message_id="msg_xyz_123")
        assert read_res["success"] is False
        assert read_res["status"] == "authorization_required"

        draft_res = await draft_email(to="user@example.com", subject="Test", body="Hello")
        assert draft_res["success"] is False
        assert draft_res["status"] == "authorization_required"


@pytest.mark.asyncio
async def test_gmail_revoked_authorization_handling():
    """Verify handling when stored token is expired or revoked by user in Google account."""
    fake_creds = MagicMock()
    fake_creds.valid = True

    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()

    # Simulate RefreshError during API call
    mock_messages.list.side_effect = RefreshError("invalid_grant: Token has been expired or revoked.")
    mock_users.messages.return_value = mock_messages
    mock_service.users.return_value = mock_users

    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.gmail.build", return_value=mock_service):
        mock_creds.return_value = fake_creds

        res = await search_gmail_messages(query="important")
        assert res["success"] is False
        assert res["status"] == "authorization_revoked"
        assert "revoked or expired" in res["error"].lower()


# ==============================================================================
# 2. Search Emails Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_search_gmail_empty_results():
    """Verify clean response when no emails match the search query."""
    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()

    mock_list_call = MagicMock()
    mock_list_call.execute.return_value = {"messages": []}
    mock_messages.list.return_value = mock_list_call
    mock_users.messages.return_value = mock_messages
    mock_service.users.return_value = mock_users

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.gmail.build", return_value=mock_service):
        mock_creds.return_value = fake_creds

        res = await search_gmail_messages(query="non_existent_subject_999")
        assert res["success"] is True
        assert res["count"] == 0
        assert res["messages"] == []
        assert "No emails found" in res["message"]


@pytest.mark.asyncio
async def test_search_gmail_valid_results():
    """Verify parsing of matched messages with headers and snippets."""
    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()

    mock_list_call = MagicMock()
    mock_list_call.execute.return_value = {
        "messages": [
            {"id": "msg_001", "threadId": "th_001"},
            {"id": "msg_002", "threadId": "th_002"},
        ]
    }
    mock_messages.list.return_value = mock_list_call

    def mock_get(userId, id, format, metadataHeaders):
        call = MagicMock()
        if id == "msg_001":
            call.execute.return_value = {
                "id": "msg_001",
                "snippet": "Here is the Q3 project update.",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "alice@company.com"},
                        {"name": "To", "value": "me@company.com"},
                        {"name": "Subject", "value": "Q3 Project Review"},
                        {"name": "Date", "value": "Tue, 14 Oct 2026 09:00:00 GMT"},
                    ]
                },
            }
        else:
            call.execute.return_value = {
                "id": "msg_002",
                "snippet": "Budget approval needed ASAP.",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "finance@company.com"},
                        {"name": "To", "value": "me@company.com"},
                        {"name": "Subject", "value": "Budget Approval Request"},
                        {"name": "Date", "value": "Wed, 15 Oct 2026 10:30:00 GMT"},
                    ]
                },
            }
        return call

    mock_messages.get.side_effect = mock_get
    mock_users.messages.return_value = mock_messages
    mock_service.users.return_value = mock_users

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.gmail.build", return_value=mock_service):
        mock_creds.return_value = fake_creds

        res = await search_gmail_messages(query="Project", max_results=5)
        assert res["success"] is True
        assert res["count"] == 2
        assert res["messages"][0]["id"] == "msg_001"
        assert res["messages"][0]["subject"] == "Q3 Project Review"
        assert res["messages"][0]["sender"] == "alice@company.com"
        assert res["messages"][1]["id"] == "msg_002"
        assert res["messages"][1]["subject"] == "Budget Approval Request"


# ==============================================================================
# 3. Read Email Message Content Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_read_gmail_message_multipart():
    """Verify read_gmail_message decodes multipart body, headers, and word count."""
    plain_text = "Hi Jarvis team,\nPlease review the deployment plan by tomorrow at 4 PM.\nThanks, Bob"
    plain_b64 = base64.urlsafe_b64encode(plain_text.encode("utf-8")).decode("utf-8")

    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()

    mock_get_call = MagicMock()
    mock_get_call.execute.return_value = {
        "id": "msg_deploy_777",
        "threadId": "th_deploy_777",
        "snippet": "Please review the deployment plan...",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "bob@contractor.org"},
                {"name": "To", "value": "lead@company.com"},
                {"name": "Cc", "value": "devops@company.com"},
                {"name": "Subject", "value": "Deployment Schedule v1.0"},
                {"name": "Date", "value": "Thu, 15 Oct 2026 14:00:00 GMT"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": plain_b64},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": base64.urlsafe_b64encode(b"<p>HTML version</p>").decode("utf-8")},
                },
            ],
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

        res = await read_gmail_message(message_id="msg_deploy_777")
        assert res["success"] is True
        assert res["id"] == "msg_deploy_777"
        assert res["sender"] == "bob@contractor.org"
        assert res["recipient"] == "lead@company.com"
        assert res["cc"] == "devops@company.com"
        assert res["subject"] == "Deployment Schedule v1.0"
        assert "Please review the deployment plan" in res["body"]
        assert res["word_count"] > 5


@pytest.mark.asyncio
async def test_read_gmail_message_html_fallback():
    """Verify HTML email without plain text is cleaned of tags."""
    html_content = "<div><p>Hello! <strong>Important announcement:</strong> server maintenance starts at midnight.</p></div>"
    html_b64 = base64.urlsafe_b64encode(html_content.encode("utf-8")).decode("utf-8")

    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()

    mock_get_call = MagicMock()
    mock_get_call.execute.return_value = {
        "id": "msg_html_123",
        "threadId": "th_html_123",
        "snippet": "Important announcement: server maintenance...",
        "payload": {
            "mimeType": "text/html",
            "headers": [
                {"name": "From", "value": "ops@infra.com"},
                {"name": "Subject", "value": "Maintenance Notice"},
            ],
            "body": {"data": html_b64},
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

        res = await read_gmail_message(message_id="msg_html_123")
        assert res["success"] is True
        assert "<div>" not in res["body"]
        assert "<p>" not in res["body"]
        assert "Important announcement: server maintenance starts at midnight." in res["body"]


# ==============================================================================
# 4. Summarize Email Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_summarize_email_structured_extraction():
    """Verify summarize_email extracts key points, action items, and recommended reply."""
    mock_read_result = {
        "success": True,
        "id": "msg_project_555",
        "sender": "sarah@techcorp.com",
        "subject": "Hackathon Project Submission Guidelines",
        "date": "Thu, 15 Oct 2026 11:00:00 GMT",
        "body": (
            "Hi team! We need to finalize the project demo video by Friday 5 PM. "
            "Please confirm who will record the voice assistant demo. "
            "Also, can you verify that the MongoDB connection works across devices? "
            "Let's win this hackathon!"
        ),
    }

    with patch("app.integrations.gmail.read_gmail_message", new_callable=AsyncMock) as mock_read:
        mock_read.return_value = mock_read_result

        summary = await summarize_email(message_id="msg_project_555")
        assert summary["success"] is True
        assert summary["subject"] == "Hackathon Project Submission Guidelines"
        assert summary["sender"] == "sarah@techcorp.com"
        assert len(summary["key_points"]) > 0
        assert len(summary["action_items"]) > 0
        # Verify action items contain detected requests/questions
        action_text = " ".join(summary["action_items"])
        assert "Please confirm" in action_text or "finalize" in action_text or "verify" in action_text
        assert "sarah" in summary["proposed_reply"].lower()


@pytest.mark.asyncio
async def test_summarize_email_with_raw_text():
    """Verify summarize_email works with direct text without message_id."""
    raw_email = "Could you please send over the latest API documentation? We have a hard deadline on Monday."
    res = await summarize_email(email_text=raw_email)
    assert res["success"] is True
    assert len(res["action_items"]) > 0
    assert "deadline" in " ".join(res["action_items"]).lower()


# ==============================================================================
# 5. Draft Email & Safety Confirmation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_draft_email_requires_explicit_confirmation():
    """Verify draft_email creates a draft and enforces user confirmation parameters."""
    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_drafts = MagicMock()

    mock_create_call = MagicMock()
    mock_create_call.execute.return_value = {
        "id": "draft_secure_123",
        "message": {"id": "msg_staged_123"},
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
            to="director@organization.org",
            subject="Status Report Submission",
            body="The multi-step autonomous workflow has achieved 100% verification."
        )

        assert res["success"] is True
        assert res["status"] == "draft_created"
        assert res["requires_user_confirmation"] is True
        assert res["action_type"] == "email_draft"
        assert res["recipient"] == "director@organization.org"
        assert res["subject"] == "Status Report Submission"
        assert "The multi-step autonomous workflow" in res["body"]
        assert "confirmation" in res["confirmation_prompt"].lower()


# ==============================================================================
# 6. Error & Quota Handling Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_gmail_missing_scopes_error_handling():
    """Verify handling when token lacks required Gmail scopes."""
    resp_mock = httplib2.Response({"status": "403"})
    http_error = HttpError(resp=resp_mock, content=b'{"error": {"message": "Insufficient Permission: Request had insufficient authentication scopes."}}')

    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()
    mock_messages.list.side_effect = http_error
    mock_users.messages.return_value = mock_messages
    mock_service.users.return_value = mock_users

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.gmail.build", return_value=mock_service):
        mock_creds.return_value = fake_creds

        res = await search_gmail_messages(query="test")
        assert res["success"] is False
        assert res["status"] == "missing_scopes"
        assert "scopes are missing" in res["error"].lower()


@pytest.mark.asyncio
async def test_gmail_quota_exceeded_handling():
    """Verify handling when Gmail API quota is exceeded."""
    resp_mock = httplib2.Response({"status": "429"})
    http_error = HttpError(resp=resp_mock, content=b'{"error": {"message": "User Rate Limit Exceeded"}}')

    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()
    mock_messages.list.side_effect = http_error
    mock_users.messages.return_value = mock_messages
    mock_service.users.return_value = mock_users

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.gmail.build", return_value=mock_service):
        mock_creds.return_value = fake_creds

        res = await search_gmail_messages(query="test")
        assert res["success"] is False
        assert res["status"] == "quota_exceeded"
        assert "quota or rate limit" in res["error"].lower()


@pytest.mark.asyncio
async def test_gmail_timeout_handling():
    """Verify handling when Gmail API call times out."""
    mock_service = MagicMock()
    mock_users = MagicMock()
    mock_messages = MagicMock()
    mock_messages.list.side_effect = socket.timeout("Operation timed out")
    mock_users.messages.return_value = mock_messages
    mock_service.users.return_value = mock_users

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("app.integrations.gmail.get_google_credentials", new_callable=AsyncMock) as mock_creds, \
         patch("app.integrations.gmail.build", return_value=mock_service):
        mock_creds.return_value = fake_creds

        res = await search_gmail_messages(query="test")
        assert res["success"] is False
        assert res["status"] == "timeout"
        assert "timed out" in res["error"].lower()


# ==============================================================================
# 7. End-to-End Dependent Workflow: Search -> Read -> Summarize -> Draft
# ==============================================================================

@pytest.mark.asyncio
async def test_gmail_end_to_end_dependent_workflow():
    """
    Test autonomous multi-step workflow:
    1. search_gmail_messages -> finds email with id 'msg_proposal_101'
    2. read_gmail_message -> reads full body
    3. summarize_email -> summarizes key points & action items
    4. draft_email -> drafts a safe reply staged for review
    5. Concluding response showing recipient, subject, and body.
    """
    tc1 = MagicMock()
    tc1.id = "call_search_mail"
    tc1.function.name = "search_gmail_messages"
    tc1.function.arguments = json.dumps({"query": "Hackathon Proposal"})

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message.tool_calls = [tc1]
    resp1.choices[0].message.content = None

    tc2 = MagicMock()
    tc2.id = "call_read_mail"
    tc2.function.name = "read_gmail_message"
    tc2.function.arguments = json.dumps({"message_id": "msg_proposal_101"})

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message.tool_calls = [tc2]
    resp2.choices[0].message.content = None

    tc3 = MagicMock()
    tc3.id = "call_summarize_mail"
    tc3.function.name = "summarize_email"
    tc3.function.arguments = json.dumps({"message_id": "msg_proposal_101"})

    resp3 = MagicMock()
    resp3.choices = [MagicMock()]
    resp3.choices[0].message.tool_calls = [tc3]
    resp3.choices[0].message.content = None

    tc4 = MagicMock()
    tc4.id = "call_draft_mail"
    tc4.function.name = "draft_email"
    tc4.function.arguments = json.dumps({
        "to": "lead@hackathon.org",
        "subject": "Re: Hackathon Proposal",
        "body": "Thank you for the feedback. We have incorporated all requested changes and staged the submission."
    })

    resp4 = MagicMock()
    resp4.choices = [MagicMock()]
    resp4.choices[0].message.tool_calls = [tc4]
    resp4.choices[0].message.content = None

    resp5 = MagicMock()
    resp5.choices = [MagicMock()]
    resp5.choices[0].message.tool_calls = None
    resp5.choices[0].message.content = (
        "I searched your Gmail, read the Hackathon Proposal email, summarized it, and created a draft reply "
        "to lead@hackathon.org with subject 'Re: Hackathon Proposal'. The draft is staged in Gmail for your review."
    )

    async def mock_search(**kwargs):
        return {
            "success": True,
            "count": 1,
            "messages": [{"id": "msg_proposal_101", "subject": "Hackathon Proposal", "sender": "lead@hackathon.org"}]
        }

    async def mock_read(**kwargs):
        return {
            "success": True,
            "id": "msg_proposal_101",
            "sender": "lead@hackathon.org",
            "subject": "Hackathon Proposal",
            "body": "Please review the proposed architecture and confirm the submission deadline.",
            "word_count": 12,
        }

    async def mock_summarize(**kwargs):
        return {
            "success": True,
            "message_id": "msg_proposal_101",
            "subject": "Hackathon Proposal",
            "sender": "lead@hackathon.org",
            "key_points": ["Review proposed architecture."],
            "action_items": ["Confirm submission deadline."],
            "proposed_reply": "Confirm receipt and address architecture.",
        }

    async def mock_draft(**kwargs):
        return {
            "success": True,
            "draft_id": "draft_proposal_reply_99",
            "status": "draft_created",
            "requires_user_confirmation": True,
            "to": "lead@hackathon.org",
            "subject": "Re: Hackathon Proposal",
            "body": kwargs.get("body"),
        }

    with patch("app.agent.get_groq_client") as mock_get_client, \
         patch.dict("app.agent.TOOL_REGISTRY", {
             "search_gmail_messages": mock_search,
             "read_gmail_message": mock_read,
             "summarize_email": mock_summarize,
             "draft_email": mock_draft,
         }):

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [resp1, resp2, resp3, resp4, resp5]
        mock_get_client.return_value = mock_client

        run = await execute_agent_command(
            "Search for the Hackathon Proposal email, read it, summarize its action items, and draft a response."
        )

        assert run.status == RunStatus.COMPLETED
        assert run.goal_achieved is True
        assert len(run.steps) == 4
        assert run.steps[0].tool_name == "search_gmail_messages"
        assert run.steps[1].tool_name == "read_gmail_message"
        assert run.steps[2].tool_name == "summarize_email"
        assert run.steps[3].tool_name == "draft_email"
        assert run.steps[3].result["status"] == "draft_created"
        assert run.steps[3].result["requires_user_confirmation"] is True
