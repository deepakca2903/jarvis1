import re
import socket
import base64
import logging
from email.mime.text import MIMEText
from typing import Dict, Any, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.exceptions import RefreshError

from app.database import get_db
from app.integrations.google_auth import (
    is_google_configured,
    get_google_credentials,
    get_google_auth_status,
)

logger = logging.getLogger(__name__)


def _strip_html_tags(html_content: str) -> str:
    """Removes HTML markup and cleans up extra whitespace."""
    if not html_content:
        return ""
    clean = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html_content, flags=re.IGNORECASE)
    clean = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"&nbsp;", " ", clean)
    clean = re.sub(r"&amp;", "&", clean)
    clean = re.sub(r"&lt;", "<", clean)
    clean = re.sub(r"&gt;", ">", clean)
    clean = re.sub(r"&quot;", '"', clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _extract_email_body(payload: Dict[str, Any]) -> str:
    """
    Extracts plain text body from a Gmail message payload.
    Traverses multipart MIME structures (parts) recursively.
    Prefers text/plain, falling back to text/html with tags stripped.
    """
    if not payload:
        return ""

    body_data = payload.get("body", {}).get("data")
    mime_type = payload.get("mimeType", "")

    # If payload itself has direct body data
    if body_data and mime_type.startswith("text/"):
        try:
            decoded = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
            if mime_type == "text/html":
                return _strip_html_tags(decoded)
            return decoded.strip()
        except Exception as e:
            logger.warning("Failed to decode top-level email body data: %s", e)

    # If multipart, traverse parts
    parts = payload.get("parts", [])
    plain_texts: List[str] = []
    html_texts: List[str] = []

    def _walk_parts(p_list):
        for part in p_list:
            m_type = part.get("mimeType", "")
            p_data = part.get("body", {}).get("data")
            if p_data:
                try:
                    decoded_text = base64.urlsafe_b64decode(p_data).decode("utf-8", errors="replace")
                    if m_type == "text/plain":
                        plain_texts.append(decoded_text.strip())
                    elif m_type == "text/html":
                        html_texts.append(_strip_html_tags(decoded_text))
                except Exception as dec_err:
                    logger.warning("Failed to decode multipart part: %s", dec_err)
            if "parts" in part and isinstance(part["parts"], list):
                _walk_parts(part["parts"])

    _walk_parts(parts)

    if plain_texts:
        return "\n\n".join(plain_texts).strip()
    if html_texts:
        return "\n\n".join(html_texts).strip()
    return ""


def _handle_gmail_error(e: Exception, operation: str) -> Dict[str, Any]:
    """
    Categorizes errors for Gmail operations:
    - Revoked authorization (RefreshError, HTTP 401)
    - Missing scopes (HTTP 403 insufficientPermissions)
    - Quota exceeded (HTTP 429, HTTP 403 quotaExceeded)
    - Timeouts (socket.timeout, TimeoutError)
    - Generic errors
    """
    logger.error("Gmail error during '%s': %s", operation, e)

    if isinstance(e, RefreshError):
        return {
            "success": False,
            "status": "authorization_revoked",
            "error": "Google authorization was revoked or expired. Please reconnect your Google account in the dashboard.",
            "operation": operation,
        }

    if isinstance(e, (socket.timeout, TimeoutError)):
        return {
            "success": False,
            "status": "timeout",
            "error": f"Gmail API request timed out during '{operation}'. Please check your network and try again.",
            "operation": operation,
        }

    if isinstance(e, HttpError):
        status_code = getattr(e.resp, "status", None) if hasattr(e, "resp") else None
        content = e.content.decode("utf-8", errors="ignore") if hasattr(e, "content") and isinstance(e.content, bytes) else str(e)
        content_lower = content.lower()

        if status_code == 401 or "invalid_grant" in content_lower:
            return {
                "success": False,
                "status": "authorization_revoked",
                "error": "Google authorization expired or was revoked. Please reconnect your account.",
                "operation": operation,
            }

        if status_code == 403:
            if "insufficient" in content_lower or "scope" in content_lower or "insufficientpermissions" in content_lower:
                return {
                    "success": False,
                    "status": "missing_scopes",
                    "error": "Gmail API scopes are missing or insufficient. Please reconnect your account with the required Gmail scopes.",
                    "operation": operation,
                }
            if any(q in content_lower for q in ["quota", "ratelimit", "userratelimit", "dailylimit"]):
                return {
                    "success": False,
                    "status": "quota_exceeded",
                    "error": "Gmail API quota or rate limit exceeded. Please try again later.",
                    "operation": operation,
                }
            return {
                "success": False,
                "status": "permission_denied",
                "error": f"Gmail API permission denied: {content}",
                "operation": operation,
            }

        if status_code == 429:
            return {
                "success": False,
                "status": "quota_exceeded",
                "error": "Gmail API quota or rate limit exceeded. Please try again later.",
                "operation": operation,
            }

        if status_code == 404:
            return {
                "success": False,
                "status": "not_found",
                "error": f"Requested Gmail resource not found for '{operation}'.",
                "operation": operation,
            }

    return {
        "success": False,
        "status": "error",
        "error": f"Gmail {operation} error: {str(e)}",
        "operation": operation,
    }


async def _check_authorization_state() -> Optional[Dict[str, Any]]:
    """Checks credentials and returns error response if unauthenticated or revoked."""
    creds = await get_google_credentials()
    if creds:
        return None

    configured = is_google_configured()
    try:
        db = get_db()
        token_doc = await db.auth_tokens.find_one({"provider": "google"})
        if token_doc:
            return {
                "success": False,
                "configured": configured,
                "authenticated": False,
                "status": "authorization_revoked",
                "error": "Google authorization was revoked or expired. Please reconnect your account in the dashboard.",
                "message": "Google authorization was revoked or expired. Please reconnect your account in the dashboard.",
            }
    except Exception:
        pass

    return {
        "success": False,
        "configured": configured,
        "authenticated": False,
        "status": "authorization_required",
        "error": "Gmail integration requires authorization. Connect your Google account in the dashboard.",
        "message": "Gmail integration requires authorization. Connect your Google account in the dashboard.",
    }


async def get_gmail_status() -> Dict[str, Any]:
    """Returns the operational and authorization status of the Gmail integration."""
    auth_status = await get_google_auth_status()
    is_ready = auth_status.get("authenticated", False)
    return {
        "name": "Gmail API",
        "status": "ready" if is_ready else "pending_setup",
        "configured": auth_status.get("configured", False),
        "authenticated": is_ready,
        "user_email": auth_status.get("user_email"),
        "auth_type": "OAuth 2.0 (Google Workspace)",
        "description": "Search authorized emails, read message content, summarize threads, and create safe drafts for confirmation.",
        "docs_url": "https://developers.google.com/gmail/api/quickstart/python",
        "setup_hint": (
            "Connected and ready."
            if is_ready
            else "Configure GOOGLE_CLIENT_ID in backend/.env, then click Connect Google Account."
        ),
    }


async def search_gmail_messages(query: str, max_results: Optional[int] = 5) -> Dict[str, Any]:
    """
    Search authorized Gmail messages by keywords, sender, or subject.
    Returns matched message snippets, subjects, senders, and dates.
    Handles empty search results cleanly.
    """
    auth_error = await _check_authorization_state()
    if auth_error:
        return auth_error

    creds = await get_google_credentials()
    clean_query = (query or "").strip()
    if not clean_query:
        return {
            "success": False,
            "status": "invalid_argument",
            "error": "Search query cannot be empty.",
        }

    limit = min(max(max_results or 5, 1), 15)

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        list_res = service.users().messages().list(
            userId="me",
            q=clean_query,
            maxResults=limit
        ).execute()

        raw_messages = list_res.get("messages", [])

        # Clean empty result handling
        if not raw_messages:
            return {
                "success": True,
                "query": clean_query,
                "count": 0,
                "messages": [],
                "message": f"No emails found matching query '{clean_query}' in Gmail.",
            }

        extracted_emails: List[Dict[str, Any]] = []

        for m in raw_messages:
            msg_id = m.get("id")
            try:
                detail = service.users().messages().get(
                    userId="me",
                    id=msg_id,
                    format="metadata",
                    metadataHeaders=["From", "To", "Subject", "Date"]
                ).execute()

                headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
                extracted_emails.append({
                    "id": msg_id,
                    "thread_id": m.get("threadId"),
                    "sender": headers.get("From", "Unknown"),
                    "recipient": headers.get("To", "Unknown"),
                    "subject": headers.get("Subject", "(No Subject)"),
                    "date": headers.get("Date"),
                    "snippet": detail.get("snippet", ""),
                })
            except Exception as item_err:
                logger.warning("Could not fetch metadata for message %s: %s", msg_id, item_err)

        return {
            "success": True,
            "query": clean_query,
            "count": len(extracted_emails),
            "messages": extracted_emails,
            "message": f"Found {len(extracted_emails)} message(s) matching '{clean_query}' in Gmail.",
        }

    except Exception as e:
        return _handle_gmail_error(e, "search_gmail_messages")


async def read_gmail_message(message_id: str) -> Dict[str, Any]:
    """
    Retrieve authorized Gmail message contents by ID.
    Decodes full payload body, headers (From, To, Cc, Subject, Date), and snippet.
    """
    auth_error = await _check_authorization_state()
    if auth_error:
        return auth_error

    clean_id = (message_id or "").strip()
    if not clean_id:
        return {
            "success": False,
            "status": "invalid_argument",
            "error": "message_id cannot be empty.",
        }

    creds = await get_google_credentials()

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        detail = service.users().messages().get(
            userId="me",
            id=clean_id,
            format="full"
        ).execute()

        payload = detail.get("payload", {})
        headers_list = payload.get("headers", [])
        headers = {h["name"]: h["value"] for h in headers_list if "name" in h and "value" in h}

        body_text = _extract_email_body(payload)
        if not body_text:
            body_text = detail.get("snippet", "")

        words = len(body_text.split())

        return {
            "success": True,
            "id": clean_id,
            "thread_id": detail.get("threadId"),
            "sender": headers.get("From", "Unknown"),
            "recipient": headers.get("To", "Unknown"),
            "cc": headers.get("Cc"),
            "subject": headers.get("Subject", "(No Subject)"),
            "date": headers.get("Date"),
            "snippet": detail.get("snippet", ""),
            "body": body_text,
            "word_count": words,
            "message": f"Successfully retrieved email '{headers.get('Subject', '(No Subject)')}' from {headers.get('From', 'Unknown')} ({words} words).",
        }

    except Exception as e:
        return _handle_gmail_error(e, "read_gmail_message")


async def summarize_email(
    message_id: Optional[str] = None,
    email_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize authorized email contents into structured key points, action items,
    and a recommended reply outline.
    Can accept a message_id (which retrieves the email automatically) or raw email_text.
    """
    sender = "Unknown"
    subject = "(No Subject)"
    date = None
    content_to_analyze = ""

    if message_id:
        email_result = await read_gmail_message(message_id)
        if not email_result.get("success"):
            return email_result

        sender = email_result.get("sender", "Unknown")
        subject = email_result.get("subject", "(No Subject)")
        date = email_result.get("date")
        content_to_analyze = email_result.get("body") or email_result.get("snippet", "")
    elif email_text:
        content_to_analyze = email_text.strip()
        subject = "User-Provided Email Text"
    else:
        return {
            "success": False,
            "status": "invalid_argument",
            "error": "Either message_id or email_text must be provided to summarize an email.",
        }

    if not content_to_analyze:
        return {
            "success": True,
            "message_id": message_id,
            "subject": subject,
            "sender": sender,
            "summary": "Email content is empty.",
            "key_points": [],
            "action_items": [],
            "proposed_reply": "Acknowledge receipt and request further clarification.",
            "message": f"Email '{subject}' has no readable body content to summarize.",
        }

    # Extract sentences for structured analysis
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content_to_analyze) if len(s.strip()) > 8]

    # Detect action items
    action_keywords = [
        "please", "could you", "can you", "need", "deadline", "by", "asap",
        "review", "approve", "confirm", "send", "update", "action", "priority", "follow up"
    ]
    action_items = [
        s for s in sentences
        if any(re.search(r"\b" + re.escape(k) + r"\b", s.lower()) for k in action_keywords) or s.endswith("?")
    ]

    key_points = sentences[:4] if sentences else [content_to_analyze[:200]]

    # Construct executive summary
    overview = " ".join(key_points[:2]).strip()
    summary_text = (
        f"Email from {sender} on '{subject}'. "
        f"Key message: {overview} "
        f"Found {len(action_items)} action item(s)."
    )

    primary_action = action_items[0] if action_items else "the updates discussed"
    proposed_reply = (
        f"Thank {sender.split('<')[0].strip()} for the email regarding '{subject}'. "
        f"Confirm receipt and address: {primary_action}."
    )

    return {
        "success": True,
        "message_id": message_id,
        "subject": subject,
        "sender": sender,
        "date": date,
        "summary": summary_text,
        "key_points": key_points[:5],
        "action_items": action_items[:5] if action_items else ["No explicit action items or deadlines detected."],
        "proposed_reply": proposed_reply,
        "message": f"Successfully generated summary for email '{subject}'.",
    }


async def draft_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Creates a safe draft email in Gmail.
    DOES NOT SEND EMAILS AUTOMATICALLY.
    Staged in Gmail Drafts folder and requires explicit user review and confirmation before sending.
    """
    auth_error = await _check_authorization_state()
    if auth_error:
        return auth_error

    clean_to = (to or "").strip()
    clean_subject = (subject or "").strip()
    clean_body = (body or "").strip()

    if not clean_to:
        return {"success": False, "status": "invalid_argument", "error": "Recipient email ('to') cannot be empty."}
    if not clean_subject:
        return {"success": False, "status": "invalid_argument", "error": "Email subject cannot be empty."}
    if not clean_body:
        return {"success": False, "status": "invalid_argument", "error": "Email body cannot be empty."}

    creds = await get_google_credentials()

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)

        message = MIMEText(clean_body)
        message["to"] = clean_to
        message["subject"] = clean_subject
        raw_encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        draft = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw_encoded}}
        ).execute()

        draft_id = draft.get("id")
        logger.info("Successfully created Gmail draft (id: %s) to %s", draft_id, clean_to)

        return {
            "success": True,
            "draft_id": draft_id,
            "recipient": clean_to,
            "to": clean_to,
            "subject": clean_subject,
            "body": clean_body,
            "status": "draft_created",
            "requires_user_confirmation": True,
            "action_type": "email_draft",
            "confirmation_prompt": f"Email draft #{draft_id} to '{clean_to}' created in Gmail. Explicit user confirmation is required before sending.",
            "message": (
                f"Draft email to '{clean_to}' with subject '{clean_subject}' created in Gmail drafts (ID: {draft_id}). "
                f"Ready for your review. Action requires explicit user confirmation before sending."
            ),
        }

    except Exception as e:
        return _handle_gmail_error(e, "draft_email")
