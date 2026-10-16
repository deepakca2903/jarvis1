import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow

from app.config import settings
from app.database import get_db

logger = logging.getLogger(__name__)

GOOGLE_SCOPES: List[str] = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


def is_google_configured() -> bool:
    """Checks whether Google Client credentials are set in environment."""
    return bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)


async def get_google_auth_status() -> Dict[str, Any]:
    """
    Returns current configuration and authentication status for Google services.
    Never exposes credentials or secrets.
    """
    configured = is_google_configured()
    if not configured:
        return {
            "configured": False,
            "authenticated": False,
            "user_email": None,
            "scopes": [],
            "message": "Google Client ID and Secret are not configured in backend/.env.",
        }

    try:
        db = get_db()
        token_doc = await db.auth_tokens.find_one({"provider": "google"})
        if not token_doc:
            return {
                "configured": True,
                "authenticated": False,
                "user_email": None,
                "scopes": [],
                "message": "Google OAuth is configured, but no user has connected an account yet.",
            }

        # Check credentials validity or refreshability
        creds = await get_google_credentials()
        is_valid = creds is not None and creds.valid
        user_email = token_doc.get("user_email")

        return {
            "configured": True,
            "authenticated": is_valid,
            "user_email": user_email,
            "scopes": token_doc.get("scopes", GOOGLE_SCOPES),
            "message": f"Connected as {user_email}" if is_valid else "Stored authorization token expired or invalid.",
        }
    except Exception as e:
        logger.error("Error checking Google auth status: %s", e)
        return {
            "configured": configured,
            "authenticated": False,
            "user_email": None,
            "scopes": [],
            "message": f"Could not verify Google auth status: {str(e)}",
        }


def create_authorization_url(redirect_uri: Optional[str] = None) -> Tuple[str, str]:
    """
    Constructs the Google OAuth 2.0 authorization URL for Drive and Gmail scopes.
    Returns (authorization_url, state).
    """
    if not is_google_configured():
        raise ValueError(
            "Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in backend/.env."
        )

    r_uri = redirect_uri or settings.GOOGLE_REDIRECT_URI
    client_config = {
        "web": {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [r_uri],
        }
    }

    flow = Flow.from_client_config(
        client_config=client_config,
        scopes=GOOGLE_SCOPES,
        redirect_uri=r_uri,
    )

    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return auth_url, state


async def exchange_code_for_tokens(code: str, redirect_uri: Optional[str] = None) -> Dict[str, Any]:
    """
    Exchanges the OAuth 2.0 authorization code for user credentials and stores
    them securely in MongoDB (auth_tokens collection).
    """
    if not is_google_configured():
        raise ValueError("Google OAuth credentials missing.")

    r_uri = redirect_uri or settings.GOOGLE_REDIRECT_URI
    client_config = {
        "web": {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [r_uri],
        }
    }

    flow = Flow.from_client_config(
        client_config=client_config,
        scopes=GOOGLE_SCOPES,
        redirect_uri=r_uri,
    )

    flow.fetch_token(code=code)
    creds = flow.credentials

    # Extract user email from ID token or userinfo API
    user_email = None
    if getattr(creds, "id_token", None) and isinstance(creds.id_token, dict):
        user_email = creds.id_token.get("email")

    if not user_email:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {creds.token}"},
                    timeout=8.0,
                )
                if res.status_code == 200:
                    user_email = res.json().get("email")
        except Exception as err:
            logger.warning("Could not fetch userinfo email: %s", err)

    token_doc = {
        "provider": "google",
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri or "https://oauth2.googleapis.com/token",
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "scopes": creds.scopes or GOOGLE_SCOPES,
        "user_email": user_email or "connected_user@gmail.com",
        "updated_at": datetime.now(timezone.utc),
    }

    db = get_db()
    await db.auth_tokens.update_one(
        {"provider": "google"},
        {"$set": token_doc},
        upsert=True
    )
    logger.info("Successfully saved Google OAuth credentials for user: %s", user_email)

    return {
        "success": True,
        "authenticated": True,
        "user_email": user_email,
        "message": f"Successfully authorized Google account ({user_email}).",
    }


async def get_google_credentials() -> Optional[Credentials]:
    """
    Retrieves stored Google Credentials from MongoDB and refreshes if expired.
    """
    if not is_google_configured():
        return None

    try:
        db = get_db()
        token_doc = await db.auth_tokens.find_one({"provider": "google"})
        if not token_doc or not token_doc.get("token"):
            return None

        creds = Credentials(
            token=token_doc.get("token"),
            refresh_token=token_doc.get("refresh_token"),
            token_uri=token_doc.get("token_uri") or "https://oauth2.googleapis.com/token",
            client_id=token_doc.get("client_id") or settings.GOOGLE_CLIENT_ID,
            client_secret=token_doc.get("client_secret") or settings.GOOGLE_CLIENT_SECRET,
            scopes=token_doc.get("scopes") or GOOGLE_SCOPES,
        )

        # Refresh if expired
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                await db.auth_tokens.update_one(
                    {"provider": "google"},
                    {
                        "$set": {
                            "token": creds.token,
                            "refresh_token": creds.refresh_token,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    }
                )
                logger.info("Refreshed Google OAuth token for %s", token_doc.get("user_email"))
            except Exception as ref_err:
                logger.error("Failed to refresh Google token: %s", ref_err)
                return None

        return creds if creds.valid else None
    except Exception as e:
        logger.error("Error retrieving Google credentials: %s", e)
        return None


async def revoke_google_auth() -> Dict[str, Any]:
    """Revokes user authorization by removing stored credentials from MongoDB."""
    try:
        db = get_db()
        res = await db.auth_tokens.delete_many({"provider": "google"})
        return {
            "success": True,
            "message": "Google authorization disconnected successfully.",
            "deleted_count": res.deleted_count,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_google_diagnostics() -> Dict[str, Any]:
    """
    Safe diagnostic inspection of Google OAuth configuration, token storage,
    refresh status, and API reachability.
    Never exposes credentials, access tokens, refresh tokens, or client secrets.
    """
    configured = is_google_configured()
    redirect_uri = settings.GOOGLE_REDIRECT_URI
    client_id_preview = (
        f"{settings.GOOGLE_CLIENT_ID[:12]}...{settings.GOOGLE_CLIENT_ID[-18:]}"
        if settings.GOOGLE_CLIENT_ID and len(settings.GOOGLE_CLIENT_ID) > 30
        else ("configured" if settings.GOOGLE_CLIENT_ID else "missing")
    )

    db = get_db()
    token_doc = None
    try:
        token_doc = await db.auth_tokens.find_one({"provider": "google"})
    except Exception as e:
        logger.error("Database query error for auth_tokens: %s", e)

    token_stored = bool(token_doc and token_doc.get("token"))
    has_refresh_token = bool(token_doc and token_doc.get("refresh_token"))
    user_email = token_doc.get("user_email") if token_doc else None
    granted_scopes = token_doc.get("scopes", []) if token_doc else []

    creds = None
    token_refresh_status = "not_stored"
    if token_stored:
        try:
            creds = await get_google_credentials()
            if creds and creds.valid:
                token_refresh_status = "active_and_valid"
            else:
                token_refresh_status = "invalid_or_revoked"
        except Exception as ref_exc:
            token_refresh_status = f"refresh_error: {str(ref_exc)}"

    # Test Gmail API connection safely without sending emails
    gmail_test: Dict[str, Any] = {"status": "unauthorized", "details": "No authorized credentials stored."}
    if creds:
        try:
            from googleapiclient.discovery import build
            service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            profile = service.users().getProfile(userId="me").execute()
            gmail_test = {
                "status": "connected",
                "email_address": profile.get("emailAddress"),
                "messages_total": profile.get("messagesTotal"),
                "details": "Gmail API connection verified successfully.",
            }
        except Exception as g_err:
            gmail_test = {
                "status": "failed",
                "error": str(g_err),
                "details": "Gmail API request failed. Check scopes and API enablement in Google Cloud Console.",
            }

    # Test Google Drive API connection safely
    drive_test: Dict[str, Any] = {"status": "unauthorized", "details": "No authorized credentials stored."}
    if creds:
        try:
            from googleapiclient.discovery import build
            service = build("drive", "v3", credentials=creds, cache_discovery=False)
            about = service.about().get(fields="user").execute()
            user_info = about.get("user", {})
            drive_test = {
                "status": "connected",
                "display_name": user_info.get("displayName"),
                "email_address": user_info.get("emailAddress"),
                "details": "Google Drive v3 API connection verified successfully.",
            }
        except Exception as d_err:
            drive_test = {
                "status": "failed",
                "error": str(d_err),
                "details": "Google Drive API request failed. Check scopes and API enablement in Google Cloud Console.",
            }

    return {
        "oauth_configured": configured,
        "client_id_preview": client_id_preview,
        "redirect_uri": redirect_uri,
        "user_authorized": bool(creds and creds.valid),
        "user_email": user_email,
        "granted_scopes": granted_scopes,
        "token_stored": token_stored,
        "has_refresh_token": has_refresh_token,
        "token_refresh_status": token_refresh_status,
        "gmail_api_test": gmail_test,
        "drive_api_test": drive_test,
        "secrets_exposed": False,
    }


async def test_gmail_draft_creation() -> Dict[str, Any]:
    """Safely creates a test email draft in Gmail staged for user confirmation."""
    from app.integrations.gmail import draft_email
    db = get_db()
    token_doc = await db.auth_tokens.find_one({"provider": "google"})
    user_email = (token_doc.get("user_email") if token_doc else None) or "me"
    
    return await draft_email(
        to=user_email,
        subject="[Jarvis Audit] Diagnostic Test Draft",
        body="This is a safe test draft generated by Jarvis AI Agent to verify Gmail API drafting capability. It has not been sent."
    )


async def test_drive_upload_verification() -> Dict[str, Any]:
    """Safely creates a tiny test file and uploads it to Google Drive to verify link."""
    from app.integrations.drive import upload_report_to_drive
    test_file_name = "drive_diagnostic_test.md"
    test_path = settings.reports_path / test_file_name
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("# Jarvis Drive Diagnostic Test\nUploaded and verified at " + datetime.now(timezone.utc).isoformat())

    return await upload_report_to_drive(
        file_path=test_file_name,
        title="Jarvis Drive Diagnostic Test"
    )

