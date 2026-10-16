import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from app.config import settings
from app.integrations.google_auth import (
    is_google_configured,
    get_google_credentials,
    get_google_auth_status,
)

logger = logging.getLogger(__name__)


async def get_drive_status() -> Dict[str, Any]:
    """Returns the operational and authorization status of the Google Drive integration."""
    auth_status = await get_google_auth_status()
    is_ready = auth_status.get("authenticated", False)
    return {
        "name": "Google Drive API",
        "status": "ready" if is_ready else "pending_setup",
        "configured": auth_status.get("configured", False),
        "authenticated": is_ready,
        "user_email": auth_status.get("user_email"),
        "auth_type": "OAuth 2.0 (Google Workspace)",
        "description": "Search authorized Drive documents, read text/docs, and upload verified reports.",
        "docs_url": "https://developers.google.com/drive/api/guides/about-sdk",
        "setup_hint": (
            "Connected and ready."
            if is_ready
            else "Configure GOOGLE_CLIENT_ID in backend/.env, then click Connect Google Drive."
        ),
    }


async def search_drive_files(query: str, page_size: int = 10) -> Dict[str, Any]:
    """
    Search authorized files in Google Drive.
    Returns matched file items with ID, name, mimeType, and webViewLink.
    """
    creds = await get_google_credentials()
    if not creds:
        return {
            "success": False,
            "configured": is_google_configured(),
            "authenticated": False,
            "status": "authorization_required",
            "message": (
                "Google Drive integration requires authorization. "
                "Configure GOOGLE_CLIENT_ID in .env and connect via the dashboard."
            ),
        }

    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        clean_q = query.replace("'", "\\'")
        q_expr = f"name contains '{clean_q}' and trashed = false"

        results = service.files().list(
            q=q_expr,
            pageSize=min(max(page_size, 1), 25),
            fields="files(id, name, mimeType, webViewLink, modifiedTime, size)",
        ).execute()

        files = results.get("files", [])
        return {
            "success": True,
            "query": query,
            "count": len(files),
            "files": files,
            "message": f"Found {len(files)} files matching '{query}' in Google Drive.",
        }
    except Exception as e:
        logger.error("Error searching Google Drive files: %s", e)
        return {
            "success": False,
            "error": f"Drive API error: {str(e)}",
            "message": "Failed to execute search on Google Drive.",
        }


async def read_drive_file(file_id: str) -> Dict[str, Any]:
    """
    Reads the content of a supported file from Google Drive.
    Exports Google Docs to plain text, or downloads text/markdown/csv files.
    """
    creds = await get_google_credentials()
    if not creds:
        return {
            "success": False,
            "configured": is_google_configured(),
            "authenticated": False,
            "status": "authorization_required",
            "message": "Google Drive authorization required to read files.",
        }

    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        meta = service.files().get(
            fileId=file_id,
            fields="id, name, mimeType, size, webViewLink"
        ).execute()

        mime_type = meta.get("mimeType", "")
        file_name = meta.get("name", file_id)

        content = ""
        if mime_type == "application/vnd.google-apps.document":
            # Export Google Doc as plain text
            export_bytes = service.files().export(fileId=file_id, mimeType="text/plain").execute()
            content = export_bytes.decode("utf-8", errors="replace")
        elif mime_type == "application/vnd.google-apps.spreadsheet":
            # Export Google Sheet as CSV
            export_bytes = service.files().export(fileId=file_id, mimeType="text/csv").execute()
            content = export_bytes.decode("utf-8", errors="replace")
        elif "text/" in mime_type or "json" in mime_type or "markdown" in mime_type or file_name.endswith((".md", ".txt", ".json", ".csv")):
            # Download text media directly
            media_bytes = service.files().get_media(fileId=file_id).execute()
            content = media_bytes.decode("utf-8", errors="replace")
        else:
            return {
                "success": False,
                "file_id": file_id,
                "name": file_name,
                "mime_type": mime_type,
                "error": f"Unsupported file type '{mime_type}' for inline text reading.",
                "web_view_link": meta.get("webViewLink"),
            }

        return {
            "success": True,
            "file_id": file_id,
            "name": file_name,
            "mime_type": mime_type,
            "content": content[:8000],  # Bound length for context window safety
            "size_chars": len(content),
            "web_view_link": meta.get("webViewLink"),
            "message": f"Successfully read file '{file_name}' from Google Drive.",
        }
    except Exception as e:
        logger.error("Error reading Google Drive file %s: %s", file_id, e)
        return {
            "success": False,
            "file_id": file_id,
            "error": f"Drive API read error: {str(e)}",
        }


async def upload_report_to_drive(
    file_path: str,
    title: Optional[str] = None,
    folder_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Uploads a generated report artifact to Google Drive and verifies that it exists.
    Returns the verified Drive file ID and webViewLink.
    """
    creds = await get_google_credentials()
    if not creds:
        return {
            "success": False,
            "configured": is_google_configured(),
            "authenticated": False,
            "status": "authorization_required",
            "message": (
                "Google Drive upload requires user authorization. "
                "Connect your account via the dashboard."
            ),
        }

    # Resolve local report file
    target_path = Path(file_path)
    if not target_path.is_absolute():
        target_path = settings.reports_path / target_path.name

    if not target_path.exists():
        return {
            "success": False,
            "error": f"Local report file '{target_path.name}' not found for upload.",
        }

    file_size = target_path.stat().st_size
    if file_size == 0:
        return {
            "success": False,
            "error": f"Local report file '{target_path.name}' is empty (0 bytes).",
        }

    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        upload_name = title or target_path.name

        file_metadata: Dict[str, Any] = {
            "name": upload_name,
            "mimeType": "text/markdown",
        }
        if folder_id:
            file_metadata["parents"] = [folder_id]

        media = MediaFileUpload(
            str(target_path),
            mimetype="text/markdown",
            resumable=True
        )

        uploaded = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name, webViewLink, size"
        ).execute()

        drive_file_id = uploaded.get("id")

        # Explicit Verification: query Drive API to verify the file was created
        verified_file = service.files().get(
            fileId=drive_file_id,
            fields="id, name, webViewLink, size"
        ).execute()

        web_link = verified_file.get("webViewLink")
        logger.info("Successfully uploaded and verified file in Google Drive: %s (id: %s)", upload_name, drive_file_id)

        return {
            "success": True,
            "verified": True,
            "drive_file_id": drive_file_id,
            "name": verified_file.get("name"),
            "web_view_link": web_link,
            "size_bytes": verified_file.get("size") or file_size,
            "message": f"Verified file '{upload_name}' uploaded to Google Drive. View Link: {web_link}",
        }
    except Exception as e:
        logger.error("Error uploading report to Google Drive: %s", e)
        return {
            "success": False,
            "verified": False,
            "error": f"Drive API upload failed: {str(e)}",
        }
