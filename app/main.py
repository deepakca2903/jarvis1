import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, status, Response, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import close_db, ping_db, init_indexes
from app.llm import check_groq_health
from app.tools import list_tasks, create_task
from app.agent import execute_agent_command

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
DIST_DIR = STATIC_DIR if (STATIC_DIR / "index.html").exists() else (FRONTEND_DIST if (FRONTEND_DIST / "index.html").exists() else None)

from app.models import (
    HealthResponse,
    DatabaseHealthResponse,
    GroqHealthResponse,
    CommandRequest,
    AgentRun,
    TaskCreate,
    ReportInfo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("jarvis")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure reports folder, test database connection & Groq
    logger.info("Initializing Jarvis AI Agent backend (Phase 2)...")
    settings.reports_path.mkdir(parents=True, exist_ok=True)

    db_status = await ping_db(timeout_ms=2500)
    if db_status["status"] == "connected":
        logger.info("MongoDB Atlas connection verified: %s", db_status["database"])
        await init_indexes()
    else:
        logger.warning(
            "MongoDB is currently offline or unreachable (%s). Backend running in degraded mode.",
            db_status.get("error")
        )

    groq_status = await check_groq_health()
    if groq_status["status"] == "connected":
        logger.info("Groq LLM verified successfully: %s (%sms)", groq_status["model"], groq_status["latency_ms"])
    else:
        logger.warning("Groq connection issue on startup: %s", groq_status.get("error"))

    yield
    # Shutdown
    logger.info("Shutting down Jarvis AI Agent backend...")
    await close_db()


app = FastAPI(
    title="Jarvis AI Agent API",
    description="Autonomous Agent backend powered by FastAPI, Groq LLM (openai/gpt-oss-20b), and MongoDB.",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="API Root")
async def root(request: Request):
    accept_header = request.headers.get("accept", "")
    if DIST_DIR and "text/html" in accept_header and (DIST_DIR / "index.html").exists():
        return FileResponse(DIST_DIR / "index.html")
    return {
        "service": "Jarvis AI Agent API",
        "version": "0.4.1",
        "model": settings.GROQ_MODEL,
        "endpoints": {
            "health": "/api/health",
            "database_health": "/api/health/db",
            "groq_health": "/api/health/groq",
            "agent_command": "/api/agent/command",
            "workflow_status": "/api/agent/workflow/{run_id}",
            "integrations_status": "/api/integrations/status",
            "tasks": "/api/tasks",
            "reports": "/api/reports",
            "docs": "/docs",
        }
    }


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Backend Health Check",
    tags=["Health"]
)
async def health_check():
    """Returns the operational status of the FastAPI service."""
    return HealthResponse()


@app.get(
    "/api/health/db",
    response_model=DatabaseHealthResponse,
    summary="MongoDB Connectivity Check",
    tags=["Health"]
)
async def db_health_check(response: Response):
    """
    Validates MongoDB connectivity with a quick round-trip ping.
    Returns HTTP 200 if connected, or HTTP 503 if disconnected.
    """
    res = await ping_db()
    if res["status"] != "connected":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return DatabaseHealthResponse(**res)


@app.get(
    "/api/health/groq",
    response_model=GroqHealthResponse,
    summary="Groq LLM Connectivity Check",
    tags=["Health"]
)
async def groq_health_endpoint(response: Response):
    """
    Performs a real, live API request to Groq using the configured GROQ_MODEL.
    Returns latency and operational status without leaking keys.
    """
    res = await check_groq_health()
    if res["status"] == "disconnected":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return GroqHealthResponse(**res)


@app.post(
    "/api/agent/command",
    response_model=AgentRun,
    summary="Execute Natural-Language Command with Groq Agent",
    tags=["Agent"]
)
async def agent_command(req: CommandRequest):
    """
    Accepts natural-language commands, prompts Groq LLM to select from allowlisted
    Python tools, executes dependent tools sequentially, and persists full history.
    """
    try:
        run_record = await execute_agent_command(command=req.command, task_id=req.task_id)
        return run_record
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Command execution error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {type(e).__name__} - {str(e)}")


@app.get(
    "/api/tasks",
    summary="List MongoDB Tasks",
    tags=["Tasks"]
)
async def get_tasks(
    status: Optional[str] = Query(None, description="Filter by status (pending, completed, etc.)"),
    priority: Optional[str] = Query(None, description="Filter by priority (low, medium, high, critical)"),
    limit: int = Query(50, ge=1, le=100)
):
    """Retrieves actual tasks stored in MongoDB."""
    return await list_tasks(status=status, priority=priority, limit=limit)


@app.post(
    "/api/tasks",
    summary="Create Task Directly in MongoDB",
    tags=["Tasks"]
)
async def post_task(req: TaskCreate):
    """Directly creates a task in MongoDB and verifies the write."""
    try:
        return await create_task(
            title=req.title,
            description=req.description,
            priority=req.priority,
            deadline=req.deadline,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/reports",
    response_model=List[ReportInfo],
    summary="List Generated Reports",
    tags=["Reports"]
)
async def get_reports():
    """Lists verified report files created on disk."""
    reports_dir = settings.reports_path
    report_files: List[ReportInfo] = []

    if not reports_dir.exists():
        return report_files

    for f in reports_dir.glob("report_*.md"):
        try:
            stat = f.stat()
            report_files.append(
                ReportInfo(
                    filename=f.name,
                    file_path=str(f),
                    download_url=f"/api/reports/{f.name}",
                    size_bytes=stat.st_size,
                    created_at=f.stat().st_ctime and f.stat().st_mtime and f.name,
                )
            )
        except Exception as err:
            logger.warning("Error reading file stat for %s: %s", f.name, err)

    # Sort newest first
    report_files.sort(key=lambda r: r.filename, reverse=True)
    return report_files


@app.get(
    "/api/reports/{filename}",
    summary="Download Generated Report",
    tags=["Reports"]
)
async def download_report(filename: str):
    """Download a verified report markdown file."""
    # Prevent path traversal attacks
    clean_name = os.path.basename(filename)
    target_path = settings.reports_path / clean_name

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")

    return FileResponse(
        path=target_path,
        media_type="text/markdown",
        filename=clean_name
    )


@app.get(
    "/api/agent/workflow/{run_id}",
    summary="Get Workflow Run Status",
    tags=["Agent"]
)
async def get_workflow_run_status(run_id: str):
    """Retrieves execution progress and step history of a specific workflow run."""
    from app.tools import workflow_status
    res = await workflow_status(run_id=run_id)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error", "Workflow run not found."))
    return res


@app.get(
    "/api/integrations/status",
    summary="Get External Integrations Status",
    tags=["Integrations"]
)
async def get_integrations_status():
    """Returns the configuration and readiness status of modular integrations."""
    from app.integrations.gmail import get_gmail_status
    from app.integrations.drive import get_drive_status
    from app.integrations.browser import get_browser_status

    g_status = await get_gmail_status()
    d_status = await get_drive_status()
    b_status = get_browser_status()

    return {
        "gmail": g_status,
        "drive": d_status,
        "browser": b_status,
    }


# ==============================================================================
# Google OAuth 2.0 User Authorization Endpoints
# ==============================================================================

@app.get(
    "/api/auth/google/status",
    summary="Get Google Account Authorization Status",
    tags=["Auth"]
)
async def google_auth_status_endpoint():
    """Returns Google connection and token validity status."""
    from app.integrations.google_auth import get_google_auth_status
    return await get_google_auth_status()


@app.get(
    "/api/auth/google/login",
    summary="Initiate Google OAuth Authorization Flow",
    tags=["Auth"]
)
async def google_auth_login():
    """Generates the Google OAuth 2.0 consent URL for Drive and Gmail scopes."""
    from app.integrations.google_auth import is_google_configured, create_authorization_url
    if not is_google_configured():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in backend/.env."
        )
    auth_url, state = create_authorization_url()
    return {"auth_url": auth_url, "state": state}


@app.get(
    "/api/auth/google/callback",
    summary="Google OAuth Callback Handler",
    tags=["Auth"]
)
async def google_auth_callback(code: str, state: Optional[str] = None):
    """Exchanges Google authorization code for tokens and redirects back to dashboard."""
    from app.integrations.google_auth import exchange_code_for_tokens
    frontend_origin = settings.CORS_ORIGINS[0] if isinstance(settings.CORS_ORIGINS, list) else "http://localhost:5173"
    try:
        res = await exchange_code_for_tokens(code=code)
        return RedirectResponse(url=f"{frontend_origin}?auth=success&email={res.get('user_email', '')}")
    except Exception as e:
        logger.error("OAuth callback exchange failed: %s", e)
        return RedirectResponse(url=f"{frontend_origin}?auth=failed&error={str(e)}")


@app.post(
    "/api/auth/google/disconnect",
    summary="Disconnect Google Account",
    tags=["Auth"]
)
async def google_auth_disconnect():
    """Disconnects the currently linked Google account and purges stored tokens."""
    from app.integrations.google_auth import revoke_google_auth
    return await revoke_google_auth()


@app.get(
    "/api/auth/google/diagnostics",
    summary="Safe Diagnostic Inspection of Google OAuth & APIs",
    tags=["Auth"]
)
async def google_auth_diagnostics():
    """Safe diagnostic report of OAuth configuration, tokens, and API readiness. Never exposes secrets."""
    from app.integrations.google_auth import get_google_diagnostics
    return await get_google_diagnostics()


@app.post(
    "/api/auth/google/test-draft",
    summary="Test Gmail API by Staging a Draft",
    tags=["Auth"]
)
async def google_auth_test_draft():
    """Creates a safe test draft in Gmail without sending to verify Gmail API."""
    from app.integrations.google_auth import test_gmail_draft_creation
    return await test_gmail_draft_creation()


@app.post(
    "/api/auth/google/test-upload",
    summary="Test Google Drive API by Uploading a Small File",
    tags=["Auth"]
)
async def google_auth_test_upload():
    """Uploads a small test file to Google Drive and verifies the link."""
    from app.integrations.google_auth import test_drive_upload_verification
    return await test_drive_upload_verification()



# Global exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "type": type(exc).__name__}
    )


# If frontend static build exists, mount /assets and provide SPA catch-all
if DIST_DIR and DIST_DIR.exists():
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def catch_all_spa(full_path: str):
        file_path = DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")

