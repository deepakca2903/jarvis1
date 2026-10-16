import os
import re
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.config import settings
from app.database import tasks_collection, get_db
from app.integrations.drive import search_drive_files, read_drive_file, upload_report_to_drive
from app.integrations.gmail import (
    search_gmail_messages,
    read_gmail_message,
    summarize_email,
    draft_email,
)
from app.integrations.browser import browse_web_page

logger = logging.getLogger(__name__)

PRIORITY_WEIGHTS = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _clean_mongo_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to ensure Mongo docs are JSON serializable (converts _id to str)."""
    if not doc:
        return {}
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    if isinstance(doc.get("updated_at"), datetime):
        doc["updated_at"] = doc["updated_at"].isoformat()
    return doc


async def create_task(
    title: str,
    description: str = "",
    priority: str = "medium",
    deadline: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create and save a new task in MongoDB.
    Verifies that the task is written to the database before reporting success.
    """
    title = (title or "").strip()
    if not title:
        raise ValueError("Task title cannot be empty.")

    clean_priority = priority.lower().strip() if priority else "medium"
    if clean_priority not in PRIORITY_WEIGHTS:
        clean_priority = "medium"

    task_id = f"task_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    task_doc = {
        "id": task_id,
        "title": title,
        "description": description or "",
        "priority": clean_priority,
        "deadline": deadline or None,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "metadata": {},
    }

    # Write to MongoDB
    col = tasks_collection()
    insert_res = await col.insert_one(task_doc)
    if not insert_res.acknowledged:
        raise RuntimeError(f"MongoDB write failed to acknowledge insert for task {task_id}")

    # Explicit Verification: query back the inserted document
    verified_doc = await col.find_one({"id": task_id})
    if not verified_doc:
        raise RuntimeError(f"Database write verification failed: Task {task_id} not found after write.")

    logger.info("Successfully created and verified task in MongoDB: %s", task_id)
    return {
        "success": True,
        "message": f"Task created and verified in database with ID {task_id}",
        "task": _clean_mongo_doc(verified_doc),
    }


async def list_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Retrieve real tasks from MongoDB matching the specified filter criteria.
    """
    query: Dict[str, Any] = {}
    if status:
        query["status"] = status.lower().strip()
    if priority:
        query["priority"] = priority.lower().strip()

    safe_limit = limit if (limit is not None and limit > 0) else 20
    col = tasks_collection()
    cursor = col.find(query).sort("created_at", -1).limit(max(1, min(safe_limit, 100)))
    raw_tasks = await cursor.to_list(length=safe_limit)
    tasks = [_clean_mongo_doc(t) for t in raw_tasks]

    logger.info("Retrieved %d tasks from MongoDB (filter: %s)", len(tasks), query)
    return {
        "success": True,
        "count": len(tasks),
        "filter": query,
        "tasks": tasks,
    }


def _parse_deadline_sort_key(deadline_str: Optional[str]) -> float:
    """Parse a deadline string to a timestamp for sorting; return high value for missing/invalid."""
    if not deadline_str:
        return 9999999999.0  # Put tasks without deadlines at the end
    try:
        # Support YYYY-MM-DD or full ISO format
        dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        # Fallback date parsing for simple YYYY-MM-DD
        try:
            dt = datetime.strptime(deadline_str[:10], "%Y-%m-%d")
            return dt.timestamp()
        except Exception:
            return 9999999998.0


async def prioritize_tasks(
    task_ids: Optional[List[str]] = None,
    sort_by: str = "deadline"
) -> Dict[str, Any]:
    """
    Deterministic Python logic to prioritize tasks by deadline or priority rank.
    """
    col = tasks_collection()
    if task_ids:
        cursor = col.find({"id": {"$in": task_ids}})
    else:
        # Default to all pending tasks
        cursor = col.find({"status": "pending"})

    raw_tasks = await cursor.to_list(length=100)
    tasks = [_clean_mongo_doc(t) for t in raw_tasks]

    if not tasks:
        return {
            "success": True,
            "count": 0,
            "message": "No tasks found to prioritize.",
            "prioritized_tasks": [],
        }

    # Deterministic sorting
    sort_criterion = sort_by.lower().strip()
    if sort_criterion == "priority":
        # Critical (4) down to Low (1), then earliest deadline, then title
        def sort_key(t):
            p_weight = PRIORITY_WEIGHTS.get(t.get("priority", "medium"), 2)
            d_time = _parse_deadline_sort_key(t.get("deadline"))
            return (-p_weight, d_time, t.get("title", ""))
    else:
        # Deadline first (earliest date first), then priority (critical > high), then title
        def sort_key(t):
            d_time = _parse_deadline_sort_key(t.get("deadline"))
            p_weight = PRIORITY_WEIGHTS.get(t.get("priority", "medium"), 2)
            return (d_time, -p_weight, t.get("title", ""))

    sorted_tasks = sorted(tasks, key=sort_key)

    # Attach deterministic rank
    for rank, task in enumerate(sorted_tasks, start=1):
        task["priority_rank"] = rank

    logger.info("Prioritized %d tasks by %s", len(sorted_tasks), sort_criterion)
    return {
        "success": True,
        "count": len(sorted_tasks),
        "sort_criteria": sort_criterion,
        "prioritized_tasks": sorted_tasks,
    }


async def generate_report(
    title: str,
    task_ids: Optional[List[str]] = None,
    status_filter: Optional[str] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Generate a markdown summary report from actual MongoDB task data.
    Verifies that the generated report file exists and is readable before reporting success.
    """
    title = (title or "Task Report").strip()
    col = tasks_collection()

    query: Dict[str, Any] = {}
    if task_ids:
        query["id"] = {"$in": task_ids}
    elif status_filter:
        query["status"] = status_filter.lower().strip()

    cursor = col.find(query).sort("created_at", -1)
    raw_tasks = await cursor.to_list(length=200)
    tasks = [_clean_mongo_doc(t) for t in raw_tasks]

    # Deterministically prioritize tasks for the report by deadline & priority
    def report_sort_key(t):
        d_time = _parse_deadline_sort_key(t.get("deadline"))
        p_weight = PRIORITY_WEIGHTS.get(t.get("priority", "medium"), 2)
        return (d_time, -p_weight, t.get("title", ""))

    sorted_tasks = sorted(tasks, key=report_sort_key)

    # Generate Report Content
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Metrics
    total_count = len(sorted_tasks)
    critical_count = sum(1 for t in sorted_tasks if t.get("priority") == "critical")
    high_count = sum(1 for t in sorted_tasks if t.get("priority") == "high")
    pending_count = sum(1 for t in sorted_tasks if t.get("status") == "pending")
    completed_count = sum(1 for t in sorted_tasks if t.get("status") == "completed")

    lines = [
        f"# 📊 Jarvis Executive Report: {title}",
        f"**Generated:** {now_str}",
        f"**Status:** Verified Generated Artifact",
        "",
        "## Summary Metrics",
        f"- **Total Tasks Analyzed:** {total_count}",
        f"- **Pending Tasks:** {pending_count}",
        f"- **Completed Tasks:** {completed_count}",
        f"- **High/Critical Priority Tasks:** {critical_count + high_count}",
        "",
    ]

    if notes:
        lines.extend([
            "## Notes",
            notes,
            "",
        ])

    lines.extend([
        "## Prioritized Task Breakdown",
        "| Rank | Task ID | Title | Priority | Deadline | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if not sorted_tasks:
        lines.append("| - | - | *No tasks found matching criteria* | - | - | - |")
    else:
        for idx, t in enumerate(sorted_tasks, start=1):
            t_id = t.get("id", "N/A")
            t_title = t.get("title", "Untitled").replace("|", "-")
            t_prio = (t.get("priority") or "medium").upper()
            t_dead = t.get("deadline") or "None"
            t_stat = (t.get("status") or "pending").upper()
            lines.append(f"| {idx} | `{t_id}` | {t_title} | **{t_prio}** | {t_dead} | {t_stat} |")

    lines.extend([
        "",
        "## Task Details",
    ])

    for idx, t in enumerate(sorted_tasks, start=1):
        desc = t.get("description") or "No description provided."
        lines.extend([
            f"### {idx}. {t.get('title')}",
            f"- **Task ID:** `{t.get('id')}`",
            f"- **Priority:** {t.get('priority')} | **Deadline:** {t.get('deadline') or 'None'}",
            f"- **Description:** {desc}",
            "",
        ])

    lines.append(f"---\n*Generated by Jarvis AI Agent &bull; Verified on disk*")
    report_content = "\n".join(lines)

    # Save to disk
    sanitized_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title.lower())[:30].strip("_")
    filename = f"report_{sanitized_title}_{now.strftime('%Y%m%d_%H%M%S')}.md"
    file_path = settings.reports_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    # Verification: check file exists, has content > 0, and is readable
    if not os.path.exists(file_path):
        raise RuntimeError(f"File verification failed: {file_path} was not created.")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise RuntimeError(f"File verification failed: {file_path} is an empty file.")

    # Read back to ensure readability
    with open(file_path, "r", encoding="utf-8") as f:
        read_back = f.read(20)
        if not read_back.startswith("#"):
            raise RuntimeError(f"File verification failed: {file_path} content is corrupted.")

    # Persist report record in MongoDB
    try:
        db = get_db()
        await db.reports.insert_one({
            "filename": filename,
            "title": title,
            "file_path": str(file_path),
            "size_bytes": file_size,
            "tasks_count": total_count,
            "created_at": now,
        })
    except Exception as err:
        logger.warning("Could not persist report metadata in MongoDB: %s", err)

    download_url = f"/api/reports/{filename}"
    logger.info("Generated and verified report: %s (%d bytes)", filename, file_size)

    return {
        "success": True,
        "filename": filename,
        "file_path": str(file_path),
        "download_url": download_url,
        "tasks_included": total_count,
        "size_bytes": file_size,
        "verified": True,
        "summary": f"Report '{title}' generated and verified on disk with {total_count} tasks.",
    }


async def verify_action(
    target_type: str,
    target: str,
    expected_state: Optional[str] = None
) -> Dict[str, Any]:
    """
    Explicitly verify whether an action or artifact actually succeeded.
    - target_type: 'file', 'task', or 'workflow'
    - target: filename/path, task_id, or run_id
    """
    target_type = (target_type or "").lower().strip()
    target = (target or "").strip()

    if target_type == "file":
        filename = os.path.basename(target)
        file_path = settings.reports_path / filename
        if not file_path.exists():
            return {
                "success": False,
                "verified": False,
                "target_type": "file",
                "target": target,
                "error": f"File '{filename}' does not exist on disk.",
            }
        size = file_path.stat().st_size
        if size == 0:
            return {
                "success": False,
                "verified": False,
                "target_type": "file",
                "target": target,
                "error": f"File '{filename}' exists but is empty (0 bytes).",
            }
        return {
            "success": True,
            "verified": True,
            "target_type": "file",
            "target": target,
            "filename": filename,
            "size_bytes": size,
            "download_url": f"/api/reports/{filename}",
            "message": f"Verified file '{filename}' exists on disk ({size} bytes).",
        }

    elif target_type == "task":
        col = tasks_collection()
        doc = await col.find_one({"$or": [{"id": target}, {"title": target}]})
        if not doc:
            return {
                "success": False,
                "verified": False,
                "target_type": "task",
                "target": target,
                "error": f"Task '{target}' was not found in MongoDB.",
            }
        clean_doc = _clean_mongo_doc(doc)
        if expected_state and clean_doc.get("status") != expected_state.lower():
            return {
                "success": False,
                "verified": False,
                "target_type": "task",
                "target": target,
                "actual_status": clean_doc.get("status"),
                "expected_status": expected_state,
                "error": f"Task status is '{clean_doc.get('status')}', expected '{expected_state}'.",
            }
        return {
            "success": True,
            "verified": True,
            "target_type": "task",
            "target": target,
            "task": clean_doc,
            "message": f"Verified task '{clean_doc.get('id')}' exists in MongoDB (status: {clean_doc.get('status')}).",
        }

    elif target_type == "workflow":
        db = get_db()
        doc = await db.execution_runs.find_one({"id": target})
        if not doc:
            return {
                "success": False,
                "verified": False,
                "target_type": "workflow",
                "target": target,
                "error": f"Workflow run '{target}' was not found in database.",
            }
        clean_doc = _clean_mongo_doc(doc)
        return {
            "success": True,
            "verified": True,
            "target_type": "workflow",
            "target": target,
            "run": clean_doc,
            "message": f"Verified workflow run '{target}' exists (status: {clean_doc.get('status')}).",
        }

    else:
        raise ValueError(f"Unsupported target_type '{target_type}'. Must be 'file', 'task', or 'workflow'.")


async def workflow_status(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve the execution status and step history of a workflow run.
    If run_id is omitted, returns the most recent workflow execution.
    """
    db = get_db()
    col = db.execution_runs
    if run_id and run_id != "latest":
        doc = await col.find_one({"id": run_id})
    else:
        doc = await col.find_one(sort=[("created_at", -1)])

    if not doc:
        return {
            "success": False,
            "error": "No workflow runs found in database.",
        }

    clean_doc = _clean_mongo_doc(doc)
    steps = clean_doc.get("steps", [])
    completed_steps = sum(1 for s in steps if s.get("status") in ["success", "retried"])
    failed_steps = sum(1 for s in steps if s.get("status") == "failed")

    return {
        "success": True,
        "run_id": clean_doc.get("id"),
        "command": clean_doc.get("command"),
        "status": clean_doc.get("status"),
        "total_steps": len(steps),
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "report_url": clean_doc.get("report_url"),
        "error": clean_doc.get("error"),
        "summary": f"Workflow {clean_doc.get('id')}: {clean_doc.get('status')} ({completed_steps}/{len(steps)} steps succeeded).",
    }


# Tool Definitions for Groq/OpenAI Function Calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create and save a new task in MongoDB. Returns the verified task document with unique ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title of the task (e.g. 'Review quarterly financials').",
                    },
                    "description": {
                        "type": ["string", "null"],
                        "description": "Detailed description or context for the task.",
                    },
                    "priority": {
                        "type": ["string", "null"],
                        "enum": ["low", "medium", "high", "critical", None],
                        "description": "Task priority level. Defaults to 'medium'.",
                    },
                    "deadline": {
                        "type": ["string", "null"],
                        "description": "Optional deadline date in ISO format, e.g. '2026-10-20' or '2026-10-20T18:00:00Z'.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "Retrieve actual tasks currently stored in MongoDB. Can filter by status ('pending', 'completed') and priority.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": ["string", "null"],
                        "enum": ["pending", "in_progress", "completed", "failed", None],
                        "description": "Optional status filter.",
                    },
                    "priority": {
                        "type": ["string", "null"],
                        "enum": ["low", "medium", "high", "critical", None],
                        "description": "Optional priority filter.",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "Maximum number of tasks to return (default 20).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prioritize_tasks",
            "description": "Prioritize a list of tasks using deterministic logic. Sorts by deadline (earliest first, nulls last) or priority level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Optional list of task IDs to prioritize. If omitted, prioritizes all pending tasks.",
                    },
                    "sort_by": {
                        "type": ["string", "null"],
                        "enum": ["deadline", "priority", None],
                        "description": "Sorting criteria: 'deadline' (default) or 'priority'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a verified Markdown report from actual task data in MongoDB and save it to disk. Returns verified download URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the report (e.g. 'Pending Tasks Prioritization Report').",
                    },
                    "task_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Optional list of specific task IDs to include.",
                    },
                    "status_filter": {
                        "type": ["string", "null"],
                        "description": "Optional filter, e.g. 'pending'.",
                    },
                    "notes": {
                        "type": ["string", "null"],
                        "description": "Optional executive summary or notes to include in the report.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_action",
            "description": "Explicitly verify that an action or artifact succeeded (e.g. check that a generated report file exists on disk and is readable, or that a task is recorded in MongoDB).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_type": {
                        "type": "string",
                        "enum": ["file", "task", "workflow"],
                        "description": "Type of target to verify: 'file' for reports on disk, 'task' for database tasks, 'workflow' for runs.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Filename, task ID, or run ID to verify.",
                    },
                    "expected_state": {
                        "type": ["string", "null"],
                        "description": "Optional expected status (e.g. 'pending', 'completed').",
                    },
                },
                "required": ["target_type", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_status",
            "description": "Check the progress and execution status of a specific workflow run or the latest run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": ["string", "null"],
                        "description": "Optional ID of the workflow run to inspect (omit to inspect the latest run).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_drive_files",
            "description": "Search authorized files in Google Drive by name or keywords. Returns file IDs, names, mimeTypes, and web view links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or file name to search in Google Drive.",
                    },
                    "page_size": {
                        "type": ["integer", "null"],
                        "description": "Maximum number of files to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_drive_file",
            "description": "Read content from a supported file in Google Drive (Google Docs, plain text, markdown, CSV, JSON). Returns text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The unique Google Drive file ID to read.",
                    },
                },
                "required": ["file_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upload_report_to_drive",
            "description": "Upload a generated report artifact to Google Drive and verify its existence. Returns verified Drive link.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path or filename of the local report to upload (e.g. filename from generate_report).",
                    },
                    "title": {
                        "type": ["string", "null"],
                        "description": "Optional title for the file in Google Drive.",
                    },
                    "folder_id": {
                        "type": ["string", "null"],
                        "description": "Optional Google Drive folder ID to upload into.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gmail_messages",
            "description": "Search authorized Gmail messages for queries, senders, or subjects. Returns matched message snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for Gmail (e.g. 'project update', 'from:manager@example.com').",
                    },
                    "max_results": {
                        "type": ["integer", "null"],
                        "description": "Max results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_gmail_message",
            "description": "Read authorized Gmail message contents by ID. Decodes full message body, headers (From, To, Cc, Subject, Date), and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "Unique Gmail message ID to read (from search_gmail_messages).",
                    },
                },
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_email",
            "description": "Summarize an email's key discussion points, action items, deadlines, and recommended reply points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": ["string", "null"],
                        "description": "The Gmail message ID to summarize.",
                    },
                    "email_text": {
                        "type": ["string", "null"],
                        "description": "Optional raw email body text to summarize if message_id is not provided.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Create a safe draft email in Gmail. Staged for user review and confirmation before sending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject line of the email.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Body content of the email.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_web_page",
            "description": "Browse a web page using automated browser to extract page title, text content, and research data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to browse (http:// or https://).",
                    },
                    "extract_summary": {
                        "type": ["boolean", "null"],
                        "description": "Whether to extract clean text summary.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

# Strict Allowlist Registry
TOOL_REGISTRY = {
    "create_task": create_task,
    "list_tasks": list_tasks,
    "prioritize_tasks": prioritize_tasks,
    "generate_report": generate_report,
    "verify_action": verify_action,
    "workflow_status": workflow_status,
    "search_drive_files": search_drive_files,
    "read_drive_file": read_drive_file,
    "upload_report_to_drive": upload_report_to_drive,
    "search_gmail_messages": search_gmail_messages,
    "read_gmail_message": read_gmail_message,
    "summarize_email": summarize_email,
    "draft_email": draft_email,
    "browse_web_page": browse_web_page,
}

