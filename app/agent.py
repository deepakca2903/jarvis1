import re
import time
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from app.config import settings
from app.llm import get_groq_client
from app.database import runs_collection, tool_steps_collection, errors_collection
from app.models import AgentRun, ToolStep, ToolStepStatus, RunStatus
from app.tools import TOOL_DEFINITIONS, TOOL_REGISTRY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Jarvis, an autonomous and highly capable AI assistant operating within a strict Python execution environment.
You have access to a verified set of Python tools. You must select and execute tools via function calling to fulfill user commands.

Operational Guidelines:
1. Task Management: Use create_task, list_tasks, prioritize_tasks, and generate_report for task operations and report generation.
2. Google Drive:
   - Use search_drive_files to locate documents by keyword or filename.
   - Use read_drive_file to read Google Docs, markdown, text, CSV, or JSON documents.
   - Use upload_report_to_drive to upload generated report artifacts to Google Drive and return verified webViewLinks.
3. Gmail:
   - Use search_gmail_messages to search authorized emails by query, sender, or subject.
   - Use read_gmail_message to retrieve full message contents, body text, and headers by message ID.
   - Use summarize_email to generate structured key discussion points, action items, and recommended reply outlines.
   - Use draft_email to create safe email drafts.
   - SAFETY CONSTRAINTS FOR GMAIL:
     * NEVER attempt or claim to send emails automatically.
     * Always inform the user that the email has been created as a draft in Gmail.
     * Always display the recipient ('to'), subject, and message body in your final response.
     * Explicit user confirmation is required before any email is actually sent.
4. Web Browsing: Use browse_web_page for authorized research and webpage text extraction.
5. Cross-App & Dependent Workflows:
   - Pass actual results between dependent steps:
     * Example: search_gmail_messages -> read_gmail_message -> summarize_email -> draft_email.
     * Example: read_drive_file -> create_task in MongoDB -> prioritize_tasks -> generate_report -> upload_report_to_drive.
   - If a prerequisite step fails or returns authorization_required, recognize the state immediately and explain the required setup clearly to the user.
   - Never invent fake tasks, fabricated email IDs, fake files, or fake Drive IDs.
6. When finished, provide a professional, helpful summary mentioning verified file names, recipient/subject/body of any drafted emails, and operational highlights.
"""

SECRET_KEYS = {"api_key", "key", "password", "secret", "token", "authorization"}


def _redact_secrets(data: Any) -> Any:
    """Recursively redacts values for any keys that match sensitive terms."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if any(sec in k.lower() for sec in SECRET_KEYS):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = _redact_secrets(v)
        return cleaned
    elif isinstance(data, list):
        return [_redact_secrets(item) for item in data]
    return data


def _clean_spoken_summary(raw_text: str, run_status: RunStatus, steps: List[ToolStep]) -> str:
    """
    Produce a concise, natural spoken voice response free of markdown tables,
    long code blocks, raw JSON, and technical headers.
    """
    if not raw_text:
        if run_status == RunStatus.COMPLETED:
            return f"I have completed your request with {len(steps)} verified operations."
        elif run_status == RunStatus.PARTIALLY_COMPLETED:
            return "Your request was partially completed. Please check your screen for details."
        else:
            return "I encountered an error executing your request. Please check your screen."

    # Remove markdown headers, bold, italics, code blocks
    text = re.sub(r"```[\s\S]*?```", "", raw_text)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\|.*?\|", "", text)  # remove table lines
    text = re.sub(r"^[#*>\-\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = " ".join(text.split())

    # Extract first 2-3 sentences for comfortable speech
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean_speech = " ".join(sentences[:2]).strip()
    return clean_speech or "Task completed successfully."


async def _call_llm_with_retry(client, max_retries: int = 3, **kwargs):
    """
    Calls Groq chat completions with automatic exponential backoff on rate limits (429/413)
    or temporary network failures.
    """
    for attempt in range(max_retries + 1):
        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = (
                "429" in err_str
                or "413" in err_str
                or "rate limit" in err_str
                or "rate_limit_exceeded" in err_str
                or "tokens per minute" in err_str
                or "tpm" in err_str
            )
            if is_rate_limit and attempt < max_retries:
                backoff_sec = 3.5 * (attempt + 1)
                logger.warning(
                    "Groq rate limit encountered (attempt %d/%d). Backing off %.1fs...",
                    attempt + 1, max_retries, backoff_sec
                )
                await asyncio.sleep(backoff_sec)
                continue
            raise exc


async def execute_agent_command(
    command: str,
    task_id: Optional[str] = None
) -> AgentRun:
    """
    Autonomous Workflow Engine.
    Executes an ordered multi-step tool-calling cycle with Groq LLM, handles dependent
    result passing, verifies goals, handles safe retries, and records execution status.
    """
    command = command.strip()
    if not command:
        raise ValueError("Command cannot be empty.")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    # Initialize execution run record in MongoDB
    run_doc: Dict[str, Any] = {
        "id": run_id,
        "command": command,
        "task_id": task_id,
        "status": RunStatus.RUNNING.value,
        "model": settings.GROQ_MODEL,
        "steps": [],
        "output": None,
        "spoken_summary": None,
        "report_url": None,
        "goal_achieved": None,
        "error": None,
        "created_at": now,
        "completed_at": None,
    }
    col_runs = runs_collection()
    await col_runs.insert_one(run_doc)

    client = get_groq_client()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": command},
    ]

    tool_steps_recorded: List[ToolStep] = []
    total_tool_calls = 0
    final_output = ""
    overall_error: Optional[str] = None
    captured_report_url: Optional[str] = None
    has_step_failure = False

    try:
        # Autonomous tool calling loop (bounded by MAX_TOOL_STEPS)
        while total_tool_calls < settings.MAX_TOOL_STEPS:
            logger.info("Agent execution turn %d for run %s...", total_tool_calls + 1, run_id)
            try:
                response = await _call_llm_with_retry(
                    client,
                    model=settings.GROQ_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_completion_tokens=1024,
                )
            except Exception as llm_err:
                logger.error("Groq API communication error: %s", llm_err)
                has_step_failure = True
                overall_error = f"Groq LLM error: {type(llm_err).__name__} - {str(llm_err)}"
                break

            choice = response.choices[0].message
            tool_calls = choice.tool_calls

            # Model decided no more tools are required
            if not tool_calls:
                final_output = choice.content or ""
                if not final_output and getattr(choice, "reasoning", None):
                    final_output = choice.reasoning
                logger.info("Agent finished workflow without further tool calls.")
                break

            # Append assistant message with tool calls
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            # Process tool calls sequentially
            for tc in tool_calls:
                if total_tool_calls >= settings.MAX_TOOL_STEPS:
                    logger.warning("Reached maximum allowed tool executions (%d). Stopping.", settings.MAX_TOOL_STEPS)
                    final_output += f"\n\n[Workflow stopped: execution limit of {settings.MAX_TOOL_STEPS} tool calls reached.]"
                    break

                total_tool_calls += 1
                step_id = f"step_{uuid.uuid4().hex[:8]}"
                tool_name = tc.function.name
                call_id = tc.id

                # Parse arguments
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as jde:
                    logger.warning("Tool arguments parse error: %s", jde)
                    args = {}

                sanitized_args = _redact_secrets(args)
                logger.info("Running tool [%s] (step %s) with arguments: %s", tool_name, step_id, sanitized_args)

                # Validate against allowlist
                if tool_name not in TOOL_REGISTRY:
                    tool_error_msg = f"Tool '{tool_name}' is not in the allowlist of permitted tools."
                    logger.error(tool_error_msg)
                    has_step_failure = True

                    await errors_collection().insert_one({
                        "run_id": run_id,
                        "step_id": step_id,
                        "tool_name": tool_name,
                        "error": tool_error_msg,
                        "created_at": datetime.now(timezone.utc),
                    })

                    step_obj = ToolStep(
                        step_id=step_id,
                        run_id=run_id,
                        tool_name=tool_name,
                        arguments=sanitized_args,
                        result=None,
                        error=tool_error_msg,
                        retry_count=0,
                        status=ToolStepStatus.FAILED,
                        duration_ms=0,
                    )
                    tool_steps_recorded.append(step_obj)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({"error": tool_error_msg}),
                    })
                    continue

                # Execute with bounded single retry
                tool_func = TOOL_REGISTRY[tool_name]
                start_step = time.perf_counter()
                retry_count = 0
                step_status = ToolStepStatus.SUCCESS
                tool_result: Optional[Any] = None
                tool_error: Optional[str] = None

                try:
                    tool_result = await tool_func(**args)
                except Exception as first_exc:
                    logger.warning("Tool %s initial attempt failed: %s. Retrying once...", tool_name, first_exc)
                    retry_count = 1
                    try:
                        await asyncio.sleep(0.4)  # Safe backoff
                        tool_result = await tool_func(**args)
                        step_status = ToolStepStatus.RETRIED
                        logger.info("Tool %s succeeded on retry.", tool_name)
                    except Exception as second_exc:
                        logger.error("Tool %s failed on retry: %s", tool_name, second_exc)
                        step_status = ToolStepStatus.FAILED
                        has_step_failure = True
                        tool_error = str(second_exc)

                        await errors_collection().insert_one({
                            "run_id": run_id,
                            "step_id": step_id,
                            "tool_name": tool_name,
                            "error": tool_error,
                            "retry_count": retry_count,
                            "created_at": datetime.now(timezone.utc),
                        })

                step_duration_ms = round((time.perf_counter() - start_step) * 1000, 2)

                # Capture report download URL if produced
                if isinstance(tool_result, dict):
                    if "download_url" in tool_result:
                        captured_report_url = tool_result["download_url"]

                step_obj = ToolStep(
                    step_id=step_id,
                    run_id=run_id,
                    tool_name=tool_name,
                    arguments=sanitized_args,
                    result=_redact_secrets(tool_result),
                    error=tool_error,
                    retry_count=retry_count,
                    status=step_status,
                    duration_ms=step_duration_ms,
                )
                tool_steps_recorded.append(step_obj)

                # Persist step in MongoDB
                await tool_steps_collection().insert_one(step_obj.model_dump())

                # Feed actual tool output back to model for next turn
                tool_content_str = json.dumps(
                    _redact_secrets(tool_result) if step_status != ToolStepStatus.FAILED else {"error": tool_error}
                )
                if len(tool_content_str) > 2500:
                    tool_content_str = tool_content_str[:2500] + '... [truncated for token limit]'

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_content_str,
                })

        # Ask model for concluding summary if loop ended on tool calls
        if not final_output and not overall_error:
            try:
                final_resp = await _call_llm_with_retry(
                    client,
                    model=settings.GROQ_MODEL,
                    messages=messages,
                    max_completion_tokens=512,
                )
                final_output = final_resp.choices[0].message.content or "Workflow operations completed."
            except Exception as final_err:
                logger.warning("Could not obtain closing summary: %s", final_err)
                final_output = f"Workflow completed with {len(tool_steps_recorded)} tool operations."

    except Exception as e:
        logger.error("Fatal workflow error in run %s: %s", run_id, e, exc_info=True)
        has_step_failure = True
        overall_error = f"{type(e).__name__}: {str(e)}"
        final_output = f"Workflow error: {overall_error}"

    # Determine accurate overall workflow status
    successful_steps = sum(1 for s in tool_steps_recorded if s.status in [ToolStepStatus.SUCCESS, ToolStepStatus.RETRIED])
    failed_steps = sum(1 for s in tool_steps_recorded if s.status == ToolStepStatus.FAILED)

    if overall_error and successful_steps == 0:
        run_status = RunStatus.FAILED
        goal_achieved = False
    elif has_step_failure:
        if successful_steps > 0:
            run_status = RunStatus.PARTIALLY_COMPLETED
            goal_achieved = False
        else:
            run_status = RunStatus.FAILED
            goal_achieved = False
    else:
        run_status = RunStatus.COMPLETED
        goal_achieved = True

    # Generate clean speech summary for voice assistant
    spoken_summary = _clean_spoken_summary(final_output, run_status, tool_steps_recorded)

    completed_at = datetime.now(timezone.utc)
    updated_run = AgentRun(
        id=run_id,
        command=command,
        task_id=task_id,
        status=run_status,
        model=settings.GROQ_MODEL,
        steps=tool_steps_recorded,
        output=final_output.strip() if final_output else None,
        spoken_summary=spoken_summary,
        report_url=captured_report_url,
        goal_achieved=goal_achieved,
        error=overall_error,
        created_at=now,
        completed_at=completed_at,
    )

    await col_runs.update_one(
        {"id": run_id},
        {"$set": updated_run.model_dump()}
    )
    logger.info("Workflow run %s finalized with status '%s' (goal_achieved=%s)", run_id, run_status.value, goal_achieved)
    return updated_run
