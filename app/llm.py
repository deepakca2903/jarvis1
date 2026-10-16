import time
import logging
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
from openai import (
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    NotFoundError,
    OpenAIError,
)
from app.config import settings

logger = logging.getLogger(__name__)

# Global client cache
_groq_client: Optional[AsyncOpenAI] = None


def get_groq_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    """Return an asynchronous OpenAI client configured for Groq API."""
    global _groq_client
    key = settings.GROQ_API_KEY if api_key is None else api_key
    if not key:
        raise ValueError("GROQ_API_KEY is not configured in backend/.env")

    # If an explicit key was provided or client not yet created
    if api_key or _groq_client is None:
        client = AsyncOpenAI(
            api_key=key,
            base_url=settings.GROQ_BASE_URL,
            timeout=settings.GROQ_TIMEOUT_SECONDS,
            max_retries=1,
        )
        if not api_key:
            _groq_client = client
        return client
    return _groq_client


def set_groq_client(client: Optional[AsyncOpenAI]) -> None:
    """Override client for testing/mocking."""
    global _groq_client
    _groq_client = client


async def check_groq_health(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Check Groq API connectivity using a real, sanitized API request.
    Validates authentication, model availability, and measures round-trip latency.
    Never exposes API keys or authorization headers.
    """
    key = settings.GROQ_API_KEY if api_key is None else api_key
    if not key or not key.strip() or key.startswith("your_"):
        return {
            "status": "disconnected",
            "model": settings.GROQ_MODEL,
            "latency_ms": None,
            "error": "GROQ_API_KEY is not configured",
            "hint": "Set a valid GROQ_API_KEY (starts with gsk_) in backend/.env",
        }

    try:
        client = get_groq_client(api_key=key)
        start_time = time.perf_counter()

        # Execute a minimal real completion request
        resp = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_completion_tokens=10,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info("Groq health check succeeded in %sms for model %s", latency_ms, settings.GROQ_MODEL)

        return {
            "status": "connected",
            "model": settings.GROQ_MODEL,
            "latency_ms": latency_ms,
            "error": None,
            "hint": None,
        }

    except AuthenticationError as err:
        logger.warning("Groq authentication failure: %s", err)
        return {
            "status": "disconnected",
            "model": settings.GROQ_MODEL,
            "latency_ms": None,
            "error": "Authentication failed: Invalid or expired Groq API key",
            "hint": "Check the GROQ_API_KEY in backend/.env (starts with gsk_).",
        }

    except RateLimitError as err:
        logger.warning("Groq rate limit: %s", err)
        return {
            "status": "degraded",
            "model": settings.GROQ_MODEL,
            "latency_ms": None,
            "error": "Groq API rate limit reached",
            "hint": "The service is temporarily throttled. Wait a few moments before retrying.",
        }

    except APITimeoutError as err:
        logger.warning("Groq request timeout: %s", err)
        return {
            "status": "disconnected",
            "model": settings.GROQ_MODEL,
            "latency_ms": None,
            "error": f"Groq request timed out after {settings.GROQ_TIMEOUT_SECONDS}s",
            "hint": "Check your internet connection or Groq API status.",
        }

    except APIConnectionError as err:
        logger.warning("Groq connection error: %s", err)
        return {
            "status": "disconnected",
            "model": settings.GROQ_MODEL,
            "latency_ms": None,
            "error": "Cannot connect to api.groq.com",
            "hint": "Verify network and firewall access to https://api.groq.com.",
        }

    except NotFoundError as err:
        logger.warning("Groq model not found: %s", err)
        return {
            "status": "disconnected",
            "model": settings.GROQ_MODEL,
            "latency_ms": None,
            "error": f"Model '{settings.GROQ_MODEL}' was not found on Groq",
            "hint": "Verify that GROQ_MODEL in backend/.env matches an available model.",
        }

    except OpenAIError as err:
        logger.warning("Groq API error: %s", err)
        # Sanitize error to avoid leaking any header or key details
        return {
            "status": "disconnected",
            "model": settings.GROQ_MODEL,
            "latency_ms": None,
            "error": f"Groq API error: {type(err).__name__}",
            "hint": "Inspect backend logs for additional non-sensitive diagnostic details.",
        }

    except Exception as err:
        logger.error("Unexpected error checking Groq health: %s", err)
        return {
            "status": "disconnected",
            "model": settings.GROQ_MODEL,
            "latency_ms": None,
            "error": f"Unexpected error: {type(err).__name__}",
            "hint": "Check backend logs for details.",
        }
