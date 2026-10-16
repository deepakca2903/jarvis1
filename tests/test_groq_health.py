import httpx
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from openai import AuthenticationError

from app.main import app
from app.llm import check_groq_health


@pytest.mark.asyncio
async def test_groq_health_endpoint_success():
    """Test /api/health/groq returns 200 and latency when Groq is reachable."""
    mock_health = {
        "status": "connected",
        "model": "openai/gpt-oss-20b",
        "latency_ms": 142.5,
        "error": None,
        "hint": None,
    }
    with patch("app.main.check_groq_health", new_callable=AsyncMock) as mock_chk:
        mock_chk.return_value = mock_health
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.get("/api/health/groq")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "connected"
        assert data["model"] == "openai/gpt-oss-20b"
        assert data["latency_ms"] == 142.5
        assert data["error"] is None


@pytest.mark.asyncio
async def test_groq_health_endpoint_invalid_key():
    """Test /api/health/groq returns 503 and safe error message on auth error."""
    mock_health = {
        "status": "disconnected",
        "model": "openai/gpt-oss-20b",
        "latency_ms": None,
        "error": "Authentication failed: Invalid or expired Groq API key",
        "hint": "Check the GROQ_API_KEY in backend/.env",
    }
    with patch("app.main.check_groq_health", new_callable=AsyncMock) as mock_chk:
        mock_chk.return_value = mock_health
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.get("/api/health/groq")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "disconnected"
        assert "Authentication failed" in data["error"]
        assert "gsk_" not in data["error"]  # Ensure no key leakage


@pytest.mark.asyncio
async def test_check_groq_health_missing_key():
    """Test check_groq_health safely handles missing or empty keys."""
    res = await check_groq_health(api_key="")
    assert res["status"] == "disconnected"
    assert "GROQ_API_KEY is not configured" in res["error"]
    assert res["latency_ms"] is None


@pytest.mark.asyncio
async def test_check_groq_health_auth_error_sanitization():
    """Test that AuthenticationError is intercepted and does not leak keys."""
    with patch("app.llm.get_groq_client") as mock_get_client:
        mock_client = AsyncMock()
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        fake_response = httpx.Response(status_code=401, request=request)
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API Key secret_12345",
            response=fake_response,
            body={"error": {"message": "Invalid API Key"}}
        )
        mock_get_client.return_value = mock_client
        res = await check_groq_health(api_key="gsk_fakekey")
        assert res["status"] == "disconnected"
        assert "secret_12345" not in res["error"]
        assert "Authentication failed" in res["error"]
