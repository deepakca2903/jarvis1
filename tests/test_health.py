import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["endpoints"]["health"] == "/api/health"


@pytest.mark.asyncio
async def test_backend_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "jarvis-agent-backend"
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_db_health_endpoint_disconnected():
    # When MongoDB ping fails or times out
    mock_ping_result = {
        "status": "disconnected",
        "database": "jarvis_db",
        "latency_ms": None,
        "uri": "mongodb://localhost:27017",
        "error": "Server selection timeout",
        "hint": "Ensure MongoDB is running.",
    }
    with patch("app.main.ping_db", new_callable=AsyncMock) as mock_ping:
        mock_ping.return_value = mock_ping_result
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/health/db")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "disconnected"
        assert "Ensure MongoDB is running" in data["hint"]


@pytest.mark.asyncio
async def test_db_health_endpoint_connected():
    # When MongoDB ping succeeds
    mock_ping_result = {
        "status": "connected",
        "database": "jarvis_db",
        "latency_ms": 1.25,
        "uri": "mongodb://localhost:27017",
        "error": None,
        "hint": None,
    }
    with patch("app.main.ping_db", new_callable=AsyncMock) as mock_ping:
        mock_ping.return_value = mock_ping_result
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/health/db")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "connected"
        assert data["database"] == "jarvis_db"
        assert data["latency_ms"] == 1.25
