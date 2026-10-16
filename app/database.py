import time
import logging
from typing import Optional, Dict, Any
from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from app.config import settings

logger = logging.getLogger(__name__)

# Global client and database instances
_client: Optional[AsyncMongoClient] = None


def get_client() -> AsyncMongoClient:
    """Return the global PyMongo asynchronous client, initializing if needed."""
    global _client
    if _client is None:
        _client = AsyncMongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=settings.DB_TIMEOUT_MS,
            connectTimeoutMS=settings.DB_TIMEOUT_MS,
        )
    return _client


def set_client(custom_client: Any) -> None:
    """Override client for testing (e.g. mongomock or test harness)."""
    global _client
    _client = custom_client


def get_db():
    """Return the application MongoDB database instance."""
    client = get_client()
    return client[settings.MONGO_DB_NAME]


def tasks_collection():
    return get_db()["tasks"]


def runs_collection():
    return get_db()["execution_runs"]


def tool_steps_collection():
    return get_db()["tool_steps"]


def errors_collection():
    return get_db()["errors"]


async def ping_db(timeout_ms: Optional[int] = None) -> Dict[str, Any]:
    """
    Ping MongoDB to verify connectivity and measure round-trip latency.
    Returns status dict without throwing uncaught exceptions.
    """
    effective_timeout = timeout_ms or settings.DB_TIMEOUT_MS
    try:
        start_time = time.perf_counter()
        client = get_client()
        # Admin command ping
        await client.admin.command("ping")
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "connected",
            "database": settings.MONGO_DB_NAME,
            "latency_ms": elapsed_ms,
            "uri": settings.MONGO_URI.split("@")[-1] if "@" in settings.MONGO_URI else settings.MONGO_URI,
            "error": None,
        }
    except ServerSelectionTimeoutError as err:
        logger.warning("MongoDB ping timeout: %s", err)
        return {
            "status": "disconnected",
            "database": settings.MONGO_DB_NAME,
            "latency_ms": None,
            "uri": settings.MONGO_URI,
            "error": f"Server selection timeout: {str(err)}",
            "hint": "Ensure MongoDB is running locally (mongod) or update MONGO_URI in backend/.env to MongoDB Atlas.",
        }
    except PyMongoError as err:
        logger.warning("MongoDB error: %s", err)
        return {
            "status": "disconnected",
            "database": settings.MONGO_DB_NAME,
            "latency_ms": None,
            "uri": settings.MONGO_URI,
            "error": str(err),
            "hint": "Check network connectivity and MongoDB credentials in backend/.env.",
        }
    except Exception as err:
        logger.error("Unexpected error during MongoDB ping: %s", err)
        return {
            "status": "disconnected",
            "database": settings.MONGO_DB_NAME,
            "latency_ms": None,
            "uri": settings.MONGO_URI,
            "error": f"{type(err).__name__}: {str(err)}",
            "hint": "Verify MongoDB configuration.",
        }


async def init_indexes() -> None:
    """Initialize necessary indexes on collections if connected."""
    try:
        db = get_db()
        await db.tasks.create_index("created_at")
        await db.tasks.create_index("status")
        await db.execution_runs.create_index("task_id")
        await db.execution_runs.create_index("created_at")
        await db.tool_steps.create_index("run_id")
        await db.errors.create_index("run_id")
        logger.info("MongoDB indexes verified.")
    except Exception as err:
        logger.warning("Could not initialize indexes (database may be unreachable during boot): %s", err)


async def close_db() -> None:
    """Close the MongoDB client connection cleanly."""
    global _client
    if _client is not None:
        try:
            await _client.close()
            logger.info("MongoDB connection closed.")
        except Exception as err:
            logger.warning("Error closing MongoDB connection: %s", err)
        finally:
            _client = None
