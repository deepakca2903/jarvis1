from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    # MongoDB Configuration
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "jarvis_db"
    DB_TIMEOUT_MS: int = 5000

    # Groq LLM Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_TIMEOUT_SECONDS: float = 35.0
    MAX_TOOL_STEPS: int = 5

    # Legacy OpenAI placeholders (preserved for backwards-compatibility)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Server & Security
    CORS_ORIGINS: Union[List[str], str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    REPORTS_DIR: str = str(BACKEND_DIR.parent / "reports")

    # Google Integrations (Drive & Gmail OAuth 2.0)
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/auth/google/callback"

    # Playwright Browser Automation & Web Search
    ENABLE_BROWSER_AUTOMATION: bool = True
    SEARCH_PROVIDER: str = "duckduckgo"
    TAVILY_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    BROWSER_TIMEOUT_MS: int = 15000
    BROWSER_HEADLESS: bool = True

    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def parse_cors_origins(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def reports_path(self) -> Path:
        p = Path(self.REPORTS_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
