"""
config.py
Environment-based configuration using Pydantic Settings.
Reads from environment variables → .env file → defaults.
"""
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field


class Environment(str, Enum):
    DEV = "development"
    STAGING = "staging"
    PROD = "production"


class Settings(BaseSettings):
    # App
    environment: Environment = Field(Environment.DEV, alias="ENVIRONMENT")
    log_level: str = Field("DEBUG", alias="LOG_LEVEL")

    # GitHub
    github_webhook_secret: str = Field(..., alias="GITHUB_WEBHOOK_SECRET")
    github_token: str = Field(..., alias="GITHUB_TOKEN")

    # Ollama
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.1:8b", alias="OLLAMA_MODEL")
    ollama_code_model: str = Field("deepseek-coder:6.7b", alias="OLLAMA_CODE_MODEL")

    # Review
    max_files_per_pr: int = Field(20, alias="MAX_FILES_PER_PR")
    token_budget_per_review: int = Field(4000, alias="TOKEN_BUDGET_PER_REVIEW")

    # Cache
    cache_similarity_threshold: float = Field(0.85, alias="CACHE_SIMILARITY_THRESHOLD")
    cache_max_size: int = Field(500, alias="CACHE_MAX_SIZE")

    # Eval gate (used in CI)
    min_eval_pass_rate: float = Field(0.80, alias="MIN_EVAL_PASS_RATE")

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PROD

    @property
    def debug_mode(self) -> bool:
        return self.environment == Environment.DEV

    model_config = {"env_file": ".env", "case_sensitive": False}


# Singleton — import this everywhere
settings = Settings()