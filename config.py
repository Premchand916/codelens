"""
config.py — Centralized configuration using Pydantic Settings.
Reads from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # GitHub webhook secret — you set this when registering the webhook on GitHub
    # [INTERNAL] GitHub uses this to sign every webhook payload with HMAC-SHA256.
    # We verify the signature to prove the request actually came from GitHub,
    # not a random attacker who found your endpoint URL.
    github_webhook_secret: str = Field(..., env="GITHUB_WEBHOOK_SECRET")

    # GitHub Personal Access Token — for posting review comments back to PRs
    github_token: str = Field(..., env="GITHUB_TOKEN")

    # Ollama config
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    ollama_code_model: str = Field(default="deepseek-coder:6.7b", env="OLLAMA_CODE_MODEL")

    # App config
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    environment: str = Field(default="development", env="ENVIRONMENT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere, don't re-instantiate
settings = Settings()