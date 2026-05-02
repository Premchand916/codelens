"""
main.py
FastAPI application entry point for CodeLens.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from llm.client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CodeLens...")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_CODE_MODEL", "deepseek-coder:6.7b")

    app.state.ollama_client = OllamaClient(
        base_url=base_url,
        model=model,
        timeout=60.0,
    )
    app.state.config = _load_config()

    ok = await app.state.ollama_client.is_available()
    if not ok:
        logger.warning("Ollama not available at startup — /ready will report degraded")

    yield

    logger.info("Shutting down CodeLens...")
    await app.state.ollama_client.close()


def _load_config():
    from types import SimpleNamespace
    return SimpleNamespace(
        webhook_secret=os.getenv("GITHUB_WEBHOOK_SECRET", ""),
    )


app = FastAPI(title="CodeLens", version="0.1.0", lifespan=lifespan)
app.include_router(router)