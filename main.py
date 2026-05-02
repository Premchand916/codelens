"""
main.py
FastAPI application entry point for CodeLens.
"""

import logging
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
    """
    Startup / shutdown lifecycle.
    # [INTERNAL] asynccontextmanager lifespan replaces @app.on_event (deprecated).
    # Code before 'yield' runs at startup. Code after runs at shutdown.
    # We initialize the Ollama client once here and store on app.state —
    # shared across all requests without creating a new client per request.
    """
    logger.info("Starting CodeLens...")
    app.state.ollama_client = OllamaClient(model="deepseek-coder:6.7b")
    app.state.config = _load_config()

    ok = await app.state.ollama_client.is_available()
    if not ok:
        logger.warning("Ollama not available at startup — /ready will report degraded")

    yield  # app runs here

    logger.info("Shutting down CodeLens...")
    await app.state.ollama_client.close()


def _load_config():
    """Minimal config — replace with Pydantic Settings in config.py."""
    import os
    from types import SimpleNamespace
    return SimpleNamespace(
        webhook_secret=os.getenv("GITHUB_WEBHOOK_SECRET", ""),
    )


app = FastAPI(title="CodeLens", version="0.1.0", lifespan=lifespan)
app.include_router(router)