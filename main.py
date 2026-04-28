"""
FastAPI application entry point for CodeLens.

Receives GitHub webhook events, verifies signatures, and accepts pull request
events for the review pipeline.
"""
import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from config import settings
from models.webhook import PRWebhookPayload


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CodeLens starting up...")
    logger.info("Environment: %s", settings.environment)
    logger.info("Ollama model: %s", settings.ollama_model)
    yield
    logger.info("CodeLens shutting down...")


app = FastAPI(
    title="CodeLens",
    description="AI-powered code review agent",
    version="0.1.0",
    lifespan=lifespan,
)


def verify_github_signature(
    payload_body: bytes,
    signature_header: str | None,
) -> bool:
    if not signature_header:
        logger.warning("Missing X-Hub-Signature-256 header")
        return False

    if not signature_header.startswith("sha256="):
        logger.warning("Signature header malformed")
        return False

    expected_signature = signature_header.removeprefix("sha256=")
    computed_signature = hmac.new(
        key=settings.github_webhook_secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed_signature, expected_signature)


@app.get("/")
async def root():
    return {"service": "codelens", "status": "ok"}


@app.get("/health")
async def health():
    return {"service": "codelens", "status": "ok"}


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
):
    payload_body = await request.body()

    if not verify_github_signature(payload_body, x_hub_signature_256):
        logger.warning("Webhook signature verification failed")
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event != "pull_request":
        logger.info("Ignoring non-PR event: %s", x_github_event)
        return JSONResponse(
            content={"status": "ignored", "event": x_github_event},
        )

    try:
        raw_payload = json.loads(payload_body)
        payload = PRWebhookPayload.model_validate(raw_payload)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON webhook body: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
    except Exception as exc:
        logger.warning("Failed to parse webhook payload: %s", exc)
        raise HTTPException(status_code=422, detail=f"Payload parse error: {exc}") from exc

    reviewable_actions = {"opened", "synchronize", "reopened", "ready_for_review"}
    if payload.action not in reviewable_actions:
        logger.info("Skipping PR action: %s", payload.action)
        return JSONResponse(
            content={"status": "skipped", "action": payload.action},
        )

    pull_request = payload.pull_request
    logger.info(
        "Accepted PR #%s [%s] | Repo: %s | +%s -%s across %s files",
        payload.number,
        payload.action,
        payload.repository.full_name,
        pull_request.additions,
        pull_request.deletions,
        pull_request.changed_files,
    )

    return JSONResponse(
        content={
            "status": "accepted",
            "pr_number": payload.number,
            "action": payload.action,
            "repo": payload.repository.full_name,
            "additions": pull_request.additions,
            "deletions": pull_request.deletions,
            "changed_files": pull_request.changed_files,
        },
    )
