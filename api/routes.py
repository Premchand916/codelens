"""
api/routes.py
FastAPI route definitions for CodeLens.
Webhook handler, manual review trigger, SSE stream, health/ready/metrics.
"""

import hashlib
import hmac
import json
import logging
import time
from typing import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.reviewer import CodeReviewer, FileReviewInput
from llm.client import OllamaClient

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store — Session 9 replaces this with Redis
# key: pr_id, value: list of progress events
_job_store: dict[str, list[str]] = {}
_review_results: dict[str, dict] = {}


# ── Webhook ──────────────────────────────────────────────────────────────────

def _verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify GitHub's HMAC-SHA256 webhook signature.

    # [INTERNAL] GitHub computes: HMAC-SHA256(secret, raw_body)
    # and sends it as X-Hub-Signature-256: sha256=<hex_digest>
    # We recompute the same HMAC and compare using hmac.compare_digest()
    # NOT == operator. Why? Timing attacks.
    # == short-circuits on first mismatch — an attacker can measure
    # response time to guess the signature byte by byte.
    # compare_digest() always takes the same time regardless of where
    # strings differ. Constant-time comparison.
    #
    # [INTERVIEW] "Why not just compare signatures with ==?"
    # → "== leaks timing information. compare_digest() is constant-time,
    #    preventing byte-by-byte brute force via response latency measurement."
    """
    expected = "sha256=" + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None),
):
    """
    Receives GitHub webhook. Validates, enqueues, responds 202 immediately.
    GitHub timeout = 10s. This handler completes in <100ms.
    """
    raw_body = await request.body()

    # Validate signature if secret configured
    webhook_secret = request.app.state.config.webhook_secret
    if webhook_secret:
        if not x_hub_signature_256:
            raise HTTPException(status_code=401, detail="Missing signature")
        if not _verify_github_signature(raw_body, x_hub_signature_256, webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")

    # Only process PR events
    if x_github_event != "pull_request":
        return Response(status_code=200, content="Event ignored")

    payload = json.loads(raw_body)
    action = payload.get("action")

    # Only review on open/reopen/sync — not on close/label/assign
    if action not in ("opened", "reopened", "synchronize"):
        return Response(status_code=200, content="Action ignored")

    pr_number = payload["pull_request"]["number"]
    repo = payload["repository"]["full_name"]
    pr_id = f"{repo}#{pr_number}"

    # Initialize job store entry
    _job_store[pr_id] = []
    _push_event(pr_id, "queued", f"PR {pr_id} queued for review")

    # Schedule background work — returns immediately
    background_tasks.add_task(
        _run_review_job,
        pr_id=pr_id,
        payload=payload,
        client=request.app.state.ollama_client,
    )

    logger.info("Webhook received for %s, review queued", pr_id)
    # 202 = accepted for async processing
    return Response(status_code=202, content=json.dumps({"pr_id": pr_id}))


# ── Background job ────────────────────────────────────────────────────────────

async def _run_review_job(pr_id: str, payload: dict, client: OllamaClient) -> None:
    """Runs after 202 is sent. GitHub connection is already closed."""
    try:
        _push_event(pr_id, "started", "Review started")

        # In production: fetch diff from GitHub API using PyGithub
        # For now: use diff from payload if present (manual trigger)
        files = _extract_files_from_payload(payload)

        reviewer = CodeReviewer(client=client)
        summary = await reviewer.review_pr(files)

        _review_results[pr_id] = {
            "files_reviewed": summary.files_reviewed,
            "files_skipped": summary.files_skipped,
            "comments": [c.model_dump() for c in summary.comments],
            "processing_time_ms": summary.processing_time_ms,
        }
        _push_event(pr_id, "completed", f"Review done: {len(summary.comments)} comments")

    except Exception as e:
        logger.error("Review job failed for %s: %s", pr_id, e)
        _push_event(pr_id, "error", str(e))


def _extract_files_from_payload(payload: dict) -> list[FileReviewInput]:
    """Extract file inputs from webhook payload. Stub — real impl uses GitHub API."""
    # Real implementation: use PyGithub to fetch PR files + diffs
    # payload["pull_request"]["url"] → GET /repos/{owner}/{repo}/pulls/{n}/files
    return []


def _push_event(pr_id: str, status: str, message: str) -> None:
    """Push a progress event to the job store."""
    event = json.dumps({"status": status, "message": message, "ts": time.time()})
    if pr_id not in _job_store:
        _job_store[pr_id] = []
    _job_store[pr_id].append(event)


# ── SSE Stream ────────────────────────────────────────────────────────────────

@router.get("/review/{pr_id}/stream")
async def stream_review(pr_id: str) -> StreamingResponse:
    """
    SSE endpoint — streams review progress to the client.

    # [SURFACE] SSE = Server-Sent Events. One-way stream: server → client.
    # [INTERNAL] SSE is HTTP/1.1 with Content-Type: text/event-stream.
    # The connection stays open. Server pushes newline-delimited events:
    #   data: {"status": "started"}\n\n
    #   data: {"status": "completed"}\n\n
    # Client (browser/curl) receives events as they arrive.
    # vs WebSocket: SSE is one-way (server→client), simpler, HTTP-native.
    # WebSocket is bidirectional — overkill for progress reporting.
    #
    # [INTERVIEW] "SSE vs WebSocket for progress streaming?"
    # → "SSE for one-way server→client push (progress, logs, notifications).
    #    WebSocket for bidirectional (chat, collaborative editing).
    #    SSE works over HTTP/1.1, auto-reconnects, simpler to implement."
    """
    return StreamingResponse(
        _event_generator(pr_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


async def _event_generator(pr_id: str) -> AsyncIterator[str]:
    """
    Async generator that yields SSE-formatted events.

    # [INTERNAL] async generator = function with 'yield' inside 'async def'.
    # Each 'yield' sends data to the client immediately.
    # FastAPI's StreamingResponse iterates this generator and flushes each chunk.
    # The \n\n at the end of each event is required by SSE spec —
    # it's the event delimiter. Single \n = field delimiter within an event.
    """
    sent_index = 0
    max_wait = 300  # 5 minute timeout

    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        events = _job_store.get(pr_id, [])

        # Yield any new events since last check
        while sent_index < len(events):
            yield f"data: {events[sent_index]}\n\n"
            sent_index += 1

        # Check if job is done
        last = events[-1] if events else None
        if last:
            last_data = json.loads(last)
            if last_data["status"] in ("completed", "error"):
                yield "data: {\"status\": \"done\"}\n\n"
                return

        # Poll every 500ms — small models are slow
        import asyncio
        await asyncio.sleep(0.5)

    yield "data: {\"status\": \"timeout\"}\n\n"


# ── Health / Ready / Metrics ──────────────────────────────────────────────────

@router.get("/")
async def root() -> dict:
    """Basic service index for humans and smoke checks."""
    return {
        "service": "CodeLens",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "docs": "/docs",
            "github_webhook": "/webhook/github",
        },
    }


@router.get("/health")
async def health() -> dict:
    """
    Liveness probe — is the process alive?
    # [INTERNAL] Liveness = "is this process running and not deadlocked?"
    # If liveness fails → Kubernetes restarts the pod.
    # Should NEVER check external dependencies (DB, Ollama) here.
    # If Ollama is down, the process is still alive — don't restart it.
    """
    return {"status": "ok", "ts": time.time()}


@router.get("/ready")
async def ready(request: Request) -> dict:
    """
    Readiness probe — is the service ready to handle traffic?
    # [INTERNAL] Readiness = "are all dependencies available?"
    # If readiness fails → load balancer stops sending traffic (but no restart).
    # Check Ollama here — if it's down, we can't review code.
    # Liveness vs Readiness: liveness = alive, readiness = useful.
    """
    client: OllamaClient = request.app.state.ollama_client
    ollama_ok = await client.is_available()
    status = "ready" if ollama_ok else "degraded"
    return {
        "status": status,
        "ollama": ollama_ok,
        "model": client.model,
    }


@router.get("/metrics")
async def metrics() -> dict:
    """Basic metrics — jobs processed, active jobs, results stored."""
    return {
        "active_jobs": len(_job_store),
        "completed_reviews": len(_review_results),
        "ts": time.time(),
    }
