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
    pr_id = f"{repo.replace('/', '_')}_{pr_number}"

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

        files = _extract_files_from_payload(payload)
        if not files:
            _push_event(pr_id, "completed", "No reviewable files found in payload")
            return

        _push_event(pr_id, "started", f"Reviewing {len(files)} file(s)...")

        reviewer = CodeReviewer(client=client)
        summary = await reviewer.review_pr(files)

        for c in summary.comments:
            _push_comment_event(pr_id, c.model_dump())

        _review_results[pr_id] = {
            "files_reviewed": summary.files_reviewed,
            "files_skipped": summary.files_skipped,
            "comments": [c.model_dump() for c in summary.comments],
            "processing_time_ms": summary.processing_time_ms,
        }
        _push_event(
            pr_id, "completed",
            f"Review done: {summary.files_reviewed} files, "
            f"{len(summary.comments)} comments in {summary.processing_time_ms:.0f}ms",
        )

    except Exception as e:
        logger.error("Review job failed for %s: %s", pr_id, e)
        _push_event(pr_id, "error", str(e))


def _push_comment_event(pr_id: str, comment: dict) -> None:
    event = json.dumps({"status": "comment", "comment": comment, "ts": time.time()})
    if pr_id not in _job_store:
        _job_store[pr_id] = []
    _job_store[pr_id].append(event)


def _extract_files_from_payload(payload: dict) -> list[FileReviewInput]:
    """Extract file inputs from webhook payload or manual submission."""
    files_data = payload.get("files", [])
    if files_data:
        return [
            FileReviewInput(
                file_path=f.get("file_path", f.get("filename", "unknown")),
                diff_chunk=f.get("diff_chunk", f.get("patch", "")),
                language=f.get("language", _guess_language(f.get("file_path", ""))),
            )
            for f in files_data
            if f.get("diff_chunk") or f.get("patch")
        ]
    return []


def _guess_language(file_path: str) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
        ".cpp": "cpp", ".c": "c", ".cs": "csharp",
    }
    for ext, lang in ext_map.items():
        if file_path.endswith(ext):
            return lang
    return "unknown"


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
    """Yields SSE events. Uses 'event: comment' for HTMX sse-swap="comment"."""
    import asyncio

    sent_index = 0
    max_wait = 120

    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        events = _job_store.get(pr_id, [])

        while sent_index < len(events):
            raw = events[sent_index]
            sent_index += 1
            evt = json.loads(raw)

            if evt["status"] in ("completed", "error"):
                html = _render_status_html(evt)
                yield f"event: comment\ndata: {html}\n\n"
                return

            if evt["status"] == "comment":
                html = _render_comment_html(evt.get("comment", {}))
                yield f"event: comment\ndata: {html}\n\n"
            else:
                html = _render_status_html(evt)
                yield f"event: comment\ndata: {html}\n\n"

        await asyncio.sleep(0.5)

    yield f"event: comment\ndata: <div class='text-yellow-400 text-sm'>Timed out waiting for review.</div>\n\n"


def _render_comment_html(c: dict) -> str:
    severity = c.get("severity", "info")
    colors = {
        "critical": "red-400", "warning": "yellow-400",
        "suggestion": "blue-400", "nitpick": "slate-400",
    }
    color = colors.get(severity, "slate-400")
    file_path = c.get("file_path", "")
    line = c.get("line_number", "")
    text = c.get("comment", "")
    category = c.get("category", "")
    loc = f"{file_path}:{line}" if line else file_path

    return (
        f'<div class="comment-card bg-panel border border-border rounded-lg p-4">'
        f'<div class="flex justify-between items-center mb-2">'
        f'<span class="text-xs font-mono text-slate-500">{loc}</span>'
        f'<span class="text-xs text-{color} font-semibold uppercase">{severity}</span>'
        f'</div>'
        f'<p class="text-sm text-slate-300">{text}</p>'
        f'{"<span class=&quot;text-xs text-slate-600 mt-1 block&quot;>" + category + "</span>" if category else ""}'
        f'</div>'
    )


def _render_status_html(evt: dict) -> str:
    status = evt.get("status", "")
    msg = evt.get("message", "")
    if status == "completed":
        return f'<div class="text-green-400 text-sm font-semibold mt-4">✓ {msg}</div>'
    if status == "error":
        return f'<div class="text-red-400 text-sm font-semibold mt-4">✗ {msg}</div>'
    return f'<div class="text-slate-500 text-sm">{msg}</div>'


# ── Manual Review Trigger ────────────────────────────────────────────────────

class ManualReviewRequest(BaseModel):
    pr_id: str
    files: list[dict]


@router.post("/review/trigger")
async def trigger_review(
    request: Request,
    body: ManualReviewRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Trigger a review manually with inline diff data. No GitHub webhook needed."""
    pr_id = body.pr_id
    _job_store[pr_id] = []
    _push_event(pr_id, "queued", f"Manual review queued: {pr_id}")

    payload = {"files": body.files}
    background_tasks.add_task(
        _run_review_job,
        pr_id=pr_id,
        payload=payload,
        client=request.app.state.ollama_client,
    )
    return {"pr_id": pr_id, "status": "queued"}


# ── Health / Ready / Metrics ──────────────────────────────────────────────────

@router.get("/api")
async def api_index() -> dict:
    """API index for humans and smoke checks."""
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
