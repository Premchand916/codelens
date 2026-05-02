# ui/routes.py
"""
UI routes — Jinja2 template rendering for CodeLens web interface.
"""
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from api.routes import (
    _job_store,
    _push_event,
    _run_review_job,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])

templates = Jinja2Templates(
    directory=Path(__file__).parent / "templates"
)

SAMPLE_FILES = [
    {
        "file_path": "src/auth.py",
        "diff_chunk": (
            "@@ -10,3 +10,5 @@\n"
            " def login(username, password):\n"
            "+    token = password  # storing raw password as token\n"
            "+    return token\n"
            "     return None"
        ),
        "language": "python",
    }
]


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@router.post("/review")
async def submit_review(
    request: Request,
    background_tasks: BackgroundTasks,
    repo_full_name: str = Form(...),
    pr_number: int = Form(...),
) -> RedirectResponse:
    pr_id = f"{repo_full_name.replace('/', '_')}_{pr_number}"
    logger.info("UI review request: %s", pr_id)

    _job_store[pr_id] = []
    _push_event(pr_id, "queued", f"Review queued: {pr_id}")

    background_tasks.add_task(
        _run_review_job,
        pr_id=pr_id,
        payload={"files": SAMPLE_FILES},
        client=request.app.state.ollama_client,
    )

    return RedirectResponse(url=f"/ui/review/{pr_id}", status_code=303)


@router.get("/review/{pr_id}", response_class=HTMLResponse)
async def review_viewer(request: Request, pr_id: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "pr_id": pr_id,
            "stream_url": f"/review/{pr_id}/stream",
        },
    )