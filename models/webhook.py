"""
models/webhook.py — Pydantic models for GitHub webhook payloads.
GitHub sends JSON — we parse and validate it here before touching it.
"""
from pydantic import BaseModel
from typing import Optional


class Repository(BaseModel):
    full_name: str        # "Premchand916/codelens"
    clone_url: str
    default_branch: str


class PullRequest(BaseModel):
    number: int
    title: str
    body: Optional[str] = None
    state: str            # "open" | "closed"
    base_ref: str         # target branch e.g. "main"
    head_ref: str         # source branch e.g. "feature/add-auth"
    additions: int
    deletions: int
    changed_files: int


class Sender(BaseModel):
    login: str            # GitHub username


class PRWebhookPayload(BaseModel):
    """
    Represents the JSON body GitHub sends when a PR event occurs.
    We only model fields we actually use — Pydantic ignores the rest.
    """
    action: str           # "opened" | "synchronize" | "reopened" | "closed"
    number: int           # PR number (redundant with pull_request.number, but top-level)
    pull_request: PullRequest
    repository: Repository
    sender: Sender