"""
Pydantic models for GitHub pull request webhook payloads.
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class GitHubUser(BaseModel):
    model_config = ConfigDict(extra="allow")

    login: str = "unknown"
    id: int | None = None


class GitHubRepository(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int | None = None
    name: str | None = None
    full_name: str
    html_url: HttpUrl | None = None
    clone_url: HttpUrl | None = None
    default_branch: str | None = None


class PullRequestRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    ref: str
    sha: str


class PullRequestBranch(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str | None = None
    ref: str
    sha: str
    repo: GitHubRepository | None = None


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int | None = None
    number: int
    title: str
    body: str | None = None
    state: str = "unknown"
    html_url: HttpUrl | None = None
    diff_url: HttpUrl | None = None
    patch_url: HttpUrl | None = None
    head: PullRequestBranch | None = None
    base: PullRequestBranch | None = None
    base_ref: str | None = None
    head_ref: str | None = None
    user: GitHubUser = Field(default_factory=GitHubUser)
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0


class PRWebhookPayload(BaseModel):
    """
    The GitHub pull_request webhook body.

    GitHub sends many fields that the app does not need yet, so these models
    keep known fields typed while allowing the rest of the payload through.
    """

    model_config = ConfigDict(extra="allow")

    action: str
    number: int
    pull_request: PullRequest
    repository: GitHubRepository
    sender: GitHubUser
    installation: dict[str, Any] | None = Field(default=None)
