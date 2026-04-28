import hashlib
import hmac
import json

import requests

from config import settings


payload = {
    "action": "opened",
    "number": 1,
    "pull_request": {
        "number": 1,
        "title": "Test PR",
        "state": "open",
        "base_ref": "main",
        "head_ref": "feature/test",
        "additions": 10,
        "deletions": 2,
        "changed_files": 3
    },
    "repository": {
        "full_name": "Premchand916/codelens",
        "clone_url": "https://github.com/Premchand916/codelens.git",
        "default_branch": "main"
    },
    "sender": {"login": "Premchand916"}
}

body = json.dumps(payload).encode("utf-8")
mac = hmac.new(
    settings.github_webhook_secret.encode("utf-8"),
    msg=body,
    digestmod=hashlib.sha256,
)
signature = f"sha256={mac.hexdigest()}"

response = requests.post(
    "http://localhost:8000/webhook/github",
    data=body,
    headers={
        "Content-Type": "application/json",
        "X-Hub-Signature-256": signature,
        "X-GitHub-Event": "pull_request",
    }
)
try:
    print(response.status_code, response.json())
except requests.JSONDecodeError:
    print(response.status_code, response.text)
