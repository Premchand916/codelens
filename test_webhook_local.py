# test_webhook_local.py — run this to verify your HMAC logic works
import hmac
import hashlib
import os
import requests
import json

SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "dev-secret")
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
mac = hmac.new(SECRET.encode("utf-8"), msg=body, digestmod=hashlib.sha256)
signature = f"sha256={mac.hexdigest()}"

BASE_URL = os.getenv("CODELENS_URL", "http://localhost:8000")
response = requests.post(
    f"{BASE_URL}/webhook/github",
    data=body,
    headers={
        "Content-Type": "application/json",
        "X-Hub-Signature-256": signature,
        "X-GitHub-Event": "pull_request",
    },
    timeout=10,
)
print(response.status_code, response.text)