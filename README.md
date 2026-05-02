# codelens
CodeLens is an AI-powered code review agent that connects to GitHub, analyzes pull request diffs, retrieves relevant coding best practices from a built-in knowledge base, and posts structured, line-level review comments — automatically.
# CodeLens — AI Code Review Agent

Automatically reviews GitHub PRs using local LLMs, hybrid RAG retrieval, and semantic caching.

## Architecture

GitHub PR → Webhook → Diff Parser → Hybrid RAG → Ollama LLM → GitHub Comments

## Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Ollama installed and running
- GitHub repo with webhook configured

---

## Quick Start

### macOS / Linux

```bash
# 1. Clone
git clone https://github.com/Premchand916/codelens
cd codelens

# 2. Create virtual env
python3 -m venv venv
source venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — add GITHUB_WEBHOOK_SECRET and GITHUB_TOKEN

# 5. Pull Ollama models
ollama pull llama3.1:8b
ollama pull deepseek-coder:6.7b

# 6. Run
python -m uvicorn main:app --reload --port 8000
```

### Windows (PowerShell)

```powershell
# 1. Clone
git clone https://github.com/Premchand916/codelens
cd codelens

# 2. Create virtual env
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install deps
pip install -r requirements.txt

# 4. Configure
copy .env.example .env
# Edit .env with your values

# 5. Pull Ollama models
ollama pull llama3.1:8b
ollama pull deepseek-coder:6.7b

# 6. Run
python -m uvicorn main:app --reload --port 8000
```

---

## Docker Compose (Recommended for Production)

```bash
# macOS / Linux / Windows (WSL2)
cp .env.example .env
# Edit .env — set OLLAMA_BASE_URL=http://ollama:11434

docker compose up --build

# Pull models into the Ollama container
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull deepseek-coder:6.7b
```

---

## GitHub Webhook Setup

1. Go to repo → Settings → Webhooks → Add webhook
2. Payload URL: `https://your-domain.com/webhook/github`
3. Content type: `application/json`
4. Secret: value from `GITHUB_WEBHOOK_SECRET` in `.env`
5. Events: **Pull requests** only

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/webhook/github` | GitHub PR events |
| POST | `/review` | Manual review trigger |
| GET | `/review/{pr_id}/stream` | SSE stream |
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness (Ollama connected?) |
| GET | `/metrics` | Cache stats, token usage |
| GET | `/eval/run` | Run eval suite |

---

## Running Tests

```bash
# macOS / Linux
pytest tests/ -v --cov=.

# Windows
python -m pytest tests/ -v --cov=.
```

## Running Eval Suite

```bash
# Dev mode (verbose terminal output)
python -m eval.evaluator --mode dev

# CI mode (outputs eval_results.json)
python -m eval.evaluator --mode ci --output eval_results.json
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_WEBHOOK_SECRET` | required | Webhook HMAC secret |
| `GITHUB_TOKEN` | required | PAT for posting comments |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Change to `http://ollama:11434` in Docker |
| `OLLAMA_MODEL` | `llama3.1:8b` | Primary model |
| `OLLAMA_CODE_MODEL` | `deepseek-coder:6.7b` | Code-specific model |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | Semantic cache hit threshold |
| `MIN_EVAL_PASS_RATE` | `0.80` | CI eval gate threshold |

---

## CI/CD

Every PR runs: **lint → tests → eval gate → Docker build check**

The eval gate blocks merge if pass rate < 80%. This prevents AI quality regressions from shipping even when unit tests pass.