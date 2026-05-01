"""
llm/client.py
Async HTTP client for Ollama local inference.
No SDK — raw httpx so you see exactly what's happening on the wire.
"""

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Wraps Ollama's /api/chat endpoint.
    
    # [SURFACE] Sends messages to a locally-running Ollama process and returns
    # the model's response as text (or streams tokens as they arrive).
    
    # [INTERNAL] Ollama exposes an HTTP API on localhost:11434.
    # Each request is: {"model": "...", "messages": [...], "stream": false}
    # The model generates tokens one-by-one (next-token prediction).
    # With stream=False, Ollama buffers all tokens and returns them at once.
    # With stream=True, it sends each token as a newline-delimited JSON chunk.
    # We use httpx.AsyncClient (not requests) because FastAPI runs on asyncio —
    # a synchronous HTTP call here would block the entire event loop.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        # httpx.AsyncClient is the async equivalent of requests.Session.
        # timeout=120 because local LLMs can be slow on first token.
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,   # 0 = deterministic, critical for code review
        max_tokens: int = 1024,
    ) -> str:
        """
        Single-shot chat completion. Returns full response as string.
        
        # [INTERNAL] temperature controls the "randomness" of token selection.
        # At each step, the model outputs a probability distribution over its
        # entire vocabulary (~32K tokens for llama3). Temperature SCALES that
        # distribution before sampling:
        #   - temp=0.0 → always pick the highest-probability token (greedy)
        #   - temp=1.0 → sample proportionally from the raw distribution
        #   - temp=2.0 → flatten distribution (more random, often incoherent)
        # For code review: temp=0.0. You want the same output for the same input.
        # For creative tasks: temp=0.7-0.9. You want variation.
        #
        # [INTERVIEW] "Why temperature=0 for code review?"
        # → "Reproducibility. If the same diff reviewed twice gives different
        #    severity ratings, your system is unreliable. temp=0 is deterministic."
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,   # Ollama's name for max_tokens
            },
        }

        logger.debug("Sending %d messages to %s", len(messages), self.model)

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()  # raises on 4xx/5xx
        except httpx.TimeoutException:
            logger.error("Ollama request timed out after %.1fs", self._client.timeout.read)
            raise
        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP error: %s — %s", e.response.status_code, e.response.text)
            raise

        data = response.json()
        # Ollama response shape: {"message": {"role": "assistant", "content": "..."}, ...}
        content = data["message"]["content"]
        logger.debug("Response tokens used: %s", data.get("eval_count", "unknown"))
        return content

    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """
        Streaming chat — yields tokens as they arrive.
        Used in Session 7 (SSE endpoint). Stubbed here for completeness.
        
        # [INTERNAL] With stream=True, Ollama sends newline-delimited JSON:
        # {"message": {"content": "def"}, "done": false}
        # {"message": {"content": " foo"}, "done": false}
        # {"message": {"content": ""}, "done": true}
        # We iterate over the response body line-by-line and yield each token.
        # This is an async generator — the 'yield' makes it one.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk["message"]["content"]
                if token:
                    yield token
                if chunk.get("done"):
                    break

    async def is_available(self) -> bool:
        """Health check — is Ollama running and the model loaded?"""
        try:
            r = await self._client.get("/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                logger.warning("Model %s not found. Available: %s", self.model, models)
            return available
        except httpx.ConnectError:
            logger.error("Ollama not running at %s", self.base_url)
            return False

    async def close(self) -> None:
        await self._client.aclose()