"""
groq_client.py
--------------
Thin wrapper around the Groq SDK that every LLM call goes through.

- Forces JSON-mode responses and parses them.
- Retries on rate limits (free tier: 8,000 tokens/minute per model) with backoff.
- Caches responses in a local SQLite file so repeated runs (evaluation,
  knowledge-base building) do not spend API quota twice.
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIStatusError, APIConnectionError

load_dotenv(Path(__file__).parent / ".env")

FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
REASONING_MODEL = os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b")
CACHE_PATH = Path(__file__).parent / ".cache" / "llm_cache.sqlite"
CACHE_ENABLED = os.getenv("LLM_CACHE", "1") != "0"
MAX_RETRIES = 6


class LLMError(RuntimeError):
    pass


class LLMQuotaError(LLMError):
    """Daily token/request quota exhausted - waiting will not help today."""


class GroqJSONClient:
    def __init__(self, api_key: str = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=key, max_retries=0)
        self._lock = threading.Lock()
        self._db = None
        if CACHE_ENABLED:
            CACHE_PATH.parent.mkdir(exist_ok=True)
            self._db = sqlite3.connect(CACHE_PATH, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def _cache_get(self, key):
        if not self._db:
            return None
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, key, value):
        if not self._db:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()

    def chat_json(self, model: str, system: str, user: str, max_tokens: int = 2000,
                  reasoning_effort: str = "low", temperature: float = 0.0,
                  max_wait: float = 90.0) -> dict:
        """max_wait: total seconds to spend waiting on rate limits before giving up
        (keep it short for interactive requests so a fallback model can take over)."""
        payload = {"model": model, "system": system, "user": user, "max_tokens": max_tokens,
                   "reasoning_effort": reasoning_effort, "temperature": temperature}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        last_err, waited = None, 0.0
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"reasoning_effort": reasoning_effort},
                )
                result = _parse_json(resp.choices[0].message.content)
                self._cache_put(key, result)
                return result
            except RateLimitError as e:
                last_err = e
                if "per day" in str(e):
                    raise LLMQuotaError(f"Daily Groq quota exhausted for {model}") from e
                delay = _retry_after(e, attempt)
                if waited + delay > max_wait:
                    raise LLMError(f"Rate limited on {model}; would need to wait {delay:.0f}s") from e
                waited += delay
                time.sleep(delay)
            except (APIConnectionError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(2 ** attempt)
            except APIStatusError as e:
                # Groq returns 400 "json_validate_failed" when the model emits bad JSON; retry once more.
                last_err = e
                if e.status_code >= 500 or "json_validate_failed" in str(e):
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(str(e)) from e
        raise LLMError(f"LLM call failed after {MAX_RETRIES} attempts: {last_err!r}")


def _parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[start:end])


def _retry_after(err: RateLimitError, attempt: int) -> float:
    header = err.response.headers.get("retry-after") if err.response is not None else None
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass
    m = re.search(r"try again in ([\d.]+)s", str(err))
    if m:
        return float(m.group(1)) + 0.5
    return min(60, 5 * 2 ** attempt)
