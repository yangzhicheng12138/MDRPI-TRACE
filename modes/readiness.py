"""Cached, read-only dependency checks for the workspace status indicator."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
import threading
import time

import httpx


def probe_dependencies(backend, transport=None):
    result = {"model": "unknown", "knowledge": "unknown"}
    if getattr(backend, "config", None) is None:
        return result
    from app.local_llm import LocalLLM
    llm = None
    try:
        llm = LocalLLM(backend.config, transport=transport)
        response = llm.client.get(llm.base_url + "/models", timeout=2)
        response.raise_for_status()
        ids = {row["id"] for row in response.json()["data"]}
        result["model"] = "ready" if llm.model in ids else "unavailable"
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        result["model"] = "unavailable"
    finally:
        if llm is not None:
            llm.close()
    try:
        url = getattr(backend, "rag_url", "")
        if url:
            with httpx.Client(timeout=2, trust_env=False, follow_redirects=False, transport=transport) as client:
                response = client.get(url.rstrip("/") + "/health")
                if response.status_code not in (200, 503):
                    response.raise_for_status()
                body = response.json()
                # The RAG health endpoint also tests the generator. Its index
                # readiness is the independent knowledge-service signal.
                index_ready = body.get("index_ready")
                result["knowledge"] = ("ready" if index_ready is True else
                                       "unavailable" if index_ready is False else "checking")
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        result["knowledge"] = "unavailable"
    return result


class ReadinessMonitor:
    """One probe at a time, shared by all browsers; never invokes generation."""
    def __init__(self, backend, *, probe=probe_dependencies, ttl=5):
        self.backend, self.probe, self.ttl = backend, probe, ttl
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workspace-readiness")
        self.lock = threading.Lock()
        self.future = None
        self.cached = None
        self.cached_at = 0.0

    def snapshot(self):
        with self.lock:
            if self.future is None and (self.cached is None or time.monotonic() - self.cached_at >= self.ttl):
                self.future = self.pool.submit(self.probe, self.backend)
            if self.future is not None:
                try:
                    self.cached = {**self.future.result(timeout=.05),
                                   "checked_at": datetime.now(timezone.utc).isoformat()}
                except TimeoutError:
                    pass
                except Exception:
                    self.cached = {"model": "unknown", "knowledge": "unknown", "checked_at": None}
                    self.future = None
                    self.cached_at = time.monotonic()
                else:
                    self.future = None
                    self.cached_at = time.monotonic()
            # Do not display a stale green indicator while a fresh probe waits.
            if self.future is not None:
                return {"model": "checking", "knowledge": "checking", "checking": True, "checked_at": None}
            return {**self.cached, "checking": False}

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
