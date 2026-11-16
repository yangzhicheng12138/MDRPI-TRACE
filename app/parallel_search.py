"""Concurrent, bounded discovery; body admission remains in WebSearch."""

from contextvars import copy_context
from copy import deepcopy
from queue import Empty, Queue
import re
from threading import Lock, Thread
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .evidence_snapshot import BudgetExceeded
from .external_evidence import ExternalFailure
from .web_search import WebSearch


def canonical_url(url):
    """Drop fragments and common tracking parameters, retain meaningful queries."""
    try:
        parts = urlsplit(url)
        if (parts.scheme != "https" or not parts.hostname or parts.port not in {None, 443}
                or parts.username or parts.password):
            return None
        host = parts.hostname.lower()
        if ":" in host:
            host = "[" + host + "]"
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in {"gclid", "fbclid", "msclkid"}]
        return urlunsplit(("https", host, parts.path or "/", urlencode(query), ""))
    except (ValueError, TypeError):
        return None


def fuse_results(engine_results, limit=12):
    """Equal-weight RRF of discovery ranks, not a claim-quality score."""
    merged = {}
    for backend, rows in engine_results:
        seen = set()
        for rank, row in enumerate(rows[:5], 1):
            key = canonical_url(row.get("url"))
            if key is None or key in seen:
                continue
            seen.add(key)
            if key not in merged:
                item = deepcopy(row)
                item.update(search_backend=backend, search_backends=[], search_ranks={}, fusion_score=0.0)
                merged[key] = item
            item = merged[key]
            item["search_backends"].append(backend)
            item["search_ranks"][backend] = rank
            item["fusion_score"] += 1 / (60 + rank)
            for field in ("title", "snippet", "publication_date"):
                if not item.get(field) and row.get(field):
                    item[field] = row[field]
    ordered = sorted(merged.values(), key=lambda row: -row["fusion_score"])[:limit]
    for row in ordered:
        row["fusion_score"] = round(row["fusion_score"], 8)
    return ordered


class _SearchBudget:
    """One engine gets one request and the shared discovery deadline."""
    def __init__(self, shared, deadline, cancel_check):
        self.shared, self.deadline = shared, deadline
        self.counts = shared.counts
        self.cancel_check = cancel_check
        self._lock, self._closed, self._used = Lock(), False, False

    def remaining(self):
        if self._closed:
            raise BudgetExceeded("Search engine closed")
        if self.cancel_check is not None:
            self.cancel_check()
        remaining = min(self.shared.remaining(), self.deadline - time.monotonic())
        if remaining <= 0:
            raise BudgetExceeded("Search engine deadline exceeded")
        return remaining

    def consume(self, kind):
        with self._lock:
            self.remaining()
            if kind != "search_requests" or self._used:
                raise BudgetExceeded("One discovery request per parallel engine")
            self.shared.consume(kind)
            self._used = True

    def close(self):
        with self._lock:
            self._closed = True


class _SearchLog:
    """Workers buffer safe request metadata; only the caller writes artifacts."""
    def __init__(self):
        self._lock, self._closed, self._records = Lock(), False, []

    def write(self, name, value):
        # Adapter attempt errors may contain arbitrary third-party text. The
        # parent instead emits one sanitized attempt per engine below.
        if name == "external_search_request":
            with self._lock:
                if not self._closed:
                    self._records.append((name, deepcopy(value)))

    def close(self):
        with self._lock:
            self._closed = True
            return self._records[:]


class ParallelSearch(WebSearch):
    provider = "ParallelSearch"

    def __init__(self, *args, primary_provider, backends, factories, timeout=8, backend="brave",
                 api_key_env=None, api_key_file=None, search_client=None, freshness="", **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(backends, (list, tuple)) or any(not isinstance(x, str) or not x for x in backends):
            raise ValueError("web_search.parallel_backends must be engine names")
        self.timeout = float(timeout)
        if not 0 < self.timeout <= 30:
            raise ValueError("Parallel search timeout must be within (0, 30] seconds")
        self.primary_provider, self.primary_backend = primary_provider, backend if primary_provider == "ddgs" else primary_provider
        self.backends = list(dict.fromkeys([self.primary_backend, *backends]))
        self.factories, self.api_key_env, self.api_key_file = factories, api_key_env, api_key_file
        self.search_client, self.freshness = search_client, freshness
        self.last_search_attempts = []

    def _read_candidates(self, discoveries, query, result):
        # Read different sites before revisiting one host. Repeated inaccessible
        # pages (and their redirects) otherwise consume the six-fetch budget.
        first, repeated, hosts = [], [], set()
        for row in discoveries:
            host = urlsplit(row["url"]).hostname
            (repeated if host in hosts else first).append(row)
            hosts.add(host)
        super()._read_candidates(first + repeated, query, result)

    def _adapter(self, backend, budget, log):
        primary = backend == self.primary_backend
        provider = self.primary_provider if primary else "ddgs"
        options = {"mode": "off", "fetcher": self.fetcher}
        if provider == "baidu":
            options.update(api_key_env=self.api_key_env, api_key_file=self.api_key_file, fallback_backends=[],
                           timeout=self.timeout, freshness=self.freshness)
        elif provider == "brave":
            options["api_key_env"] = self.api_key_env
        else:
            options.update(backend=backend, fallback_backends=[], timeout=self.timeout,
                           search_client=self.search_client, freshness=self.freshness)
        return self.factories[provider](budget, log, None, **options)

    @staticmethod
    def _failure(exc):
        text = str(exc)
        if re.fullmatch(r"(?:external_http_\d{3}|external_[a-z_]+(?::[A-Za-z_]+)?|invalid_baidu_[a-z_]+|baidu_api_error|ddgs_backend_unavailable:[a-z]+)", text):
            return text
        return "search_failed:" + type(exc).__name__

    def _discover(self, query):
        self._check_cancelled()
        slots = self.budget.available("search_requests")
        if not slots:
            raise BudgetExceeded("search_requests budget exhausted")
        deadline = time.monotonic() + min(self.timeout, self.budget.remaining())
        queue, workers, results = Queue(), [], {}
        self.last_search_attempts = []
        skipped = {}
        for backend in self.backends:
            child_budget, child_log = _SearchBudget(self.budget, deadline, self.cancel_check), _SearchLog()
            adapter = self._adapter(backend, child_budget, child_log)
            error = adapter._configuration_error()
            if error:
                skipped[backend] = {"provider": adapter.provider, "backend": backend, "status": "configuration_error", "error": error}
                continue
            if len(workers) >= slots:
                skipped[backend] = {"provider": adapter.provider, "backend": backend, "status": "budget_skipped"}
                continue
            workers.append((backend, adapter, child_budget, child_log))

        def work(backend, adapter):
            started = time.monotonic()
            attempt = {"provider": adapter.provider, "backend": backend}
            rows, fatal = [], None
            try:
                rows = adapter._discover(query)
                adapter.budget.remaining()
                attempt.update(status="search_completed" if rows else "no_results", result_count=len(rows))
            except BudgetExceeded:
                attempt.update(status="timeout", error="search_engine_deadline_or_budget")
            except Exception as exc:
                attempt.update(status="technical_failure", error=self._failure(exc))
            except BaseException as exc:
                fatal = exc  # Cancellation is control flow, never a search failure.
            attempt["seconds"] = round(time.monotonic() - started, 3)
            queue.put((backend, rows, attempt, fatal, time.monotonic()))

        try:
            for backend, adapter, _, _ in workers:
                self._check_cancelled()
                context = copy_context()
                Thread(target=context.run, args=(work, backend, adapter), daemon=True,
                       name="web-search-" + backend).start()
            pending = {worker[0] for worker in workers}
            while pending:
                self._check_cancelled()
                self.budget.remaining()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    backend, rows, attempt, fatal, finished = queue.get(timeout=min(.05, remaining))
                except Empty:
                    continue
                if fatal is not None:
                    raise fatal
                if finished <= deadline:
                    results[backend] = (rows, attempt)
                pending.discard(backend)
            self._check_cancelled()
        finally:
            for _, _, child_budget, _ in workers:
                child_budget.close()
            records = {backend: child_log.close() for backend, _, _, child_log in workers}

        workers_by_backend = {backend: adapter for backend, adapter, _, _ in workers}
        batches = []
        for backend in self.backends:
            for name, value in records.get(backend, []):
                self.log.write(name, value)
            if backend in skipped:
                attempt = skipped[backend]
            elif backend in results:
                rows, attempt = results[backend]
                batches.append((backend, rows))
            else:
                attempt = {"provider": workers_by_backend[backend].provider, "backend": backend,
                           "status": "timeout", "error": "search_engine_timeout", "seconds": round(self.timeout, 3)}
            self.last_search_attempts.append(attempt)
            self.log.write("external_search_attempt", attempt)
        fused = fuse_results(batches)
        self.log.write("parallel_search_fusion", {"strategy": "parallel", "query": query, "engines": self.backends,
            "attempts": self.last_search_attempts, "result_count": len(fused), "results": fused,
            "ranking": "RRF(k=60), equal engine weight; discovery rank is not evidence quality"})
        if not fused and any(a["status"] in {"technical_failure", "timeout", "configuration_error"} for a in self.last_search_attempts):
            raise ExternalFailure("parallel_search_no_usable_results")
        return fused
