"""Official standalone DeepSeek Search/Fetch APIs with bounded Baidu fallback."""
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import re
import time
from urllib.parse import urlsplit

from .evidence_snapshot import BudgetExceeded, RunBudget
from .external_evidence import ExternalFailure, validate_public_url
from .web_content import access_failure
from .web_search import BaiduSearch, WebSearch, page_excerpt

SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"


def safe_error(exc):
    value = str(exc)
    if re.fullmatch(r"(?:external_[a-z_]+(?::[A-Za-z_]+)?|external_http_\d{3}|deepseek_[a-z_]+)", value):
        return value
    return "deepseek_request_failed:" + type(exc).__name__


def result_url(value):
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
                or parts.port not in {None, 443} or any(ord(c) < 32 for c in value)
                or parts.hostname == "localhost" or parts.hostname.endswith((".local", ".internal"))):
            return None
        try:
            if not ipaddress.ip_address(parts.hostname).is_global:
                return None
        except ValueError:
            pass
        return value.split("#", 1)[0]
    except ValueError:
        return None


class DeepSeekSearch(BaiduSearch):
    provider = "DeepSeekSearch"
    max_query_chars = 128

    def __init__(self, *args, api_key_env="DEEPSEEK_API_KEY", api_key_file=None,
                 timeout=8, fallback_options=None, freshness="", **kwargs):
        super().__init__(*args, api_key_env=api_key_env, api_key_file=api_key_file,
                         timeout=timeout, fallback_backends=[], freshness=freshness, **kwargs)
        self.timeout = float(timeout)
        if not 0 < self.timeout <= 30:
            raise ValueError("DeepSeek timeout must be within (0, 30] seconds")
        self.fallback_options = fallback_options
        self._used_fallback = False
        self.last_discoveries = []

    def _configuration_error(self):
        if not self._api_key() and self.fallback_options is None:
            return "missing_search_api_key:" + self.api_key_env
        return None

    def _request(self, endpoint, parameters, kind):
        self._check_cancelled()
        key = self._api_key()
        if not key:
            raise ExternalFailure("deepseek_missing_api_key")
        self.log.write("external_search_request" if kind == "search_requests" else "external_page_request",
                       {"provider": self.provider, "endpoint": endpoint, "parameters": parameters,
                        "request_count_scope": kind, "api_key_required": True})
        # Do not copy arbitrary error messages or response headers into artifacts.
        try:
            data, _ = self.fetcher(endpoint, self.budget, kind, validator=validate_public_url,
                method="POST", body=json.dumps(parameters, ensure_ascii=False).encode(),
                timeout_seconds=min(self.timeout, self.budget.remaining()),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            self._check_cancelled()
            response = json.loads(data)
        except BudgetExceeded:
            raise
        except Exception as exc:
            raise ExternalFailure(safe_error(exc)) from None
        if not isinstance(response, dict) or "error" in response:
            raise ExternalFailure("deepseek_invalid_response")
        return response

    def _fallback_discover(self, query):
        if self._used_fallback or self.fallback_options is None:
            return []
        self._used_fallback = True
        fallback = BaiduSearch(self.budget, self.log, None, mode="off", fetcher=self.fetcher,
                               freshness=self.freshness, **self.fallback_options)
        try:
            rows = fallback._discover(query)
        except BudgetExceeded:
            raise
        except Exception as exc:
            if not fallback.last_search_attempts:
                fallback.last_search_attempts = [{"provider": fallback.provider, "backend": "baidu",
                    "status": "technical_failure", "error": safe_error(exc)}]
            rows = []
        self.last_search_attempts.extend(fallback.last_search_attempts)
        self.last_discoveries.extend(rows)
        return rows

    def _discover(self, query):
        self._check_cancelled()
        self.last_search_attempts, self.last_discoveries = [], []
        self._used_fallback = False
        query = " ".join(query.split())
        if not query:
            raise ExternalFailure("deepseek_empty_query")
        suffix = ""
        if self.freshness in {"d", "w", "m", "y"}:
            days = {"d": 1, "w": 7, "m": 31, "y": 366}[self.freshness]
            suffix = " after:" + (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        prepared = query[:self.max_query_chars - len(suffix)].rstrip() + suffix
        if prepared != query:
            self.log.write("external_query_preparation", {"provider": self.provider,
                "input_chars": len(query), "query_chars": len(prepared), "freshness": self.freshness})
        started = time.monotonic()
        attempt = {"provider": self.provider, "backend": "deepseek"}
        rows = []
        try:
            response = self._request(SEARCH_URL, {"query": prepared, "count": 5, "need_content": True},
                                     "search_requests")
            values = response.get("results")
            if not isinstance(values, list):
                raise ExternalFailure("deepseek_invalid_results")
            seen = set()
            for value in values[:5]:
                if not isinstance(value, dict):
                    raise ExternalFailure("deepseek_invalid_result")
                url = result_url(value.get("url"))
                host = urlsplit(url).hostname if url else ""
                paper = re.search(r"/paper/([a-f0-9]{32})(?:[/?]|$)", url or "", re.I)
                identity = ("cnki:" + paper[1].lower()) if paper and host.endswith((".cnki.net", ".cnki.cn")) else url
                if not url or identity in seen:
                    continue
                seen.add(identity)
                for key in ("title", "snippet", "summary", "content"):
                    if value.get(key) is not None and not isinstance(value[key], str):
                        raise ExternalFailure("deepseek_invalid_result")
                rows.append({"title": (value.get("title") or "")[:500], "url": url,
                    "snippet": (value.get("summary") or value.get("snippet") or "")[:4000],
                    "body_content": value.get("content") or "", "material_level": "search_snippet",
                    "publication_date": value.get("date") or "", "site_name": value.get("site_name"),
                    "search_backend": "deepseek", "search_backends": ["deepseek"],
                    "authority": value.get("authority"), "rerank_score": value.get("rerank_score")})
            attempt.update(status="search_completed" if rows else "no_results", result_count=len(rows))
        except BudgetExceeded:
            raise
        except Exception as exc:
            attempt.update(status="technical_failure", error=safe_error(exc))
        attempt["seconds"] = round(time.monotonic() - started, 3)
        self.last_search_attempts.append(attempt)
        self.log.write("external_search_attempt", attempt)
        self.last_discoveries = rows[:]
        return rows or self._fallback_discover(prepared)

    def _admit_content(self, text, metadata, query, result, method):
        url = metadata["url"]
        candidate, status = page_excerpt(text.encode(), metadata, url,
            query+'\n'+getattr(self,'request_question',query), body_text=text)
        event = {"source": url, "final_url": url, "status": status, "reader": method}
        result["events"].append(event)
        self.log.write("external_admission", {**event,
            "decision": "candidate_for_reflection" if candidate else "reject",
            "evidence_id": candidate["evidence_id"] if candidate else None})
        if candidate and candidate["source_identity"] not in {s["source_identity"] for s in result["evidence"]}:
            backend=metadata.get("search_backend") or "deepseek"
            candidate.update(retrieval_method="registry_record" if metadata.get("registry_verified") and metadata.get("resolved_material_level") in {"article_abstract","article_record","full_text_excerpt"} else method,
                             discovered_url=metadata.get("resolved_from") or url, search_backend=backend,
                             search_backends=list(dict.fromkeys(["deepseek",backend])), site_name=metadata.get("site_name"))
            if metadata.get("publication_date"):
                candidate.update(publication_date=metadata["publication_date"],
                                 publication_date_source="search_result")
            for key in ("doi","pmid","pmcid","article_title","authors","journal","bibliographic_year",
                        "first_publication_date","publication_types","document_kind","registry_verified",
                        "publication_date_source","resolved_from","year","source_type","volume","issue","pages"):
                if metadata.get(key) is not None: candidate[key]=metadata[key]
            if metadata.get("resolved_material_level"):
                candidate["material_level"]=metadata["resolved_material_level"]
            from .evidence_constraints import eligibility, document_keys
            decision=eligibility(candidate,getattr(self,"request_question",query))
            if not decision["eligible"]:
                self.log.write("source_candidate_rejected",{"url":url,**decision})
                return
            existing=set().union(*(document_keys(s) for s in result["evidence"])) if result["evidence"] else set()
            if document_keys(candidate) & existing:
                self.log.write("source_duplicate_skipped",{"url":url,"identifiers":sorted(document_keys(candidate))})
                return
            candidate["source_quality"]=decision
            result["evidence"].append(candidate)

    def _read_candidates(self, discoveries, query, result):
        from .source_resolution import SourceResolver
        if not hasattr(self,"source_resolver"): self.source_resolver=SourceResolver(self)
        discoveries=self.source_resolver.expand(discoveries,query,getattr(self,"request_question",query))
        primary = [row for row in discoveries if row.get("body_content") or row.get("search_backend") == "deepseek"]
        if not primary:
            return WebSearch._read_candidates(self, discoveries, query, result)
        # Search content is the actual supplied page body; summaries stay snippets.
        for row in primary:
            self._check_cancelled()
            if row.get("body_content"):
                self._admit_content(row["body_content"], row, query, result, "deepseek_search_content")
                if len(result["evidence"]) >= 2:
                    return
        # Reserve the remaining fetch budget for fallback sites when Fetch cannot help.
        pending = [row for row in primary if not row.get("body_content") and not access_failure(row["url"])]
        for row in pending[:2]:
            self._check_cancelled()
            try:
                validate_public_url(row["url"])
                response = self._request(FETCH_URL, {"url": row["url"]}, "fetch_requests")
                content = response.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ExternalFailure("deepseek_empty_fetch_content")
                metadata = {**row, "title": response.get("title") or row["title"]}
                self._admit_content(content, metadata, query, result, "deepseek_fetch")
            except BudgetExceeded:
                break
            except Exception as exc:
                event = {"source": row["url"], "status": "technical_failure",
                         "category": "deepseek_fetch", "error": safe_error(exc)}
                result["events"].append(event)
                self.log.write("external_fetch_failure", event)
            if len(result["evidence"]) >= 2:
                return
        if not result["evidence"]:
            fallback = self._fallback_discover(query)
            if fallback:
                WebSearch._read_candidates(self, fallback, query, result)


def daily_search(query, log, config, *, freshness="", fetcher=None, browser_reader=None, original_question=None):
    """Same Search/Fetch and provenance for chat; no extra planning model call."""
    from .web_search import create_web_search
    from modes.tasks import check_cancelled
    options = {"mode": "off", "freshness": freshness, "cancel_check": check_cancelled,
               "browser_reader": browser_reader}
    if fetcher is not None:
        options["fetcher"] = fetcher
    web = create_web_search(config, RunBudget(45), log, None, **options)
    web.request_question=original_question or query
    result = {"evidence": [], "events": []}
    try:
        discoveries = web._discover(query)
        log.write("chat_web_results", {"engine": "deepseek", "search_strategy": "fallback",
            "results": [{k: v for k, v in row.items() if k != "body_content"} for row in discoveries]})
        web._read_candidates(discoveries, query, result)
    except Exception as exc:
        result["events"].append({"provider": "web", "status": "unavailable", "error": safe_error(exc)})
    from .evidence_constraints import deduplicate_documents, eligibility
    sources = deduplicate_documents(result["evidence"])[:2]
    selected = {s["source_identity"] for s in sources}
    for row in web.last_discoveries:
        if len(sources) >= 3:
            break
        if row["url"] in selected or not row.get("snippet") or access_failure(row["url"]):
            continue
        text = "Search-result snippet only (page body not read):\n" + row["snippet"][:2000]
        import hashlib
        digest = hashlib.sha256(text.encode()).hexdigest()
        snippet={"evidence_id": "EXT_SNIPPET_" + hashlib.sha256((row["url"] + text).encode()).hexdigest()[:16],
            "origin": "external", "source_type": "web_page", "title": row["title"],
            "source_url": row["url"], "source_identity": row["url"], "text": text,
            "content_sha256": digest, "material_level": "search_snippet",
            "publication_date": row.get("publication_date"), "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "search_backend": row.get("search_backend"), "search_backends": row.get("search_backends", []),
            "location": "search_snippet"}
        if not eligibility(snippet,web.request_question)["eligible"]: continue
        sources.append(snippet)
        selected.add(row["url"])
    events = [{**row, "provider": "search_attempt", "search_provider": row.get("provider"),
               "engine": row["backend"]} for row in web.last_search_attempts]
    events += result["events"]
    events.append({"provider": "web", "status": "available" if sources else "unavailable",
        "query": query, "full_pages": sum(s["material_level"] != "search_snippet" for s in sources),
        "snippets": sum(s["material_level"] == "search_snippet" for s in sources)})
    log.write("chat_web_material", {"sources": sources, "events": events})
    return sources, events
