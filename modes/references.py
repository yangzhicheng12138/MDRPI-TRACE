"""Deduplicated reference cards and bounded original-link checks; no LLM calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import re
import threading
import time
from urllib.parse import quote, urlsplit

from app.citation_format import (REFERENCE_STANDARD, _doi, _identities, format_reference,
                                 original_url, reference_metadata, merge_reference_sources, reference_quality, reference_type)
from app.evidence_snapshot import RunBudget
from app.external_evidence import safe_get, validate_public_url

_cache = {}
_lock = threading.Lock()


merge_sources = merge_reference_sources


def check_link(url, *, fetcher=safe_get):
    if not url:
        return {"status": "missing", "url": "", "message": "暂无法提供原文链接"}
    with _lock:
        cached = _cache.get(url)
        if cached and cached[0] > time.monotonic() and fetcher is safe_get:
            return dict(cached[1])
    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        _, final = fetcher(url, RunBudget(8), "fetch_requests", validator=validate_public_url,
                           method="HEAD", timeout_seconds=8)
        result = {"status": "verified", "url": url, "resolved_url": final,
                  "checked_at": checked_at, "message": "链接已核验；全文可用性取决于来源权限"}
    except Exception as exc:
        # Paywalls can block the publisher's HEAD response even for a registered
        # DOI. Confirm the identifier with its registry instead of inventing a URL.
        verified = False
        if urlsplit(url).hostname == "doi.org":
            try:
                from urllib.parse import unquote
                doi = unquote(urlsplit(url).path.lstrip("/"))
                data, final = fetcher("https://api.crossref.org/works/" + quote(doi, safe=""),
                    RunBudget(5), "fetch_requests", validator=validate_public_url, timeout_seconds=5)
                registered = json.loads(data).get("message", {}).get("DOI", "")
                verified = registered.casefold() == doi.casefold()
            except Exception:
                pass
        result = ({"status": "verified", "url": url, "checked_at": checked_at,
                   "verification_basis": "crossref_registration", "message": "DOI登记已核验；出版社全文可能需要访问权限"}
                  if verified else {"status": "unavailable", "url": "", "checked_at": checked_at,
                                    "message": "暂无法提供已核验的原文链接", "reason": str(exc)[:120]})
    if fetcher is safe_get:
        with _lock:
            if len(_cache) >= 512:
                _cache.pop(next(iter(_cache)))
            _cache[url] = (time.monotonic() + (86400 if result["status"] == "verified" else 60), dict(result))
    return result


def reference_cards(sources, *, verify=False, checker=None):
    checker = checker or check_link
    groups = merge_sources(sources)
    urls = [original_url(group["source"]) for group in groups]
    if verify:
        # Bound network work even when restoring an unusually large old session.
        with ThreadPoolExecutor(max_workers=4) as pool:
            links = list(pool.map(checker, urls[:12]))
        links += [{"status": "pending", "url": "", "message": "原文链接尚未核验"} for _ in urls[12:]]
    else:
        links = [{"status": "pending" if url else "missing", "url": "", "message": "原文链接尚未核验" if url else "暂无法提供原文链接"} for url in urls]
    rows = []
    for number, (group, link) in enumerate(zip(groups, links), 1):
        source = group["source"]
        quality = reference_quality(source)
        kind = reference_type(source)
        # Keep all excerpt locations for audit consumers. The research-facing
        # panel only needs explicit PDF page locations, never HTML/JATS offsets.
        pdf_locations = [] if kind == "EB" else [location for location in group["locations"]
            if re.match(r"^PDF\s*(?:第\s*\d|pages?\s+\d)", str(location), re.I)]
        resource_type = source.get("standard_category") or {"J": "期刊论文", "M": "图书章节" if source.get("container_title") else "图书",
            "S": "标准文献", "R": "报告", "EB": "网页", "Z": "海报" if source.get("source_type") == "poster" else "其他资料"}.get(kind, "")
        rows.append({"number": number, "title": source.get("title") or "题名待补充",
                     "reference": format_reference(source), "standard": REFERENCE_STANDARD,
                     "year": source.get("year", ""), "source_name": source.get("journal") or source.get("site_name") or source.get("organization") or source.get("publisher") or "",
                     "doi": _doi(source), "locations": group["locations"], "excerpt_count": len(group["positions"]),
                     "reference_type": kind, "pdf_locations": pdf_locations,
                     "standard_number": source.get("standard_number", ""),
                     "resource_type": resource_type,
                     "standard_number_note": source.get("standard_number_note", ""),
                     "publication_date": source.get("publication_date", ""), "effective_date": source.get("effective_date", ""),
                     "display_metadata_verification": source.get("display_metadata_verification"),
                     "missing_metadata": quality["missing"], "metadata_warnings": quality["warnings"],
                     "metadata_status": quality["status"], "link": link})
    return {"standard": REFERENCE_STANDARD, "references": rows, "source_count": len(sources), "document_count": len(rows)}
