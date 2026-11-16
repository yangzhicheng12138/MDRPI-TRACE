"""Small daily tools using Open-Meteo, DDGS and the existing public page reader.

General chat accepts explicitly labelled search snippets; the clinical research
evidence pipeline is deliberately unchanged. See vendor/README.md for provenance.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import operator
import re
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from langchain_core.tools import tool

from app.evidence_snapshot import RunBudget
from app.external_evidence import safe_get, validate_public_url
from app.web_content import access_failure
from .tasks import check_cancelled


@tool
def calculate(expression: str) -> dict:
    """Evaluate numeric arithmetic with + - * / // % ** and parentheses, not code.

    Percentages must be written as division by 100. Results are numeric, no network.
    """
    if len(expression) > 200:
        raise ValueError("Arithmetic expression is too long.")
    tree = ast.parse(expression, mode="eval")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                  ast.Mod: operator.mod, ast.Pow: operator.pow}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Decimal(ast.get_source_segment(expression, node))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is outside the calculator range.")
            value = operations[type(node.op)](left, right)
        else:
            raise ValueError("Use numeric arithmetic only; names and function calls are not supported.")
        if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Result is outside the calculator range.")
        return value
    value = visit(tree.body)
    return {"expression": expression, "result": int(value) if value == value.to_integral() else float(value)}


def source(title, url, text, material="web_page_excerpt"):
    return {"evidence_id": "CHAT_" + uuid4().hex[:12], "origin": "external",
            "source_type": "web_page", "title": title, "source_url": url,
            "source_identity": url, "text": text, "material_level": material,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "location": material}


WEATHER_CODES = {
    0: "Clear sky / 晴", 1: "Mainly clear / 大部晴朗", 2: "Partly cloudy / 局部多云",
    3: "Overcast / 阴", 45: "Fog / 雾", 48: "Depositing rime fog / 雾凇",
    51: "Light drizzle / 小毛毛雨", 53: "Moderate drizzle / 中等毛毛雨", 55: "Dense drizzle / 浓毛毛雨",
    56: "Light freezing drizzle / 轻冻毛毛雨", 57: "Dense freezing drizzle / 强冻毛毛雨",
    61: "Slight rain / 小雨", 63: "Moderate rain / 中雨", 65: "Heavy rain / 大雨",
    66: "Light freezing rain / 轻冻雨", 67: "Heavy freezing rain / 强冻雨",
    71: "Slight snow / 小雪", 73: "Moderate snow / 中雪", 75: "Heavy snow / 大雪",
    77: "Snow grains / 米雪", 80: "Slight rain showers / 小阵雨",
    81: "Moderate rain showers / 中等阵雨", 82: "Violent rain showers / 强阵雨",
    85: "Slight snow showers / 小阵雪", 86: "Heavy snow showers / 强阵雪",
    95: "Thunderstorm / 雷暴", 96: "Thunderstorm with slight hail / 雷暴伴小冰雹",
    99: "Thunderstorm with heavy hail / 雷暴伴强冰雹",
}


def weather(location, days=7, language="en", *, client=None, log=None):
    """Geocode an explicit/contextual location, then get current + daily weather."""
    if not location.strip():
        return [], [{"provider": "weather", "status": "needs_location",
                     "message": "Ask which city; a timezone is not a user location."}]
    own_client = client is None
    client = client or httpx.Client(timeout=10, trust_env=False)
    try:
        response = client.get("https://geocoding-api.open-meteo.com/v1/search",
                              params={"name": location, "count": 3, "language": language})
        response.raise_for_status()
        places = response.json().get("results", [])
        if log:
            log.write("weather_geocoding", {"query": location, "url": str(response.url), "results": places})
        if not places:
            return [], [{"provider": "weather", "status": "location_not_found", "location": location}]
        place = places[0]
        label = ", ".join(dict.fromkeys(str(place[k]) for k in ("name", "admin1", "country") if place.get(k)))
        params = {"latitude": place["latitude"], "longitude": place["longitude"],
                  "timezone": "auto", "forecast_days": max(7, min(16, days)),
                  "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
                  "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max"}
        response = client.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        body = response.json()
        if log:
            log.write("weather_forecast", {"url": str(response.url), "data": body})
        current = dict(body.get("current", {}))
        current["description"] = WEATHER_CODES.get(current.get("weather_code"), "Unknown")
        daily = body.get("daily", {})
        rows = [{key: values[i] for key, values in daily.items()} for i in range(len(daily.get("time", [])))]
        for row in rows:
            row["description"] = WEATHER_CODES.get(row.get("weather_code"), "Unknown")
        if not current.get("time") or not rows:
            raise ValueError("Weather response has no current time or daily forecast.")
        data = {"location": label, "requested_location": location,
                "timezone": body.get("timezone"), "current": current,
                "current_units": body.get("current_units", {}), "daily": rows,
                "daily_units": body.get("daily_units", {}),
                "note": "Open-Meteo model-based weather, not a station observation. Dates/times are local to this location; do not extrapolate beyond supplied dates."}
        item = source("Open-Meteo — " + label, str(response.url), json.dumps(data, ensure_ascii=False), "structured_weather")
        return [item], [{"provider": "weather", "status": "available", "location": label,
                        "timezone": data["timezone"], "current_time": current["time"],
                        "dates": [r["time"] for r in rows]}]
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return [], [{"provider": "weather", "status": "unavailable", "location": location,
                     "error": str(exc)[:400]}]
    finally:
        if own_client:
            client.close()


def excerpt(text, query="", limit=6500):
    """Rank multilingual paragraphs without requiring English word matches."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    latin = set(re.findall(r"[a-z0-9]{2,}", query.lower()))
    chinese = [s for s in re.findall(r"[\u4e00-\u9fff]+", query)]
    terms = latin | {s[i:i+2] for s in chinese for i in range(max(1, len(s)-1))}
    ranked = sorted(range(len(paragraphs)), key=lambda i: -sum(t in paragraphs[i].lower() for t in terms))
    selected, size = {}, 0
    for index in ranked:
        if size >= limit:
            break
        part = paragraphs[index][:limit-size]
        selected[index] = part
        size += len(part) + 2
    return "\n\n".join(selected[i] for i in sorted(selected))


def read_page(url, query="", title="", *, browser_reader=None, fetcher=safe_get):
    """Reuse the project's HTTPS transport and installed text/PDF/browser readers."""
    budget = RunBudget(25)
    try:
        publication_date = None
        data, final_url = fetcher(url, budget, "fetch_requests", validator=validate_public_url)
        # Observed failure: a search article redirected to a sign-in screen.
        login_paths = {"/login", "/signin", "/sign-in", "/auth/login"}
        if urlsplit(final_url).path.rstrip('/').lower() in login_paths and urlsplit(url).path != urlsplit(final_url).path:
            raise ValueError("page_redirected_to_login")
        if data.startswith(b"%PDF-"):
            import fitz
            with fitz.open(stream=data, filetype="pdf") as document:
                text = "\n".join(page.get_text() for page in document)
            material = "pdf_excerpt"
        elif re.search(br"<(?:!doctype|html|body|head)\b", data[:3000], re.I):
            from trafilatura import extract, extract_metadata
            text = extract(data, url=final_url, include_comments=False, include_tables=True) or ""
            if not text and browser_reader and b"<script" in data.lower():
                data, final_url = browser_reader(final_url, budget)
                text = extract(data, url=final_url, include_comments=False, include_tables=True) or ""
            meta = extract_metadata(data)
            title = title or (meta.title if meta else "")
            publication_date = meta.date if meta else None
            material = "web_page_excerpt"
        else:
            text, material = data.decode("utf-8", errors="replace"), "web_page_excerpt"
        text = excerpt(text, query)
        if not text:
            raise ValueError("page_body_unavailable")
        failure = access_failure(final_url, text)
        if failure:
            raise ValueError(failure)
        item = source(title or final_url, final_url, text, material)
        if publication_date:
            item["publication_date"] = publication_date
        return item, {"provider": "page", "status": "available", "url": final_url}
    except Exception as exc:
        return None, {"provider": "page", "status": "unavailable", "url": url, "error": str(exc)[:400]}


def search_web(query, log, *, backends=("brave", "duckduckgo", "google"), freshness="", browser_reader=None,
               search_client=None, page_reader=read_page, config=None, api_fetcher=None, original_question=None):
    """Configured discovery; body if available, otherwise labelled snippets."""
    from ddgs import DDGS
    from ddgs.engines import ENGINES
    events, candidates, readings = [], {}, {}
    if not query.strip():
        return [], [{"provider": "web", "status": "unavailable", "error": "empty_search_query"}]
    from app.web_search import create_web_search, search_provider
    from app.evidence_snapshot import RunBudget
    provider = search_provider(config) if config is not None else "ddgs"
    if provider == "deepseek":
        from app.deepseek_search import daily_search
        return daily_search(query, log, config, freshness=freshness, fetcher=api_fetcher,
                            browser_reader=browser_reader, original_question=original_question)
    adapter = None
    if provider != "ddgs" or config is not None and config.get("web_search", "strategy") == "parallel":
        options = {"mode": "off", "cancel_check": check_cancelled}
        if api_fetcher is not None:
            options["fetcher"] = api_fetcher
        if provider == "baidu":
            options.update(search_client=search_client, freshness=freshness)
        adapter = create_web_search(config, RunBudget(45), log, None, **options)
        backends = [provider]
    region = "cn-zh" if re.search(r"[\u4e00-\u9fff]", query) else "us-en"
    for engine in dict.fromkeys(backends):
        if adapter is None and engine not in ENGINES["text"]:
            events.append({"provider": "search_attempt", "engine": engine, "status": "unavailable", "error": "unsupported_engine"})
            continue
        parameters = {"query": query, "region": region, "max_results": 4, "backend": engine}
        if freshness:
            parameters["timelimit"] = freshness
        log.write("chat_web_request", {"provider": adapter.provider if adapter else "DDGS", "parameters": parameters})
        started = time.monotonic()
        try:
            if adapter is None:
                with (search_client or DDGS)(timeout=6) as client:
                    rows = client.text(**parameters)
            else:
                try:
                    discoveries = adapter._discover(query)
                finally:
                    events.extend({**row, "search_provider": row.get("provider", "DDGS"), "provider": "search_attempt", "engine": row["backend"]}
                                  for row in getattr(adapter, "last_search_attempts", []))
                rows = [{"href": row["url"], "title": row["title"], "body": row["snippet"],
                         "date": row.get("publication_date", ""), "search_backend": row.get("search_backend", engine)}
                        | {k: row[k] for k in ("search_backends", "search_ranks", "fusion_score") if k in row}
                        for row in discoveries]
            rows = [r for r in rows if r.get("href", "").startswith("https://")]
            actual_engine = rows[0].get("search_backend", engine) if rows else engine
            log.write("chat_web_results", {"engine": actual_engine, "configured_provider": provider,
                "search_strategy": config.get("web_search", "strategy", default="fallback") if config else "fallback", "results": rows})
            events.append({"provider": "search_attempt", "engine": actual_engine, "status": "available" if rows else "no_results",
                           "seconds": round(time.monotonic() - started, 3)})
        except Exception as exc:
            events.append({"provider": "search_attempt", "engine": engine, "status": "unavailable", "error": str(exc)[:300],
                           "seconds": round(time.monotonic() - started, 3)})
            continue
        # Network reads are independent. Only the calling thread writes TurnLog.
        def read(row):
            return row["href"], page_reader(row["href"], query, row.get("title", ""), browser_reader=browser_reader)
        pending = [r for r in rows[:2] if r['href'] not in readings]
        check_cancelled()
        with ThreadPoolExecutor(max_workers=2) as pool:
            for url, value in pool.map(read, pending):
                readings[url] = value
                events.append({**value[1], "provider": "page_attempt"})
        for row in rows[:3]:
            check_cancelled()
            item, _ = readings.get(row['href'], (None, None))
            if item is None and row.get("body"):
                item = source(row.get("title") or row["href"], row["href"],
                              "Search-result snippet only (page body not read):\n" + row["body"][:2000], "search_snippet")
            if item:
                if row.get("date") and not item.get("publication_date"):
                    item["publication_date"] = row["date"]
                item["search_backend"] = row.get("search_backend", engine)
                for key in ("search_backends", "search_ranks", "fusion_score"):
                    if key in row:
                        item[key] = row[key]
                candidates[row['href']] = item
        if any(s['material_level'] != 'search_snippet' for s in candidates.values()):
            break
        # Snippets remain usable as labelled snippets, but try the next engine
        # for accessible text if all pages were denied or redirected to login.
    sources = sorted(candidates.values(), key=lambda s: s['material_level'] == 'search_snippet')[:3]
    log.write("chat_web_material", {"sources": sources, "events": events})
    return sources, events + [{"provider": "web", "status": "available" if sources else "unavailable",
                              "query": query, "full_pages": sum(s["material_level"] != "search_snippet" for s in sources),
                              "snippets": sum(s["material_level"] == "search_snippet" for s in sources)}]
