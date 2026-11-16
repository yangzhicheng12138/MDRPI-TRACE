"""Tool selection for ordinary chat, separate from the research/teaching flows."""
from datetime import datetime
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

DEFAULT_TIMEZONE = "Asia/Shanghai"
RETRIEVAL_QUERY_CHARACTERS = 12000


def prepare_retrieval_query(question, limit=RETRIEVAL_QUERY_CHARACTERS):
    """Send full supported input; preserve the current tail if a bound is needed."""
    if not isinstance(question, str):
        raise ValueError("检索问题必须是文字。")
    if limit < 128:
        raise ValueError("检索文字上限过小。")
    shortened = len(question) > limit
    if shortened:
        marker = "\n【中间背景已省略；保留开头和末尾当前问题】\n"
        head = (limit - len(marker)) // 3
        tail = limit - len(marker) - head
        query = question[:head] + marker + question[-tail:]
    else:
        query = question
    return {"query": query, "input_characters": len(question), "query_characters": len(query),
            "shortened": shortened, "strategy": "head_and_current_tail" if shortened else "full_input"}


def validate_timezone(name):
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError("时区无效，请使用 Asia/Shanghai、UTC、Europe/London 等 IANA 时区名。") from exc
    return name


@tool
def get_current_time(timezone: str = DEFAULT_TIMEZONE) -> dict:
    """Read the actual system clock in an IANA timezone. No internet is needed."""
    # Adapted from MCP servers/time TimeServer.get_current_time (license in vendor/).
    current_time = datetime.now(ZoneInfo(validate_timezone(timezone)))
    return {"timezone": timezone, "datetime": current_time.isoformat(timespec="seconds"),
            "day_of_week": current_time.strftime("%A"), "is_dst": bool(current_time.dst())}


def clock_answer(value, question):
    instant = datetime.fromisoformat(value["datetime"])
    offset = instant.strftime("%z")
    zone = value["timezone"]
    stamp = instant.strftime("%Y-%m-%d %H:%M:%S")
    if re.search(r"[\u4e00-\u9fff]", question):
        day = "一二三四五六日"[instant.weekday()]
        label = "北京时间" if zone == "Asia/Shanghai" else zone
        return f"现在是 **{stamp}，星期{day}**（{label}，UTC{offset[:3]}:{offset[3:]}）。"
    return f"It is **{stamp}, {value['day_of_week']}** ({zone}, UTC{offset[:3]}:{offset[3:]})."


def simple_clock_question(text):
    """Fast path only for a whole current-time/date question; not word matching."""
    text = re.sub(r"[\s，,。.!！?？]", "", text).lower()
    return bool(re.fullmatch(
        r"(?:你好|您好|hi|hello)?(?:请问|请告诉我|告诉我|你知道|知道)?(?:现在(?:的)?时间(?:是多少)?|现在几点(?:了)?|今天(?:是)?(?:几号|什么日期|星期几)|当前(?:的)?(?:时间|日期))(?:吗|么)?", text)
        or re.fullmatch(r"(?:hi|hello)?(?:what(?:'s|is)the(?:current)?time(?:now)?|whattimeisit(?:now)?|what(?:'s|is)(?:today'?sdate|thecurrentdate))", text))


def allowed_tools(policy, attachments, memory_profile=""):
    from .tool_runtime import permitted
    return permitted(policy, attachments, memory_profile)


def normalize_plan(plan, policy, attachments, memory_profile=""):
    available = allowed_tools(policy, attachments, memory_profile)
    requested = list(dict.fromkeys(plan["tools"]))
    selected = [name for name in requested if name in available]
    result = {"location": "", "location_text": "", "forecast_days": 7, "urls": [], "expression": "", "freshness": "",
              **plan, "tools": selected, "blocked_tools": [t for t in requested if t not in available]}
    if "calls" in plan:
        result["calls"] = [call for call in plan["calls"] if call["name"] in available]
    return result


def require_current_news_search(plan, question, timezone=DEFAULT_TIMEZONE):
    """Repair a missed current-news tool choice; permissions still apply afterward."""
    # Quoted text and code may mention news in translation/writing examples.
    text = re.sub(r'```.*?```|`[^`]*`|“[^”]*”|‘[^’]*’|"[^"\n]*"', "", question, flags=re.S)
    if re.search(r"(?:不要|不用|别|无需|禁止|不需要).{0,12}(?:联网|上网|搜索|检索)|"
                 r"\b(?:do not|don't|without|avoid)\b.{0,25}\b(?:search|browse|internet|web|online)\b", text, re.I):
        return plan
    if re.search(r"新闻(?:的)?(?:定义|概念|写作|历史|起源)|虚构|编造|杜撰|"
                 r"\b(?:fictional|imaginary|definition|history of news|news writing)\b", text, re.I):
        return plan
    news = re.search(r"新闻|资讯|时事|头条|热点|热搜|\b(?:news|headlines|current events)\b", text, re.I)
    current = re.search(r"今天|今日|最近|近期|近日|近来|这几天|本周|这周|本月|今年|最新|实时|当前|现在|刚刚|"
                        r"\b(?:today|recent|latest|current|breaking|this (?:week|month|year))\b", text, re.I)
    if not news or not current or "web_search" in plan["tools"]:
        return plan
    freshness = ""
    for pattern, window in ((r"今天|今日|刚刚|\b(?:today|breaking)\b", "d"),
                            (r"本周|这周|\bthis week\b", "w"),
                            (r"本月|\bthis month\b", "m"),
                            (r"今年|\bthis year\b", "y")):
        if re.search(pattern, text, re.I):
            freshness = window
            break
    query = question.strip()
    if freshness == "d":
        query += " " + get_current_time.invoke({"timezone": timezone})["datetime"][:10]
    result = {**plan, "tools": [*plan["tools"], "web_search"], "query": query,
              "freshness": freshness, "routing_override": "current_news_requires_search"}
    if "calls" in plan:
        result["calls"] = [*plan["calls"], {"name": "web_search", "args": {"query": query, "freshness": freshness}}]
    return result


def require_explicit_external_tools(plan, question, *, add_required=True):
    """Honor direct read/search requests and restrictions if the planner misses them.

    Ambiguous information needs still belong to the model. This only compiles
    explicit user intent into existing tools; their permissions/URL checks remain.
    """
    from .tool_runtime import calls_from_plan, from_calls
    direct = re.sub(r'```.*?```|`[^`]*`|“[^”]*”|‘[^’]*’|"[^"\n]*"', "", question, flags=re.S)
    url_pattern = r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&()*+,;=%]+"
    direct = re.sub(url_pattern, "", direct)
    negative = r"(?:不要|不用|别|无需|禁止|不需要)|\b(?:do not|don't|without|avoid)\b"
    gap = r"[^，,。.!?！？;；\n]{0,10}"
    no_network = re.search(r"(?:" + negative + r")" + gap + r"(?:联网|上网|网络|\b(?:browsing|internet|online|network)\b)", direct, re.I)
    no_search = no_network or re.search(r"(?:" + negative + r")" + gap + r"(?:搜索|检索|\bsearch\b)", direct, re.I)
    no_read = no_network or re.search(r"(?:" + negative + r")" + gap + r"(?:读取|打开|浏览|\b(?:read|open|visit)\b)", direct, re.I)
    blocked = ({"weather", "web_search", "read_url"} if no_network else
               ({"web_search"} if no_search else set()) | ({"read_url"} if no_read else set()))
    calls = calls_from_plan(plan, question)
    removed = [call["name"] for call in calls if call["name"] in blocked]
    calls = [call for call in calls if call["name"] not in blocked]
    # Questions about how to use a tool, supplied-text transformations and code
    # examples are not requests to actually contact the site/search engine.
    explanatory = re.match(r"\s*(?:(?:请|帮我|麻烦你?)\s*)?(?:翻译|改写|润色|校对|解释|说明|"
        r"(?:写|编写|生成).{0,12}(?:代码|脚本|程序)|如何|怎样|怎么|"
        r"translate\b|rewrite\b|explain\b|how\b|write\b.{0,24}(?:code|script))", direct, re.I)
    required = []
    if add_required and not explanatory:
        explicit_search = re.search(r"(?:联网|上网)(?:搜索|搜一下|查找|检索|查询)|"
            r"(?:请|帮我|麻烦你?)(?:在网上|在网络上)(?:搜|查|检索)|"
            r"\b(?:search|look up|check)\b.{0,45}\b(?:online|the web|the internet)\b|"
            r"\b(?:browse|search) the (?:web|internet)\b", direct, re.I)
        if explicit_search and not no_search and not any(c["name"] == "web_search" for c in calls):
            # Never compile a whole patient narrative into a public search. The
            # model must supply a short, de-identified query for clinical tasks.
            clinical = re.search(r"患者|患儿|病例|病历|\b(?:patient|clinical case)\b", direct, re.I)
            if not clinical:
                required.append({"name": "web_search", "args": {"query": question.strip()[:1000], "freshness": ""}})
        read_requested = re.search(r"读取|阅读|打开|浏览|查看|概括|总结|摘要|核实|验证|"
                                  r"\b(?:read|open|visit|summari[sz]e|verify|check)\b", direct, re.I)
        if read_requested and not no_read:
            urls = []
            for match in re.finditer(url_pattern, question):
                url = match[0].rstrip(".,;!?")
                while url.endswith(")") and url.count(")") > url.count("("):
                    url = url[:-1]
                if url not in urls:
                    urls.append(url)
            planned = {url for c in calls if c["name"] == "read_url" for url in c["args"].get("urls", [])}
            missing = [url for url in urls if url not in planned][:3]
            if missing:
                required.append({"name": "read_url", "args": {"urls": missing, "query": question[:1000]}})
    if not required and not removed:
        return plan
    return {**plan, **from_calls({"calls": calls + required, "reason": plan.get("reason", "")}),
            "routing_override": "explicit_external_request", "per_request_blocked_tools": removed}


def ground_search_freshness(plan, question):
    """A planner cannot invent a publication window for an undated question.

    Clinical durations, the runtime clock and dates in generated queries are
    not requests for recently published sources. Historical dates remain in
    the query rather than becoming a relative search-engine recency filter.
    """
    if "web_search" not in plan.get("tools", []):
        return plan
    text = re.sub(r'```.*?```|`[^`]*`|“[^”]*”|‘[^’]*’|"[^"\n]*"', "", question, flags=re.S)
    # Require a publication/news context so "the patient has used a mask for
    # the past 24 hours" cannot restrict an evidence search to one day.
    publication = r"新闻|资讯|头条|热点|热搜|发表|发布|报道|出版|文献|研究|指南|更新|新进展|新发现|\b(?:news|headlines|published|released|papers?|studies|guidelines?|updates?|research|publications?)\b"
    windows = set()
    patterns = {"d": r"今天|今日|当天|(?:过去|最近|近)\s*24\s*小时|\b(?:today|past 24 hours)\b",
                "w": r"本周|这周|(?:最近|过去|近)\s*(?:一周|1周|7天|七天)|\b(?:this week|past week|past 7 days)\b",
                "m": r"本月|这个月|(?:最近|过去|近)\s*(?:一个月|一月|1个月|30天)|\b(?:this month|past month|past 30 days)\b",
                "y": r"今年|(?:最近|过去|近)\s*(?:一年|1年)|\b(?:this year|past year|past 12 months)\b"}
    for clause in re.split(r"[，,。.!?！？;；\n]", text):
        for window, pattern in patterns.items():
            for match in re.finditer(pattern, clause, re.I):
                after = clause[match.end():match.end()+60]
                source = re.search(publication, after, re.I)
                patient = re.search(r"使用|佩戴|症状|疼痛|诉|留置|住院|患者|患儿|\b(?:admitted|using|treated|pain|symptoms?|patients?)\b", after, re.I)
                before = clause[max(0, match.start()-40):match.start()]
                if (source and (not patient or source.start() < patient.start())) or re.search(r"(?:发表|发布|报道|出版|\b(?:published|released|updated)\b)[^，。!?；\n]{0,25}$", before, re.I):
                    windows.add(window)
    desired = next(iter(windows)) if len(windows) == 1 else ""
    # Multi-period comparisons are expressed by their queries; applying one
    # shared window would silently remove one side of the requested comparison.
    calls = plan.get("calls", [])
    if plan.get("freshness", "") == desired and all(call.get("args", {}).get("freshness", "") == desired for call in calls if call.get("name") == "web_search"):
        return plan
    result = {**plan, "freshness": desired, "freshness_grounding": "explicit_user_publication_window" if desired else "no_explicit_publication_window"}
    if "calls" in plan:
        result["calls"] = [{**call, "args": {**call["args"], "freshness": desired}} if call["name"] == "web_search" else call for call in calls]
    return result


def ground_weather_location(plan, messages):
    """Do not turn the server's timezone or an assistant guess into user location."""
    if "weather" not in plan["tools"] or not plan["location"]:
        return plan
    quote = plan.get("location_text", "").strip()
    if quote and any(quote.casefold() in m["content"].casefold() for m in messages if m["role"] == "user"):
        return plan
    result = {**plan, "location": "", "location_text": "", "location_note": "No user-supplied place found; ask for the city."}
    if "calls" in plan:
        result["calls"] = [{**call, "args": {**call["args"], "location": "", "location_text": ""}}
                           if call["name"] == "weather" else call for call in plan["calls"]]
    return result


class PlannedWebQuery:
    """Reuse existing web search/readers with the chat planner's general query.

    No second MDRPI-specific query-planning model call in ordinary chat.
    """
    def __init__(self, query):
        self.query = query

    def call(self, prompt, payload, schema, name, **kwargs):
        if name != "web_query_plan":
            raise ValueError("Unexpected web planning stage: " + name)
        return {"search_query": self.query, "verification_focus": "Answer the user's current question using retrieved web content."}


def source_notes(events):
    notes = []
    for event in events:
        provider, status = event.get("provider", ""), event.get("status")
        if provider == "upload" and event.get("partial"):
            notes.append("文件较长，本轮使用与问题相关的选段；可继续追问具体章节。")
        if provider == "local" and status == "query_shortened":
            notes.append("病例较长，本轮检索保留开头和末尾当前问题，中间部分背景已省略；可拆分追问。")
        if provider == "local" and status == "insufficient_evidence":
            notes.append("本次知识库检索未找到足以支持当前问题的可核验证据；不能据此作确定结论。")
        elif provider == "local" and status == "invalid_model_output":
            notes.append("本次知识库候选引用未通过校验，未将该结果当作已核实资料；可稍后重试。")
        elif provider == "local" and status == "possible_conflict":
            notes.append("本次知识库资料可能存在建议差异，需分别核对适用条件；不能自动合并为统一结论。")
        elif provider == "local" and status in {"busy", "429"}:
            notes.append("知识库回答服务正在处理另一任务；本次未取得该回答，可稍后重试。")
        elif provider == "local" and status == "input_rejected":
            notes.append("本次知识库检索未完成：" + (event.get("message") or "输入未通过校验，请缩短或拆分后重试。"))
        elif provider == "performance" and (event.get("message") or status != "complete"):
            notes.append(event.get("message") or "本轮性能分析未全部完成，请查看停止原因与未解决审查项。")
        if status in {"unavailable", "index_unavailable", "model_unavailable", "technical_failure"}:
            if provider in {"web", "DDGS", "BraveSearch", "BaiduSearch", "ParallelSearch", "DeepSeekSearch"}:
                notes.append("网页搜索暂不可用，本次未取得可核实的网页资料。")
            elif provider == "local":
                notes.append("MDRPI 知识库暂时无法检索；本次回答没有获得该知识库资料。")
            elif provider == "router":
                if not event.get("fallback_tools"):
                    notes.append("本轮未能完成工具选择，回答未使用外部检索。")
            elif provider == "clock":
                notes.append("未能识别请求的时区，请使用明确的城市或 IANA 时区名。")
            elif provider == "weather" and not any(e.get("fallback_for") == "weather" and e.get("status") == "available" for e in events):
                notes.append("天气数据服务暂不可用；请以本次实际取得的网页资料为准，未取得资料时无法核实天气。")
            elif provider == "page":
                notes.append("该网页正文暂时无法读取，回答不会把未读内容当作已核实资料。")
        elif provider in {"web", "DDGS", "BraveSearch", "BaiduSearch", "ParallelSearch", "DeepSeekSearch"} and status in {"no_evidence", "no_results"}:
            notes.append("本次网页搜索没有取得可用正文，尚未核实所问的实时信息。")
        elif provider == "web" and status == "available" and event.get("full_pages") == 0 and event.get("snippets", 0):
            notes.append("本次只取得搜索结果摘要，未读到网页正文；摘要不等于已阅读全文。")
        elif provider == "local" and status == "out_of_scope":
            notes.append("该问题不属于 MDRPI 知识库范围，回答将依据其他可用资料或模型知识。")
    return list(dict.fromkeys(notes))
