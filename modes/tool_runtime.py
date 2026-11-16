"""Typed ordinary-chat tools and bounded observation-driven execution."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time

from jsonschema import Draft202012Validator

from .tasks import check_cancelled


def parameters(**fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def text(limit=1000):
    return {"type": "string", "maxLength": limit}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    permission: str
    handler: object


@dataclass
class Result:
    tool: str
    args: dict
    status: str = "available"
    data: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    events: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)

    def observation(self):
        args = {k: v[:400] + " [excerpt]" if k in {"code", "content"} and isinstance(v, str) and len(v) > 400 else v
                for k, v in self.args.items()}
        return {"tool": self.tool, "args": args, "status": self.status, "data": self.data,
                "events": self.events[-6:], "artifacts": self.artifacts,
                "source_count": len(self.sources), "sources_omitted": max(0, len(self.sources) - 4),
                "sources": [{"title": s.get("title", ""), "url": s.get("source_url", ""),
                    "evidence_id": s.get("evidence_id", ""), "location": s.get("location", ""),
                    "material_level": s.get("material_level", ""), "text": s.get("text", "")[:1800],
                    "preview_truncated": len(s.get("text", "")) > 1800,
                    **{key: s[key] for key in ("population", "device_type", "retrieval_support_status",
                        "support_quotes", "retrieval_assessment", "source_quality") if key in s}}
                    for s in self.sources[:4]]}


@dataclass
class Context:
    backend: object
    state: dict
    log: object
    emit: object
    file_sources: object
    runtime: dict
    root: object = None
    deadline: float | None = None


def _clock(ctx, args):
    from .chat_tools import get_current_time, DEFAULT_TIMEZONE
    value = get_current_time.invoke({"timezone": args["timezone"] or ctx.state.get("timezone", DEFAULT_TIMEZONE)})
    return Result("clock", args, data={"requested_clock": value},
                  events=[{"provider": "clock", "status": "available", **value}])


def _calculator(ctx, args):
    from .daily_tools import calculate
    ctx.emit("status", "正在计算…")
    # Preserve the arithmetic meaning of common pasted/display operators. This
    # avoids a whole model-repair round for e.g. 238×0.85 while retaining the
    # existing numeric AST evaluator; no names, functions or code are admitted.
    expression = args["expression"].translate(str.maketrans(
        "×÷−＋－＊／（）．０１２３４５６７８９", "*/-+-*/().0123456789")).strip()
    value = calculate.invoke({"expression": expression})
    if expression != args["expression"]:
        value["original_expression"] = args["expression"]
    return Result("calculator", args, data={"calculation": value},
                  events=[{"provider": "calculator", "status": "available", **value}])


def _knowledge(ctx, args):
    scoped=getattr(ctx.backend,'sources_for_request',None)
    found, notices = (scoped(args['query'],ctx.runtime['question'],'local',ctx.log,ctx.emit) if scoped
                      else ctx.backend.sources(args["query"], "local", ctx.log, ctx.emit))
    return Result("knowledge_base", args, sources=found, events=notices)


def _files(ctx, args):
    found, notices = ctx.file_sources(ctx.state, args["query"])
    return Result("files", args, sources=found, events=notices)


def _web(ctx, args):
    options = {"freshness": args["freshness"]} if args["freshness"] else {}
    scoped=getattr(ctx.backend,'web_sources_for_request',None)
    if scoped:
        found,notices=scoped(args['query'],ctx.runtime['question'],ctx.log,ctx.emit,
                            remaining_seconds=max(1,ctx.deadline-time.monotonic()),**options)
    else:
        found, notices = ctx.backend.web_sources(args["query"], ctx.log, ctx.emit, **options)
    return Result("web_search", args, sources=found, events=notices)


def _daily(ctx, args, name):
    from .language import select_language
    defaults = {"query": "", "location": "", "location_text": "", "forecast_days": 7, "urls": []}
    plan = {**defaults, **args, "tools": [name]}
    found, notices = ctx.backend.daily_sources(plan, ctx.log, ctx.emit,
        language=select_language(ctx.runtime["question"]))
    return Result(name, args, sources=found, events=notices)


def registry():
    tools = [
        Tool("clock", "Read the real current time in an IANA timezone; no network.",
             parameters(timezone=text(80)), "always", _clock),
        Tool("calculator", "Calculate a numeric expression exactly; no names or remembered live rates.",
             parameters(expression=text(500)), "always", _calculator),
        Tool("knowledge_base", "Search the installed MDRPI assessment/prevention/nursing knowledge base.",
             parameters(query=text(12000)), "knowledge", _knowledge),
        Tool("files", "Find relevant excerpts in this conversation's uploaded documents.",
             parameters(query=text()), "files", _files),
        Tool("web_search", "Search current/public external information and read matching pages. freshness: d/w/m/y or empty.",
             parameters(query=text(), freshness={"type": "string", "enum": ["", "d", "w", "m", "y"]}), "network", _web),
        Tool("weather", "Get current weather and forecasts. location_text must quote the user's place; empty location asks for it.",
             parameters(location=text(160), location_text=text(160),
                        forecast_days={"type": "integer", "minimum": 1, "maximum": 16}),
             "network", lambda ctx, args: _daily(ctx, args, "weather")),
        Tool("read_url", "Read supplied/discovered public HTTPS URLs; do not invent addresses.",
             parameters(urls={"type": "array", "items": text(2000), "maxItems": 3}, query=text()),
             "network", lambda ctx, args: _daily(ctx, args, "read_url")),
    ]
    from .workspace import extra_tools
    tools.extend(extra_tools())
    from .preferences import extra_tools as preference_tools
    tools.extend(preference_tools())
    return {t.name: t for t in tools}


def permitted(policy, attachments, memory_profile=""):
    permissions = {"always"}
    if policy in {"auto", "local", "web"}:
        permissions.add("knowledge")
    if policy in {"auto", "web"}:
        permissions.add("network")
    if any(f.get("active", True) for f in attachments):
        permissions.add("files")
    if memory_profile:
        permissions.add("memory")
    return [name for name, tool in registry().items() if tool.permission in permissions
            and (name != "analyze_table" or any(f.get("active", True) and
                f.get("name", "").lower().endswith((".csv", ".xlsx")) for f in attachments))]


def descriptions(names):
    return [{"name": t.name, "description": t.description, "parameters": t.parameters}
            for name, t in registry().items() if name in names]


def plan_schema(names):
    choices = [{"type": "object", "properties": {"name": {"const": name}, "args": tool.parameters},
                "required": ["name", "args"], "additionalProperties": False}
               for name, tool in registry().items() if name in names]
    return parameters(reason=text(240), calls={"type": "array", "items": {"anyOf": choices}, "maxItems": 3})


def from_calls(plan):
    """Keep the existing plan fields for stored sessions and UI diagnostics."""
    result = {"location": "", "location_text": "", "forecast_days": 7, "urls": [],
              "expression": "", "freshness": "", "query": "", "timezone": "", **plan}
    if "calls" not in plan:
        return result  # Older adapters/session plans retain the flat tool contract.
    result["tools"] = list(dict.fromkeys(c["name"] for c in plan["calls"]))
    for call in reversed(plan["calls"]):
        result.update({k: v for k, v in call["args"].items() if k in result and k not in {"tools", "calls", "reason"}})
    return result


def calls_from_plan(plan, question):
    if "calls" in plan:
        return plan["calls"]
    query = plan.get("query") or question
    args = {"clock": {"timezone": plan.get("timezone", "")},
            "calculator": {"expression": plan.get("expression", "")},
            "knowledge_base": {"query": query}, "files": {"query": query},
            "web_search": {"query": query, "freshness": plan.get("freshness", "")},
            "weather": {"location": plan.get("location", ""), "location_text": plan.get("location_text", ""),
                        "forecast_days": plan.get("forecast_days", 7)},
            "read_url": {"query": query, "urls": plan.get("urls", [])}}
    return [{"name": name, "args": args.get(name, {})} for name in plan["tools"]]


def execute(ctx, call):
    check_cancelled()
    name, args = call["name"], call["args"]
    try:
        tool = registry()[name]
        Draft202012Validator(tool.parameters).validate(args)
        if name in {"create_document", "export_table", "revise_document_title", "python"}:
            from .workspace import output_request_text
            question = ctx.runtime.get("question", "")
            direct = _direct_request(question)
            no_execution = re.search(r"不要执行|不用执行|不必执行|只(?:解释|说明|写代码)|仅(?:解释|说明|写代码)|"
                                     r"\b(?:without executing|explain only|do not execute|don't execute)\b", direct, re.I)
            output_example = (name != "python" and not output_request_text(question)
                              and re.search(r"生成|创建|导出|下载|\b(?:create|generate|export|download)\b", question, re.I))
            if no_execution or output_example:
                raise ValueError("本轮只要求说明、引用或明确禁止执行，未运行文件写入工具。")
        result = tool.handler(ctx, args)
        if result.status == "available" and result.events and not result.sources and not result.data and not result.artifacts:
            result.status = result.events[-1].get("status", "no_evidence")
    except Exception as exc:
        result = Result(name, args, status="unavailable", data={"error": str(exc)[:500]},
                        events=[{"provider": name, "status": "unavailable", "error": str(exc)[:500]}])
    ctx.log.write("tool_result", result.observation())
    check_cancelled()
    return result


def _direct_request(question):
    return re.sub(r'```.*?```|`[^`]*`|“[^”]*”|‘[^’]*’|"[^"\n]*"', "", question, flags=re.S)


def finish_contract(ctx):
    """Record completion from this turn's tool observations and saved bytes."""
    from .workspace import (artifact_path, inspect_artifact, requested_file_formats,
                            requested_output_name, preserve_table_requested, output_request_text)
    from .preferences import requested_mutations, MUTATION_PATTERNS
    question = ctx.runtime.get("question", "")
    direct = _direct_request(question)
    observations = ctx.runtime.get("tool_observations", [])
    selected = set(ctx.runtime.get("selected_tools", []))
    formats = requested_file_formats(direct)
    preserve = preserve_table_requested(direct)
    file_tools = {"create_document", "export_table", "revise_document_title"}
    output_request = output_request_text(question)
    file_requested = bool(formats or selected & file_tools or any(o.get("tool") in file_tools for o in observations)
                          or re.search(r"(?:生成|创建|制作|导出|下载|写成|另存为|提供).{0,40}(?:文件|文档|报告|图片|图表)|"
                                       r"\b(?:create|generate|export|download|provide)\b.{0,60}\b(?:file|document|report|chart|image)\b", output_request, re.I))
    # A request to explain code or translate a sentence does not authorize writes.
    if re.search(r"不要执行|不用执行|仅(?:解释|说明|写代码)|只(?:解释|说明|写代码)|\b(?:without executing|explain only)\b", direct, re.I):
        file_requested = bool(any(o.get("artifacts") for o in observations))
    successful_ids = {a.get("artifact_id") for o in observations if o.get("status") == "available"
                      for a in o.get("artifacts", [])}
    verified, invalid = [], []
    for artifact in ctx.runtime.get("artifacts", []):
        identity = artifact.get("artifact_id")
        validation = artifact.get("validation", {})
        if identity not in successful_ids or validation.get("valid") is not True:
            invalid.append({"name": artifact.get("name", ""), "reason": validation.get("error", "文件工具未成功")})
            continue
        try:
            if ctx.root is None:
                raise ValueError("没有可核对的文件存储")
            saved = artifact_path(ctx.root, ctx.state["session_id"], identity)
            if hashlib.sha256(saved.read_bytes()).hexdigest() != artifact.get("sha256"):
                raise ValueError("保存的文件与生成时的校验值不同")
            actual = inspect_artifact(saved)
            if actual.get("valid") is not True or actual.get("format_verified") is not True or actual.get("format") != validation.get("format"):
                raise ValueError(actual.get("error", "实际文件格式校验失败"))
            if validation.get("preserved_original_values") is True and actual.get("matrix_sha256") != validation.get("matrix_sha256"):
                raise ValueError("完整表格矩阵校验值不同")
            verified.append({"artifact_id": identity, "name": artifact["name"], "url": artifact["url"],
                             "format": actual["format"], "validation": validation})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append({"name": artifact.get("name", ""), "reason": str(exc)[:240]})
    requested_name = requested_output_name(question)
    matching = [a for a in verified if not formats or a["format"] in formats]
    if preserve:
        matching = [a for a in matching if a["validation"].get("preserved_original_values") is True]
    if requested_name:
        requested_path = Path(requested_name.replace("\\", "/"))
        stem = requested_path.stem[:100]
        matching = [a for a in matching if Path(a["name"]).stem == stem
                    and (not requested_path.suffix or a["name"] == stem + requested_path.suffix)]
    missing = [format_ for format_ in formats if not any(a["format"] == format_ for a in matching)]
    files_completed = bool(matching) and not missing
    if files_completed:
        file_reason = "verified_current_turn_artifacts"
    elif invalid:
        file_reason = "实际输出校验失败：" + "；".join(a["reason"] for a in invalid[:3])
    elif preserve and verified:
        file_reason = "现有输出未通过完整保留原表格的校验"
    elif missing and verified:
        file_reason = "尚未生成所需格式：" + "、".join(missing)
    elif requested_name and verified:
        file_reason = "实际文件名与所要求的文件名不一致"
    else:
        file_reason = "本轮没有生成并校验通过的所需文件"
    actions = requested_mutations(question)
    mutations = [o for o in observations if o.get("tool") in MUTATION_PATTERNS]
    preference_requested = bool(actions or any(o.get("status") == "available" for o in mutations))
    writes = [o.get("data", {}).get("preference_write", {}) for o in mutations
              if o.get("status") == "available" and o.get("data", {}).get("preference_write", {}).get("write_confirmed") is True]
    profile = ctx.state.get("memory_profile", "")
    confirmed = bool(writes) and all(any(w.get("action") == action for w in writes) for action in actions)
    preference_reason = "本轮没有完成并读取确认的偏好修改"
    if confirmed:
        try:
            from .preferences import read
            data = read(ctx.root, profile)
            final = {}
            cleared = False
            for write in writes:
                if write.get("profile") != profile:
                    confirmed = False
                if write.get("action") == "forget_preference" and write.get("all"):
                    final.clear(); cleared = True
                else:
                    final[write.get("key")] = write.get("value") if write.get("action") == "remember_preference" else None
            values = {item["key"]: item["value"] for item in data["items"]}
            confirmed = confirmed and all(values.get(key) == value if value is not None else key not in values
                                           for key, value in final.items())
            if cleared:
                confirmed = confirmed and set(values) <= {key for key, value in final.items() if value is not None}
            preference_reason = "verified_current_turn_preference_writes" if confirmed else "偏好修改后的持久化读取核对未通过"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            confirmed = False
            preference_reason = "偏好持久化读取失败：" + str(exc)[:240]
    elif not profile and preference_requested:
        preference_reason = "尚未选择偏好档案，无法保存跨会话偏好；请在设置中填写偏好档案名称"
    need_prose = bool(re.search(r"解释|说明|分析|总结|概括|比较|对比|为什么|如何|怎么|建议|评估|计算|统计|平均|均值|合计|总计|求和|"
                               r"\b(?:explain|why|how|summari[sz]e|compare|analy[sz]e|calculate|statistics|mean|average|total|sum)\b", direct, re.I))
    contract = {"schema_version": "completion_contract_v1", "requires_guard": file_requested or preference_requested,
                "need_prose": need_prose,
                "files": {"requested": file_requested, "requested_format": formats[-1] if formats else "",
                          "requested_formats": formats, "requested_name": requested_name, "preserve_original": preserve,
                          "completed": files_completed, "artifacts": matching, "verified_artifacts": verified,
                          "invalid_artifacts": invalid, "missing_formats": missing, "reason": file_reason},
                "preferences": {"requested": preference_requested, "actions": actions, "completed": confirmed,
                                "profile": profile, "writes": writes, "reason": preference_reason}}
    contract["completed"] = ((not file_requested or files_completed) and (not preference_requested or confirmed))
    ctx.runtime["completion_contract"] = contract
    ctx.log.write("completion_contract", contract)
    return contract


def _without_completion_prose(answer, runtime):
    """Keep explanation while receipts alone report file/memory completion."""
    from .workspace import render_artifact_links
    contract = runtime.get("completion_contract", {})
    verified = contract.get("files", {}).get("verified_artifacts", [])
    answer = render_artifact_links(answer, runtime.get("previous_generated_files", []) + verified)
    # Remove a completion aside as a unit before splitting inline code. Otherwise
    # '- Total: 60 (created `plot.png` and `data.xlsx`)' loses the number and leaves
    # dangling conjunctions after its filename fragments are individually removed.
    code_pattern = r"(```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`+[^`\n]*`+)"
    code_ranges = [match.span() for match in re.finditer(code_pattern, answer)]
    answer = re.sub(r"[（(][^()（）\n]{0,300}(?:已生成|已创建|已导出|已保存|\b(?:created|generated|exported|saved)\b)"
                    r"[^()（）\n]{0,300}[）)]",
                    lambda match: match[0] if any(start <= match.start() < end for start, end in code_ranges) else "",
                    answer, flags=re.I)
    answer = re.sub(r"\[([^\]\n]*)\]\(([^)\s]+)\)",
                    lambda match: "" if re.search(r"下载|download", match[1], re.I)
                    or re.search(r"\.(?:docx|xlsx|csv|pdf|pptx?|txt|png)(?:[?#]|$)", match[2], re.I)
                    else match[0], answer)
    parts = re.split(code_pattern, answer)
    for index in range(1, len(parts), 2):
        if not parts[index].startswith(("```", "~~~")) and re.search(r"\.(?:docx|xlsx|csv|pdf|pptx?|txt|png)\b|sandbox:|/mnt/data/|/artifacts/", parts[index], re.I):
            parts[index] = ""
    file_claim = re.compile(r"(?:已|成功|完成|为你|为您|给你|给您|准备好)"
                            r"[^\n。！？!?]{0,35}(?:创建|生成|制作|导出|保存)|"
                            r"(?:文件名|下载链接|下载地址|文件格式)\s*[：:]|"
                            r"(?:文件|文档|表格|Word|Excel|DOCX|XLSX|CSV)[^\n。！？!?]{0,30}"
                            r"(?:已(?:经)?(?:创建|生成|制作|导出|保存|完成|就绪)|创建完成|生成完成|保存成功|可下载|准备好)|"
                            r"\b(?:I (?:have |successfully )?(?:created|generated|exported|saved|prepared)|"
                            r"(?:file|document|report|download) (?:is|has been|was) (?:ready|created|generated|saved)|"
                            r"(?:created|generated|exported|saved) (?:the |a |your )?(?:file|document|report)|"
                            r"download (?:link|here)|here (?:is|are) (?:the |your )?(?:file|document|report))\b", re.I)
    memory_claim = re.compile(r"(?:已|成功|我会|我将|以后会|今后会|从现在起)[^\n。！？!?]{0,24}(?:记住|记下|偏好|保存在|记忆)|"
                              r"\b(?:I (?:have |will |'ll )?(?:saved|remembered|remember|save) (?:your|the) preference|"
                              r"saved to (?:your )?memory|preference (?:was|is|has been) saved)\b", re.I)
    for index in range(0, len(parts), 2):
        kept = []
        for clause in re.split(r"(?<=[，,。！？!?；;\n])|(?<=\.)\s+(?=[A-Z])", parts[index]):
            if (file_claim.search(clause) or memory_claim.search(clause)
                    or re.search(r"sandbox:|file:/|/mnt/data/|/artifacts/", clause, re.I)
                    or re.fullmatch(r"\s*(?:[-*#>]+\s*)?(?:下载|生成文件|附件|download|generated files)[：:\s]*", clause, re.I)):
                continue
            kept.append(clause)
        parts[index] = "".join(line for line in "".join(kept).splitlines(keepends=True)
            if not re.fullmatch(r"\s*(?:[-*#]+\s*)?(?:\*\*)?[^：:\n]{0,30}"
                                r"(?:下载链接|下载文件|文件下载|图表|柱状图|折线图|饼图)"
                                r"(?:\*\*)?[：:]?(?:\*\*)?\s*", line.rstrip("\r\n"), re.I))
    clean = "".join(parts).strip()
    return re.sub(r"\n{3,}", "\n\n", clean)


def table_summary(runtime):
    """Render measured simple table aggregates without another generative pass."""
    observations = runtime.get("tool_observations", [])
    if not observations or any(o.get("tool") != "analyze_table" or o.get("status") != "available" for o in observations):
        return ""
    english = not bool(re.search(r"[\u3400-\u9fff]", runtime.get("question", "")))
    labels = {"sum": "Total" if english else "合计", "mean": "Mean" if english else "平均数"}
    rows, coverage = [], []
    for observation in observations:
        data = observation.get("data", {}).get("workspace_result", {})
        operation = data.get("operation")
        if operation not in labels or not isinstance(data.get("truncated"), bool):
            return ""
        coverage.append((data.get("rows_loaded"), data["truncated"]))
        for item in data.get("result", []):
            value = item.get(operation)
            if type(value) not in {int, float} or not isinstance(item.get("index"), str):
                return ""
            column = item["index"].replace("|", "\\|").replace("\n", " ")
            rows.append(f"| {labels[operation]} | {column} | {value} |")
    if not rows or len(rows) > 20 or len(set(coverage)) != 1:
        return ""
    loaded, truncated = coverage[0]
    if type(loaded) is not int or loaded < 0:
        return ""
    if truncated:
        scope = (f"Statistics cover the {loaded} loaded rows only; the input exceeded the read limit."
                 if english else f"统计基于实际载入的 {loaded} 行；源表格超出读取上限，结果不代表完整表格。")
    else:
        scope = f"Statistics cover all {loaded} data rows." if english else f"统计覆盖表格全部 {loaded} 行数据。"
    header = "| Statistic | Column | Value |" if english else "| 指标 | 字段 | 结果 |"
    return scope + "\n\n" + "\n".join([header, "| --- | --- | --- |", *rows])


def guard_response(answer, runtime):
    """Keep ordinary explanation, with deterministic file/memory receipts."""
    from .workspace import render_artifact_links
    contract = runtime.get("completion_contract", {})
    files, preferences = contract.get("files", {}), contract.get("preferences", {})
    english = not bool(re.search(r"[\u3400-\u9fff]", runtime.get("question", "")))
    paragraphs = []
    if files.get("requested"):
        if files.get("completed"):
            links = " · ".join(f"[{a['name']}]({a['url']})" for a in files["artifacts"])
            paragraphs.append(("Created and checked this turn's files: " if english else "本轮已生成并校验文件：") + links)
            if files.get("preserve_original"):
                measured = files["artifacts"][0]["validation"]
                paragraphs.append((f"Preserved all {measured.get('original_rows', 0)} original rows and their column order as cell text values."
                                   if english else f"已保留全部 {measured.get('original_rows', 0)} 行原始单元格文字值及行列顺序。"))
        else:
            paragraphs.append(("The requested file task was not completed. " if english else "本次文件任务未完成。") + files.get("reason", ""))
            if files.get("verified_artifacts"):
                paragraphs.append(("Other checked outputs from this turn: " if english else "本轮其他已通过校验的输出：")
                                  + " · ".join(f"[{a['name']}]({a['url']})" for a in files["verified_artifacts"]))
    if preferences.get("requested"):
        if preferences.get("completed"):
            receipts = []
            for write in preferences["writes"]:
                if write["action"] == "remember_preference":
                    receipts.append(write["key"] + "：" + write["value"])
                else:
                    receipts.append(("cleared all preferences" if english else "已清空全部偏好") if write.get("all")
                                    else ("deleted " if english else "已删除 ") + write["key"])
            paragraphs.append(("Saved and read back preferences in profile " if english else "已持久化并读取核对偏好档案 ")
                              + preferences["profile"] + "：" + "；".join(receipts))
        else:
            paragraphs.append(("The preference change was not completed. " if english else "本次偏好修改未完成。") + preferences.get("reason", ""))
    if paragraphs:
        runtime["response_guard"] = {"applied": True, "reason": "completion_contract_receipt", "completed": contract.get("completed", False)}
        prose = _without_completion_prose(answer, runtime)
        if prose:
            paragraphs.append(prose)
        return "\n\n".join(paragraphs)
    verified = files.get("verified_artifacts", [])
    clean = render_artifact_links(answer, runtime.get("previous_generated_files", []) + verified)
    # Catch unsolicited completion claims even if the planner omitted the task.
    fake_file = re.search(r"(?:已|成功).{0,16}(?:生成|创建|导出|保存).{0,20}(?:文件|文档|报告)|"
                          r"\b(?:I (?:have )?(?:created|generated|exported)|file (?:is|has been) ready)\b|sandbox:/|/mnt/data/", answer, re.I)
    fake_memory = re.search(r"(?:已|成功).{0,12}(?:记住|记下|保存.{0,8}偏好)|"
                            r"\b(?:I (?:have )?(?:saved|remembered) (?:your|the) preference|saved to (?:your )?memory)\b", answer, re.I)
    if (fake_file and not verified) or (fake_memory and not preferences.get("completed")):
        runtime["response_guard"] = {"applied": True, "reason": "unsupported_completion_claim", "completed": False}
        notice = ("This turn has no verified file creation or persisted preference change."
                  if english else "本轮未确认生成可下载文件或完成偏好持久化修改。")
        prose = _without_completion_prose(answer, runtime)
        return notice + ("\n\n" + prose if prose else "")
    return clean


def call_identity(call):
    """Ignore cosmetic query changes; preserve case-sensitive URLs/code/file data."""
    value = {**call, "args": dict(call.get("args", {}))}
    if call.get("name") in {"web_search", "knowledge_base", "files"}:
        query = value["args"].get("query")
        if isinstance(query, str):
            value["args"]["query"] = " ".join(query.casefold().split())
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run(ctx, plan, *, rounds=3, max_calls=8, seconds=180):
    from .chat_tools import normalize_plan, ground_weather_location, ground_search_freshness, require_explicit_external_tools
    started, seen, observations, sources, events, artifacts = time.monotonic(), set(), [], [], [], []
    ctx.deadline = started + seconds
    question, policy = ctx.runtime["question"], ctx.state["policy"]
    stop, completed_rounds = "complete", 0
    for round_no in range(rounds):
        check_cancelled()
        plan = ground_weather_location(normalize_plan(plan, policy, ctx.state.get("attachments", []),
            ctx.state.get("memory_profile", "")), ctx.state["messages"])
        plan = ground_search_freshness(plan, question)
        # Continuations may select new calls, but cannot undo a direct per-turn
        # network restriction. Do not re-add already satisfied required tools.
        restricted = require_explicit_external_tools(plan, question, add_required=False)
        if restricted.get("per_request_blocked_tools"):
            blocked = set(restricted["per_request_blocked_tools"])
            plan = {**plan, "tools": [name for name in plan["tools"] if name not in blocked]}
            if "calls" in plan:
                plan["calls"] = [call for call in plan["calls"] if call["name"] not in blocked]
        calls = calls_from_plan(plan, question)
        if not calls:
            break
        completed_rounds = round_no + 1
        ctx.log.write("tool_round", {"round": completed_rounds, "plan": plan})
        results = []
        for call in calls:
            if len(seen) >= max_calls or time.monotonic() - started >= seconds:
                stop = "call_limit" if len(seen) >= max_calls else "time_limit"
                break
            key = call_identity(call)
            if key in seen:
                ctx.log.write("tool_duplicate", call)
                continue
            seen.add(key)
            result = execute(ctx, call)
            results.append(result)
            sources.extend(result.sources); events.extend(result.events); artifacts.extend(result.artifacts)
            ctx.runtime["artifacts"] = artifacts
            ctx.runtime.update(result.data)
            observations.append(result.observation())
            ctx.runtime["tool_observations"] = observations
            if "preferences" in result.data:
                ctx.state["preferences"] = result.data["preferences"]
        # Preserve the existing weather-service fallback without needing another model.
        if (any(r.tool == "weather" and r.status == "unavailable" for r in results)
                and "web_search" not in plan["tools"] and policy in {"auto", "web"} and len(seen) < max_calls
                and time.monotonic() < ctx.deadline):
            fallback = {"name": "web_search", "args": {"query": plan.get("query") or question, "freshness": ""}}
            key = call_identity(fallback)
            if key not in seen:
                seen.add(key); result = execute(ctx, fallback)
                result.events = [{**e, "fallback_for": "weather"} for e in result.events]
                sources.extend(result.sources); events.extend(result.events); observations.append(result.observation())
                results.append(result)
                ctx.runtime["tool_observations"] = observations
                ctx.runtime["fallback_tools"] = ["web_search"]
        if stop != "complete":
            break
        if not results:
            stop = "duplicate_calls"
            break
        if (plan.get("terminal_tools") and plan.get("planner") in {"literal_document", "explicit_table", "explicit_title_revision"}
                and all(r.status == "available" and r.artifacts for r in results)):
            ctx.log.write("tool_terminal_task", {"planner": plan["planner"], "reason": "All explicitly requested deterministic file operations succeeded."})
            break
        dynamic = {r.tool for r in results} - {"clock", "calculator", "weather"}
        next_plan = getattr(ctx.backend, "continue_chat", None)
        # A successful standalone calculation needs no extra model call. But a
        # calculation followed by a requested chart/file is not a finished task;
        # invalid arithmetic arguments also need a chance to be corrected.
        repairable = any(r.status == "unavailable" and r.tool in {"calculator", "clock"} for r in results)
        contract = finish_contract(ctx) if next_plan is not None else {}
        pending = contract.get("requires_guard") and not contract.get("completed")
        if (not dynamic and not repairable and not pending) or next_plan is None:
            break
        if round_no + 1 >= rounds:
            stop = "round_limit"
            break
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            stop = "time_limit"
            break
        try:
            plan = next_plan({**ctx.state, "chat_task_progress": ctx.runtime.get("completion_contract", {})},
                             observations, ctx.log, ctx.emit, remaining_seconds=remaining)
        except Exception as exc:
            ctx.log.write("tool_continuation_error", {"error": str(exc)[:500]})
            stop = "continuation_unavailable"
            break
    from app.evidence_constraints import deduplicate_documents
    sources=deduplicate_documents(sources,retain_local_passages=True)
    selected, identities = [], set()
    for source in sources:
        identity = (source.get("evidence_id"), source.get("source_url"), source.get("text"))
        if identity not in identities:
            identities.add(identity); selected.append(source)
    ctx.runtime.update(tool_observations=observations, tool_outcomes=events,
        tool_loop={"rounds": completed_rounds, "calls": len(seen), "stop_reason": stop,
                   "seconds": round(time.monotonic() - started, 3)})
    ctx.log.write("tool_loop_complete", ctx.runtime["tool_loop"])
    finish_contract(ctx)
    return selected, events, artifacts
