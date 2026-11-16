"""Hide library provenance in reader prose while retaining original run records."""
from copy import deepcopy
from collections import Counter
import json
import re

from app.six_step_schema import STEPS, DISCLAIMER
from app.citation_format import numbered_references, format_reference


NUMERIC_CITATION = re.compile(r"(?<![A-Za-z0-9_])\[(\d+(?:\s*[,，]\s*\d+)*)\](?!\()")
INTERNAL_SOURCE = r"DOC\d+(?:_P\d+_C\d+(?:/C\d+)*)?|EXT_(?:WEB_[A-Za-z0-9]+|PMC\d+_[A-Za-z0-9]+)"
REFERENCE_HEADING = re.compile(r"(?m)^\s*(?:#{1,6}\s*)?(?:参考资料(?:\s*/\s*References)?|参考文献|References)\s*$")


def performance_sources(result):
    """Project the actual final citations, preserving their verification fields."""
    sources, seen = [], set()
    for source in result.get("citations", []):
        if not isinstance(source, dict):
            continue
        identity = (source.get("evidence_id"), source.get("source_url"), source.get("text"))
        if identity in seen:
            continue
        seen.add(identity)
        sources.append(deepcopy(source))
    return sources


def performance_status(result):
    """Analysis quality is separate from successful completion of an HTTP turn."""
    issues = [row for row in result.get("unresolved_issues", []) if isinstance(row, dict)]
    status = result.get("status") or "not_recorded"
    return {"status": status, "stop_reason": result.get("stop_reason") or "not_recorded",
            "is_partial": status != "complete" or bool(issues) or bool(result.get("errors")),
            "unresolved_issue_count": len(issues),
            "unresolved_issue_severities": dict(Counter(row.get("severity") or "unknown" for row in issues)),
            "error_count": len(result.get("errors", [])), "source_count": len(performance_sources(result)),
            "three_checks_completed": bool(result.get("audit_summary", {}).get("three_checks_completed")),
            "measurement": "system self-audit observations; not confirmed clinical errors"}


def performance_notice(result):
    observed = performance_status(result)
    if not observed["is_partial"]:
        return ""
    label = {"partial": "分析尚未完全通过自检", "conflict": "分析存在尚未解决的资料冲突",
             "insufficient": "资料不足，分析未完成", "technical_failure": "分析遇到技术问题，流程未完成"}.get(
                 observed["status"], "分析完整性尚未确认")
    reason = {"revision_limit": "已达到修订次数上限", "budget_exceeded": "已达到运行预算",
              "time_limit": "已达到时间上限", "source_changed_before_publication": "发布前来源校验失败"}.get(
                  observed["stop_reason"], observed["stop_reason"])
    details = []
    if reason != "not_recorded":
        details.append("停止原因：" + reason)
    if observed["unresolved_issue_count"]:
        details.append("自动自检仍有 " + str(observed["unresolved_issue_count"]) + " 项未解决提示；这不等于已确认的临床错误数量")
    return "**" + label + "。**" + (" " + "；".join(details) + "。" if details else "") + " 以下保留已生成的内容，请结合原始记录核查。"


def performance_fields(result):
    """State fields shared by the reference panel, notices and exports."""
    sources = performance_sources(result)
    events = [deepcopy(row) for row in result.get("web", {}).get("events", []) if isinstance(row, dict)]
    notice = performance_notice(result)
    if notice:
        events.append({"provider": "performance", "status": performance_status(result)["status"], "message": notice})
    return {"sources": sources, "sources_loaded": True, "source_events": events,
            "performance_status": performance_status(result)}


def prose_parts(text):
    """Do not change quoted programming examples, including open code fences."""
    return re.split(r"(```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`+[^`\n]*`+)", text)


def without_inference_labels(text):
    """Hide automatic inference badges in performance prose, including old replies."""
    parts = prose_parts(text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"(?:病例推断[（(]需临床确认[）)]|一般推断[（(]待证据核实[）)])\s*[:：]\s*",
            "", parts[i])
        parts[i] = re.sub(r"[（(]\s*(?:需临床确认|待证据核实)\s*[）)]", "", parts[i])
    return "".join(parts)


def is_knowledge_source(source):
    if source.get("source_type") == "user_upload" or source.get("origin") == "external":
        return False
    return bool(source.get("kb_version") or source.get("origin") == "local" or
                re.match(r"DOC\d+", str(source.get("doc_id") or source.get("chunk_id") or source.get("evidence_id") or "")))


def clean_prose(text, *, hidden_numbers=(), internal_ids=(), hide_all_numbers=False, labels=None):
    """Remove provenance markers, preserving measurements, conditions and code."""
    hidden = set(hidden_numbers)
    known = "|".join(re.escape(str(s)) for s in sorted(set(internal_ids), key=lambda s: len(str(s)), reverse=True) if s)
    token = r"(?:" + INTERNAL_SOURCE + ("|" + known if known else "") + r")"
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + token + r"(?![A-Za-z0-9_])")
    grouped = re.compile(r"[\[(（]\s*(?:(?:来源|依据|引用|事实|source|evidence)\s*[:：]\s*)?" +
                         token + r"(?:\s*[,，;；/、–—-]\s*" + token + r")*\s*[\])）]", re.I)
    parts = prose_parts(text)
    for i in range(0, len(parts), 2):
        part = parts[i]
        part = re.sub(r"\[[^\]\n]*\]\(/v1/sources/DOC\d+(?:#page=\d+)?\)", "", part)
        part = re.sub(r"/v1/sources/DOC\d+(?:#page=\d+)?", "", part)
        part = grouped.sub("", part)
        # A source named inside a sentence still has a grammatical role. Hide
        # parenthetical markers above, but leave readable wording in that role.
        part = pattern.sub(lambda m: "病例资料" if m[0] in internal_ids and re.fullmatch(r"F\d+", m[0]) else "相关文献", part)
        for identifier, label in (labels or {}).items():
            part = part.replace("[" + identifier + "]", label)
            part = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(identifier) + r"(?![A-Za-z0-9_])", lambda _: label, part)
        wrapped = re.compile(r"[（(]\s*(?:\[\d+(?:\s*[,，]\s*\d+)*\][\s,，;；、]*)+[)）]")
        part = wrapped.sub(lambda m: "" if all(hide_all_numbers or int(n) in hidden for n in re.findall(r"\d+", m[0])) else m[0], part)
        def replace(match):
            values = [int(n) for n in re.findall(r"\d+", match[1])]
            remaining = [n for n in values if not hide_all_numbers and n not in hidden]
            return match[0] if remaining == values else ("[" + ", ".join(map(str, remaining)) + "]" if remaining else "")
        part = NUMERIC_CITATION.sub(replace, part)
        if part != parts[i]:
            part = re.sub(r"[ \t]+(?=[。，、；;,.!?！？])", "", part)
        parts[i] = part
    return "".join(parts)


def without_bibliography(text):
    """Remove numbered reference rows, retaining subsequent warnings/content."""
    parts = prose_parts(text)
    for i in range(0, len(parts), 2):
        match = REFERENCE_HEADING.search(parts[i])
        if match:
            tail = re.sub(r"(?m)^[ \t]*(?:-\s*)?\[\d+\][^\n]*(?:\n|$)", "", parts[i][match.end():]).strip()
            parts[i] = parts[i][:match.start()].rstrip() + ("\n\n" + tail if tail else "")
    return "".join(parts).rstrip()


def saved_answer(text, sources):
    """Old replies numbered documents in citation order, not retrieval order."""
    library = [s for s in sources if is_knowledge_source(s)]
    if not library:
        return text
    hidden = set()
    parts = prose_parts(text)
    for i in range(0, len(parts), 2):
        heading = REFERENCE_HEADING.search(parts[i])
        if not heading:
            continue
        def reference_row(match):
            if any(s.get("title") and s["title"].rstrip(".。") in match[2] for s in library):
                hidden.add(int(match[1]))
                return ""
            return match[0]
        tail = re.sub(r"(?m)^[ \t]*(?:-\s*)?\[(\d+)\]([^\n]*)(?:\n|$)", reference_row, parts[i][heading.end():])
        if not re.search(r"(?m)^[ \t]*(?:-\s*)?\[\d+\]", tail):
            parts[i] = parts[i][:heading.start()].rstrip() + ("\n\n" + tail.strip() if tail.strip() else "")
        else:
            parts[i] = parts[i][:heading.end()] + tail
    return clean_prose("".join(parts).rstrip(), hidden_numbers=hidden)


def performance_claim_text(claim, step, *, include_details=True):
    """Use the answer body as prose rather than exposing its rubric categories."""
    text = without_inference_labels(claim.get("text", "")).strip()
    if step != "prioritize":
        fields = "|".join(re.escape(label) for label in STEPS[step][1].values())
        fields += r"|(?:措施|行动|复评指标)\s*\d+"
        text = re.sub(r"^\s*(?:[-*]\s+|\d+[.)、]\s*)", "", text)
        text = re.sub(r"^(?:\*\*)?(?:" + fields + r")(?:\*\*)?\s*[:：](?:\*\*)?\s*", "", text)
    details = [claim.get(key, "").strip().rstrip("。；;") for key in ("conditions", "rationale")] if include_details else []
    details = [detail for detail in details if detail and detail not in text]
    if details:
        text += "（" + "；".join(details) + "）"
    return text


def sentence(text):
    text = text.strip().strip("；; ").rstrip("。")
    return text + "。" if text else ""


def narrative_goals(text):
    # Only move an explicitly labelled goal; measurements and other brackets
    # remain exactly as written in the model answer.
    return re.sub(r"[（(](?:\*\*)?目标(?:\*\*)?\s*[:：](?:\*\*)?([^（()）]+)[）)]",
                  lambda match: "，以" + match[1].strip(), text)


def plain_measure_title(text):
    return re.sub(r"^\*\*([^*\n]+)\*\*$", r"\1", text.strip())


def measure_name(text):
    text = re.sub(r"[（(](?:\*\*)?目标(?:\*\*)?\s*[:：](?:\*\*)?[^（()）]+[）)]", "", text)
    first = re.split(r"[。；;]", text, maxsplit=1)[0].strip()
    explicit = re.match(r"^([^:：]+)[:：]", first)
    prefix = plain_measure_title(explicit[1]) if explicit else ""
    return prefix if short_measure_title(prefix) else plain_measure_title(first)


def prose_fields(text):
    """Read explicit prose labels; unlabelled sentences remain untouched."""
    text = re.sub(r"(?<=[。；;])\s*(?:\*\*)?记录(?:\*\*)?(?!单|表|\s*[:：])", "记录：", text)
    labels = r"前置条件|适用条件|条件|约束|执行角色|核查|升级条件|复评时机|结果记录|记录要求|记录"
    # Models may bold the whole field, including its value. Remove only that
    # wrapper before splitting fields, so its closing stars do not leak out.
    text = re.sub(r"\*\*((?:" + labels + r")\s*[:：][^*\n]+)\*\*", r"\1", text)
    pattern = re.compile(r"(?<!\w)(?:\*\*)?(" + labels + r")(?:\*\*)?\s*[:：](?:\*\*)?\s*")
    matches = list(pattern.finditer(text))
    if not matches:
        return text, {}
    main = text[:matches[0].start()].strip("；; ")
    fields = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        name = ("条件" if match[1] == "适用条件" else
                "记录" if match[1] in {"结果记录", "记录要求"} else match[1])
        value = text[match.end():end].strip("；; \n").rstrip("。")
        if value:
            fields.setdefault(name, []).append(value)
    return main, {key: "；".join(dict.fromkeys(values)) for key, values in fields.items()}


def narrative_paragraphs(parts, *, limit=360):
    """Break long prose at whole claims, keeping each explanation intact."""
    paragraphs, current = [], ""
    for part in parts:
        if not part:
            continue
        if current and len(current) + len(part) + 1 > limit:
            paragraphs.append(current)
            current = ""
        current = current + " " + part if current else part
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def coherent_cues(value):
    """Group recorded claims by their existing coverage or explicit prose label."""
    claims = value.get("claims", [])
    coverage = value.get("coverage", {})
    groups = {key: [] for key in ("patient", "device", "care", "tissue")}
    labels = {key: STEPS["recognize"][1][key] for key in groups}
    assigned = set()
    for claim in claims:
        raw = without_inference_labels(claim.get("text", "")).strip()
        explicit = next((key for key, label in labels.items()
                         if re.match(r"^\s*(?:[-*]\s+|\d+[.)、]\s*)?(?:\*\*)?" + re.escape(label)
                                     + r"(?:\*\*)?\s*[:：]", raw)), None)
        keys = [explicit] if explicit else [key for key in groups
            if claim["claim_id"] in coverage.get(key, {}).get("claim_ids", [])]
        text = performance_claim_text(claim, "recognize")
        for key in keys:
            if text and text not in groups[key]:
                groups[key].append(text)
                assigned.add(claim["claim_id"])
    paragraphs = ["**" + labels[key] + "**：" + " ".join(texts)
                  for key, texts in groups.items() if texts]
    # Older replies may have no coverage mapping. Preserve their full prose
    # without guessing which clinical category it belongs to.
    remaining = [performance_claim_text(c, "recognize") for c in claims if c["claim_id"] not in assigned]
    if remaining:
        paragraphs.append(narrative_paragraphs(list(dict.fromkeys(remaining))))
    return "\n\n".join(p for p in paragraphs if p)


def short_measure_title(name):
    return (
        0 < len(name) <= 36
        and not name.startswith(("若", "如果", "当", "仅当", "只有", "除非"))
        and name not in {"目标", "条件", "适用条件", "约束"}
        and not re.search(r"[\n`*_#\[\]<>。；;]|目标\s*[:：]", name)
        and name.count("（") == name.count("）")
        and name.count("(") == name.count(")")
    )


def solution_paragraph(number, main, details):
    """Reuse a short existing measure name; never invent or truncate a title."""
    goal_pattern = re.compile(r"[（(](?:\*\*)?目标(?:\*\*)?\s*[:：](?:\*\*)?([^（()）]+)[）)]")
    first, separator, rest = main.partition("。")
    explicit = re.match(r"^([^:：]+)[:：]\s*(.*)$", main, re.S)
    if explicit and short_measure_title(plain_measure_title(goal_pattern.sub("", explicit[1]))):
        first, separator, rest = explicit[1], "：", explicit[2]
    first = first.strip()
    name = measure_name(first)
    safe_title = separator and short_measure_title(name)
    title = "方案" + str(number) + ("：" + name if safe_title else "")
    if safe_title and plain_measure_title(goal_pattern.sub("", first)) == name:
        # The action is already in the heading. Keep its explicit purpose in
        # the description, without repeating the action or losing the goal.
        goals = "；".join(match[1].strip() for match in goal_pattern.finditer(first))
        body = sentence("目的是" + goals) if goals else ""
        body += sentence(narrative_goals(rest.strip()))
    else:
        body = sentence(narrative_goals(main))
    body += "".join(sentence(value) for value in details if value)
    return "**" + title + "**" + ("  \n" + body if body else "")


def coherent_solutions(claims):
    measures, scope = [], []
    for number, claim in enumerate(claims, 1):
        text = performance_claim_text(claim, "solutions", include_details=False)
        main, fields = prose_fields(text)
        name = measure_name(main)
        conditions = [fields.get("前置条件", ""), fields.get("条件", ""), claim.get("conditions", "")]
        constraints = [fields.get("约束", "")]
        details = list(dict.fromkeys(value.strip().rstrip("。；;") for value in conditions + constraints if value.strip()))
        if details:
            # Keep each condition tied to its own measure at the section end.
            scope.append("**方案" + str(number) + "**：" + sentence((name + "时，" if short_measure_title(name) else "") + "；".join(details)))
        explanation = []
        if claim.get("rationale") and claim["rationale"] not in main:
            explanation.append(claim["rationale"])
        for key, value in fields.items():
            if key not in {"前置条件", "条件", "约束"}:
                explanation.append(value)
        measures.append(solution_paragraph(number, main, explanation))
    body = "\n\n".join(measures)
    return body + ("\n\n**实施条件与注意事项**  \n" + "  \n".join(scope) if scope else "")


def coherent_actions(claims):
    actions = []
    for claim in claims:
        main, fields = prose_fields(performance_claim_text(claim, "action"))
        timing = re.match(r"^(立即|短期|待确认)行动\s*[:：]\s*", main)
        when = {"立即": "立即", "短期": "短期内", "待确认": "对于待确认事项"}.get(timing[1], "") if timing else ""
        if timing:
            main = main[timing.end():]
        main = narrative_goals(main)
        role = fields.pop("执行角色", "").replace("+", "与")
        if role:
            subject = re.split(r"[（(]", role, maxsplit=1)[0].strip()
            if main.startswith(subject):
                main = role + main[len(subject):]
            elif main.startswith(("若", "如果", "当")):
                main = sentence(main) + sentence("该项工作由" + role + "负责")
            else:
                main = role + "负责" + main
        if when:
            if when == "立即" and role and main.startswith(role):
                rest = main[len(role):]
                modal = re.match(r"^(需要|可以|负责|应|需|可)", rest)
                main = role + (modal[0] + "立即" + rest[modal.end():] if modal else "立即" + rest)
            else:
                main = when + "，" + main
        action = [sentence(main)]
        prerequisite = fields.pop("前置条件", "")
        if prerequisite:
            action.append(sentence("实施前提是" + prerequisite))
        check = fields.pop("核查", "")
        if check:
            action.append(sentence("核查时，" + check))
        # Preserve fields we do not interpret, rather than silently deleting
        # a qualification or assigning it to another person's operation.
        action.extend(sentence(value) for value in fields.values())
        paragraph = "".join(text for text in action if text)
        if paragraph:
            actions.append(paragraph)
    return "\n\n".join(str(i) + ". " + text for i, text in enumerate(actions, 1))


def coherent_outcomes(claims):
    timing, observations, escalation, records = [], [], [], []
    for claim in claims:
        original = without_inference_labels(claim.get("text", "")).strip()
        main, fields = prose_fields(performance_claim_text(claim, "evaluate"))
        timing.append(sentence(narrative_goals(fields.pop("复评时机", ""))))
        record = fields.pop("记录", "")
        observation = []
        if main:
            if re.match(r"^(?:\*\*)?复评指标\s*\d+", original):
                main = ("复评时观察" if not observations else "还需观察") + main.rstrip("。")
            if record and re.search(r"(?:应|需|要|并|及时)记录|^记录", record):
                observation.extend((sentence(main), sentence(record)))
            else:
                observation.append(sentence(main.rstrip("。") + ("，并记录" + record if record else "")))
        elif record:
            records.append(sentence("复评记录包括" + record))
        trigger = fields.pop("升级条件", "")
        if trigger:
            prefix = ("出现以下情况时：" if re.match(r"^1[）.)、]", trigger) else
                      "" if trigger.startswith(("若", "如果", "当", "如", "一旦")) else "若")
            has_response = re.search(r"(?:应|需|要|立即).{0,8}(?:通知|报告|升级|调整|更换|评估|呼叫|重新)", trigger)
            target = observation if main else escalation
            target.append(("**升级条件**：" if not main else "")
                          + sentence((prefix if main else "") + trigger + ("" if has_response else "，需升级评估")))
        observation.extend(sentence(value) for value in fields.values())
        paragraph = "".join(text for text in observation if text)
        if paragraph:
            observations.append(paragraph)
    when = "".join(dict.fromkeys(text for text in timing if text))
    groups = ["**复评时机**：" + when if when else ""]
    groups.extend(observations)
    groups.extend(("".join(escalation), "".join(records)))
    return "\n\n".join(dict.fromkeys(text for text in groups if text))


def performance_references(result):
    """List only used documents, with recorded locations and supporting steps."""
    order = [step for step in STEPS if step in result["six_steps"]]
    numbers, references = numbered_references(result, order)
    used = {eid for value in result["six_steps"].values()
            for c in value.get("claims", []) for eid in c.get("evidence_ids", [])}
    step_names = {"recognize": "识别", "analyze": "分析", "prioritize": "排序",
                  "solutions": "方案", "action": "行动", "evaluate": "复评"}
    rows = []
    for number, source in references:
        pages, supports = set(), []
        for excerpt in result.get("citations", []):
            eid = excerpt.get("evidence_id")
            if eid not in used or numbers.get(eid) != number:
                continue
            for p in excerpt.get("source_pages") or []:
                if str(p).isdigit() and int(p) > 0:
                    pages.add(int(p))
        for step in order:
            if any(numbers.get(eid) == number for c in result["six_steps"][step].get("claims", [])
                   for eid in c.get("evidence_ids", [])):
                supports.append(step_names[step])
        details = (["涉及：" + "、".join(supports)] if supports else [])
        if pages:
            details.append("原文位置：PDF 第 " + "、".join(map(str, sorted(pages))) + " 页")
        rows.append(str(number) + ". " + format_reference(source)
                    + ("  \n" + "；".join(details) + "。" if details else ""))
    return "### 可核查证据来源\n\n" + "\n\n".join(rows) if rows else ""


def render_performance_prose(result, *, references=""):
    """Six readable sections with the actual publication/self-audit boundary."""
    sections = []
    if notice := performance_notice(result):
        sections.append(notice)
    if result.get("errors"):
        web_categories = {"search", "network", "configuration", "page_fetch_or_parse", "browser_reader", "fulltext_resolution"}
        reviewed_web_failure = (result.get("audit_summary", {}).get("three_checks_completed")
                                and all(error.get("category") in web_categories for error in result["errors"]))
        sections.append("联网补充资料出现异常；以下保留已生成并完成反思审查的研究回答。" if reviewed_web_failure else
                        "反思流程未完成，以下保留已生成的研究回答。")
    for step, (title, _) in STEPS.items():
        value = result["six_steps"].get(step, {})
        paragraphs = [performance_claim_text(claim, step) for claim in value.get("claims", [])]
        paragraphs = list(dict.fromkeys(text for text in paragraphs if text))
        if not paragraphs:
            continue
        formatter = {"solutions": coherent_solutions, "action": coherent_actions, "evaluate": coherent_outcomes}.get(step)
        body = (coherent_cues(value) if step == "recognize" else
                formatter(value["claims"]) if formatter else
                "\n\n".join(paragraphs) if step == "prioritize" else narrative_paragraphs(paragraphs))
        sections.append("### " + title + "\n\n" + body)
    if references:
        sections.append(references)
    sections.append(DISCLAIMER)
    parts = prose_parts("\n\n".join(sections))
    for i in range(0, len(parts), 2):
        # CommonMark needs whitespace after a bold span ending in a closing
        # parenthesis before Chinese prose, or it can display literal stars.
        parts[i] = re.sub(r"(\*\*[^*\n]+[）)])\*\*(?=\w)", r"\1** ", parts[i])
    return "".join(parts)


def performance_answer(result):
    """Render draft, revision and restored replies as prose without changing evidence."""
    if not result.get("six_steps"):
        text = result.get("answer") or json.dumps(result, ensure_ascii=False)
        hidden = {int(n) for n in re.findall(r"(?m)^[ \t]*(?:-\s*)?\[(\d+)\]", text)} if REFERENCE_HEADING.search(text) else set()
        body = without_inference_labels(clean_prose(without_bibliography(text), hidden_numbers=hidden))
        notice = performance_notice(result)
        return notice + "\n\n" + body if notice else body
    public = deepcopy(result)
    hidden = set(numbered_references(result, public["six_steps"])[0].values())
    identifiers = [s.get(k, "") for s in result.get("citations", []) for k in ("evidence_id", "doc_id", "chunk_id")]
    identifiers += [f.get("fact_id", "") for f in result.get("facts", {}).get("facts", [])]
    labels = {}
    names = {"recognize": "识别线索", "analyze": "分析", "prioritize": "问题", "solutions": "措施", "action": "行动", "evaluate": "复评指标"}
    for key, step in public["six_steps"].items():
        for i, claim in enumerate(step["claims"], 1):
            number = re.search(r"\.c(\d+)$", claim["claim_id"])
            labels[claim["claim_id"]] = names.get(key, "项目") + (number[1] if number else str(i))
            if key == "solutions":
                target = {"text": clean_prose(claim.get("text", ""), internal_ids=identifiers, hidden_numbers=hidden)}
                label = re.split(r"[。；;]", performance_claim_text(target, key), maxsplit=1)[0].strip()
                if label:
                    labels[claim["claim_id"]] = label
    for step in public["six_steps"].values():
        for claim in step["claims"]:
            for key in ("text", "conditions", "rationale"):
                claim[key] = clean_prose(claim.get(key, ""), internal_ids=identifiers, hidden_numbers=hidden, labels=labels)
            claim["evidence_ids"] = []
        for row in step.get("coverage", {}).values():
            row["reason"] = clean_prose(row.get("reason", ""), internal_ids=identifiers, hidden_numbers=hidden, labels=labels)
    public["citations"] = []
    return render_performance_prose(public, references=performance_references(result))


class SourceStream:
    """Publish complete lines so split model tokens cannot expose library IDs."""
    def __init__(self, sources, emit):
        self.emit = emit
        self.hidden = [i for i, source in enumerate(sources, 1) if is_knowledge_source(source)]
        self.raw = ""
        self.published_end = 0

    def __call__(self, kind, value):
        if kind != "token" or not self.hidden:
            self.emit(kind, value)
            return
        self.raw += value
        end = self.raw.rfind("\n") + 1
        if end > self.published_end:
            self.published_end = end
            self.emit("answer_text", clean_prose(self.raw[:end], hidden_numbers=self.hidden))

    def finish(self):
        if self.hidden and self.raw:
            self.emit("answer_text", clean_prose(self.raw, hidden_numbers=self.hidden))
