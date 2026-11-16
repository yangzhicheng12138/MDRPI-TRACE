"""Turn explicit prose review decisions into the existing internal contracts.

The model supplies the clinical judgments. This codec supplies only structure,
identity checks and provenance; it never infers a passing judgment from silence.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import re

from jsonschema import Draft202012Validator, ValidationError

from .question_generation_audit import audited_review_schema, validate_audited_review, evidence_text_field
from .question_generation_schema import blind_review_schema, validate_review


class TextReviewFormatError(ValueError):
    """A short, safe location and format diagnosis, without source content."""

    def __init__(self, location, reason):
        self.location = location
        self.reason = reason
        super().__init__(f"{location}：{reason}")


_VERDICTS = dict(zip(("通过", "需修订", "需补证", "拒绝"),
                     ("accept", "revise", "needs_evidence", "reject")))
_KINDS = dict(zip(("医学错误", "证据不符", "材料缺口", "表述歧义", "考核不符", "格式问题"),
                  ("medical_error", "evidence_mismatch", "material_gap", "ambiguity",
                   "assessment_mismatch", "format")))
_SUPPORT = {"充分支持": "supported", "支持": "supported", "有限支持": "limited",
            "不支持": "unsupported", "不确定": "uncertain"}
_CASE = {"一致": "consistent", "无依据": "unsupported", "不支持": "unsupported", "不确定": "uncertain"}
_ALIGNMENT = {"一致": "aligned", "对齐": "aligned", "不一致": "misaligned", "不匹配": "misaligned", "不确定": "uncertain"}
_DISTINCT = {"有区分": "distinct", "无重复": "distinct", "重复": "overlap",
             "不适用": "not_applicable", "不确定": "uncertain"}

_LABELS = {
    "结论": "verdict", "病例事实": "case_status", "病例事实说明": "case_note",
    "目标评分": "alignment_status", "目标评分说明": "alignment_note",
    "重复判断": "distinct_status", "重复判断说明": "distinct_note",
    "问题": "issue", "类型": "kind", "位置": "location", "原句": "text",
    "原因": "explanation", "建议": "requested_change", "来源": "evidence_ids",
    "证据核查": "check", "评分点核查": "check", "评分点": "check",
    "评分点支持": "check_status", "评分点理由": "rationale", "理由": "rationale",
    "核查来源": "source", "来源核查": "source", "来源支持": "source_status",
    "依据概述": "quote", "证据概述": "quote", "原文": "quote",
    "补证": "gap", "检索词": "query", "补证理由": "reason",
    "总结": "summary", "可答性": "answerable", "独立试答": "attempted_answer", "歧义": "ambiguity",
}
_NAMES = {value: key for key, value in _LABELS.items()}
_NAMES.update(check="证据核查", source="核查来源", rationale="评分点理由")
_NONE = {"无", "无。"}
_GLOBAL = {"verdict", "case_status", "case_note", "alignment_status", "alignment_note",
           "distinct_status", "distinct_note", "summary"}
_ISSUE_FIELDS = {"kind", "location", "text", "explanation", "requested_change", "evidence_ids"}
_QUALITY = {"case_status", "case_note", "alignment_status", "alignment_note", "distinct_status",
            "distinct_note", "check", "check_status", "rationale", "source", "source_status", "quote"}


def review_text_instructions(*, audited: bool, blind: bool = False, evidence_mode="verbatim") -> str:
    """A prose-only report protocol, with no schema or serialization request."""
    evidence_text_field(evidence_mode)
    if blind:
        return """只给出中文文字盲审报告，使用下面固定标签，不用表格或竖线。
可答性：可答 或 不可答（必须明确选择）
独立试答：仅依据已给学生可见材料独立作答；确实无法作答时写无。
歧义：无
有歧义时用具体内容替代“无”，每条重复“歧义：”，至多5条。
标签后可分行写正文。不得省略可答性、独立试答或歧义，不得从教师答案推断可答性。"""
    quality = """
病例事实：一致 或 无依据 或 不确定
病例事实说明：说明病例事实是否被擅自添加或改变。
目标评分：一致 或 不一致 或 不确定
目标评分说明：说明考核目标、任务和评分点是否对齐。
重复判断：有区分 或 重复 或 不适用 或 不确定
重复判断说明：依据相似候选明确判断；有相似候选不能写不适用。
""" if audited else "\n"
    checks = """
证据核查：R1
评分点支持：充分支持 或 有限支持 或 不支持 或 不确定
评分点理由：解释原文如何支持该评分点及病例应用，或为何支持不足。
核查来源：该评分点实际引用的来源编号
来源支持：充分支持 或 有限支持 或 不支持 或 不确定
原文：从该来源逐字连续摘录，不要改写、拼接、省略号或添加引号；没有可用原文写无。
每个评分点重复一个“证据核查”组；该评分点引用的每个来源都重复一个
“核查来源/来源支持/原文”组，必须逐一覆盖，不得漏查或借用其他评分点的来源。
每评分点至多16个来源。充分支持的来源必须给出原文；无原文不能标为充分支持。
""" if audited else "\n"
    if audited and evidence_mode == "semantic":
        checks = checks.replace(
            "原文：从该来源逐字连续摘录，不要改写、拼接、省略号或添加引号；没有可用原文写无。",
            "依据概述：用自己的话忠实概括该来源与评分点有关的意思，可缩写、翻译或综合同一来源的相关内容，不必逐字摘录；没有可用依据写无。"
        ).replace("核查来源/来源支持/原文", "核查来源/来源支持/依据概述").replace(
            "充分支持的来源必须给出原文；无原文不能标为充分支持。",
            "充分支持的来源必须说明依据；概述须符合来源含义和适用条件，不因字面不同认定错误。"
        )
    return """只给出中文文字审核报告，按下列固定标签作答，不用表格或竖线。
标签必须保留；示例中的备选状态必须明确选一个，不得省略判断或用沉默表示通过。
结论：通过 或 需修订 或 需补证 或 拒绝
""" + quality + """
问题：1
类型：医学错误 或 证据不符 或 材料缺口 或 表述歧义 或 考核不符 或 格式问题
位置：具体字段或评分点
原句：待修改原句，没有原句写无
原因：问题原因
建议：具体修改建议
来源：相关实际来源编号，多个用顿号分开；此标签可省略，无引用时由程序记空
每个问题重复上述“问题”组，至多8条。没有问题时仅写“问题：无”，不能直接省略。
""" + checks + """
补证：1
检索词：需要补充证据的具体检索词
补证理由：说明补什么依据以及为什么需要
每个补证要求重复上述“补证”组；也可在同一补证段落重复完整的“检索词/补证理由”配对，
程序会逐对记录，合计至多3条。无需补证时仅写“补证：无”。
总结：简述最终判断。
通过必须没有未解决问题和补证需求；需修订或拒绝必须列问题；需补证必须列检索词及理由。
说明、原因、建议、原句、原文、总结可以跨行写。保留真实临床判断，材料缺失不能自动算医学错误。"""


@dataclass
class _Field:
    name: str
    value: str
    line: int

    @property
    def where(self):
        return f"第{self.line}行·{_NAMES[self.name]}"


def _unmark(value):
    value = value.strip()
    for marker in ("**", "__", "`"):
        if value.startswith(marker) and value.endswith(marker) and len(value) > len(marker) * 2:
            return value[len(marker):-len(marker)].strip()
    return value


def _line_field(line):
    """Recognize formatting around labels without normalizing quoted content."""
    value = line.strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"^(?:[-+*]\s+|[（(]?\d+[）).、．]\s*)", "", value)
    # '**标签：** 正文' emphasizes the colon along with the label.
    emphasized = re.match(r"^(\*\*|__)([^:：]{1,30})[:：]\1\s*(.*)$", value)
    if emphasized:
        label = re.sub(r"\s+", "", emphasized[2])
        if label in _LABELS:
            return _LABELS[label], emphasized[3]
    for candidate in (value, _unmark(value)):
        match = re.match(r"^(.{1,30}?)\s*[:：]\s*(.*)$", candidate)
        if match:
            label = re.sub(r"\s+", "", _unmark(match[1]))
            if label in _LABELS:
                return _LABELS[label], match[2]
    value = _unmark(value)
    # Markdown headings such as '### 问题 1' and '## 证据核查 R1'.
    match = re.fullmatch(r"(问题|补证|证据核查|评分点核查|评分点|核查来源|来源核查)\s*([A-Za-z0-9_-]+|[一二三四五六七八九十]+)", value)
    if match:
        return _LABELS[match[1]], match[2]
    if value in _LABELS:
        return _LABELS[value], ""
    return None


def _fields(content):
    result = []
    current = None
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            if current:
                current.value += "\n"
            continue
        parsed = _line_field(line)
        if parsed:
            current = _Field(*parsed, line=number)
            result.append(current)
        elif current is None:
            if re.fullmatch(r"\s*#{0,6}\s*(?:审核报告|盲审报告|审核意见)\s*", line):
                continue
            raise TextReviewFormatError(f"第{number}行", "无法识别固定标签")
        else:
            # Unknown headings/field labels must not silently swallow an issue.
            # A literal source, cited sentence or independent answer can itself
            # contain labels such as '血压：' and therefore retains raw lines.
            looks_like_field = re.match(r"\s*(?:[-+*]\s+|\d+[.、]\s*)?(?:\*\*)?[^\s:：]{1,16}[:：]", line)
            if line.lstrip().startswith("#") or (looks_like_field and current.name not in {"quote", "text", "attempted_answer"}):
                raise TextReviewFormatError(f"第{number}行", "无法识别固定标签")
            current.value += "\n" + line
    for field in result:
        field.value = field.value.strip()
    return result


def _put(target, field):
    if field.name in target:
        raise TextReviewFormatError(field.where, "标签重复")
    target[field.name] = field


def _need(target, names, location):
    for name in names:
        if name not in target:
            raise TextReviewFormatError(location + "·" + _NAMES[name], "缺少必填标签")
    return {name: target[name].value for name in names}


def _choice(field, mapping):
    value = _unmark(re.sub(r"[。.]$", "", field.value.strip()))
    value = re.sub(r"[。.]$", "", value)
    if value in mapping:
        return mapping[value]
    if value in mapping.values():
        return value
    raise TextReviewFormatError(field.where, "未知或矛盾状态；必须明确选择一个")


def _record_header(field):
    value = _unmark(field.value)
    if value in _NONE:
        return False
    if not re.fullmatch(r"(?:第\s*)?(?:[1-9]\d*|[一二三四五六七八九十]+)(?:\s*[条项])?", value):
        raise TextReviewFormatError(field.where, "应写条目编号或无")
    return True


def _identities(field):
    value = _unmark(field.value)
    if value in _NONE:
        return []
    result = re.split(r"[\s,，、;；]+", value)
    if not result or any(not item for item in result):
        raise TextReviewFormatError(field.where, "来源编号格式不完整")
    return result


def _blank_none(value):
    return "" if value in _NONE else value


def _blind(fields):
    single, notes, explicit_none = {}, [], False
    for field in fields:
        if field.name in {"answerable", "attempted_answer"}:
            _put(single, field)
        elif field.name == "ambiguity":
            if _unmark(field.value) in _NONE:
                if notes or explicit_none:
                    raise TextReviewFormatError(field.where, "歧义无与其他条目重复或矛盾")
                explicit_none = True
            else:
                if explicit_none:
                    raise TextReviewFormatError(field.where, "歧义无与具体歧义矛盾")
                notes.append(field.value)
        else:
            raise TextReviewFormatError(field.where, "盲审报告包含其他审核标签")
    values = _need(single, ("answerable", "attempted_answer"), "盲审报告")
    if not notes and not explicit_none:
        raise TextReviewFormatError("盲审报告·歧义", "缺少必填标签")
    return {"answerable": _choice(single["answerable"], {"可答": True, "不可答": False}),
            "attempted_answer": _blank_none(values["attempted_answer"]), "ambiguity_notes": notes}


def _review(fields, *, audited):
    global_fields, issues, gaps, checks = {}, [], [], []
    seen_issues = seen_gaps = False
    issues_none = gaps_none = False
    active = None
    check = None
    source = None
    for field in fields:
        name = field.name
        if not audited and name in _QUALITY:
            raise TextReviewFormatError(field.where, "基础审核包含额外质量核查标签")
        if name in _GLOBAL:
            _put(global_fields, field)
        elif name == "issue":
            present = _record_header(field)
            if issues_none or (not present and seen_issues):
                raise TextReviewFormatError(field.where, "问题无与其他问题条目矛盾或重复")
            seen_issues, issues_none = True, not present
            active = {} if present else None
            if present:
                issues.append(active)
        elif name == "gap":
            present = _record_header(field)
            if gaps_none or (not present and seen_gaps):
                raise TextReviewFormatError(field.where, "补证无与其他补证条目矛盾或重复")
            seen_gaps, gaps_none = True, not present
            active = {} if present else None
            if present:
                gaps.append(active)
        elif name == "check":
            check = {"header": field, "fields": {}, "sources": []}
            checks.append(check)
            active, source = None, None
        elif name == "source":
            if check is None:
                raise TextReviewFormatError(field.where, "来源核查缺少所属评分点")
            source = {"header": field, "fields": {}}
            check["sources"].append(source)
        elif name in {"check_status", "rationale"}:
            if check is None:
                raise TextReviewFormatError(field.where, "缺少所属评分点")
            _put(check["fields"], field)
        elif name in {"source_status", "quote"}:
            if source is None:
                raise TextReviewFormatError(field.where, "缺少所属核查来源")
            _put(source["fields"], field)
        elif name in _ISSUE_FIELDS or name in {"query", "reason"}:
            if active is None:
                raise TextReviewFormatError(field.where, "缺少所属问题或补证条目")
            # A field cannot migrate into a differently shaped record.
            if (name in _ISSUE_FIELDS and (not issues or active is not issues[-1])) or (name in {"query", "reason"} and (not gaps or active is not gaps[-1])):
                raise TextReviewFormatError(field.where, "标签与当前条目类型不符")
            # A complete second search pair is another gap even when the
            # writer does not repeat the section heading. Never overwrite an
            # earlier query or pair a new query with the earlier reason.
            if name == "query" and "query" in active:
                if "reason" not in active:
                    raise TextReviewFormatError(field.where, "上一补证条目缺少补证理由")
                active = {}
                gaps.append(active)
            _put(active, field)
        else:
            raise TextReviewFormatError(field.where, "标签不属于当前审核报告")
    values = _need(global_fields, ("verdict", "summary"), "审核报告")
    if not seen_issues:
        raise TextReviewFormatError("审核报告·问题", "缺少问题条目或明确的无")
    if not seen_gaps:
        raise TextReviewFormatError("审核报告·补证", "缺少补证条目或明确的无")
    result = {"verdict": _choice(global_fields["verdict"], _VERDICTS), "issues": [],
              "evidence_gaps": [], "summary": values["summary"]}
    for index, item in enumerate(issues, 1):
        values = _need(item, ("kind", "location", "text", "explanation", "requested_change"), f"问题{index}")
        result["issues"].append({**values, "kind": _choice(item["kind"], _KINDS),
                                 "text": _blank_none(values["text"]),
                                 "evidence_ids": _identities(item["evidence_ids"]) if "evidence_ids" in item else []})
    for index, item in enumerate(gaps, 1):
        values = _need(item, ("query", "reason"), f"补证{index}")
        result["evidence_gaps"].append(values)
    if audited:
        audit = {}
        for target, stem, mapping in (("case_facts", "case", _CASE),
                                       ("assessment_alignment", "alignment", _ALIGNMENT),
                                       ("distinctness", "distinct", _DISTINCT)):
            _need(global_fields, (stem + "_status", stem + "_note"), "审核报告")
            audit[target] = {"status": _choice(global_fields[stem + "_status"], mapping),
                             "note": global_fields[stem + "_note"].value}
        audit["evidence_support"] = []
        for index, item in enumerate(checks, 1):
            values = _need(item["fields"], ("check_status", "rationale"), f"证据核查{index}")
            row = {"criterion_id": _unmark(item["header"].value),
                   "status": _choice(item["fields"]["check_status"], _SUPPORT),
                   "rationale": values["rationale"], "sources": []}
            for source_index, item_source in enumerate(item["sources"], 1):
                location = f"证据核查{index}·来源{source_index}"
                source_fields = item_source["fields"]
                values = _need(source_fields, ("source_status", "quote"), location)
                row["sources"].append({"evidence_id": _unmark(item_source["header"].value),
                                       "status": _choice(source_fields["source_status"], _SUPPORT),
                                       "quote": _blank_none(values["quote"])})
            audit["evidence_support"].append(row)
        result["quality_audit"] = audit
    return result


# Only a repeated dot page ornament at the start of a PDF line is removable.
# Clinical numbers, list numbering, punctuation and arbitrary words stay intact.
_PDF_PAGE = re.compile(r"\n[ \t\u3000]*(?:[·⋅][ \t\u3000]*){2,}[0-9]{1,5}(?=[ \t\u3000\r\n])[ \t\u3000]*(?:\r?\n)?")


def _view(text, *, pdf=False):
    skipped = set()
    if pdf:
        for match in _PDF_PAGE.finditer(text):
            skipped.update(range(match.start(), match.end()))
    positions = [i for i, character in enumerate(text) if not character.isspace() and i not in skipped]
    return "".join(text[i] for i in positions), positions


def _original_span(quote, text, location):
    if not quote.strip():
        return quote
    candidates = [quote]
    pairs = {"“": "”", '"': '"', "‘": "’", "'": "'", "`": "`"}
    stripped = quote.strip()
    if len(stripped) > 2 and pairs.get(stripped[0]) == stripped[-1]:
        candidates.append(stripped[1:-1])
    for candidate in candidates:
        wanted, _ = _view(candidate)
        if not wanted:
            continue
        for pdf in (False, True):
            normalized, positions = _view(text, pdf=pdf)
            start = normalized.find(wanted)
            if start >= 0:
                return text[positions[start]:positions[start + len(wanted) - 1] + 1]
    raise TextReviewFormatError(location, "原文不是指定来源的连续摘录")


def _restore_quotes(value, evidence, question):
    sources = {row["evidence_id"]: row["text"] for row in evidence["sources"]}
    criteria = {row["id"]: set(row["evidence_ids"]) for row in question["scoring_points"]}
    checks = value.get("quality_audit", {}).get("evidence_support", [])
    if not isinstance(checks, list):
        return
    for index, check in enumerate(checks, 1):
        location = f"证据核查{index}"
        if not isinstance(check, dict) or not isinstance(check.get("sources"), list):
            continue  # The existing schema produces a safe field location below.
        if check.get("criterion_id") not in criteria:
            raise TextReviewFormatError(location + "·评分点", "未知评分点编号")
        for source_index, source in enumerate(check["sources"], 1):
            where = f"{location}·来源{source_index}"
            if not isinstance(source, dict):
                continue
            identity = source.get("evidence_id")
            if not isinstance(identity, str) or identity not in sources:
                raise TextReviewFormatError(where, "未知来源编号")
            if identity not in criteria[check["criterion_id"]]:
                raise TextReviewFormatError(where, "来源不属于当前评分点的引用")
            if isinstance(source.get("quote"), str):
                source["quote"] = _original_span(source["quote"], sources[identity], where + "·原文")


def _json_compat(content):
    candidate = content.strip()
    if candidate.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", candidate, re.DOTALL | re.IGNORECASE)
        if not match:
            raise TextReviewFormatError("审核报告", "无法识别围栏中的兼容报告")
        candidate = match[1].strip()
    if not candidate.startswith("{"):
        return None

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TextReviewFormatError("兼容报告", "存在重复字段")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise TextReviewFormatError("兼容报告", "包含非法数值")

    try:
        return json.loads(candidate, object_pairs_hook=unique, parse_constant=invalid_constant)
    except json.JSONDecodeError as exc:
        raise TextReviewFormatError(f"第{exc.lineno}行", "兼容报告未完整闭合或格式错误") from None


def _validation_location(error):
    # Schema paths contain only fixed contract keys and array offsets; never
    # include jsonschema's message, which can embed the entire source excerpt.
    allowed = {"answerable", "attempted_answer", "ambiguity_notes", "verdict", "issues", "evidence_gaps",
               "summary", "kind", "location", "text", "explanation", "evidence_ids", "requested_change",
               "query", "reason", "quality_audit", "case_facts", "assessment_alignment", "distinctness",
               "status", "note", "evidence_support", "criterion_id", "sources", "evidence_id", "quote", "rationale"}
    path = "报告"
    for part in error.absolute_path:
        if isinstance(part, int):
            path += f"[{part + 1}]"
        elif part in allowed:
            path += "." + part
    return path


def parse_review_text(content, *, evidence, question, audited: bool, blind: bool = False,
                      evidence_mode="verbatim") -> dict:
    """Parse a complete report using the selected evidence explanation policy.

    Full valid legacy object replies remain readable. No missing clinical
    decision, issue, gap, criterion or actual citation is synthesized.
    """
    evidence_text_field(evidence_mode)
    if not isinstance(content, str) or not content.strip():
        raise TextReviewFormatError("审核报告", "文字回复为空或类型错误")
    value = _json_compat(content)
    if value is None:
        fields = _fields(content)
        value = _blind(fields) if blind else _review(fields, audited=audited)
    value = deepcopy(value)
    try:
        if blind:
            Draft202012Validator(blind_review_schema()).validate(value)
        elif audited:
            if evidence_mode == "semantic" and isinstance(value, dict):
                # Saved reports may use the former "原文/quote" label. Store
                # their evidence explanation as a summary, never a verified quote.
                audit = value.get("quality_audit")
                checks = audit.get("evidence_support") if isinstance(audit, dict) else None
                for check in checks if isinstance(checks, list) else []:
                    sources = check.get("sources") if isinstance(check, dict) else None
                    for source in sources if isinstance(sources, list) else []:
                        if isinstance(source, dict) and "quote" in source and "summary" not in source:
                            source["summary"] = source.pop("quote")
            Draft202012Validator(audited_review_schema(
                [row["evidence_id"] for row in evidence["sources"]], question,
                evidence_mode=evidence_mode)).validate(value)
            if evidence_mode == "verbatim":
                _restore_quotes(value, evidence, question)
            validate_audited_review(value, evidence, question, evidence_mode=evidence_mode)
        else:
            validate_review(value, [row["evidence_id"] for row in evidence["sources"]])
    except ValidationError as exc:
        raise TextReviewFormatError(_validation_location(exc), "未通过审核结构、覆盖或结论一致性校验") from None
    return value
