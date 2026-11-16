"""Independent six-step contracts and deterministic publication gates."""

from __future__ import annotations

from copy import deepcopy
import re

from .local_llm import InvalidModelOutput
from .citation_format import display_text, reference_for_display, numbered_references


SCHEMA_VERSION = "mdrpi.six_steps.v2"
DISCLAIMER = "本回答仅用于研究和教学，不构成对真实患者的护理医嘱；如用于临床，需由具备资质的医护人员结合患者实际情况判断。"
STEPS = {
    "recognize": ("1. Recognize Cues｜识别线索", {
        "patient": "患者因素", "device": "器械因素", "care": "护理过程因素", "tissue": "局部组织线索",
        "negatives": "明确否定", "unknowns": "未知资料", "questions": "需要补充的问题"}),
    "analyze": ("2. Analyze Cues｜分析线索", {
        "relationships": "关键线索与风险关系", "mdrpi": "MDRPI支持程度及不确定性（不作自动诊断或分期）",
        "tissue_type": "皮肤或黏膜或待确认", "differential": "需鉴别的其他病因", "information": "判断所缺资料"}),
    "prioritize": ("3. Prioritize Hypotheses｜问题排序", {
        "problems": "带编号的问题", "priority": "优先级或无法排序", "rationale": "排序理由",
        "facts": "相关病例事实", "evidence": "排序的文献依据"}),
    "solutions": ("4. Generate Solutions｜生成方案", {
        "measures": "带编号的具体措施", "goals": "措施目标", "conditions": "适用条件",
        "function": "不得损害治疗或监测功能的约束", "evidence": "措施的文献依据"}),
    "action": ("5. Take Action｜采取行动", {
        "links": "关联措施编号", "order": "相对先后顺序及前置依赖", "timing": "立即行动或待确认的时机",
        "roles": "执行角色、合格人员或团队协作（权限无依据时明确待确认）", "checks": "操作后的核查"}),
    "evaluate": ("6. Evaluate Outcomes｜评价结果", {
        "sites": "复查皮肤或黏膜部位", "function": "器械治疗及监测功能", "physiology": "患者生理指标",
        "timing": "复评时间或触发事件", "escalation": "升级触发条件", "records": "可核查的结果记录与证据来源"}),
}
STATUSES = ["supported", "partial", "missing", "not_applicable", "conflict"]


def obj(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def string(limit=300):
    return {"type": "string", "minLength": 1, **({"maxLength": limit} if limit is not None else {})}


def strings(values=None, maximum=12):
    item = string(180)
    if values:
        item["enum"] = list(values)
    return {"type": "array", "items": item, "uniqueItems": True, "maxItems": maximum if values != [] else 0}


def step_schema(step, fact_ids, evidence_ids, previous_ids, *, research_mode=False):
    ids = [f"{step}.c{i}" for i in range(1, 7)]
    claim = obj({
        "claim_id": {"type": "string", "enum": ids}, "text": string(None if research_mode else 150),
        "kind": {"type": "string", "enum": ["case_fact", "source_statement", "case_inference", "model_inference", "uncertainty"]},
        "fact_ids": strings(fact_ids), "evidence_ids": strings(evidence_ids, 4),
        "conditions": string(None if research_mode else 80), "rationale": string(None if research_mode else 100),
        "depends_on": strings(previous_ids, 8),
    })
    requirement = obj({"status": {"type": "string", "enum": STATUSES},
                       "reason": string(None if research_mode else 160), "claim_ids": strings(ids, 6)})
    if research_mode:
        # Lean generation embeds these explanations in the actual text. Empty
        # metadata does not declare clinical support; the content is audited.
        claim["properties"]["conditions"]["minLength"] = 0
        claim["properties"]["rationale"]["minLength"] = 0
        requirement["properties"]["reason"]["minLength"] = 0
        # Repeating a valid reference does not change the model's statement.
        # Keep its raw list; ID membership and claim/dependency checks still apply.
        for field in ("fact_ids", "evidence_ids", "depends_on"):
            claim["properties"][field].pop("uniqueItems")
        requirement["properties"]["claim_ids"].pop("uniqueItems")
    kinds = ["case_fact", "uncertainty"] if step == "recognize" and not research_mode else ["case_fact", "source_statement", "case_inference", "model_inference", "uncertainty"]
    branches = []
    for kind in kinds:
        branch = deepcopy(claim)
        p = branch["properties"]
        p["kind"]["enum"] = [kind]
        if kind == "case_fact":
            p["fact_ids"]["minItems"] = 1
            p["evidence_ids"]["maxItems"] = 0
        elif kind in {"source_statement", "case_inference"}:
            if not evidence_ids or (kind == "case_inference" and not fact_ids):
                continue
            p["evidence_ids"]["minItems"] = 1
            if kind == "case_inference":
                p["fact_ids"]["minItems"] = 1
        if kind == "case_fact" and not fact_ids:
            continue
        branches.append(branch)
    return obj({"claims": {"type": "array", "items": {"anyOf": branches}, "maxItems": 6 if research_mode else 4},
                "coverage": obj({key: deepcopy(requirement) for key in STEPS[step][1]})})


def empty_steps(reason="未取得经核验内容"):
    return {step: {"claims": [], "coverage": {
        key: {"status": "missing", "reason": reason, "claim_ids": []} for key in requirements}}
        for step, (_, requirements) in STEPS.items()}


def all_claims(steps):
    return {claim["claim_id"]: claim for value in steps.values() for claim in value["claims"]}


def validate_steps(steps, facts, evidence, *, complete=True, research_mode=False):
    from jsonschema import validate, ValidationError
    if (complete and set(steps) != set(STEPS)) or not set(steps) <= set(STEPS):
        raise InvalidModelOutput("Unexpected six-step fields")
    facts = {item["fact_id"]: item for item in facts["facts"]}
    previous = []
    graph = {}
    # JSON object key order does not define the six-step dependency order.
    for step in STEPS:
        if step not in steps:
            continue
        value = steps[step]
        try:
            validate(value, step_schema(step, list(facts), list(evidence), previous, research_mode=research_mode))
        except ValidationError as exc:
            raise InvalidModelOutput(f"Invalid {step} contract: {exc.message}") from exc
        ids = [item["claim_id"] for item in value["claims"]]
        if len(set(ids)) != len(ids):
            raise InvalidModelOutput("Duplicate claim IDs")
        for claim in value["claims"]:
            cid = claim["claim_id"]
            if claim["kind"] == "case_fact" and (not claim["fact_ids"] or claim["evidence_ids"]):
                raise InvalidModelOutput("Case facts require original facts, not literature")
            # In research, keep the original prose and let the existing case
            # audit judge attribution/classification. A semantic concern in a
            # valid body must not regenerate the entire six-step draft.
            if not research_mode and claim["kind"] == "case_fact" and re.search(r"\bDOC\d{3}_P\d{4}_C\d{3,4}\b", claim["text"]):
                raise InvalidModelOutput("Case fact contains literature attribution")
            if claim["kind"] in {"source_statement", "case_inference"} and not claim["evidence_ids"]:
                raise InvalidModelOutput("Clinical claim lacks evidence")
            if claim["kind"] == "case_inference" and not claim["fact_ids"]:
                raise InvalidModelOutput("Case inference lacks case facts")
            # An unknown observation can support explicit uncertainty (for example,
            # inability to rank). The case and evidence audits must check that it
            # was not inverted into a positive or negative patient finding.
            if not set(claim["depends_on"]) <= set(previous + ids) or cid in claim["depends_on"]:
                raise InvalidModelOutput("Dangling or self dependency")
            if not research_mode and step in {"action", "evaluate"} and claim["kind"] in {"source_statement", "case_inference"}:
                expected = "solutions." if step == "action" else ("solutions.", "action.")
                if not any(dep.startswith(expected) for dep in claim["depends_on"]):
                    raise InvalidModelOutput("Action/outcome lacks a measure dependency")
            graph[cid] = claim["depends_on"]
        referenced = set()
        for row in value["coverage"].values():
            if not set(row["claim_ids"]) <= set(ids):
                raise InvalidModelOutput("Coverage references missing claim")
            if row["status"] in {"supported", "partial", "conflict"} and not row["claim_ids"]:
                raise InvalidModelOutput("Nonempty heading is not coverage")
            referenced.update(row["claim_ids"])
        # Research retains additional body text for the independent claim
        # audits; an omitted coverage link alone does not invalidate the step.
        if not research_mode and set(ids) - referenced:
            raise InvalidModelOutput("Unmapped claims")
        previous += ids
    visited = set()
    def visit(node, ancestors):
        if node in ancestors:
            raise InvalidModelOutput("Cyclic dependency")
        if node in visited:
            return
        for dep in graph[node]:
            visit(dep, ancestors | {node})
        visited.add(node)
    for node in graph:
        visit(node, set())


def block_claims(steps, blocked, reason):
    """Remove transitive dependents and downgrade every affected requirement."""
    result = deepcopy(steps)
    blocked = set(blocked)
    while True:
        extra = {cid for cid, claim in all_claims(result).items() if set(claim["depends_on"]) & blocked}
        if extra <= blocked:
            break
        blocked |= extra
    for value in result.values():
        value["claims"] = [c for c in value["claims"] if c["claim_id"] not in blocked]
        for row in value["coverage"].values():
            if set(row["claim_ids"]) & blocked:
                row["claim_ids"] = [cid for cid in row["claim_ids"] if cid not in blocked]
                row.update(status="partial" if row["claim_ids"] else "missing", reason=reason)
    return result, sorted(blocked)


def _render_claim(c, numbers):
    text = display_text(c["text"], numbers)
    if c["kind"] == "case_inference":
        text = "病例推断（需临床确认）：" + text
    if c["kind"] == "model_inference":
        text = "一般推断（待证据核实）：" + text
    if c["kind"] in {"source_statement", "case_inference", "model_inference"}:
        details = [label + display_text(c[key], numbers) for key, label in (("conditions", "适用条件："), ("rationale", "简要依据：")) if c[key]]
        if details:
            text += "（" + "；".join(details) + "）"
    refs = list(dict.fromkeys(numbers[eid] for eid in c["evidence_ids"] if eid in numbers))
    if refs:
        text += " [" + ", ".join(map(str, refs)) + "]"
    return text


def render_answer(result):
    lines = []
    numbers, references = numbered_references(result, STEPS)
    research = result.get("answer_mode") == "research_six_steps"
    if research and result.get("errors"):
        web_categories = {"search", "network", "configuration", "page_fetch_or_parse", "browser_reader", "fulltext_resolution"}
        reviewed_web_failure = (result.get("audit_summary", {}).get("three_checks_completed")
                                and all(error.get("category") in web_categories for error in result["errors"]))
        notice = ("联网补充资料出现异常；以下保留已生成并完成反思审查的研究回答。" if reviewed_web_failure else
                  "反思流程未完成，以下保留已生成的研究回答；仍需结合运行记录中的未解决问题复核。")
        lines.extend([notice, ""])
    for step, (title, names) in STEPS.items():
        lines.append(title)
        value = result["six_steps"][step]
        claims = {c["claim_id"]: c for c in value["claims"]}
        if research:
            # Coverage maps one claim to several requirements. It is review
            # metadata, not a request to repeat the complete answer each time.
            lines.extend("- " + _render_claim(c, numbers) for c in claims.values())
            gaps = [(names[key], row["reason"]) for key, row in value["coverage"].items()
                    if row["status"] in {"partial", "missing", "conflict"}]
            if gaps:
                lines.append("待补充与限制：")
                lines.extend(f"- {label}：{display_text(reason, numbers)}" for label, reason in gaps)
            lines.append("")
            continue
        displayed = set()
        for key, row in value["coverage"].items():
            texts = []
            for cid in row["claim_ids"]:
                texts.append(_render_claim(claims[cid], numbers))
                displayed.add(cid)
            state = {"supported": "已覆盖", "partial": "部分", "missing": "待补充",
                     "conflict": "存在分歧", "not_applicable": "不适用"}[row["status"]]
            lines.append(f"- {names[key]}（{state}）：" + ("；".join(texts) or display_text(row["reason"], numbers)))
        for cid, c in claims.items():
            if cid not in displayed:
                lines.append("- 补充内容：" + _render_claim(c, numbers))
        lines.append("")
    if references:
        lines.extend(["参考资料", ""])
        lines.extend(f"- [{number}] {reference_for_display(source)}" for number, source in references)
        lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)
