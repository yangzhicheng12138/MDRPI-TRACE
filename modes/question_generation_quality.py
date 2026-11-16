"""Explainable, local-only review hints for generated development questions.

This module does not decide clinical correctness or reject a question. Text
overlap is a candidate for independent review, not a semantic duplicate verdict.
It deliberately excludes the fixed case stem from duplicate similarity.

Case provenance checks need ``item_plan['materials']`` with ``id/case_facts``.
Only explicit case-attributed new abbreviations are surfaced; this is neither a
diagnosis vocabulary nor an attempt to infer disease truth from missing sources.
"""
from difflib import SequenceMatcher
import re
import unicodedata


_HIGH_TASK = .88
_JOINT_TASK = .65
_JOINT_ANSWER = .82

# These are educational task distinctions, not a medical diagnosis blacklist.
# They prevent a shared question template from making "assess" and "intervene"
# duplicates merely because the surrounding case words are identical.
_TASK_GOALS = {
    "assessment": r"评估目标|评估内容|评估项目|补充评估|评估什么|assessment (?:targets?|goals?|findings?)|what.{0,15}assess",
    "intervention": r"护理措施|干预措施|干预方案|护理方案|制定方案|采取什么措施|care plan|interventions?|nursing actions?",
    "priority": r"优先问题|优先处理|问题排序|首要问题|priority|prioriti[sz]",
    "classification": r"如何分期|判断分期|损伤分期|属于哪一期|分类依据|classif|staging",
    "mechanism": r"解释.{0,15}(?:机制|关系|原因)|发生机制|受力机制|explain.{0,20}(?:mechanism|relationship|cause)",
    "reassessment": r"复评指标|复评时间|复查指标|评价结果|评价疗效|reassessment|evaluate.{0,15}outcomes?",
}

_PATIENT_ROLE = re.compile(r"患者|患儿|病人|新生儿|早产儿|婴儿|\b(?:patient|child|infant|neonate|newborn)\b", re.I)
_PATIENT_ASSERTION = re.compile(
    r"本例|该(?:名|位)?(?:患者|患儿|早产儿|新生儿|婴儿)|此(?:名|位)?(?:患者|患儿)|"
    r"(?:患者|患儿|早产儿|新生儿|婴儿).{0,8}(?:由于|因为|因|患有|已确诊|已诊断|合并|伴有)|"
    r"(?:因为|由于)(?:本例|该|此|这名|这位)?(?:患者|患儿|病人|早产儿|新生儿|婴儿)|"
    r"\b(?:this|our) (?:patient|child|infant|neonate)|"
    r"\bthe (?:patient|child|infant|neonate).{0,20}(?:has|had|diagnosed|due to|treated for)\b", re.I)

# Language-level scope markers keep teaching suggestions, differential diagnoses
# and explicitly uncertain/negated examples out of patient-fact assertions.
_NON_ASSERTION = re.compile(
    r"可能|假设|推测|尚未确诊|未确诊|未明确|未确定|待(?:评估|核实|确认)|"
    r"未(?:提供|提及|说明|给出|记录)|没有提供|资料不足|不能|不应|不可|不宜|不要|并非|"
    r"不意味着|不代表|未必|是否|鉴别|排查|建议|应当|应该|需(?:要)?评估|"
    r"应(?:定期|及时|进一步)?(?:评估|复核|记录|检查|询问|监测|避免|遵循|维持|保持)|"
    r"若|如果|例如|比如|对于|一般而言|预防|以防|风险|"
    r"\b(?:may|might|could|possible|possibly|suspect|hypothes\w*|uncertain|"
    r"not|cannot|can't|should|consider|whether|differential|if|unless|"
    r"for example|such as|in general|risk|prevention|assess for)\b", re.I)

_CASE_QUOTE = re.compile(
    r"(?:题面|题干|病例(?:原文|资料)?|案例(?:原文|资料)?)"
    r"(?:中)?\s*(?:已|明确|确实)?(?:提示|指出|写明|写道|描述|说明|记录|提供|显示|提到|给出)"
    r"(?:了|为)?\s*[:：]?\s*"
    r"(?:“(?P<double>[^”\n]{2,140})”|‘(?P<single>[^’\n]{2,140})’|"
    r"\"(?P<ascii_double>[^\"\n]{2,140})\"|'(?P<ascii_single>[^'\n]{2,140})')")
_NON_ASSERTED_ATTRIBUTION = re.compile(r"如果|假如|假设|若|不能|不应|不可|不要|并非|不是|没有|未曾|无法")
_CAUSAL_PATIENT = re.compile(r"(?:因为|由于)(?:本例|该|此|这名|这位)?(?:患者|患儿|病人|早产儿|新生儿|婴儿)")
_CAUSAL_QUALIFIER = re.compile(r"可能|也许|或许|若|如果|假如|假设|推测|未确诊|尚未确诊|是否|排查|"
                             r"不应把|不能把|不得把|不能认定|不应认定")


def _text(value):
    return value if isinstance(value, str) else ""


def _normalize(value):
    return "".join(character for character in unicodedata.normalize("NFKC", _text(value)).casefold()
                   if character.isalnum())


def _remove_citations(text, question):
    for evidence_id in question.get("evidence_ids", []):
        if isinstance(evidence_id, str) and evidence_id:
            text = text.replace(evidence_id, "")
    return text


def _answer_parts(question):
    # Equivalent alternatives are a set, not an ordering contract. Sorting avoids
    # disguising duplicate content by changing rubric/alternative order.
    answer = _remove_citations(_text(question.get("reference_answer")), question)
    criteria = []
    for point in question.get("scoring_points", []):
        if not isinstance(point, dict):
            continue
        value = _text(point.get("criterion"))
        alternatives = sorted(_text(row) for row in point.get("acceptable_answers", []) if isinstance(row, str))
        criteria.append(value + " " + " ".join(alternatives))
    return _normalize(answer), _normalize(" ".join(sorted(criteria)))


def _similarity(left, right):
    if not left or not right:
        return 0.
    if left == right:
        return 1.
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _answer_similarity(left, right):
    answer, points = left
    older_answer, older_points = right
    if answer and older_answer:
        score = _similarity(answer, older_answer)
        # Generic repeated rubrics must not drown out two clinically different
        # reference answers. The separate similarities are lexical only.
        return .7 * score + .3 * _similarity(points, older_points) if points and older_points else score
    return _similarity(points, older_points)


def _goals(task):
    return {name for name, pattern in _TASK_GOALS.items() if re.search(pattern, task, re.I)}


def _previous(value, number):
    if not isinstance(value, dict):
        return None, None
    question = value.get("question") if isinstance(value.get("question"), dict) else value
    identity = value.get("item_id") or value.get("id") or question.get("item_id") or f"previous_{number}"
    return str(identity), question


def _duplicate_candidates(question, previous_questions):
    task_raw = _text(question.get("task"))
    task, answer = _normalize(task_raw), _answer_parts(question)
    goals = _goals(task_raw)
    candidates, seen = [], set()
    for number, previous in enumerate(previous_questions or (), 1):
        item_id, older = _previous(previous, number)
        if older is None or item_id in seen:
            continue
        previous_task = _normalize(older.get("task"))
        if not task or not previous_task:
            continue
        previous_goals = _goals(_text(older.get("task")))
        if goals and previous_goals and not goals & previous_goals:
            # Explicitly different educational tasks can share almost all of
            # their wording and still be valid questions over one fixed case.
            continue
        task_score = _similarity(task, previous_task)
        answer_score = _answer_similarity(answer, _answer_parts(older))
        short_task = min(len(task), len(previous_task)) < 12
        high_task_overlap = task_score >= _HIGH_TASK
        joint_overlap = task_score >= _JOINT_TASK and answer_score >= _JOINT_ANSWER
        if not high_task_overlap and not joint_overlap:
            continue
        score = task_score if high_task_overlap else .65 * task_score + .35 * answer_score
        if answer_score < .5:
            comparison = "答案要点文字差异较大，可能是同一任务的不同观点或评分冲突，请核对后决定。"
        else:
            comparison = "答案要点文字也有重合，但文字相近不证明结论一致，需核对否定、条件和考点。"
        candidates.append({"item_id": item_id, "similarity": round(score, 6),
            "reason": f"作答任务相似度{task_score:.3f}，答案要点相似度{answer_score:.3f}；"
                      + comparison
                      + ("任务文字较短，不能仅凭共用问句认定重复，需结合选项与考点复核。" if short_task else "")
                      + "固定病例正文未计入相似度；这只是重复/覆盖重叠候选，不自动拒题。"})
        seen.add(item_id)
    return sorted(candidates, key=lambda row: (-row["similarity"], row["item_id"]))


def _abbreviations(text):
    # Generic typography, not a list of disease labels. Numbers, one-letter
    # choice labels and mixed-case clinical measures are deliberately excluded.
    return {match.group() for match in re.finditer(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9]{1,11}(?![A-Za-z0-9_])", text)
            if sum(character.isalpha() for character in match.group()) >= 2}


def _clauses(text):
    # A later recommendation must not hide an earlier asserted patient fact in
    # the same sentence ("this infant has XYZ, assess the skin next").
    return [part.strip() for part in re.split(r"[。！？!?；;，,\n]|<br\s*/?>", text) if part.strip()]


def _negative_abbreviation(clause, abbreviation):
    pattern = (r"(?:无(?:明显|相关|已知)?|没有|未见|否认|不是|不属于|排除|"
               r"\bno\s+|\bwithout\s+)\s*" + re.escape(abbreviation))
    return bool(re.search(pattern, clause, re.I))


def _answer_fields(question):
    result = [("task", _text(question.get("task"))),
              ("reference_answer", _text(question.get("reference_answer")))]
    for number, point in enumerate(question.get("scoring_points", [])):
        if not isinstance(point, dict):
            continue
        result.append((f"scoring_points[{number}].criterion", _text(point.get("criterion"))))
        for index, value in enumerate(point.get("acceptable_answers", [])):
            result.append((f"scoring_points[{number}].acceptable_answers[{index}]", _text(value)))
    correct = set(question.get("correct_options", []))
    for option in question.get("options", []):
        if isinstance(option, dict) and option.get("label") in correct:
            result.append(("options." + option["label"], _text(option.get("text"))))
    return result


def _case_quote_checks(location, text, source_cases):
    normalized_sources = [_normalize(facts) for _, facts in source_cases]
    checks = []
    for match in _CASE_QUOTE.finditer(text):
        # Check only the lead-in, not a later qualification. For example,
        # "题面提示‘尺寸需核对’但未明确参数" still asserts the quoted premise.
        lead = re.split(r"[。！？!?；;，,\n（(]", text[:match.start()])[-1]
        if _NON_ASSERTED_ATTRIBUTION.search(lead):
            continue
        quote = next(value for value in match.groupdict().values() if value is not None)
        normalized_quote = _normalize(quote)
        if not normalized_quote or any(normalized_quote in source for source in normalized_sources):
            continue
        checks.append({"kind": "case_fact_provenance", "location": location, "text": match.group(),
            "reason": f"此处把“{quote}”归为题面/病例提示，但未在本题所选原病例中找到该短句；"
                      "请核查是已有事实的合理概括还是新增了患者事实。非逐字复述不等于幻觉或医学错误，"
                      "此提示不自动拒题，也不要求删除合理建议。"})
    return checks


def _provenance_checks(question, item_plan):
    if not isinstance(item_plan, dict):
        return []
    materials = item_plan.get("materials", [])
    source_cases = [(str(row.get("id", "source")), row["case_facts"])
                    for row in materials if isinstance(row, dict)
                    and isinstance(row.get("case_facts"), str) and row["case_facts"].strip()]
    if not source_cases:
        return []
    stem = _text(question.get("stem"))
    matching = [(identity, facts) for identity, facts in source_cases if _normalize(facts) == _normalize(stem)]
    if not matching:
        return [{"kind": "case_fact_provenance", "location": "stem", "text": stem,
            "reason": "题干未与任一所选病例正文匹配，请核对病例事实来源；这里只报告来源差异，"
                      "不判断新增内容的医学真伪，也不能替代现有固定题干守卫。"}]
    # Use the selected case alone. A diagnosis occurring only in another
    # selected material must not become a known fact of this patient.
    known = _abbreviations(" ".join(facts for _, facts in matching))
    checks, seen = [], set()
    for location, text in _answer_fields(question):
        clean = _remove_citations(text, question)
        checks.extend(_case_quote_checks(location, clean, matching))
        for clause in _clauses(clean):
            causal = _CAUSAL_PATIENT.search(clause)
            if causal and causal.start() and not _CAUSAL_QUALIFIER.search(clause[:causal.start()]):
                # A recommendation can have a separately asserted factual
                # rationale: "不应延误支持，因为早产儿需...治疗XYZ". Keep
                # uncertainty/condition markers that scope over the cause, but
                # do not let the recommendation's "不应" hide that rationale.
                clause = clause[causal.start():]
            if (not _PATIENT_ROLE.search(clause) or not _PATIENT_ASSERTION.search(clause)
                    or _NON_ASSERTION.search(clause)):
                continue
            for abbreviation in sorted(_abbreviations(clause) - known):
                if abbreviation in seen or _negative_abbreviation(clause, abbreviation):
                    continue
                checks.append({"kind": "case_fact_provenance", "location": location, "text": clause,
                    "reason": f"此句把新增简称“{abbreviation}”与本例患者联系，但该简称未见于所选病例正文；"
                              "请核对它是否是已有病名的等价简称、有依据的解释，或被新增为该患者的已知诊断/病史。"
                              "简称缺失不等于医学错误；不要仅因此扣分或要求删除正确内容。"})
                seen.add(abbreviation)
    return checks


def assess_question_quality(question, previous_questions, item_plan=None):
    """Return local review hints; never infer an approval or medical verdict.

    ``previous_questions`` accepts item rows with ``item_id/question`` or plain
    question packages. ``item_plan.materials`` supplies immutable case sources.
    This lexical screen cannot establish semantic uniqueness or comprehensively
    recognize newly invented clinical facts; the independent reviewer owns both.
    """
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    if previous_questions is not None and not isinstance(previous_questions, (list, tuple)):
        raise ValueError("previous_questions must be a list of question packages")
    return {"duplicate_candidates": _duplicate_candidates(question, previous_questions),
            "review_checks": _provenance_checks(question, item_plan)}
