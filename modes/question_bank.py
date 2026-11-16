"""Versioned teaching content, separate from formal evaluation cases."""
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from .language import localized

BANK_PATH = Path(__file__).resolve().parents[1] / "data/teaching_bank/openrn.json"
RUBRIC_PATH = BANK_PATH.with_name("rubrics.json")
SPECIALTY_PATH = BANK_PATH.with_name("mdrpi_specialty.json")
PROJECT_KEY = "project_authored_source_grounded"
ADAPTED_KEY = "project_adapted_source_reference"
CATEGORIES = {
    "mdrpi": ("MDRPI 专项练习", "MDRPI focused practice"),
    "skin": ("压力性损伤与皮肤护理", "Pressure injury and skin care"),
    "devices": ("器械相关护理", "Device-related nursing"),
    "original": ("项目虚构练习", "Original fictional practice"),
}


def load_bank():
    from .standardized_cases import load_standardized_cases
    rubrics = load_rubrics()
    bank = {}
    for path in (BANK_PATH, SPECIALTY_PATH):
        for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
            if item["id"] in bank:
                raise ValueError("Duplicate teaching question ID: " + item["id"])
            rubric = rubrics[item["id"]]
            if rubric["basis_question"] != item["case"] or rubric["basis_answer"] != item["answer"]["en"]:
                raise ValueError("Teaching rubric does not match question/key: " + item["id"])
            bank[item["id"]] = {**item, "rubric": rubric}
    for case_id, item in load_standardized_cases().items():
        if case_id in bank:
            raise ValueError("Duplicate teaching question ID: " + case_id)
        bank[case_id] = item
    from .content_review import apply_review, load_reviews
    ledger = load_reviews()
    reviewed_bank = {}
    for key, item in bank.items():
        reviewed = apply_review(item, ledger=ledger)
        if reviewed.get("answer_origin") == ADAPTED_KEY:
            # Keep the publisher's unchanged key separately from the reviewed
            # teaching reference that includes project explanations/safety.
            reviewed["publisher_answer"] = deepcopy(item.get("answer", {}))
        reviewed_bank[key] = reviewed
    bank = reviewed_bank
    from .bank_expansion import approved_expansions, practice_item
    for item in approved_expansions():
        if item["id"] in bank:
            raise ValueError("Duplicate published review item ID: " + item["id"])
        bank[item["id"]] = practice_item(item)
    from .bank_expansion import attach_objective_rules, display_classification, load_objective_rules
    objective_rules = load_objective_rules()
    return {key:attach_objective_rules(display_classification(item), rules=objective_rules)
            for key,item in bank.items()}


@lru_cache(maxsize=1)
def load_rubrics():
    rubrics = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))["items"]
    for item in json.loads(SPECIALTY_PATH.read_text(encoding="utf-8"))["items"]:
        if item["id"] in rubrics:
            raise ValueError("Duplicate teaching rubric ID: " + item["id"])
        rubrics[item["id"]] = item["rubric"]
    return rubrics


def assessment_item(state):
    """Attach explicit contracts to unchanged old question snapshots only.

    Never rewrite an old case, answer, history or clinical rubric. Objective UI
    metadata and explicit term aliases can be read from the current reviewed
    item when the stored question and both reference languages are unchanged.
    """
    item = deepcopy(state.get("bank_item") or {})
    if not item.get("id"):
        return item
    from .teaching import CASES
    current = CASES.get(item["id"], {})
    if (current.get("case") != item.get("case") or current.get("case_zh") != item.get("case_zh")
            or current.get("answer") != item.get("answer")):
        return item
    objective_keys = ("question_type", "options", "options_en", "correct_options",
                      "correct_truth", "correct_order", "objective_rule_source")
    # Existing UI/key metadata is part of the question snapshot. Attach a
    # missing contract only when its existing fields agree with the source.
    objective_matches = all(key not in item or item[key] == current.get(key) for key in objective_keys)
    if objective_matches and current.get("question_type") in {"true_false", "ordering"}:
        from .bank_expansion import objective_rule_fingerprint
        candidate = {**item, **{key: current[key] for key in objective_keys if key not in item and key in current}}
        objective_matches = (bool(current.get("objective_rule_source"))
                             and objective_rule_fingerprint(candidate)
                             == current["objective_rule_source"].get("source_sha256"))
    for key in (*objective_keys, "summary_constraints"):
        if key not in item and key in current and (key == "summary_constraints" or objective_matches):
            item[key] = deepcopy(current[key])
    if item.get("rubric") and current.get("rubric"):
        latest = {row["id"]: row for row in current["rubric"]["required"]}
        for row in item["rubric"]["required"]:
            candidate = latest.get(row["id"], {})
            if all(row.get(key) == candidate.get(key) for key in ("criterion", "acceptable", "incorrect")):
                for key in ("accepted_terms", "term_slot"):
                    if key in candidate:
                        row[key] = deepcopy(candidate[key])
    return item


def question_rubric(state):
    item = assessment_item(state)
    from .rubric import objective_question_type, objective_rubric
    if objective_question_type(item) in {"true_false", "ordering"}:
        return objective_rubric(item, item.get("rubric"))
    if item.get("rubric"):
        return item["rubric"]
    # Old sessions keep their original question/key; attach only a rubric written
    # for that exact question/key, without rewriting the stored student history.
    candidate = load_rubrics().get(item.get("id"))
    if candidate and candidate["basis_question"] == item.get("case") and candidate["basis_answer"] == item.get("answer", {}).get("en"):
        return candidate
    return None


def rubric_context(rubric, language):
    result = {"version": rubric["version"], "required": [
        {"id": row["id"], **{key: row[key][language] for key in ("criterion", "acceptable", "incorrect")}}
        for row in rubric["required"]], "optional_extensions": rubric["optional_extensions"][language]}
    ids = {row["id"] for row in rubric["required"]}
    for source, target in zip(rubric["required"], result["required"]):
        for key in ("accepted_terms", "term_slot"):
            if source.get(key):
                target[key] = deepcopy(source[key])
    result["assessment_contract"] = {
        "scope": "Only these required criteria. Accept stated equivalents and authored aliases; optional detail cannot block completion.",
        "error": "Incorrect requires an explicit conflicting learner claim, not omitted explanation or an inferred claim.",
        "provenance": "Case facts, references and teacher hints are not learner statements; keep uncertainty and attempt order intact."}
    if rubric.get("misconceptions"):
        result["common_misconceptions"] = [
            {"id": row["id"], "criterion_id": row["criterion_id"],
             "statement": row["statement"][language], "feedback": row["feedback"][language]}
            for row in rubric["misconceptions"] if row["criterion_id"] in ids]
    return result


def bank_hint_plan(item, language, focus):
    """Authored hints target one outstanding criterion; old banks keep global hints."""
    pathways = item.get("hint_pathways", {})
    if pathways:
        target = focus if focus in pathways else next(iter(pathways))
        return {"hints": pathways[target][language]}, "project_authored_criterion_hints", target
    return {"hints": item["hints"][language]}, "textbook_adapted_hints", "all"


def objective_hint_cues(item, language, focus="all"):
    """Read authored hints in the same form used for display and accounting."""
    raw, origin, focus = bank_hint_plan(item, language, focus)
    cues = []
    for row in raw["hints"]:
        if not row.get("cue"):
            continue
        explanation = row.get("explanation", "").replace("请结合题面说明为何不能按该说法判断或操作。", "")
        explanation = explanation.replace("Explain why it should not guide judgment or action; ", "")
        cues.append(localized(language, "**线索：** ", "**Cue:** ") + row["cue"] +
                    ("\n\n" + localized(language, "**解释：** ", "**Explanation:** ") + explanation if explanation else ""))
    return cues, origin, focus


def bank_origin(item):
    if item.get("kind") == "standardized_case":
        return "standardized_source"
    if item.get("kind") == "fictional_case":
        return "practice_library"
    return "specialty_bank" if item.get("answer_origin") in {PROJECT_KEY, "ai_reviewed_source_grounded"} else "textbook_bank"


def reference_context(state):
    item = state.get("bank_item")
    if not item or not item.get("has_reference_answer", True):
        return None
    language = state.get("language", "zh")
    if item.get("answer_origin") == "ai_reviewed_source_grounded":
        return {"id": item["id"], "original_answer": item["answer"]["en"], "localized_answer": item["answer"][language],
                "answer_origin": item["answer_origin"], "review_status": item.get("review_status"),
                "source": item.get("review_references") or item["source"],
                "question_scope": "Use only this question/current step. AI-reviewed educational reference; case uncertainty and safe-treatment conditions remain binding."}
    result = {"id": item["id"], "question_scope": "Answer only the supplied question, not an entire care plan.",
            "original_answer": item["answer"]["en"], "localized_answer": item["answer"][language],
            "source": item["source"], "translation_note": "Chinese wording and hints are project adaptations; the English key is from the cited textbook."}
    if item.get("answer_origin") == PROJECT_KEY:
        result.update(answer_origin=PROJECT_KEY, scope=item["scope"],
                      translation_note="Both languages are project-authored practice guidance grounded in the cited sources, not a publisher's answer key or an independently expert-validated rubric.")
    elif item.get("answer_origin") == ADAPTED_KEY:
        publisher = item.get("publisher_answer") or {}
        result.update(answer_origin=ADAPTED_KEY,
                      original_answer=publisher.get("en"), original_localized_answer=publisher.get(language),
                      teaching_reference=item["answer"][language],
                      translation_note="The teaching reference combines the cited publisher's original answer with project-authored feature, purpose or safety supplements. The supplements, including English text, are not additions to the publisher's answer key.",
                      review_note=item.get("review_note", ""), reference_note=item.get("reference_note", ""),
                      supplement_sources=item.get("review_references") or item.get("content_review", {}).get("references", []))
    return result


def reference_answer_text(state):
    from .teaching_copy import clean_teaching_copy
    item = state.get("bank_item")
    if not item or not item.get("has_reference_answer", True):
        return ""
    language = state.get("language", "zh")
    source = item.get("source") or {}
    if item.get("answer_origin") == "ai_reviewed_source_grounded":
        references = item.get("review_references") or source["references"]
        links = "\n".join("- " + (ref["citation"] + f" [查看原文]({ref['url']})" if ref.get("citation") else f"[{ref['title']}]({ref['url']})") + " — " + ref["locator"] for ref in references)
        return clean_teaching_copy("\n\n".join([localized(language, "### 参考答案与要点", "### Reference answer and criteria"),
                            item["answer"][language], localized(language, "依据：", "Source basis:"), links]))
    if item.get("answer_origin") == PROJECT_KEY:
        parts = [localized(language, "### 专项练习参考要点", "### Reference points for this focused exercise"),
                 localized(language, "以下要点由本项目依据所列资料编写，与智能体反馈分开展示。",
                           "These project-authored points are grounded in the sources below and shown separately from AI feedback."),
                 item["answer"][language]]
        misconceptions = item["rubric"].get("misconceptions", [])
        if misconceptions:
            parts.append(localized(language, "**常见误解提醒**", "**Common misconceptions to consider**") + "\n\n" +
                         "\n".join("- " + row["feedback"][language] for row in misconceptions))
        links = [f"[{ref['title']}]({ref['url']}) — {ref['locator']}" for ref in source["references"]]
        parts.append(localized(language, "编写依据：", "Source basis: ") + "\n\n" + "\n".join("- " + link for link in links))
        return clean_teaching_copy("\n\n".join(parts))
    adapted = item.get("answer_origin") == ADAPTED_KEY
    title = (localized(language, "### 教材答案与项目补充要点", "### Textbook answer and project supplements") if adapted else
             localized(language, "### 教材参考答案", "### Textbook reference answer"))
    note = (localized(language, "以下教学参考包含教材原答案，以及本项目依据所列资料补充的特征、用途或安全要点；补充内容不属于出版原答案。",
                      "This teaching reference includes the original textbook answer and project supplements on features, purpose or safety grounded in the cited sources. The supplements are not part of the publisher's answer key.") if adapted else
            localized(language, "以下为原教材配套答案的中文译文；与上面的智能体反馈分开展示。", "The following is the textbook's supplied answer, shown separately from the AI feedback."))
    attribution = f"{source['book']} — {source['section']}, {source['question_number']}. {source['authors']}."
    links = (f"[{localized(language, '题目原文', 'Original question')}]({source['question_url']}) · "
             f"[{localized(language, '答案原文', 'Original answer')}]({source['answer_url']}) · "
             f"[{source['license']}]({source['license_url']})")
    parts = [title, note]
    publisher = item.get("publisher_answer", {}).get(language) if adapted else None
    if publisher:
        parts.append(localized(language, "**教材原答案（中文译文）**\n\n", "**Original textbook answer**\n\n") + publisher)
        parts.append(localized(language, "**教学参考与项目补充**\n\n", "**Teaching reference with project supplements**\n\n") + item["answer"][language])
    else:
        parts.append(item["answer"][language])
    parts.append(attribution + "\n\n" + links)
    notes = [item["reference_note"]] if item.get("reference_note") else []
    for authored_note in notes:
        if authored_note not in item["answer"][language]:
            parts.append(authored_note)
    if adapted:
        references = item.get("review_references") or item.get("content_review", {}).get("references", [])
        extra = [f"[{ref['title']}]({ref['url']}) — {ref.get('locator', '')}" for ref in references
                 if ref.get("url") and ref["url"] not in {source.get("question_url"), source.get("answer_url")}]
        if extra:
            parts.append(localized(language, "项目补充依据：", "Project supplement sources: ") + "\n\n" + "\n".join("- " + link for link in extra))
    return clean_teaching_copy("\n\n".join(parts))
