"""Check learner quotation provenance and record when actual hints were seen."""
from copy import deepcopy
import re
import unicodedata

from .language import localized
from .teaching import student_answers, is_student_message


def quote_catalog(answers):
    """Give the model stable choices; application code supplies the original text."""
    catalog = []
    for attempt, answer in enumerate(answers, 1):
        clauses = [part.strip() for part in re.split(r"(?<=[。！？；;\n])|(?<=[.!?])(?=\s|$)", answer) if part.strip()]
        if len(clauses) > 64:
            # Do not silently discard the tail of a long answer.
            clauses = clauses[:63] + [answer[answer.find(clauses[63]):].strip()]
        for index, text in enumerate(clauses, 1):
            catalog.append({"id": f"A{attempt}S{index}", "attempt": attempt, "text": text})
    return catalog


def resolve_evidence(raw, catalog):
    result = deepcopy(raw)
    lookup = {row["id"]: row for row in catalog}
    rows = list(result.get("criteria", {}).values()) if "criteria" in result else [
        row for kind in ("strengths", "improvements", "gaps", "material_errors")
        for row in result.get(kind, []) if isinstance(row, dict)]
    for row in rows:
        evidence = []
        for citation in row.get("evidence", []):
            original = lookup.get(citation.get("quote_id"))
            if original:
                evidence.append({"attempt": original["attempt"], "quote": original["text"]})
            else:
                evidence.append({"attempt": 0, "quote": "", "rejected_quote_id": citation.get("quote_id")})
        row["evidence"] = evidence
    result["evidence_method"] = "learner_sentence_ids"
    return result


def enforce_status_contract(raw, rubric, catalog, case, language):
    """Reject ungrounded error/scope claims without converting them into a pass.

    An incorrect decision must identify an actual learner claim and the exact
    scoped rule it conflicts with. This checks attribution and rule provenance,
    not whether a clinical interpretation is true. Raw outputs remain in logs.
    """
    result = deepcopy(raw)
    lookup = {row["id"]: row["text"] for row in catalog}
    attempts = {row["id"]: row["attempt"] for row in catalog}
    latest_attempt = max(attempts.values(), default=0)
    warnings = deepcopy(result.get("status_contract_warnings", []))
    for criterion in rubric["required"]:
        row = result.get("criteria", {}).get(criterion["id"])
        if not row or row.get("decision_origin") == "authored_choice_set":
            continue
        cited, cited_attempts = [], []
        for evidence in row.get("evidence", []):
            if evidence.get("quote_id") in lookup:
                cited.append(lookup[evidence["quote_id"]])
                cited_attempts.append(attempts[evidence["quote_id"]])
            elif evidence.get("quote") and any(
                    evidence.get("attempt") == clause["attempt"] and evidence["quote"] in clause["text"]
                    for clause in catalog):
                cited.append(evidence["quote"])
                cited_attempts.append(evidence["attempt"])
        if row.get("supersedes_previous") and latest_attempt not in cited_attempts:
            warnings.append({"criterion": criterion["id"], "reason": "supersession_without_current_learner_words",
                             "model_decision": deepcopy(row)})
            row["supersedes_previous"] = False
        reason = None
        remaining = row.get("remaining_requirement", "")
        if remaining and remaining not in criterion["criterion"][language]:
            # A model sometimes copies the exact authored requirement and
            # appends a neutral label or examples of equivalent wording. That
            # is a formatting violation, not evidence of a different task.
            # Do not accept arbitrary additions such as another required list.
            authored = criterion["criterion"][language]
            core = re.sub(r"\s*[（(]\s*(?:如|例如|比如|e\.g\.?|for example|such as)[\s\S]*[）)]\s*$", "", remaining, flags=re.I).strip()
            core = re.sub(r"\s*[,，]\s*(?:例如|比如|e\.g\.?|for example|such as)\b[\s\S]*$", "", core, flags=re.I).strip()
            neutral = {"", "的功能", "的用途", "的主要用途", "的含义", " function", " purpose", " meaning"}
            if any(core.casefold() == (authored + suffix).casefold() for suffix in neutral):
                row["remaining_requirement"] = authored
                warnings.append({"criterion": criterion["id"], "reason": "authored_requirement_example_normalized",
                                 "model_remaining_requirement": remaining})
            else:
                reason = "requirement_outside_current_criterion"
        if row["status"] == "incorrect":
            claim = row.get("error_quote", "").strip()
            rule = row.get("conflict_rule_quote", "").strip()
            rules = [criterion[key][language] for key in ("criterion", "acceptable", "incorrect")]
            if (row.get("decision_basis") != "explicit_conflict" or not claim
                    or not any(claim in text for text in cited) or not rule
                    or not any(rule in text for text in rules + [case])):
                reason = "unconfirmed_actual_conflicting_claim"
        if reason:
            warnings.append({"criterion": criterion["id"], "reason": reason, "model_decision": deepcopy(row)})
            row.update(status="uncertain", decision_origin="status_contract_guard",
                       explanation=localized(language, "判分依据未能与本项要求及实际学生原话核对，暂不计为实际错误。",
                                             "The decision could not be grounded in this criterion and the learner's words; no actual error is recorded."),
                       remaining_requirement="", error_quote="", conflict_rule_quote="", supersedes_previous=False)
    result["status_contract_warnings"] = warnings
    return result


def check_concept_name_scope(raw, item, rubric, catalog, language):
    """A closed glossary name is not an explanation of its separate feature.

    This guard recognizes only complete canonical-name clauses. It never marks
    a purpose/clinical explanation correct by keywords. Other substantive text
    remains semantic; a verifier must cite that text, not merely the term name.
    """
    result = deepcopy(raw)
    if item.get("kind") != "knowledge_card":
        return result
    named = next((row for row in rubric["required"] if row["criterion"]["en"].startswith("Identify the concept:")), None)
    if not named:
        return result
    aliases = list((item.get("publisher_answer") or {}).values())
    aliases += [value.split(":" if ":" in value else "：", 1)[-1]
                for value in named["criterion"].values()]
    compact = lambda value: re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()
    names = {compact(alias.strip("。.!！;； \t\n")) for alias in aliases if alias}

    def clause_kind(text):
        text = unicodedata.normalize("NFKC", text).strip()
        teacher = re.match(r"^(?:引用)?(?:老师|教师|提示)(?:说|写|是|提到|给出)?\s*[:：]?\s*|"
                           r"^(?:the\s+)?(?:teacher|tutor|hint)\s+(?:said|says|wrote|is|named)\s*[:：]?\s*", text, re.I)
        if teacher:
            text = text[teacher.end():]
        text = text.strip("。.!！;；:： \t\n\"'“”‘’`")
        text = re.sub(r"^(?:(?:我的)?(?:答案|概念|名称|名字)(?:是|为)?\s*[:：]?\s*|这是\s*|是\s*|"
                      r"(?:(?:my|the)\s+)?(?:answer|concept|name)\s+(?:is|was)\s+|it(?:\s+is|'s)\s+|this\s+is\s+|the\s+)", "", text, flags=re.I)
        text = re.sub(r"^the\s+", "", text, flags=re.I).strip("。.!！;；:： \t\n\"'“”‘’`")
        return ("teacher_name" if teacher else "name") if compact(text) in names else None

    kinds = {row["id"]: clause_kind(row["text"]) for row in catalog}
    all_names = bool(catalog) and all(kinds[row["id"]] or is_acknowledgement(row["text"]) for row in catalog)
    all_teacher = all_names and any(value == "teacher_name" for value in kinds.values()) and not any(value == "name" for value in kinds.values())
    for criterion in rubric["required"]:
        row = result.get("criteria", {}).get(criterion["id"])
        if not row:
            continue
        if criterion["id"] == named["id"]:
            if not all_teacher:
                continue
            missing = True
        else:
            cited = [kinds.get(e.get("quote_id")) for e in row.get("evidence", [])]
            if not cited and row.get("evidence"):
                cited = [clause_kind(e.get("quote", "")) for e in row["evidence"]]
            only_name_quotes = bool(cited) and all(cited)
            if not all_names and not (row.get("status") in {"met", "partial"} and only_name_quotes):
                continue
            missing = all_names
        row.update(status="missing" if missing else "uncertain", evidence=[],
                   decision_basis="not_found" if missing else "ambiguous",
                   decision_origin="concept_name_only_scope" if missing else "concept_name_quote_insufficient",
                   explanation=localized(language, "这里只给出了术语名称或引用的教师名称，没有用实际作答说明本项要求的含义或用途。",
                                         "Only a term name or quoted tutor name is supplied; no learner explanation of this required feature or purpose is available."))
        row.pop("error_quote", None)
        row.pop("conflict_rule_quote", None)
        if missing:
            row["remaining_requirement"] = criterion["criterion"][language]
        else:
            row.pop("remaining_requirement", None)
    return result


def check_authored_term_slots(raw, rubric, catalog, language):
    """Positive closed name/list slots with explicit authored bilingual aliases.

    Never applies aliases to care plans or reasoning. Negated, conditional,
    interrogative and quoted tutor clauses remain semantic model decisions.
    """
    result = deepcopy(raw)
    latest = max((row["attempt"] for row in catalog), default=0)
    for criterion in rubric["required"]:
        aliases = criterion.get("accepted_terms", [])
        if criterion.get("term_slot") != "availability_list" or not aliases:
            continue
        def alias_match(text):
            return any(re.search(r"(?<![A-Za-z])" + re.escape(alias) + r"(?![A-Za-z])", text, re.I) for alias in aliases)
        # A new denial/uncertainty about the named item cannot be overruled by
        # an old positive list; its meaning belongs to semantic assessment.
        if any(alias_match(c["text"]) and re.search(
                r"不|没有|无需|未|假如|如果|可能|考虑|\?|？|\b(?:not|no|without|if|maybe|might|consider)\b", c["text"], re.I)
               for c in catalog if c["attempt"] == latest):
            continue
        for clause in reversed(catalog):
            text = clause["text"]
            if not alias_match(text) or re.search(
                    r"不|没有|无需|未|假如|如果|可能|考虑|老师|教师|提示|引用|\?|？|[“\"]|"
                    r"\b(?:not|no|without|if|maybe|might|consider|teacher|tutor|hint)\b", text, re.I):
                continue
            # Lists in this supplies question and explicit availability wording
            # are closed naming slots. A long explanatory paragraph is not.
            if len(text) > 260 or not (re.search(r"备|准备|有|放|包括|用品|[、,，]|\b(?:have|available|supplies|include)\b", text, re.I)
                                      or any(text.strip("。.;； ").casefold() == alias.casefold() for alias in aliases)):
                continue
            result["criteria"][criterion["id"]] = {
                "status": "met", "evidence": [{"quote_id": clause["id"]}],
                "explanation": localized(language, "所列用品名称符合本题已发布的双语术语映射。",
                                         "The listed supply uses an accepted name in this question's published bilingual mapping."),
                "decision_origin": "authored_bilingual_term_slot"}
            break
    return result


def summary_boundaries(state):
    """Only version-bound exclusions authored for this exact bank item."""
    item = state.get("bank_item") or {}
    return deepcopy(item.get("summary_constraints") or item.get("rubric", {}).get("summary_constraints") or {})


def guard_teacher_explanation(text, state, language):
    """Do not publish an expansion that repeats an authored excluded direction."""
    boundaries = summary_boundaries(state)
    matched = []
    # These are authored exclusions for one question, never grading keywords.
    # A reminder saying not to perform the excluded assessment is consistent
    # with that boundary. Inspect each clause so an earlier negation cannot
    # hide a later positive recommendation.
    clauses = [part.strip() for part in re.split(r"(?<=[。.!?！？;；\n])|[，,]", text) if part.strip()]
    for phrase in boundaries.get("excluded_recommendations", []):
        if not phrase:
            continue
        for clause in clauses:
            occurrences = list(re.finditer(re.escape(phrase), clause, re.I))
            if not occurrences:
                continue
            for occurrence in occurrences:
                prefix = clause[:occurrence.start()]
                suffix = clause[occurrence.end():]
                if re.search(r"(?:不要|不应|不能|不得|无需|不需|不可|避免|禁止|不适用|不宜)[^。;；\n]{0,50}$|"
                             r"\b(?:do not|don't|must not|should not|never|avoid|not appropriate to)\b[^.!?;\n]{0,100}$|"
                             r"\bnot\s+(?:(?:be|been|being|to|a|an|as|automatically|directly)\s+)*"
                             r"(?:classified|classified as|classify|staged|stage|apply|applied|considered|assessed|recommended)\b[^.!?;\n]{0,80}$", prefix, re.I):
                    continue
                # Some exclusions include the action itself (e.g. "stage
                # mucosal findings"); its preceding "do not" still governs it.
                if re.search(r"\b(?:do not|don't|must not|should not|never|avoid|not)\s+"
                             r"(?:(?:a|an|as|to|be|been|being|automatically|directly)\s+)*$", prefix, re.I):
                    continue
                if re.match(r"\s*[^。;；\n]{0,15}(?:不适用|不用于|不能用于)", suffix):
                    continue
                suffix = suffix.lstrip(" \t\"'”’`")
                if re.match(r"(?:does not|do not|is not|are not|cannot|must not)\s+"
                            r"(?:apply|be applied|applicable|appropriate|valid|used)\b|"
                            r"(?:is|are|would be)\s+(?:incorrect|inappropriate|invalid|unsupported|not appropriate)\b", suffix, re.I):
                    continue
                matched.append(phrase)
                break
            if phrase in matched:
                break
    if re.search(r"学生(?:已|准确|正确|提到|指出|提出|表达|识别|理解|掌握|补充|回答|说)|"
                 r"(?:后续|第二次|再次)(?:的)?回答(?:中|后)?(?:补充|完善|说明)|"
                 r"\b(?:the (?:student|learner)|you)\s+(?:has |have |had |already )?"
                 r"(?:said|stated|identified|demonstrated|understood|mastered|improved|added|corrected)\b", text, re.I):
        matched.append("unverified_learner_attribution_in_teacher_prose")
    if not matched:
        return text, []
    from .question_bank import reference_answer_text
    guidance = boundaries.get("guidance", {})
    message = localized(language, "本次扩展讲解与本题公开边界不一致，已保留作答记录与参考要点。",
                        "The generated expansion conflicted with this question's published boundary. Your answer record and reference points are retained.")
    # This affects prose only. It never changes criteria, advancement or marks.
    return "\n\n".join(part for part in [message, guidance.get(language, ""), reference_answer_text(state)] if part), matched


def check_named_concept(raw, item, rubric, catalog, language):
    """Positive canonical-name check for authored glossary cards only.

    This closed answer slot uses the paired textbook term; it never grades a
    care plan, purpose, diagnosis or clinical reasoning by keyword presence.
    Denials, questions, quoted tutor words and conflicting later naming are
    left to the model, rather than being marked correct by this check.
    """
    result = deepcopy(raw)
    if item.get("kind") != "knowledge_card":
        return result, False
    criterion = next((row for row in rubric["required"] if row["criterion"]["en"].startswith("Identify the concept:")), None)
    if not criterion or criterion["id"] not in result.get("criteria", {}):
        return result, False
    row = result["criteria"][criterion["id"]]
    lookup = {clause["id"]: clause for clause in catalog}
    cited_attempts = [lookup[e["quote_id"]]["attempt"] for e in row.get("evidence", []) if e.get("quote_id") in lookup]
    aliases = list((item.get("publisher_answer") or item.get("answer", {})).values())
    latest_attempt = max((clause["attempt"] for clause in catalog), default=0)
    patterns = [r"(?<![A-Za-z])" + r"\s*".join(re.escape(part) for part in unicodedata.normalize("NFKC", alias).split()) + r"(?![A-Za-z])"
                for alias in aliases]
    # A later question/denial about the same authored name makes the closed
    # positive check inapplicable. Let semantic assessment resolve it; an old
    # matching name must not overrule the learner's latest wording.
    for clause in catalog:
        if clause["attempt"] != latest_attempt:
            continue
        text = unicodedata.normalize("NFKC", clause["text"])
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match and (text.rstrip().endswith(("?", "？")) or re.search(
                    r"\b(?:not|no|isn't|isnt|if|maybe|might|tutor|teacher)\b|不|非|如果|可能|老师|教师|提示|引用", text[:match.start()], re.I)):
                return result, False
    for clause in reversed(catalog):
        if row["status"] == "incorrect" and cited_attempts and clause["attempt"] < max(cited_attempts):
            continue
        text = unicodedata.normalize("NFKC", clause["text"])
        if text.rstrip().endswith(("?", "？")):
            continue
        if clause["attempt"] < latest_attempt and any(re.search(r"(?:correct|correction|更正|纠正|改正|说错).*(?:named|called|name is|it is the|名称|名字|叫|应为|应该是)", c["text"], re.I)
                                                       for c in catalog if c["attempt"] == latest_attempt):
            continue
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if not match:
                continue
            prefix = text[:match.start()]
            if re.search(r"\b(?:not|no|isn't|isnt|if|maybe|might|tutor|teacher)\b|不|非|如果|可能|老师|教师|提示|引用", prefix, re.I):
                continue
            result["criteria"][criterion["id"]] = {
                "status": "met", "evidence": [{"quote_id": clause["id"]}],
                "explanation": localized(language, "你已给出对应概念名称；用途在另一要点核对。",
                                         "You have named the corresponding concept; its purpose is checked separately."),
                "decision_origin": "paired_textbook_term"}
            return result, row["status"] != "met"
    return result, False


def is_acknowledgement(answer):
    """A standalone conversational acknowledgement supplies no subject answer.

    Match the whole utterance: a substantive explanation after agreement must
    still be assessed, and earlier substantive attempts remain available.
    """
    return bool(re.fullmatch(
        r"(?:yes|ok|okay|i agree(?: with (?:you|that|what you said))?|understood|got it|thanks|"
        r"是的|好的|好|嗯|明白了|知道了|我同意(?:你说的|你的说法|你的观点)?|对|谢谢)[。.!！?？\s]*",
        answer.strip(), re.I))


def checked_quotes(citations, answers):
    accepted, rejected = [], []
    for citation in citations:
        attempt, quote = citation.get("attempt"), citation.get("quote", "")
        if isinstance(attempt, int) and not isinstance(attempt, bool) and isinstance(quote, str):
            quote = quote.strip()
            acknowledgement = is_acknowledgement(answers[attempt - 1]) if 1 <= attempt <= len(answers) else False
            if 1 <= attempt <= len(answers) and quote and quote in answers[attempt - 1] and not acknowledgement:
                accepted.append({"attempt": attempt, "quote": quote})
                continue
        rejected.append(citation)
    return accepted, rejected


def hint_events(state):
    if state.get("hint_events"):
        return deepcopy(state["hint_events"])
    # Old/synthetic records can contain an explicitly typed hint without the
    # new event field. Never treat summaries, case bodies or feedback as hints.
    events, answer_count = [], 0
    for message in state.get("messages", []):
        if is_student_message(message, state):
            answer_count += 1
        elif message.get("role") == "assistant" and (message.get("kind") == "hint" or
                message.get("kind") == "help" and re.match(r"^\s*\*\*(?:提示|Hint)\*\*", message.get("content", ""), re.I)):
            events.append({"answer_count": answer_count, "focus": "all", "text": message["content"], "origin": "saved_typed_hint"})
    return events


def annotate_review(review, state, *, origin="answer"):
    result = deepcopy(review)
    count = len(student_answers(state))
    result.update(answer_count=count, review_origin=origin)
    hints = hint_events(state)
    references = state.get("reference_views") or []
    explanations = state.get("teaching_dialogue_events") or []
    events = hints + references + explanations
    previous = {row["id"]: row for row in state.get("reviews", [{}])[-1].get("criterion_checks", [])} if state.get("reviews") else {}
    for row in result.get("criterion_checks", []):
        relevant = [event for event in events if event.get("focus") in {row["id"], "all", "none"}]
        supported = bool(row.get("evidence")) and row["status"] in {"met", "partial", "incorrect"}
        assisted = supported and any(evidence["attempt"] > event["answer_count"]
                                     for event in relevant for evidence in row["evidence"])
        # Preserve the observable fact that this point had already been
        # covered before a later hint; later quoted additions do not erase it.
        before = previous.get(row["id"])
        if before and before["status"] == "met" and before.get("attribution") == "student":
            assisted = False
        row["attribution"] = "teacher_help" if assisted else "student" if supported else "not_demonstrated"
        after_reference = assisted and any(evidence["attempt"] > event["answer_count"]
            for event in references if event in relevant for evidence in row["evidence"])
        after_explanation = assisted and any(evidence["attempt"] > event["answer_count"]
            for event in explanations if event in relevant for evidence in row["evidence"])
        row["answer_context"] = ("after_reference" if after_reference else "after_explanation" if after_explanation
                                 else "after_hint" if assisted else "without_requested_hint" if supported else "not_demonstrated")
        if origin == "recheck":
            row["newly_demonstrated"] = False
            row["judgment_revised"] = bool(before and before["status"] != row["status"])
    if origin == "recheck":
        result["improvements"] = []
        result["claim_checks"] = [row for row in result.get("claim_checks", []) if row["kind"] != "improvements"]
    for row in result.get("claim_checks", []):
        relevant = [event for event in events if event.get("focus") in {row["dimension"], "all", "none"}]
        assisted = any(evidence["attempt"] > event["answer_count"] for event in relevant for evidence in row["evidence"])
        row["attribution"] = "teacher_help" if assisted else "student" if row["evidence"] else "not_demonstrated"
        after_reference = assisted and any(evidence["attempt"] > event["answer_count"]
            for event in references if event in relevant for evidence in row["evidence"])
        after_explanation = assisted and any(evidence["attempt"] > event["answer_count"]
            for event in explanations if event in relevant for evidence in row["evidence"])
        row["answer_context"] = ("after_reference" if after_reference else "after_explanation" if after_explanation
                                 else "after_hint" if assisted else "without_requested_hint" if row["evidence"] else "not_demonstrated")
    result["assistance"] = {"hints_seen": len(hints), "reference_answers_seen": len(references), "explanations_seen": len(explanations),
        "covered_without_requested_hint": sum(row["status"] == "met" and row["attribution"] == "student" for row in result.get("criterion_checks", [])),
        "covered_after_hint": sum(row["status"] == "met" and row["attribution"] == "teacher_help" for row in result.get("criterion_checks", []))}
    return result


def ground_formative_review(raw, case, answers, previous_reviews, language):
    """Ground claims for free cases without inventing a fixed clinical answer key.

    Verbatim case quotations constrain the requested task; their clinical
    interpretation still belongs to the model, not a keyword grading rule.
    """
    from .teaching import normalize_review
    if not any(isinstance(item, dict) for key in ("strengths", "improvements", "gaps", "material_errors") for item in raw.get(key, [])):
        # Old/control backends retain their contract. The production schema
        # always supplies structured claims, even when all arrays are empty.
        if not raw.get("quote_backed"):
            return normalize_review(raw, language)
    result = {**raw, "grading_mode": "case_formative", "evidence_warnings": [], "claim_checks": []}
    for kind in ("strengths", "improvements", "gaps", "material_errors"):
        result[kind] = []
        for claim in raw.get(kind, []):
            evidence, rejected = checked_quotes(claim.get("evidence", []), answers)
            task_quote = claim.get("task_quote", "").strip()
            needs_quote = kind != "gaps"
            if rejected or (needs_quote and not evidence) or (kind == "gaps" and (not task_quote or task_quote not in case)):
                result["evidence_warnings"].append({"kind": kind, "claim": claim, "reason": "unconfirmed_quote_or_question_scope"})
                continue
            if kind == "improvements" and (not previous_reviews or not any(e["attempt"] == len(answers) for e in evidence)):
                continue
            row = {"kind": kind, "text": claim["text"], "dimension": claim["dimension"],
                   "evidence": evidence, "task_quote": task_quote}
            result["claim_checks"].append(row)
            result[kind].append(claim["text"])
    if result["evidence_warnings"]:
        result["ready_for_summary"] = False
        result["feedback"] = localized(language,
            "这次评价有内容尚未与原话核对清楚。你可以补充说明，或点击“重新核对”让我重读已有回答。",
            "Some judgements could not be linked to the original wording. You can clarify your reasoning or choose Recheck to review your existing answers.")
    result["ready_for_summary"] = bool(result["ready_for_summary"] and not result["material_errors"] and not result["gaps"])
    result["language"] = language
    result["answer_count"] = len(answers)
    return result


def formative_text(review, language):
    parts = [localized(language, "**本次回答反馈**", "**Feedback on your answer**")]
    for kind, zh, en in [("strengths", "已经做到", "What your answer shows"),
                         ("improvements", "这次补充", "What your supplement adds"),
                         ("material_errors", "需要纠正", "Corrections"), ("gaps", "还可完善", "Remaining points")]:
        rows = [row for row in review.get("claim_checks", []) if row["kind"] == kind]
        if not rows:
            continue
        parts.append("**" + localized(language, zh, en) + "**")
        for row in rows:
            text = row["text"]
            for evidence in row["evidence"]:
                text += "\n\n" + localized(language, f"第{evidence['attempt']}次作答原话：", f"Your words in answer {evidence['attempt']}: ")
                text += "\n> " + evidence["quote"].replace("\n", "\n> ")
            if row.get("attribution") == "teacher_help":
                context = row.get("answer_context")
                text += "\n\n" + (localized(language, "这段作答在查看参考答案后提交。", "This answer was submitted after viewing the reference answer.")
                    if context == "after_reference" else localized(language, "这段作答在教学讲解后提交；讲解不占提示次数。", "This answer was submitted after a teaching explanation, which does not use a hint opportunity.")
                    if context == "after_explanation" else localized(language, "这段作答在查看对应提示后提交。", "This answer was submitted after viewing a hint for this point."))
            if kind == "gaps" and row["task_quote"]:
                text += "\n\n" + localized(language, "对应题目要求：", "Question wording: ") + row["task_quote"]
            parts.append(text)
    if review.get("evidence_warnings"):
        parts.append(localized(language, "有评价未能定位到对应原话，暂不计为你的错误或进步，可重新核对。",
                               "Some judgements could not be located in the original wording and are not counted as errors or progress. You can request a recheck."))
    if review.get("ready_for_summary"):
        parts.append(localized(language, "当前问题的主要要求已覆盖，可以进入总结。", "Your answer covers the main requirements; we can summarize."))
    assistance = review.get("assistance", {})
    if assistance.get("hints_seen"):
        parts.append(localized(language, f"本题已查看 {assistance['hints_seen']} 条提示；提示使用与作答记录分别保存。",
                               f"You have viewed {assistance['hints_seen']} hints; hint use is recorded separately from your answers."))
    if assistance.get("explanations_seen"):
        parts.append(localized(language, f"本题已获得 {assistance['explanations_seen']} 次教学讲解，不占用提示机会。",
                               f"You have received {assistance['explanations_seen']} teaching explanations, without using hint opportunities."))
    return "\n\n".join(parts)
