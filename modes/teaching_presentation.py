"""Read-only learner evidence annotations and compact teaching feedback.

Only saved judgements and exact, attempt-bound learner quotations may produce
highlights. Display fields never replace the original dialogue or its evidence.
"""
from copy import deepcopy
import re

from .language import localized
from .teaching import help_intent, is_student_message, student_answers
from .teaching_router import fast_teaching_route


_STATUSES = {"met", "partial", "missing", "incorrect", "uncertain"}
_QUOTE_HEADER = re.compile(
    r"^(?:\*\*)?(?:(?:当前步骤|实际)?第\s*\d+\s*次作答(?:原话)?|"
    r"Your words in (?:this step's )?answer\s+\d+|Actual answer\s+\d+)\s*[:：]?", re.I)
_FEEDBACK_TITLE = re.compile(
    r"^\s*\*\*(?:本题逐项核对|本次回答反馈|Check against this question|Feedback on your answer)\*\*", re.I | re.M)
_ERROR_COMPARISON = re.compile(
    r"(?:请对照实际原话与本项规则[:：]|Compare your actual words with this criterion's rule:)"
    r"[ \t]*\n(?P<claim>>[^\n]*(?:\n>[^\n]*)*)", re.I)


def _label(check, language):
    value = check.get("criterion") or check.get("label") or ""
    return (value.get(language) or value.get("zh") or value.get("en") or "") if isinstance(value, dict) else str(value)


def _utf16(text, index):
    # JavaScript string offsets count UTF-16 units; an emoji occupies two.
    return len(text[:index].encode("utf-16-le", errors="surrogatepass")) // 2


def _message_identity(left, right):
    if (left.get("role") != right.get("role") or left.get("kind") != right.get("kind")
            or left.get("content") != right.get("content")):
        return False
    if (left.get("at") or right.get("at")) and left.get("at") != right.get("at"):
        return False
    if ("turn_id" in left or "turn_id" in right) and left.get("turn_id") != right.get("turn_id"):
        return False
    return True


def _message_map(record, messages, cursor):
    """Bind each round in display order, including repeated identical answers."""
    mapping = {}
    for source_index, original in enumerate(record.get("messages") or []):
        for target_index in range(cursor, len(messages)):
            if _message_identity(original, messages[target_index]):
                mapping[source_index] = target_index
                cursor = target_index + 1
                break
    return mapping, cursor


def _answer_rows(record, mapping):
    answers = student_answers(record)
    candidates = [(index, message) for index, message in enumerate(record.get("messages") or [])
                  if is_student_message(message, record)]
    # Missing or mismatched history is not a reason to shift a quotation to a
    # later identical answer. Keep the saved attempt positions and skip it.
    if len(candidates) != len(answers):
        return [{"text": text, "attempt": index} for index, text in enumerate(answers, 1)]
    rows = []
    for attempt, (text, candidate) in enumerate(zip(answers, candidates), 1):
        source_index, message = candidate
        row = {"text": text, "attempt": attempt}
        actual = message.get("answer_text", message.get("content", ""))
        content = message.get("content", "")
        if actual == text and source_index in mapping:
            row.update(source_index=source_index, message_index=mapping[source_index], message_at=message.get("at", ""))
            # Preserve historical attempt numbering, but never paint an old
            # mistakenly submitted standalone help request as a correct answer.
            if fast_teaching_route(text) is not None:
                row["help_control"] = True
            elif content == text:
                row["content_start"] = 0
            elif isinstance(content, str) and text and content.count(text) == 1:
                row["content_start"] = content.index(text)
        rows.append(row)
    return rows


def _scopes(record, answer_rows):
    """Map clinical local attempts only when their full ordered history agrees."""
    flow = record.get("clinical_flow") or {}
    if not flow:
        return {"": {"answers": answer_rows, "offset": 0}}, False
    steps = flow.get("steps") or []
    expected = [text for step in steps for text in step.get("answers") or []]
    exact = expected == [row["text"] for row in answer_rows]
    scopes, offset = {}, 0
    for step in steps:
        texts = step.get("answers") or []
        rows = ([{**row, "attempt": index} for index, row in enumerate(answer_rows[offset:offset + len(texts)], 1)]
                if exact else [{"text": text, "attempt": index} for index, text in enumerate(texts, 1)])
        scopes[step.get("id", "")] = {"answers": rows, "offset": offset, "step": step}
        offset += len(texts)
    return scopes, True


def _evidence(check, answers, *, visible_count=None):
    found = []
    for citation in check.get("evidence") or []:
        if not isinstance(citation, dict):
            continue
        attempt, quote = citation.get("attempt"), citation.get("quote")
        if (not isinstance(attempt, int) or isinstance(attempt, bool) or not 1 <= attempt <= len(answers)
                or visible_count is not None and attempt > visible_count
                or not isinstance(quote, str) or not quote.strip()):
            continue
        answer = answers[attempt - 1]
        if quote not in answer["text"] or "message_index" not in answer or answer.get("help_control"):
            continue
        found.append((answer, quote))
    return found


def _latest_checks(record, scopes, guided):
    if not guided:
        latest = (record.get("reviews") or [{}])[-1]
        return [("", latest.get("criterion_checks") or [])]
    return [(step_id, scope["step"].get("criterion_checks") or
             ((scope["step"].get("reviews") or [{}])[-1].get("criterion_checks") or []))
            for step_id, scope in scopes.items()]


def _highlight(record, messages, scopes, guided, language):
    ranges = {}
    for step_id, checks in _latest_checks(record, scopes, guided):
        for check in checks:
            if check.get("status") != "met":
                continue
            criterion = {"id": str(check.get("id", "")), "label": _label(check, language)}
            for answer, quote in _evidence(check, scopes[step_id]["answers"]):
                if "content_start" not in answer:
                    continue
                # Saved citations have no occurrence index. Repeated wording
                # can occur in opposing statements, so only a unique exact
                # quote may select a green span. The answer link remains valid.
                start = answer["text"].index(quote)
                if answer["text"].find(quote, start + 1) >= 0:
                    continue
                end = start + len(quote)
                ranges.setdefault(answer["message_index"], []).append({
                    "start": answer["content_start"] + start, "end": answer["content_start"] + end,
                    "criteria": [criterion], "step_id": step_id})
    for message_index, spans in ranges.items():
        merged = []
        for span in sorted(spans, key=lambda value: (value["start"], value["end"])):
            if merged and span["step_id"] == merged[-1]["step_id"] and span["start"] <= merged[-1]["end"]:
                previous = merged[-1]
                previous["end"] = max(previous["end"], span["end"])
                known = {row["id"] for row in previous["criteria"]}
                previous["criteria"].extend(row for row in span["criteria"] if row["id"] not in known)
            else:
                merged.append(deepcopy(span))
        content = messages[message_index].get("content", "")
        messages[message_index]["annotations"] = [{**span, "start": _utf16(content, span["start"]),
                                                   "end": _utf16(content, span["end"])} for span in merged]


def _without_learner_quotes(text):
    """Remove repeated labelled learner blockquotes, preserving teacher prose."""
    lines, result, skipping = str(text or "").splitlines(), [], False
    for line in lines:
        stripped = line.strip()
        if _QUOTE_HEADER.match(stripped):
            skipping = True
            continue
        if skipping:
            if not stripped or stripped.startswith(">"):
                continue
            skipping = False
        result.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()


def _summary_content(text, language, verified_quotes):
    lines, result, in_record = str(text or "").splitlines(), [], False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+(?:本轮作答记录|Your answer record)\b", stripped, re.I):
            in_record = True
            result.append(localized(language,
                "原始作答及要点标记保留在上方对话中；本轮记录与判定已保存。",
                "Your original answers and evidence marks remain in the conversation above. This round's answers and judgements are saved."))
            result.append("")
            continue
        if in_record:
            if re.match(r"^#{1,6}\s+", stripped) or re.match(
                    r"^\*\*(?:参考答案|本题参考|教材原答案|教学参考|Reference answer|Original textbook answer|Teaching reference)", stripped, re.I):
                in_record = False
            else:
                continue
        result.append(line)
    return _clean_feedback_notes(_without_learner_quotes("\n".join(result)), verified_quotes, language)


def _dequote(text, verified_quotes, language):
    # Remove full, quoted answer repetitions; retain short clinical concepts.
    for quote in sorted(set(verified_quotes), key=len, reverse=True):
        if len(quote.strip()) < 8:
            continue
        for left, right in (("“", "”"), ('"', '"'), ("‘", "’"), ("'", "'")):
            text = text.replace(left + quote + right, localized(language, "该作答", "this answer"))
        if len(quote.strip()) >= 18:
            text = text.replace(quote, localized(language, "该作答", "this answer"))
    return text


def _clean_feedback_notes(text, verified_quotes, language):
    if not _FEEDBACK_TITLE.search(text):
        return text
    # Only labelled assessment notes in the grading portion are compacted.
    # Reference and tutor sections after their heading retain their full prose.
    following = re.search(r"^#{1,6}\s+", text, re.M)
    end = following.start() if following else len(text)
    prefix, suffix = text[:end], text[end:]
    prefix = re.sub(r"((?:核对说明|Assessment note)\s*[:：]\s*)([^\n]+)",
                    lambda match: match[1] + _dequote(match[2], verified_quotes, language), prefix, flags=re.I)
    return prefix + suffix


def _note(check, verified, language):
    text = _without_learner_quotes(check.get("explanation", ""))
    quotes = [quote for _, quote in verified]
    if _ERROR_COMPARISON.search(text):
        def remove_claim(match):
            claim = "\n".join(re.sub(r"^> ?", "", line) for line in match["claim"].splitlines()).strip()
            if claim and (claim == check.get("error_quote") or any(claim in quote for quote in quotes)):
                return localized(language, "请对照本项规则修正：", "Correct the answer against this criterion's rule:")
            return match[0]
        text = _ERROR_COMPARISON.sub(remove_claim, text)
        # The remaining block is the authored rule, even if it happens to
        # share words with an answer. Preserve it verbatim, including short
        # safety statements and negations; do not run generic dequoting here.
        return text
    else:
        text = _dequote(text, quotes, language)
    return text[:420].rstrip() + ("…" if len(text) > 420 else "")


def _review_rows(record, scopes, guided):
    reviews = record.get("reviews") or []
    if not reviews and guided:
        reviews = [review for scope in scopes.values() for review in scope["step"].get("reviews") or []]
    result = []
    for index, review in enumerate(reviews):
        step_id = review.get("step_id", "") if guided else ""
        if step_id not in scopes:
            continue
        count = review.get("answer_count")
        if not isinstance(count, int) or isinstance(count, bool):
            # Quoting an early answer does not date a later review. Without a
            # saved count/snapshot, do not attach it to an earlier tutor bubble.
            continue
        result.append({"index": index, "review": review, "step_id": step_id,
                       "global_count": scopes[step_id]["offset"] + count, "answer_count": count})
    return result


def _feedback_view(review, step_id, scope, language, guided):
    count = review.get("answer_count")
    count = count if isinstance(count, int) and not isinstance(count, bool) else len(scope["answers"])
    criteria = []
    for check in review.get("criterion_checks") or []:
        if check.get("status") not in _STATUSES:
            continue
        verified = _evidence(check, scope["answers"], visible_count=count)
        references = []
        for answer, _ in verified:
            reference = {"attempt": answer["attempt"], "message_at": answer["message_at"]}
            if reference not in references:
                references.append(reference)
        criteria.append({"id": str(check.get("id", "")), "label": _label(check, language), "status": check["status"],
                         "note": _note(check, verified, language), "answer_context": check.get("answer_context", ""),
                         "evidence_refs": references, "newly_demonstrated": bool(check.get("newly_demonstrated")),
                         "judgment_revised": bool(check.get("judgment_revised"))})
    covered = sum(row["status"] == "met" for row in criteria)
    passed = bool(review.get("ready_for_summary") and not review.get("material_errors"))
    if passed:
        kind = "next_step" if guided and scope.get("step", {}).get("id") != "evaluate" else "summary"
        action = localized(language, "本步要点已覆盖，可以进入下一步。", "This step is covered; continue to the next step.") if kind == "next_step" else localized(language, "本题要点已覆盖，可以结束并总结。", "The required points are covered; finish and summarize.")
    else:
        pending = next((row for row in criteria if row["status"] != "met"), None)
        kind = "continue_answer"
        action = (localized(language, "可以只补充：", "You can add only: ") + pending["label"] +
                  localized(language, "；已覆盖的内容无需重写。", ". You do not need to repeat covered points.")) if pending else localized(language, "可以继续补充作答，也可以询问需要解释的部分。", "Continue your answer or ask about the part you would like explained.")
    return {"covered": covered, "total": len(criteria), "passed": passed, "step_id": step_id,
            "criteria": criteria, "next_action": {"kind": kind, "text": action}}


def _compact(view, language, fallback):
    if not view["criteria"]:
        return _without_learner_quotes(fallback)
    heading = localized(language, f"**已覆盖 {view['covered']}/{view['total']} 个要点**",
                        f"**Required points covered: {view['covered']}/{view['total']}**")
    labels = {"met": ("已覆盖", "Covered"), "partial": ("还需补充", "Add more"),
              "missing": ("尚未覆盖", "Not covered"), "incorrect": ("需要纠正", "Correct this"),
              "uncertain": ("待确认", "Unconfirmed")}
    rows = [heading]
    for criterion in view["criteria"]:
        line = "- " + criterion["label"] + " · " + localized(language, *labels[criterion["status"]])
        if criterion["note"]:
            line += "\n  " + criterion["note"]
        rows.append(line)
    rows.append(view["next_action"]["text"])
    return "\n\n".join(rows)


def _project_round(record, messages, mapping, language):
    answer_rows = _answer_rows(record, mapping)
    scopes, guided = _scopes(record, answer_rows)
    _highlight(record, messages, scopes, guided, language)
    saved = _review_rows(record, scopes, guided)
    used, latest, last_answer_index, last_feedback_index = set(), {}, -1, -1
    current_step = next(iter(scopes), "")
    step_ids = list(scopes)
    for source_index, original in enumerate(record.get("messages") or []):
        if source_index not in mapping:
            continue
        message = messages[mapping[source_index]]
        if message.get("role") == "user":
            if is_student_message(original, record):
                last_answer_index = source_index
            continue
        if guided and message.get("kind") == "step_question":
            position = step_ids.index(current_step) if current_step in step_ids else -1
            if position + 1 < len(step_ids):
                current_step = step_ids[position + 1]
        kind = message.get("kind")
        if kind == "summary":
            verified_quotes = {quote for saved_row in saved
                               for check in (saved_row["review"].get("criterion_checks") or []) + (saved_row["review"].get("claim_checks") or [])
                               for answer, quote in _evidence(check, scopes[saved_row["step_id"]]["answers"])
                               if answer.get("source_index", source_index) < source_index}
            # A summary can include actual tutor teaching and allowed reference
            # material. Keep those sections and remove only repeated evidence.
            message["display_content"] = _summary_content(message.get("content", ""), language, verified_quotes)
            continue
        if kind == "help" and (_FEEDBACK_TITLE.search(message.get("content", "")) or
                               any(_QUOTE_HEADER.match(line.strip()) for line in message.get("content", "").splitlines())):
            # Older feedback requests were saved as generic help. Compact only
            # their explicit evidence text; never attach a guessed later grade.
            message["display_content"] = _without_learner_quotes(message.get("content", ""))
            continue
        if kind not in {"feedback", "recheck"}:
            continue
        step_id = message.get("step_id", current_step if guided else "")
        if step_id not in scopes:
            continue
        visible = sum("source_index" in answer and answer["source_index"] < source_index for answer in answer_rows)
        row, snapshot = None, message.get("review_snapshot")
        if isinstance(snapshot, dict):
            snapshot_step = snapshot.get("step_id", step_id)
            snapshot_count = snapshot.get("answer_count")
            if (snapshot_step not in scopes or snapshot_step != step_id
                    or not isinstance(snapshot_count, int) or isinstance(snapshot_count, bool) or snapshot_count < 0
                    or snapshot_count > len(scopes[snapshot_step]["answers"])
                    or scopes[snapshot_step]["offset"] + snapshot_count > visible):
                message["display_content"] = _without_learner_quotes(message.get("content", ""))
                continue
            row = {"review": snapshot, "step_id": snapshot_step}
        else:
            index = message.get("review_index")
            if isinstance(index, int) and not isinstance(index, bool):
                row = next((value for value in saved if value["index"] == index and value["step_id"] == step_id
                            and value["global_count"] <= visible), None)
            if row is None:
                eligible = [value for value in saved if value["step_id"] == step_id and value["global_count"] <= visible
                            and value["index"] not in used]
                previous_users = [(index, value) for index, value in enumerate((record.get("messages") or [])[:source_index])
                                  if value.get("role") == "user"]
                request_index, preceding = previous_users[-1] if previous_users else (-1, {})
                recheck = kind == "recheck" or (request_index > last_feedback_index and preceding.get("kind") == "help"
                                               and help_intent(preceding.get("content", "")) == "recheck")
                if last_answer_index > last_feedback_index or recheck:
                    row = next((value for value in eligible if value["global_count"] == visible
                                and (not recheck or value["review"].get("review_origin") == "recheck")), None)
                if row is None:
                    row = latest.get(step_id)
        if row is None:
            message["display_content"] = _without_learner_quotes(message.get("content", ""))
            continue
        if row["step_id"] not in scopes:
            continue
        step_id = row["step_id"]
        if "index" in row:
            used.add(row["index"])
        latest[step_id] = row
        last_feedback_index = source_index
        view = _feedback_view(row["review"], step_id, scopes[step_id], language, guided)
        if view["criteria"]:
            message["feedback_view"] = view
            message["display_content"] = _compact(view, language, message.get("content", ""))
        else:
            # Open formative tasks have no fixed criterion denominator. Keep
            # their actual prose instead of rendering an invented 0/0 card.
            message["display_content"] = _without_learner_quotes(message.get("content", ""))


def annotate_messages(state, messages):
    """Return a deep-copy projection; original learner and teacher text stays intact."""
    projected = deepcopy(messages)
    for message in projected:
        message.pop("annotations", None)
        message.pop("feedback_view", None)
        message.pop("display_content", None)
        if message.get("role") == "user":
            message["display_content"] = message.get("content", "")
    if state.get("mode") not in {None, "teaching"}:
        return projected
    if state.get("practice_history"):
        from .learning import records
        rounds = records(state)
    else:
        rounds = [state]
    cursor = 0
    for record in rounds:
        mapping, cursor = _message_map(record, projected, cursor)
        _project_round(record, projected, mapping, record.get("language", state.get("language", "zh")))
    if state.get("mode") == "teaching":
        from .teaching_copy import clean_teaching_copy
        learner_words = [value for record in rounds for row in record.get("messages") or []
                         if row.get("role") == "user" for value in (row.get("content"), row.get("answer_text")) if value]
        for message in projected:
            if message.get("role") == "assistant":
                message["display_content"] = clean_teaching_copy(message.get("display_content", message.get("content", "")), learner_words)
    return projected
