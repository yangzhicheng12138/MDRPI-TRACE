"""Continue practice using saved assessments, without another model call.

OATutor's replaceable item-selection approach is a design reference. These
rules use this project's categories/scenarios, not uncalibrated BKT scores.
"""
from copy import deepcopy

from .language import localized
from .teaching import CASES, invitation, practice_title, student_answers
from .question_bank import CATEGORIES, PROJECT_KEY, ADAPTED_KEY


ACTION_LABELS = {
    "next": ("下一题", "Next question"), "similar": ("同类练习", "Related exercise"),
    "review": ("复习遗漏", "Review pending points"), "retry_exercise": ("重练本题", "Repeat this exercise"),
    "finish": ("结束学习", "End learning"), "resume": ("继续学习", "Continue learning"),
    "due": ("到期复习", "Review due questions"),
}
OUTCOME_LABELS = {
    "in_progress": ("进行中", "In progress"), "covered": ("本题要点已覆盖", "Required points covered"),
    "needs_review": ("待复习", "To review"), "unassessed": ("尚无评价", "Not assessed"),
}


RECORD_FIELDS = (
    "case", "case_origin", "bank_item", "language", "attempts", "reviews",
    "messages", "sources", "source_events", "attachments", "summary", "stop_reason",
    "stage", "turn_id", "updated_at", "exercise_started_at", "review_of",
    "help_cache", "hint_status", "last_feedback", "interruptions",
    "hint_events", "reference_views", "reference_cache", "teacher_explanation", "recheck_for_answer",
    "answer_events", "teaching_dialogue_events", "input_route_history",
    "clinical_flow", "case_spec", "case_intake", "learning_owner_id",
    "policy", "web_mode",
)


def exercise_record(state):
    record = deepcopy({k: state[k] for k in RECORD_FIELDS if k in state})
    record["exercise_id"] = state.get("exercise_id") or state["session_id"] + ":first"
    record["practice_root_id"] = state.get("practice_root_id") or record["exercise_id"]
    return record


def records(state):
    return state.get("practice_history", []) + ([exercise_record(state)] if state.get("case") else [])


def item_key(record):
    return (record.get("bank_item") or {}).get("id") or record["practice_root_id"]


def row_for(record, language):
    item = record.get("bank_item") or {}
    review = next(iter(record.get("reviews", [])[-1:]), {})
    checks = deepcopy(review.get("criterion_checks", []))
    complete = record.get("stage") == "complete"
    outcome = ("in_progress" if not complete else "needs_review" if not review and record.get("reference_views") else "unassessed" if not review else
               "covered" if review.get("ready_for_summary") and not review.get("material_errors") else "needs_review")
    if record.get("clinical_flow"):
        from .learning_analytics import record_metrics
        measurement = record_metrics(record)
        outcome = "covered" if measurement["completion_status"] == "completed" else "interrupted" if measurement["completion_status"] == "interrupted" else "in_progress"
        complete = measurement["completion_status"] == "completed"
    # Keep the saved criterion status and student quotation intact. No new
    # clinical judgement, grading or inference from a tutor explanation.
    gaps = [r["criterion"][language] for r in checks if r["status"] != "met"] if checks else list(
        dict.fromkeys(review.get("material_errors", []) + review.get("gaps", [])))
    return {"exercise_id": record["exercise_id"], "item_key": item_key(record),
            "bank_id": item.get("id", ""), "category": item.get("category", ""),
            "title": practice_title(item, language) if item else " ".join(record["case"].split())[:72],
            "outcome": outcome, "complete": complete, "ended":record.get("stage")=="complete", "attempt_count": len(student_answers(record)),
            "criteria": checks, "gaps": gaps, "review_of": record.get("review_of", ""),
            "assistance": deepcopy(review.get("assistance", {})),
            "updated_at": record.get("updated_at", ""), "stop_reason": record.get("stop_reason", "")}


def learning_progress(state):
    language = state.get("language", "zh")
    rows = [row_for(r, language) for r in records(state)]
    latest = {r["item_key"]: r for r in rows}
    # An unfinished repeat cannot erase the previous completed assessment.
    completed = {r["item_key"]: r for r in rows if r["complete"] or r.get("ended")}
    for row in rows:
        newer = completed.get(row["item_key"])
        if row["complete"] and newer["exercise_id"] != row["exercise_id"]:
            row["superseded_by"] = newer["exercise_id"]
            row["latest_outcome"] = newer["outcome"]
    queue = [r for r in completed.values() if r["outcome"] in {"needs_review", "unassessed", "interrupted"}]
    from .learning_analytics import aggregate_records
    return {"session_id": state["session_id"], "ended": state.get("practice_ended", False),"analytics":aggregate_records(records(state)),
            "completed_exercises": sum(r["complete"] for r in rows), "unique_questions": len(latest),
            "covered_questions": sum(r["outcome"] == "covered" for r in completed.values()),
            "review_count": len(queue), "review_queue": queue, "exercises": rows,
            "language": language}


def progress_text(progress):
    language = progress["language"]
    done, unique, count = (progress[k] for k in ("completed_exercises", "unique_questions", "review_count"))
    intro = localized(language,
        f"已完成 {done} 轮练习，涉及 {unique} 道题；{count} 道题待复习。",
        f"Completed {done} exercise rounds across {unique} questions; {count} questions to review.")
    note = localized(language, "根据本会话已保存的评价整理；结束练习不代表全部答对。",
                     "Based on saved assessments in this session; finishing an exercise does not mean every point was correct.")
    return intro + "\n\n" + note


def choose_exercise(state, action, exercise_id=""):
    """Return an existing bank item or saved exercise, plus a visible reason."""
    progress = learning_progress(state)
    language = state.get("language", "zh")
    history = records(state)
    if action in {"review", "retry_exercise"}:
        if action == "review":
            if not progress["review_queue"]:
                raise ValueError(localized(language, "当前没有待复习题。可以选择下一题或查看学习记录。",
                                           "There are no pending review questions. Choose the next question or view your learning record."))
            exercise_id = progress["review_queue"][0]["exercise_id"]
        record = next((r for r in history if r["exercise_id"] == exercise_id and r["stage"] == "complete"), None)
        if record is None:
            raise ValueError(localized(language, "未找到本会话已完成的练习。", "No completed exercise with that ID exists in this session."))
        reason = localized(language, "重新练习已保存的题目；本轮重新作答，之前的原文与评价保留在学习记录中。",
                           "Repeat the saved question with a fresh answer; previous answers and assessments remain in your learning record.")
        return record.get("bank_item") or {}, record, reason
    seen = {r["bank_id"] for r in progress["exercises"] if r["bank_id"]}
    candidates = [item for key, item in CASES.items() if item["has_reference_answer"] and key not in seen
                  and (state.get("clinical_flow") or not item.get("six_step_rubric"))]
    current = state.get("bank_item") or {}
    if action == "similar":
        candidates = [i for i in candidates if current.get("category") and i["category"] == current["category"]]
        if not candidates:
            raise ValueError(localized(language, "当前题目没有尚未练习的同类题。可以选择下一题，或在学习进度中重练旧题。",
                                       "No untried question shares this topic. Choose the next question or repeat an earlier exercise from Learning progress."))
    if not candidates:
        raise ValueError(localized(language, "现有带答案的题目都已练习。可以复习遗漏，或在学习进度中选择旧题重练。",
                                   "All available answer-paired questions have been tried. Review pending points or repeat a saved exercise from Learning progress."))
    weak_categories = {r["category"] for r in progress["review_queue"] if r["category"]}
    def priority(item):
        shared_skills = set(current.get("skill_ids", [])) & set(item.get("skill_ids", []))
        return (0 if action == "next" and item["category"] in weak_categories else 1,
                0 if shared_skills else 1,
                0 if current.get("scenario_id") and item.get("scenario_id") == current["scenario_id"] else 1,
                0 if item["category"] == current.get("category") else 1)
    item = min(candidates, key=priority)
    topic = localized(language, *CATEGORIES[item["category"]])
    related = bool(set(current.get("skill_ids", [])) & set(item.get("skill_ids", [])))
    reason = (localized(language, f"从“{topic}”选择同一知识点的新情境，练习将判断应用到另一种器械。",
                        f"An untried question from {topic} applies a shared skill in a different device scenario.") if related else
              localized(language, f"从“{topic}”选择尚未做过的练习题；优先待复习主题或同一情境。",
                        f"An untried question from {topic}, prioritizing topics to review or the same scenario."))
    return item, None, reason


def exercise_intro(item, case, language):
    """One renderer shared by initial bank selection and subsequent exercises."""
    from .rubric import objective_question_type, objective_invitation, objective_question_text
    if objective_question_type(item):
        reviewed = item.get("answer_origin") == "ai_reviewed_source_grounded"
        label = localized(language, "专业知识练习", "Practice content") if reviewed else localized(language, "教材配套练习", "Textbook exercise")
        parts = [label, objective_question_text(case), objective_invitation(objective_question_type(item), language),
                 localized(language, "总结时可查看参考答案与依据。", "The reference answer and sources appear in the summary.")]
        if not reviewed and item.get("source", {}).get("question_url"):
            source = item["source"]
            parts.append(localized(language, "题目来源：", "Question source: ") + f"[{source['book']}]({source['question_url']}) · {source['authors']} · {source['license']}")
        return "\n\n".join(parts)
    if item.get("answer_origin") == "ai_reviewed_source_grounded":
        label = "标准化案例" if item.get("kind") == "standardized_case" else "项目练习案例" if item.get("kind") == "fictional_case" else "专业知识练习"
        ending = invitation(case, language) if item.get("six_step_rubric") else localized(language, "请先回答本题，可按自己的表达组织思路。", "Answer this question in your own words.")
        return "\n\n".join([localized(language, label, "Practice content"), case,
                            localized(language, "可先尝试作答，或逐条获取提示；看完三条提示后可查看参考答案，答对后也可复盘。",
                                      "Try answering or request hints one at a time. View the reference after three hints, or review it after a correct answer."), ending])
    if item.get("kind") == "standardized_case":
        return "\n\n".join([localized(language, "标准化案例源材料 · 待专家审核", "Imported standardized case · pending expert review"),
            case, localized(language, "本例保留原始题面和编号。练习反馈用于形成性学习，未提供专家评分标准。", "The original case and ID are retained. Feedback supports formative learning; no expert scoring key was supplied."), invitation(case, language)])
    bank_item = item if item.get("has_reference_answer") else {}
    if bank_item and item.get("answer_origin") == PROJECT_KEY:
        goals = "；".join(item["learning_objectives"][language])
        return "\n\n".join([localized(language, "MDRPI 专项练习 · 项目编写", "MDRPI focused practice · project-authored"),
                             case, localized(language, "练习重点：", "Learning focus: ") + goals,
                             localized(language, "先回答本题即可，可按自己的表达说明理由；看完三条提示后可查看参考要点，答对后也可复盘。",
                                       "Answer in your own words, with your reasoning. View reference points after three hints, or review them after a correct answer.")])
    if bank_item and item.get("answer_origin") == ADAPTED_KEY:
        prompt = localized(language, "请先回答本题，可补充理由；看完三条提示或答对后，可查看分开展示的教材原答案与项目补充要点。",
                           "Answer this question; you may add your reasoning. After three hints or a correct answer, view the publisher's key and project supplements in separate sections.")
    else:
        prompt = localized(language, "请先回答本题，可补充理由；看完三条提示后可查看教材参考答案，答对后也可复盘。",
                           "Answer this question; you may add your reasoning. View the textbook reference after three hints, or review it after a correct answer.") if bank_item else invitation(case, language)
    label = localized(language, "教材配套练习", "Textbook exercise") if bank_item else ""
    text = (label + "\n\n" + case + "\n\n" + prompt).strip()
    if bank_item:
        source = bank_item["source"]
        text += "\n\n" + localized(language, "题目来源：", "Question source: ") + f"[{source['book']}]({source['question_url']}) · {source['authors']} · {source['license']}"
    return text
