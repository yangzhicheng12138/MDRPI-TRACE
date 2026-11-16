"""Reference access for subjective practice, applied without changing snapshots."""
from copy import deepcopy
import re

from .language import localized


REFERENCE_ACCESS_VERSION = "subjective_reference_access_20261013_v1"
_REFERENCE_HEADINGS = (
    "### 教材参考答案", "### 教材答案与项目补充要点", "### 专项练习参考要点",
    "### 参考答案与要点", "### Textbook reference answer", "### Textbook answer and project supplements",
    "### Reference points for this focused exercise", "### Reference answer and criteria",
)


def _verified_answer(answers, reviews):
    if not answers or not reviews:
        return False
    review = reviews[-1]
    if not review.get("ready_for_summary") or review.get("material_errors"):
        return False
    checks = review.get("criterion_checks") or []
    if checks:
        for row in checks:
            if row.get("status") != "met" or not row.get("evidence"):
                return False
            for evidence in row["evidence"]:
                index, quote = evidence.get("attempt"), evidence.get("quote")
                if (isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(answers)
                        or not isinstance(quote, str) or not quote or quote not in answers[index - 1]):
                    return False
        return True
    # Legacy open tasks have no authored criteria. Only a grounded review with
    # actual learner quotations can establish a correct-answer recap.
    return bool(review.get("quote_backed") and any(row.get("evidence") for row in review.get("claim_checks", [])))


def _historical_reference(state, step_index, step):
    if step and step.get("reference_views"):
        return True
    views = state.get("reference_views") or []
    if not step and views:
        return True
    if step and any(row.get("step_id") == step["id"] for row in views):
        return True
    # Old references were typed messages before separate exposure events were
    # introduced. Full published-key headings establish prior full disclosure;
    # a completed stage or an ordinary teacher explanation does not.
    inferred_index = 0
    for row in state.get("messages") or []:
        if row.get("role") != "assistant":
            continue
        content = row.get("content", "")
        if not isinstance(content, str):
            continue
        if row.get("kind") in {"case", "step_question"}:
            match = re.search(r"(?:第\s*([1-6])\s*/\s*6|Step\s+([1-6])\s*/\s*6)", content, re.I)
            if match:
                inferred_index = int(match[1] or match[2]) - 1
        if (row.get("kind") == "summary" and any(title in content for title in _REFERENCE_HEADINGS)
                and "### 本步参考答案" not in content and "### Reference answer for this step" not in content):
            return True
        if row.get("kind") != "reference":
            continue
        if not step:
            return True
        if row.get("step_id") == step["id"] or not row.get("step_id") and inferred_index == step_index:
            return True
    return False


def reference_access(state, step_index=None):
    """Three shown hints unlock each subjective question/current clinical step.

    A quote-backed correct answer permits recap. Prior actual reference views
    remain available. Merely ending a question never unlocks its reference.
    Objective feedback keeps its existing fixed-answer contract.
    """
    from .rubric import objective_question_type
    from .question_bank import assessment_item
    from .teaching import student_answers
    from .learner_evidence import hint_events
    item = assessment_item(state)
    if state.get("mode", "teaching") != "teaching" or objective_question_type(item):
        return {"allowed": True, "hints_seen": 0, "remaining": 0, "reason": "objective_or_not_teaching"}
    if not state.get("case"):
        return {"allowed": False, "hints_seen": 0, "remaining": 3, "reason": "no_active_exercise"}
    flow = state.get("clinical_flow") or {}
    step = None
    if flow:
        index = flow.get("current_step") if step_index is None else step_index
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(flow.get("steps") or []):
            raise ValueError("Invalid clinical step for reference access.")
        step_index, step = index, flow["steps"][index]
        events, answers, reviews = step.get("hints") or [], step.get("answers") or [], step.get("reviews") or []
    else:
        if step_index is not None:
            raise ValueError("This question has no clinical steps.")
        events, answers, reviews = hint_events(state), student_answers(state), state.get("reviews") or []
    seen = min(3, len(events))
    reason = ("previously_viewed" if _historical_reference(state, step_index, step) else
              "answered_correctly" if _verified_answer(answers, reviews) else
              "three_hints_seen" if seen >= 3 else "hints_required")
    return {"allowed": reason != "hints_required", "hints_seen": seen,
            "remaining": max(0, 3 - seen), "reason": reason}


def all_reference_steps_accessible(state):
    flow = state.get("clinical_flow") or {}
    return (all(reference_access(state, index)["allowed"] for index in range(len(flow.get("steps") or [])))
            if flow else reference_access(state)["allowed"])


def has_verified_answer(state, step_index=None):
    """Whether the saved passing review is supported by real learner quotes."""
    flow = state.get("clinical_flow") or {}
    if flow:
        index = flow["current_step"] if step_index is None else step_index
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(flow["steps"]):
            return False
        step = flow["steps"][index]
        return _verified_answer(step.get("answers") or [], step.get("reviews") or [])
    from .teaching import student_answers
    return _verified_answer(student_answers(state), state.get("reviews") or [])


def reference_locked_text(state):
    access = reference_access(state)
    if state.get("clinical_flow") and access["allowed"] and not all_reference_steps_accessible(state):
        return localized(state.get("language", "zh"),
            "当前步骤的参考可以回看，其他尚未答对或未看完三条提示的步骤仍未解锁。结束学习不会一次展示所有步骤的完整答案。",
            "The current step's reference is available. Other steps remain locked until correctly answered or all three hints are viewed. Ending practice does not reveal every step's full answer.")
    return localized(state.get("language", "zh"),
        f"参考答案尚未解锁。需要先看完当前问题的 3 条提示，目前还剩 {access['remaining']} 次。结束练习不会直接显示答案；你可以在当前页继续获取剩余提示。",
        f"The reference answer is not unlocked. View all 3 hints for this task; {access['remaining']} remain. Ending practice does not reveal the key. You can get the remaining hints on this page.")


def _hide_rubric_key(rubric):
    if not isinstance(rubric, dict):
        return
    for key in ("basis_answer", "reference_answer", "required", "misconceptions", "optional_extensions",
                "summary_constraints", "acceptable", "incorrect", "follow_up"):
        rubric.pop(key, None)


def public_practice_item(item):
    """Expose catalogue material without its unpublished answer/scoring key.

    Catalogue and review browsing are independent of a learner's current
    exercise. Their responses therefore expose questions, provenance and review
    history, while the saved exercise controls any later answer disclosure.
    """
    public = deepcopy(item)
    for key in ("answer", "publisher_answer", "reference_answer", "reference_note", "review_note",
                "scoring", "formal_scoring", "must_answer_points", "correct_options", "correct_truth", "correct_order",
                "truth_value", "objective_rule_source", "hints", "hint_pathways", "summary_constraints"):
        public.pop(key, None)
    _hide_rubric_key(public.get("rubric"))
    container = public.get("six_step_rubric")
    if isinstance(container, dict):
        _hide_rubric_key(container)
        for step in container.get("steps") or []:
            _hide_rubric_key(step)
    return public


def public_teaching_state(state):
    """Redact undisclosed keys in a returned copy; keep internal/history snapshots."""
    public = deepcopy(state)
    # A different current mode/type must not expose an earlier locked question.
    # Keep objective fields and its message contract unchanged while projecting
    # each completed teaching record against its own saved access evidence.
    if "practice_history" in state:
        public["practice_history"] = [public_teaching_state({**record, "mode": "teaching"})
                                      for record in state.get("practice_history") or []]
    if state.get("mode", "teaching") != "teaching":
        return public
    from .rubric import objective_question_type
    from .question_bank import assessment_item
    if objective_question_type(assessment_item(state)):
        return public
    access = reference_access(state)
    public["reference_access"] = access
    flow = state.get("clinical_flow") or {}
    all_allowed = all_reference_steps_accessible(state)
    bank = public.get("bank_item") or {}
    # Hints are released through actual hint events, not a hidden preloaded list.
    for key in ("hints", "hint_pathways"):
        bank.pop(key, None)
    if not all_allowed:
        for key in ("answer", "publisher_answer", "reference_answer", "reference_note", "review_note", "summary_constraints", "content_review",
                    "scoring", "formal_scoring", "must_answer_points"):
            bank.pop(key, None)
        _hide_rubric_key(bank.get("rubric"))
    if flow:
        output_flow = public["clinical_flow"]
        policies = [reference_access(state, index) for index in range(len(flow["steps"]))]
        for index, step in enumerate(output_flow["steps"]):
            step["reference_access"] = policies[index]
        for container in (output_flow.get("rubric_snapshot"), bank.get("six_step_rubric")):
            if not isinstance(container, dict):
                continue
            for row in container.get("steps") or []:
                index = next((i for i, step in enumerate(flow["steps"]) if step["id"] == row.get("id")), None)
                if index is None or not policies[index]["allowed"]:
                    _hide_rubric_key(row)
        contracts = output_flow.get("task_rubric_snapshot", {}).get("steps", {})
        for index, step in enumerate(flow["steps"]):
            if not policies[index]["allowed"]:
                _hide_rubric_key(contracts.get(step["id"]))
    if not all_allowed:
        public.pop("help_cache", None)
        public.pop("reference_cache", None)
    if not all_allowed:
        for source in public.get("sources") or []:
            if isinstance(source, dict):
                for key in ("text", "raw_text", "content", "body"):
                    source.pop(key, None)
    if state.get("messages"):
        from .teaching_presentation import annotate_messages
        public["messages"] = annotate_messages({**state, "mode": "teaching", "practice_history": []}, state["messages"])
    return public


def public_teaching_payload(value):
    """Apply the policy to state/record envelopes returned by HTTP endpoints."""
    if isinstance(value, dict):
        if "case" in value and "bank_item" in value:
            return public_teaching_state(value)
        return {key: public_teaching_payload(row) for key, row in value.items()}
    if isinstance(value, list):
        return [public_teaching_payload(row) for row in value]
    return value
