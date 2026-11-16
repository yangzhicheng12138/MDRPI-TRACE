"""Read-only summaries of saved learner evidence, never model-created grades.

``value`` is the percentage of required points covered in saved judgements.
It is not a validated clinical competency score. A step without an answer and
an assessment has no measurement (``None``), rather than an invented zero.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import re


SCHEMA_VERSION = "learning_analytics_v1"
MEASURE = "required_point_coverage"
STEP_LABELS = {
    "recognize": "识别线索", "analyze": "分析线索", "prioritize": "确定优先问题",
    "solutions": "制定解决方案", "action": "采取行动", "evaluate": "评价结果",
}
DIFFICULTIES = ("基础", "进阶", "复杂")
STATUSES = ("met", "partial", "incorrect", "uncertain", "missing")
_ACKNOWLEDGEMENT = re.compile(
    r"(?:yes|ok|okay|i agree|understood|got it|thanks|是的|好的|好|嗯|明白了|知道了|我同意|对|谢谢)[。.!！?？\s]*", re.I)


def normalize_difficulty(value):
    return {"综合": "进阶", "basic": "基础", "beginner": "基础", "advanced": "进阶",
            "intermediate": "进阶", "complex": "复杂"}.get(value, value if value in DIFFICULTIES else "")


def _strings(value):
    if isinstance(value, str):
        return [value] if value else []
    return list(dict.fromkeys(item for item in (value or []) if isinstance(item, str) and item))


def case_metadata(record):
    """Normalize metadata already recorded at intake; do not infer patient facts."""
    item = record.get("bank_item") or {}
    flow = record.get("clinical_flow") or {}
    metadata = {**(item.get("classification") or {}), **(item.get("scope") or {}),
                **item, **(flow.get("case_metadata") or {}), **(record.get("case_spec") or {}),
                **(record.get("custom_spec") or {}), **(record.get("case_metadata") or {})}
    population = metadata.get("populations") or metadata.get("population")
    devices = metadata.get("device_types") or metadata.get("devices") or metadata.get("device_type")
    difficulty = normalize_difficulty(metadata.get("difficulty") or metadata.get("level") or metadata.get("source_difficulty"))
    case_id = item.get("id") or record.get("case_id") or record.get("practice_root_id")
    if not case_id:
        case_id = "custom:" + hashlib.sha256(str(record.get("case", "")).encode()).hexdigest()
    title = item.get("title") or metadata.get("title")
    if not title:
        title = "AI生成练习案例" if record.get("case_origin") == "generated_case" else "自有脱敏案例"
    return {"case_id": case_id, "title": str(title).split(" / ")[0], "difficulty": difficulty,
            "populations": _strings(population), "device_types": _strings(devices),
            "category": metadata.get("category", ""), "case_origin": record.get("case_origin", "")}


def _answer_texts(step):
    answers = []
    for item in step.get("answers") or []:
        text = item if isinstance(item, str) else item.get("text", item.get("content", "")) if isinstance(item, dict) else ""
        # Keep positions for the one-based quote citations. Acknowledgements
        # can remain in the history but do not substantiate a required point.
        answers.append(text if isinstance(text, str) else "")
    return answers


def _valid_evidence(evidence, answers):
    if not isinstance(evidence, list) or not evidence:
        return False
    for citation in evidence:
        if not isinstance(citation, dict):
            return False
        attempt, quote = citation.get("attempt"), citation.get("quote")
        if (not isinstance(attempt, int) or isinstance(attempt, bool) or not 1 <= attempt <= len(answers)
                or not isinstance(quote, str) or not quote.strip() or quote.strip() not in answers[attempt - 1]
                or _ACKNOWLEDGEMENT.fullmatch(answers[attempt - 1].strip())):
            return False
    return True


def step_measure(step):
    """Use the latest cumulative assessment, preserving its actual denominator."""
    answers = _answer_texts(step)
    substantive = [text for text in answers if text.strip() and not _ACKNOWLEDGEMENT.fullmatch(text.strip())]
    reviews = step.get("reviews") or []
    latest = reviews[-1] if reviews and isinstance(reviews[-1], dict) else {}
    checks = step.get("criterion_checks") or latest.get("criterion_checks") or []
    # A saved hint is an event, not evidence of a student answer or an ability.
    hints = len(step.get("hints") or [])
    references = len(step.get("reference_views") or [])
    explanations = len(step.get("teaching_dialogue_events") or [])
    result = {"value": None, "sample_count": 0, "met": 0, "required": 0,
              **{status: 0 for status in STATUSES if status != "met"}, "hint_count": hints,
              "reference_answer_count": references, "explanation_count": explanations,
              "independent_met": 0, "assisted_met": 0, "attempt_count": len(substantive),
              "status": step.get("status", "pending"), "criteria": [],
              "rubric_version": latest.get("rubric_version") or step.get("rubric_version") or ""}
    if not substantive or not isinstance(checks, list) or not checks:
        return result
    seen = set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or check.get("status") not in STATUSES or check.get("required") is False:
            continue
        identity = check.get("id") or str(index)
        if identity in seen:
            continue
        seen.add(identity)
        status = check["status"]
        if status in {"met", "partial", "incorrect"} and not _valid_evidence(check.get("evidence"), answers):
            status = "uncertain"
        label = check.get("criterion") or check.get("label") or identity
        if isinstance(label, dict):
            label = label.get("zh") or label.get("en") or identity
        result["criteria"].append({"id": identity, "label": str(label), "status": status})
        result["required"] += 1
        result[status] += 1
        if status == "met":
            assisted = check.get("attribution") == "teacher_help" or check.get("answer_context") in {"after_hint", "after_reference", "after_explanation"}
            if (hints or references or explanations) and not check.get("attribution") and not check.get("answer_context"):
                # Old records without explicit assistance attribution cannot
                # support a claim of independent mastery after seeing a hint.
                assisted = True
            result["assisted_met" if assisted else "independent_met"] += 1
    if result["required"]:
        result["value"] = round(100 * result["met"] / result["required"], 1)
        result["sample_count"] = 1
    return result


def _timestamp(record):
    return record.get("completed_at") or record.get("updated_at") or record.get("exercise_started_at") or ""


def _chronology(record, index):
    try:
        date = datetime.fromisoformat(_timestamp(record).replace("Z", "+00:00"))
        return (date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date).timestamp(), index
    except (ValueError, TypeError):
        return float("-inf"), index


def unique_records(records):
    """Idempotent rechecks replace a round, while repeated case attempts remain."""
    selected = {}
    for index, record in enumerate(records):
        if isinstance(record, dict) and (record.get("case") or record.get("bank_item") or record.get("clinical_flow")):
            selected[record.get("exercise_id") or f"unnamed:{index}"] = (index, record)
    return [record for index, record in sorted(selected.values(), key=lambda pair: _chronology(pair[1], pair[0]))]


def record_metrics(record):
    """A compact per-round record shared by reports and recommendations."""
    flow = record.get("clinical_flow") or {}
    steps = {step.get("id"): step for step in flow.get("steps", []) if isinstance(step, dict)}
    measured = [{"id": identity, "label": label, **step_measure(steps.get(identity, {}))}
                for identity, label in STEP_LABELS.items()]
    if flow:
        status = flow.get("status", "in_progress")
        completed = status == "completed" and all(steps.get(identity, {}).get("status") == "passed" for identity in STEP_LABELS)
        outcome = "completed" if completed else "interrupted" if status == "interrupted" else "in_progress"
    else:
        # Old completed exercises are real history, but cannot supply six-step
        # measurements or justify advancing the difficulty of guided cases.
        outcome = "completed" if record.get("stage") == "complete" else "in_progress"
        if record.get("stop_reason") in {"learner_ended", "early_end", "interrupted"}:
            outcome = "interrupted"
    measured_steps = [step for step in measured if step["required"]]
    required, met = sum(step["required"] for step in measured), sum(step["met"] for step in measured)
    hints = sum(step["hint_count"] for step in measured)
    references = sum(len(step.get("reference_views") or []) for step in steps.values()) if flow else len(record.get("reference_views") or [])
    explanations = sum(len(step.get("teaching_dialogue_events") or []) for step in steps.values()) if flow else len(record.get("teaching_dialogue_events") or [])
    if not flow:
        hints = len(record.get("hint_events") or [])
    versions = [step["rubric_version"] for step in measured_steps]
    scoring_version = flow.get("scoring_version") or record.get("scoring_version") or ""
    if not scoring_version and versions and all(versions):
        scoring_version = "|".join(f"{step['id']}:{step['rubric_version']}" for step in measured_steps)
    from .teaching import student_answers
    return {"exercise_id": record.get("exercise_id", ""), **case_metadata(record),
            "at": _timestamp(record), "completion_status": outcome, "guided": bool(flow),
            "six_steps": measured, "answered_steps": sum(step["attempt_count"] > 0 for step in measured),
            "passed_steps": sum(step["status"] in {"passed", "awaiting_next"} for step in measured),
            "required": required, "met": met, "value": round(100 * met / required, 1) if required else None,
            "hint_count": hints, "reference_answer_count": references, "explanation_count": explanations,
            "attempt_count": sum(step["attempt_count"] for step in measured) if flow else len(student_answers(record)),
            "scoring_version": scoring_version,
            "low_hint_mastery": bool(outcome == "completed" and flow and len(measured_steps) == 6
                                      and all(step["value"] == 100 for step in measured) and hints <= 1 and references == 0 and explanations == 0
                                      and all(step["assisted_met"] == 0 for step in measured))}


def _coverage(history, key):
    values = []
    for row in history:
        labels = row[key] if isinstance(row[key], list) else [row[key]] if row[key] else []
        values.extend(labels)
    return [{"id": identity, "rounds": values.count(identity),
             "completed_rounds": sum(row["completion_status"] == "completed" and
                                      (identity in row[key] if isinstance(row[key], list) else row[key] == identity)
                                      for row in history)} for identity in sorted(set(values))]


def _trend(history):
    scored = [row for row in history if row["value"] is not None and row["scoring_version"]
              and row["completion_status"] == "completed"]
    version = scored[-1]["scoring_version"] if scored else ""
    comparable = [row for row in scored if row["scoring_version"] == version]
    if len(comparable) < 2:
        return {"available": False, "scoring_version": version, "sample_count": len(comparable),
                "previous": None, "recent": None, "delta": None,
                "reason": "同一评价版本下至少需要两轮已完成且有实际判断的训练。"}
    # Disjoint windows avoid a record contributing to both sides of a change.
    window = min(3, len(comparable) // 2)
    previous = comparable[-2 * window:-window]
    recent = comparable[-window:]
    old = round(sum(row["value"] for row in previous) / window, 1)
    new = round(sum(row["value"] for row in recent) / window, 1)
    return {"available": True, "scoring_version": version, "sample_count": 2 * window,
            "previous": old, "recent": new, "delta": round(new - old, 1), "window_rounds": window,
            "reason": "比较同一评价版本的两组不重叠已完成训练；显示必答点覆盖变化。"}


def aggregate_records(records):
    history = [record_metrics(record) for record in unique_records(records)]
    six_steps, weak = [], Counter()
    weak_labels = {}
    for identity, label in STEP_LABELS.items():
        measurements = [next(step for step in row["six_steps"] if step["id"] == identity) for row in history]
        axis = {"id": identity, "label": label, "value": None}
        for key in ("sample_count", "met", "required", "partial", "incorrect", "uncertain", "missing", "hint_count", "reference_answer_count", "explanation_count", "independent_met", "assisted_met", "attempt_count"):
            axis[key] = sum(step[key] for step in measurements)
        axis["value"] = round(100 * axis["met"] / axis["required"], 1) if axis["required"] else None
        six_steps.append(axis)
        for step in measurements:
            for criterion in step["criteria"]:
                if criterion["status"] != "met":
                    key = identity, criterion["id"], criterion["status"]
                    weak[key] += 1
                    weak_labels[key] = criterion["label"]
    counts = Counter(row["completion_status"] for row in history)
    return {"schema_version": SCHEMA_VERSION, "measure": MEASURE,
            "measure_label": "已记录必答点覆盖率（非经验证的临床能力评分）",
            "counts": {"training_rounds": len(history), "completed_rounds": counts["completed"],
                       "interrupted_rounds": counts["interrupted"], "in_progress_rounds": counts["in_progress"],
                       "unique_cases": len({row["case_id"] for row in history}),
                       "completed_cases": len({row["case_id"] for row in history if row["completion_status"] == "completed"}),
                       "answered_steps": sum(row["answered_steps"] for row in history),
                       "hint_count": sum(row["hint_count"] for row in history),
                       "reference_answer_count": sum(row["reference_answer_count"] for row in history),
                       "explanation_count": sum(row["explanation_count"] for row in history)},
            "six_steps": six_steps,
            "coverage": {"difficulty": _coverage(history, "difficulty"), "populations": _coverage(history, "populations"),
                         "device_types": _coverage(history, "device_types"), "categories": _coverage(history, "category")},
            "weak_points": [{"step_id": key[0], "criterion_id": key[1], "status": key[2],
                             "label": weak_labels[key], "count": count} for key, count in weak.most_common(12)],
            "history": history, "trend": _trend(history)}
