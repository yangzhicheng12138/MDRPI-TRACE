"""Strict, development-only contracts for source-grounded question creation.

These contracts do not publish to the teaching bank. Source approval, immutable
case facts, actual page reads and model-role separation belong to the controller.
"""
from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError


POPULATIONS = ("adult", "pediatric", "neonatal")
DIFFICULTIES = ("基础", "进阶", "复杂")
QUESTION_TYPES = ("short_answer", "single_choice")
ISSUE_KINDS = (
    "medical_error", "evidence_mismatch", "material_gap", "ambiguity",
    "assessment_mismatch", "format",
)
VERDICTS = ("accept", "revise", "needs_evidence", "reject")


def _object(properties, required=None):
    return {
        "type": "object", "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _text(limit=500, *, minimum=1):
    return {"type": "string", "minLength": minimum, "maxLength": limit,
            **({"pattern": r"\S"} if minimum else {})}


def _list(items, *, minimum=0, maximum=8, unique=False):
    return {
        "type": "array", "items": items, "minItems": minimum,
        "maxItems": maximum, **({"uniqueItems": True} if unique else {}),
    }


def _evidence(evidence_ids, *, minimum=0):
    ids = list(dict.fromkeys(evidence_ids))
    if not ids or any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("At least one nonempty, actually read evidence ID is required")
    return _list({"type": "string", "enum": ids}, minimum=minimum, maximum=16, unique=True)


def _gap_schema():
    return _object({"query": _text(180), "reason": _text(400)})


def blueprint_schema():
    """Input contract; reviewed development-source membership is checked upstream."""
    properties = {
        "topic": _text(120), "learning_objective": _text(400),
        "population": {"type": "string", "enum": list(POPULATIONS)},
        "device_type": _text(100),
        "difficulty": {"type": "string", "enum": list(DIFFICULTIES)},
        "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
        "count": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
        "material_ids": _list(_text(160), minimum=1, maximum=5, unique=True),
        "requirements": {**_text(2000, minimum=0), "default": ""},
    }
    return _object(properties, [key for key in properties if key not in {"count", "requirements"}])


def validate_blueprint(value):
    """Normalize a copy, apply optional defaults, then fail closed on bad inputs."""
    result = deepcopy(value)
    if isinstance(result, dict):
        result.setdefault("count", 1)
        result.setdefault("requirements", "")
        for key, item in result.items():
            if isinstance(item, str):
                result[key] = item.strip()
        if isinstance(result.get("material_ids"), list):
            result["material_ids"] = [item.strip() if isinstance(item, str) else item
                                      for item in result["material_ids"]]
    Draft202012Validator(blueprint_schema()).validate(result)
    return result


def anchor_plan_schema(evidence_ids):
    """Optional separate planning contract; the normal generation call plans inline."""
    return _object({
        "learning_objective": _text(400), "planned_task": _text(500),
        "required_criteria": _list(_object({
            "id": {"type": "string", "enum": ["R1", "R2", "R3"]},
            "criterion": _text(350), "evidence_ids": _evidence(evidence_ids, minimum=1),
        }), minimum=2, maximum=3),
        "difficulty_rationale": _text(300),
        "evidence_gaps": _list(_gap_schema(), maximum=3),
    })


def question_schema(evidence_ids, question_type):
    """One question, including its teacher-only answer and scoring package."""
    if question_type not in QUESTION_TYPES:
        raise ValueError("Unsupported question type")
    single_choice = question_type == "single_choice"
    option = _object({"label": {"type": "string", "enum": list("ABCDE")}, "text": _text(350)})
    return _object({
        "title": _text(100), "stem": _text(2400), "task": _text(500),
        "options": _list(option, minimum=3 if single_choice else 0,
                         maximum=5 if single_choice else 0),
        "correct_options": _list({"type": "string", "enum": list("ABCDE")},
                                 minimum=1 if single_choice else 0,
                                 maximum=1 if single_choice else 0, unique=True),
        "reference_answer": _text(1800),
        "scoring_points": _list(_object({
            "id": {"type": "string", "enum": ["R1", "R2", "R3"]},
            "criterion": _text(400),
            "acceptable_answers": _list(_text(350), minimum=1, maximum=4, unique=True),
            "evidence_ids": _evidence(evidence_ids, minimum=1),
        }), minimum=1 if single_choice else 2, maximum=1 if single_choice else 3),
        "hints": _list(_text(240), minimum=3, maximum=3, unique=True),
        "common_misconceptions": _list(_text(300), minimum=1, maximum=4, unique=True),
        "evidence_ids": _evidence(evidence_ids, minimum=1),
    })


def validate_question(value, evidence_ids, question_type):
    """Enforce cross-field identity constraints in addition to the JSON schema."""
    Draft202012Validator(question_schema(evidence_ids, question_type)).validate(value)
    labels = [item["label"] for item in value["options"]]
    if len(labels) != len(set(labels)):
        raise ValidationError("Option labels must be unique")
    texts = [" ".join(item["text"].split()).casefold() for item in value["options"]]
    if len(texts) != len(set(texts)):
        raise ValidationError("Option texts must be distinct")
    if any(label not in labels for label in value["correct_options"]):
        raise ValidationError("Correct option must name an existing option")
    criterion_ids = [item["id"] for item in value["scoring_points"]]
    if len(criterion_ids) != len(set(criterion_ids)):
        raise ValidationError("Scoring-point IDs must be unique")
    cited_ids = set(value["evidence_ids"])
    if any(not set(item["evidence_ids"]).issubset(cited_ids) for item in value["scoring_points"]):
        raise ValidationError("Question evidence_ids must include every scoring-point reference")
    return deepcopy(value)


def blind_review_schema():
    """The blind checker must not receive the answer, rubric, hints or full review."""
    return _object({
        "answerable": {"type": "boolean"},
        "attempted_answer": _text(1400, minimum=0),
        "ambiguity_notes": _list(_text(350), maximum=5),
    })


def review_schema(evidence_ids):
    """Reviewer decisions separate incorrect medicine from absent documentation."""
    issue = _object({
        "kind": {"type": "string", "enum": list(ISSUE_KINDS)},
        "location": _text(160), "text": _text(700, minimum=0),
        "explanation": _text(600), "evidence_ids": _evidence(evidence_ids),
        "requested_change": _text(500),
    })
    schema = _object({
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "issues": _list(issue, maximum=8),
        "evidence_gaps": _list(_gap_schema(), maximum=3),
        "summary": _text(600),
    })
    schema["allOf"] = [
        {"if": {"properties": {"verdict": {"const": "accept"}}},
         "then": {"properties": {"issues": {"maxItems": 0}, "evidence_gaps": {"maxItems": 0}}}},
        {"if": {"properties": {"verdict": {"const": "needs_evidence"}}},
         "then": {"properties": {"evidence_gaps": {"minItems": 1}}}},
        {"if": {"properties": {"verdict": {"enum": ["revise", "reject"]}}},
         "then": {"properties": {"issues": {"minItems": 1}}}},
    ]
    return schema


def validate_review(value, evidence_ids):
    """A pass cannot hide an unresolved issue or an evidence gap."""
    Draft202012Validator(review_schema(evidence_ids)).validate(value)
    return deepcopy(value)
