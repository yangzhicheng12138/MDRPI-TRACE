"""Reviewed teaching additions bound to immutable source questions and answer keys."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .bank_review import AI_CHECKS, ai_reviewer, configured_reviewers

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/teaching_bank/content_reviews.json"
STEP_IDS = ("recognize", "analyze", "prioritize", "solutions", "action", "evaluate")
ALLOWED_ADDITIONS = {"answer", "answer_origin", "has_reference_answer", "six_step_rubric", "must_answer_points",
                     "review_references", "review_note", "missing_fields", "rubric", "case_zh", "reference_note",
                     "question_type", "training_flow", "options", "options_en", "correct_options",
                     "summary_constraints"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_reviews(path=PATH):
    if not Path(path).exists():
        return {"schema_version": "content_review_v1", "items": {}}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "content_review_v1" or not isinstance(data.get("items"), dict):
        raise ValueError("Unsupported content review ledger.")
    return data


def validate_review(item, record, registry=None):
    if record.get("base_fingerprint") != fingerprint(item):
        raise ValueError("Source question or key changed after review.")
    if record.get("review_fingerprint") != fingerprint({k: v for k, v in record.items() if k != "review_fingerprint"}):
        raise ValueError("Reviewed additions changed after review.")
    if record.get("status") != "approved_ai" or record.get("reviewer_kind") != "ai":
        raise ValueError("This record is not an AI approval.")
    from .bank_review import _timestamp
    _timestamp(record.get("reviewed_at"))
    profile = ai_reviewer(configured_reviewers() if registry is None else registry, record.get("reviewer_id"), record["reviewed_at"])
    if record.get("reviewer_profile_fingerprint") != fingerprint(profile):
        raise ValueError("Reviewer authorization changed.")
    if not all(isinstance(record.get("checks", {}).get(key), str) and record["checks"][key].strip() for key in AI_CHECKS):
        raise ValueError("Item-specific review findings are required.")
    if not record.get("references"):
        raise ValueError("Review references are required.")
    additions = record.get("additions", {})
    if not isinstance(additions, dict) or not set(additions) <= ALLOWED_ADDITIONS:
        raise ValueError("Review may not overwrite source identity or unrelated fields.")
    if "summary_constraints" in additions:
        constraint = additions["summary_constraints"]
        if not isinstance(constraint, dict) or not isinstance(constraint.get("excluded_recommendations"), list) or not all(
                isinstance(text, str) and text.strip() for text in constraint["excluded_recommendations"]):
            raise ValueError("Summary exclusions must be explicit reviewed recommendation phrases.")
        if not isinstance(constraint.get("guidance"), dict) or not all(constraint["guidance"].get(lang) for lang in ("zh", "en")):
            raise ValueError("Reviewed summary constraints need localized positive guidance.")
    if any(key in additions for key in ("question_type", "options", "options_en", "correct_options")):
        kind = additions.get("question_type")
        expected = {"case_mcq": "single_choice", "case_sequence": "ordering"}.get(item.get("kind"))
        options, correct = additions.get("options"), additions.get("correct_options")
        if kind != expected or additions.get("training_flow") != "short_question":
            raise ValueError("Legacy objective metadata must preserve the source question kind.")
        if not isinstance(options, dict) or len(options) < 2 or not all(
                isinstance(label, str) and len(label) == 1 and "A" <= label <= "Z"
                and isinstance(text, str) and text.strip() for label, text in options.items()):
            raise ValueError("Reviewed objective options need explicit stable labels and text.")
        if not isinstance(correct, list) or not correct or len(set(correct)) != len(correct) or not set(correct) <= set(options):
            raise ValueError("Reviewed objective keys must identify the supplied options.")
        if (kind == "single_choice" and len(correct) != 1) or (kind == "ordering" and set(correct) != set(options)):
            raise ValueError("Reviewed objective key does not match its question type.")
        if "options_en" in additions and (not isinstance(additions["options_en"], dict)
                or set(additions["options_en"]) != set(options)
                or not all(isinstance(text, str) and text.strip() for text in additions["options_en"].values())):
            raise ValueError("Localized objective options must retain the same labels.")
    if item.get("kind") in {"standardized_case", "fictional_case"}:
        if "case_zh" in additions:
            raise ValueError("Reviewed case additions cannot rewrite the original narrative.")
        rubric = additions.get("six_step_rubric", {})
        if not rubric.get("version") or tuple(row.get("id") for row in rubric.get("steps", [])) != STEP_IDS:
            raise ValueError("A reviewed case requires all six steps.")
        for step in rubric["steps"]:
            if not step.get("reference_answer", {}).get("zh") or len(step.get("required", [])) != 2:
                raise ValueError("Each reviewed step requires an answer and two explicit requirements.")
            for row in step["required"]:
                if not all(isinstance(row.get(key), dict) and all(row[key].get(lang) for lang in ("zh", "en")) for key in ("criterion", "acceptable", "incorrect", "follow_up")):
                    raise ValueError("Reviewed requirements must have complete localized fields.")
                if not row.get("reference_ids") or not set(row["reference_ids"]) <= {r["id"] for r in record["references"]}:
                    raise ValueError("Required-point sources do not match this review.")
    return record


def apply_review(item, *, ledger=None, registry=None):
    """Fail closed on stale additions while retaining the original practice item."""
    record = (load_reviews() if ledger is None else ledger)["items"].get(item.get("id"))
    if not record:
        return item
    try:
        validate_review(item, record, registry)
    except (ValueError, KeyError, TypeError):
        return {**item, "review_status": "approval_invalidated", "content_review": {"status": "approval_invalidated"}}
    result = deepcopy(item)
    result.update(deepcopy(record["additions"]))
    result.update(review_status="approved_ai", content_review={key: deepcopy(record[key]) for key in
                  ("status", "reviewer_id", "reviewer_kind", "reviewed_at", "review_fingerprint", "checks", "references")})
    return result
