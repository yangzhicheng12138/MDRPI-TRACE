"""Version-bound content reviews with explicit human or owner-authorized AI identity."""
from copy import deepcopy
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "data/teaching_bank/mdrpi_review_queue.json"
REVIEWERS_PATH = ROOT / "data/teaching_bank/reviewers.json"
SCHEMA_VERSION = "teaching_bank_review_v1"
WORKFLOW_FIELDS = {"content_fingerprint", "review_status", "formal_allowed", "review_history", "duplicate_warnings"}
AI_CHECKS = ("source_accuracy", "answer_correctness", "population_and_device", "safety", "scoring_consistency")


def configured_reviewers():
    """Only the repository owner's registry grants AI publication permission."""
    return json.loads(REVIEWERS_PATH.read_text(encoding="utf-8")) if REVIEWERS_PATH.exists() else {}


def ai_reviewer(registry, actor, timestamp):
    profile = (registry or {}).get(actor)
    if not isinstance(profile, dict) or profile.get("kind") != "ai" or profile.get("owner_authorized") is not True or profile.get("active") is not True:
        raise ValueError("AI approval requires an active owner-authorized AI reviewer.")
    for field in ("name", "authorization_record", "authorized_by", "authorized_at"):
        _required(profile.get(field), "AI reviewer " + field)
    _timestamp(profile["authorized_at"])
    if datetime.fromisoformat(timestamp.replace("Z", "+00:00")) < datetime.fromisoformat(profile["authorized_at"].replace("Z", "+00:00")):
        raise ValueError("AI review cannot predate owner authorization.")
    return profile


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def content_fingerprint(item):
    """Bind all clinical, source, licence and author fields to the reviewed version."""
    return _digest({key: value for key, value in item.items() if key not in WORKFLOW_FIELDS})


def normalize_question(text):
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in value if character.isalnum())


def _text(item):
    return item.get("case_text") or item.get("case_zh") or item.get("case") or ""


def _required(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " is required.")
    return value


def _timestamp(value):
    _required(value, "timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must use ISO 8601.") from error
    if timestamp.tzinfo is None or timestamp > datetime.now(timezone.utc):
        raise ValueError("timestamp must include a timezone and cannot be in the future.")
    return value


def _url(value):
    parts = urlsplit(_required(value, "source URL"))
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or any(character.isspace() for character in value):
        raise ValueError("Source links must be public HTTPS links without credentials.")
    if parts.hostname == "localhost" or parts.hostname.endswith(".local"):
        raise ValueError("Local source URLs are not public references.")
    try:
        address = ipaddress.ip_address(parts.hostname)
    except ValueError:
        address = None
    if address and not address.is_global:
        raise ValueError("Private source IP addresses are not public references.")
    return value


def validate_item(item):
    if not isinstance(item, dict):
        raise ValueError("Teaching item must be an object.")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,119}", item.get("id", "")):
        raise ValueError("A stable lowercase teaching item ID is required.")
    if type(item.get("version")) is not int or item["version"] < 1:
        raise ValueError("Item version must be a positive integer.")
    for field in ("title", "case_text", "content_origin", "author_id", "clinical_scope", "license", "source_type"):
        _required(item.get(field), field)
    if not normalize_question(item["case_text"]):
        raise ValueError("Question must include meaningful text.")
    if item.get("kind") not in {"knowledge_exercise", "case_question", "standardized_case"} or item.get("category") != "mdrpi":
        raise ValueError("Explicit supported professional teaching-item kind and category are required.")
    if item.get("difficulty") not in {"基础", "进阶", "复杂"}:
        raise ValueError("Unsupported case difficulty.")
    populations = item.get("populations")
    if not isinstance(populations, list) or not populations or len(set(populations)) != len(populations) or not set(populations) <= {"adult", "pediatric", "neonatal"}:
        raise ValueError("Explicit clinical population applicability is required.")
    taxonomy = json.loads((ROOT / "data/teaching_bank/catalog_taxonomy.json").read_text(encoding="utf-8"))
    devices = item.get("device_types")
    if not isinstance(devices, list) or not devices or len(set(devices)) != len(devices) or not set(devices) <= set(taxonomy["devices"]):
        raise ValueError("Explicit supported medical-device types are required.")
    if not isinstance(item.get("learning_objectives"), dict):
        raise ValueError("Learning objectives must be a language-keyed object.")
    goals = item["learning_objectives"].get("zh")
    if not isinstance(goals, list) or not goals or not all(isinstance(goal, str) and goal.strip() for goal in goals):
        raise ValueError("Learning objectives are required.")
    _url(item.get("license_url"))
    if not isinstance(item.get("source"), dict):
        raise ValueError("Source metadata must be an object.")
    sources = item["source"].get("references")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source references with locators and licences are required.")
    ids = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Source reference must be an object.")
        for field in ("id", "title", "edition", "locator", "license", "use_type"):
            _required(source.get(field), "reference " + field)
        if source["id"] in ids:
            raise ValueError("Duplicate source reference ID.")
        ids.add(source["id"])
        _url(source.get("url"))
        if not set(populations) <= set(source.get("populations", [])):
            raise ValueError("Reference applicability does not cover the item populations.")
        if source["use_type"] not in {"evidence_link", "adapted_cc_text"}:
            raise ValueError("Reference use must distinguish evidence links from reused text.")
        if source["use_type"] == "adapted_cc_text":
            if not source["license"].startswith("CC BY"):
                raise ValueError("Reused content requires a compatible explicit licence.")
            _url(source.get("license_url"))
        _timestamp(source.get("checked_at"))
    fields = item.get("field_sources", {})
    if not isinstance(fields, dict):
        raise ValueError("Field-level provenance must be an object.")
    if not all(isinstance(fields.get(key), str) and fields[key].strip() for key in ("case_text", "learning_objectives", "populations", "device_types", "difficulty")):
        raise ValueError("Field-level content and classification provenance is required.")
    if item.get("question_type"):
        from .bank_expansion import QUESTION_TYPES, TOPICS
        if item["question_type"] not in QUESTION_TYPES or item.get("topic") not in TOPICS:
            raise ValueError("Explicit supported question type and topic are required.")
        if item.get("question_type") == "clinical_case":
            STEP_IDS = ("recognize", "analyze", "prioritize", "solutions", "action", "evaluate")
            rubric = item.get("six_step_rubric") or {}
            if item.get("training_flow") != "six_step" or item["kind"] != "case_question" or not rubric.get("version"):
                raise ValueError("Clinical practice cases must declare their six-step training flow.")
            if tuple(step.get("id") for step in rubric.get("steps", [])) != STEP_IDS:
                raise ValueError("All six case steps must be provided in order.")
            for step in rubric["steps"]:
                if not step.get("reference_answer", {}).get("zh") or len(step.get("required", [])) != 2:
                    raise ValueError("Each case step requires an answer and two explicit requirements.")
                for point in step["required"]:
                    if not point.get("reference_ids") or not set(point["reference_ids"]) <= ids:
                        raise ValueError("Case step sources must be present in the reviewed source list.")
                    if not all(isinstance(point.get(field), dict) and all(point[field].get(lang) for lang in ("zh", "en")) for field in ("criterion", "acceptable", "incorrect", "follow_up")):
                        raise ValueError("Case point definitions must be complete.")
        if item["question_type"] in {"single_choice", "multiple_choice", "ordering"}:
            options, key = item.get("options"), item.get("correct_options")
            if not isinstance(options, dict) or len(options) < 3 or not all(re.fullmatch(r"[A-Z]", label) and isinstance(text, str) and text.strip() for label,text in options.items()):
                raise ValueError("Choice/ordering questions require distinct labelled options.")
            if not isinstance(key, list) or not key or len(set(key)) != len(key) or not set(key) <= set(options):
                raise ValueError("Answer choices must bind the supplied options.")
            if item["question_type"] == "single_choice" and len(key) != 1 or item["question_type"] == "multiple_choice" and len(key) < 2:
                raise ValueError("Single and multiple choice keys must match their question type.")
            if item["question_type"] == "ordering" and set(key) != set(options):
                raise ValueError("Ordering keys must include all supplied options once.")
        if item["question_type"] == "true_false" and type(item.get("truth_value")) is not bool:
            raise ValueError("Judgement questions require an explicit boolean answer.")
    return deepcopy(item)


def empty_queue():
    return {"schema_version": SCHEMA_VERSION, "items": [], "events": []}


def _event(queue, item, action, actor, timestamp, reason, details=None):
    if queue["events"] and datetime.fromisoformat(timestamp.replace("Z", "+00:00")) < datetime.fromisoformat(queue["events"][-1]["timestamp"].replace("Z", "+00:00")):
        raise ValueError("Audit events cannot move backwards in time.")
    event = {"event_id": len(queue["events"]) + 1, "item_id": item["id"], "item_version": item["version"],
             "content_fingerprint": content_fingerprint(item), "action": action, "actor": actor,
             "timestamp": timestamp, "reason": reason, "details": details or {},
             "previous_event_hash": queue["events"][-1]["event_hash"] if queue["events"] else ""}
    event["event_hash"] = _digest(event)
    queue["events"].append(event)
    item.setdefault("review_history", []).append(deepcopy(event))


def validate_queue(queue):
    if queue.get("schema_version") != SCHEMA_VERSION or not isinstance(queue.get("items"), list) or not isinstance(queue.get("events"), list):
        raise ValueError("Unsupported review queue schema.")
    previous, events = "", {}
    for number, event in enumerate(queue["events"], 1):
        digest = _digest({key: value for key, value in event.items() if key != "event_hash"})
        if event.get("event_id") != number or event.get("previous_event_hash") != previous or event.get("event_hash") != digest:
            raise ValueError("Review audit log is inconsistent.")
        previous = digest
        events[number] = event
    ids = set()
    for item in queue["items"]:
        validate_item(item)
        if item["id"] in ids:
            raise ValueError("Duplicate queue item ID.")
        ids.add(item["id"])
        for event in item.get("review_history", []):
            if event != events.get(event.get("event_id")) or event.get("item_id") != item["id"]:
                raise ValueError("Item review history does not match the audit log.")
    return queue


def load_review_queue(path=QUEUE_PATH):
    return validate_queue(json.loads(Path(path).read_text(encoding="utf-8")))


def duplicate_checks(item, existing_items, threshold=0.82):
    target = normalize_question(_text(item))
    warnings = []
    for candidate in existing_items:
        other = normalize_question(_text(candidate))
        if not other:
            continue
        if target == other:
            raise ValueError("Normalized duplicate question: " + candidate.get("id", "existing_unlabelled_question"))
        ratio = SequenceMatcher(None, target, other, autojunk=False).ratio()
        if ratio >= threshold:
            warnings.append({"item_id": candidate.get("id", "existing_unlabelled_question"), "similarity": round(ratio, 4), "kind": "near_duplicate_requires_review"})
    return warnings


def import_items(queue, incoming, *, existing_items=(), actor="content_import", timestamp=None):
    """Validate the full batch first; a rejected row never partially publishes it."""
    result = deepcopy(validate_queue(queue))
    timestamp = _timestamp(timestamp or datetime.now(timezone.utc).isoformat())
    _required(actor, "import actor")
    incoming = list(incoming)
    if not incoming:
        raise ValueError("Import batch is empty.")
    outside = list(existing_items)
    for raw in incoming:
        item = validate_item(raw)
        if any(item["id"] == candidate.get("id") for candidate in result["items"] + outside):
            raise ValueError("Duplicate stable teaching item ID: " + item["id"])
        item = {key: value for key, value in item.items() if key not in WORKFLOW_FIELDS}
        item.update(review_status="pending_review", formal_allowed=False, review_history=[])
        item["duplicate_warnings"] = duplicate_checks(item, result["items"] + outside)
        item["content_fingerprint"] = content_fingerprint(item)
        _event(result, item, "import", actor, timestamp, "Validated source-bound draft import.")
        result["items"].append(item)
    return result


def _reviewer(registry, expert_id, item):
    profile = (registry or {}).get(expert_id)
    if not isinstance(profile, dict) or profile.get("independent") is not True or profile.get("identity_verified") is not True:
        raise ValueError("Approval requires an independently verified expert in the configured registry.")
    for field in ("name", "qualifications", "verification_record", "verified_by"):
        _required(profile.get(field), "expert " + field)
    _timestamp(profile.get("verified_at"))
    if expert_id in set(item.get("author_ids", [])) | {item["author_id"]}:
        raise ValueError("Content authors cannot independently approve their own item.")
    if not set(item["populations"]) <= set(profile.get("populations", [])):
        raise ValueError("Expert applicability does not cover the item populations.")
    if not set(item["device_types"]) <= set(profile.get("device_types", [])):
        raise ValueError("Expert applicability does not cover the item devices.")
    return profile


def _scoring(item):
    scoring = item.get("formal_scoring")
    if not isinstance(scoring, dict) or not scoring.get("version") or not isinstance(scoring.get("reference_answer"), str) or not scoring["reference_answer"].strip():
        raise ValueError("Formal approval requires a source-bound reference answer and scoring version.")
    points = scoring.get("must_answer_points")
    if not isinstance(points, list) or not points or not all(isinstance(point, dict) and point.get("id") and point.get("criterion") and point.get("acceptable") and point.get("incorrect") for point in points):
        raise ValueError("Formal approval requires explicit auditable scoring criteria.")
    if len({point["id"] for point in points}) != len(points):
        raise ValueError("Formal scoring criterion IDs must be unique.")
    references = {reference["id"] for reference in item["source"]["references"]}
    if not scoring.get("reference_ids") or not set(scoring["reference_ids"]) <= references:
        raise ValueError("Formal scoring must identify its supporting sources.")


def review_item(queue, item_id, decision, record, *, reviewers=None):
    """Approve the exact version, recording AI review separately from expert review."""
    result = deepcopy(validate_queue(queue))
    item = next((row for row in result["items"] if row["id"] == item_id), None)
    if item is None:
        raise ValueError("Unknown teaching item ID.")
    if decision not in {"approve", "reject", "request_revision"}:
        raise ValueError("Unsupported review decision.")
    for field in ("actor", "reason", "basis", "review_record_id"):
        _required(record.get(field), "review " + field)
    timestamp = _timestamp(record.get("timestamp"))
    fingerprint = content_fingerprint(item)
    if item.get("content_fingerprint") != fingerprint:
        raise ValueError("Changed content must be submitted as a higher-version revision before review.")
    if type(record.get("item_version")) is not int or record["item_version"] != item["version"] or record.get("content_fingerprint") != fingerprint:
        raise ValueError("Review must bind the current item version and content fingerprint.")
    if any(event["details"].get("review_record_id") == record["review_record_id"] for event in result["events"]):
        raise ValueError("Duplicate review record ID.")
    details = {key: deepcopy(record[key]) for key in ("basis", "review_record_id", "item_version", "content_fingerprint")}
    if decision == "approve" and record.get("reviewer_kind") == "ai":
        _scoring(item)
        profile = ai_reviewer(configured_reviewers() if reviewers is None else reviewers, record["actor"], timestamp)
        if not set(item["populations"]) <= set(profile.get("populations", [])) or not set(item["device_types"]) <= set(profile.get("device_types", [])):
            raise ValueError("AI review authorization does not cover the item scope.")
        checks = record.get("checks", {})
        if not isinstance(checks, dict) or not all(isinstance(checks.get(key), str) and checks[key].strip() for key in AI_CHECKS):
            raise ValueError("AI approval requires item-specific source, answer, scope, safety and scoring checks.")
        if set(record.get("duplicate_reviewed", [])) != {row["item_id"] for row in item.get("duplicate_warnings", [])}:
            raise ValueError("Reviewer must explicitly review all near-duplicate warnings.")
        details.update(reviewer_kind="ai", reviewer_name=profile["name"], reviewer_profile_fingerprint=_digest(profile),
                       authorization_record=profile["authorization_record"], checks=deepcopy(checks),
                       duplicate_reviewed=record.get("duplicate_reviewed", []))
    elif decision == "approve":
        _scoring(item)
        expert_id = _required(record.get("expert_id"), "expert ID")
        if record["actor"] != expert_id:
            raise ValueError("Approval actor must be the independently identified expert.")
        profile = _reviewer(reviewers, expert_id, item)
        if datetime.fromisoformat(timestamp.replace("Z", "+00:00")) < datetime.fromisoformat(profile["verified_at"].replace("Z", "+00:00")):
            raise ValueError("Review cannot predate expert identity verification.")
        warnings = item.get("duplicate_warnings", [])
        if warnings and set(record.get("duplicate_reviewed", [])) != {row["item_id"] for row in warnings}:
            raise ValueError("Expert must explicitly review all near-duplicate warnings.")
        details.update(expert_id=expert_id, expert_profile_fingerprint=_digest(profile),
                       expert_name=profile["name"], qualifications=profile["qualifications"],
                       verification_record=profile["verification_record"], independent=True,
                       duplicate_reviewed=record.get("duplicate_reviewed", []))
    item["review_status"] = {"approve": "approved", "reject": "rejected", "request_revision": "revision_requested"}[decision]
    if decision == "approve" and record.get("reviewer_kind") == "ai":
        item["review_status"] = "approved_ai"
    item["formal_allowed"] = decision == "approve"
    item["content_fingerprint"] = fingerprint
    _event(result, item, decision, record["actor"], timestamp, record["reason"], details)
    return result


def revise_item(queue, item_id, replacement, *, editor, reason, timestamp=None, existing_items=()):
    result = deepcopy(validate_queue(queue))
    old = next((row for row in result["items"] if row["id"] == item_id), None)
    if old is None:
        raise ValueError("Unknown teaching item ID.")
    new = validate_item(replacement)
    if new["id"] != old["id"] or new["version"] <= old["version"]:
        raise ValueError("A revision must preserve the stable ID and increase its version.")
    _required(editor, "revision editor")
    _required(reason, "revision reason")
    timestamp = _timestamp(timestamp or datetime.now(timezone.utc).isoformat())
    new = {key: value for key, value in new.items() if key not in WORKFLOW_FIELDS}
    candidates = [row for row in result["items"] if row["id"] != item_id] + list(existing_items)
    new.update(review_status="pending_review", formal_allowed=False, review_history=deepcopy(old["review_history"]),
               duplicate_warnings=duplicate_checks(new, candidates))
    new["content_fingerprint"] = content_fingerprint(new)
    _event(result, new, "revise", editor, timestamp, reason, {"previous_version": old["version"], "previous_fingerprint": old["content_fingerprint"]})
    result["items"][result["items"].index(old)] = new
    return result


def formal_allowed(item, reviewers=None):
    """Recheck reviewed content, scoring and current reviewer authorization."""
    try:
        validate_item(item)
        _scoring(item)
        history = item.get("review_history") or []
        event = history[-1]
        details = event["details"]
        digest = content_fingerprint(item)
        if item.get("review_status") not in {"approved", "approved_ai"} or item.get("formal_allowed") is not True or event.get("action") != "approve":
            return False
        if item.get("content_fingerprint") != digest or event.get("content_fingerprint") != digest or event.get("item_version") != item["version"]:
            return False
        if details.get("content_fingerprint") != digest or details.get("item_version") != item["version"] or not details.get("basis") or not details.get("review_record_id"):
            return False
        if event.get("event_hash") != _digest({key: value for key, value in event.items() if key != "event_hash"}):
            return False
        _timestamp(event["timestamp"])
        if item["review_status"] == "approved_ai":
            profile = ai_reviewer(configured_reviewers() if reviewers is None else reviewers, event["actor"], event["timestamp"])
            if details.get("reviewer_kind") != "ai" or details.get("reviewer_profile_fingerprint") != _digest(profile):
                return False
            if not set(item["populations"]) <= set(profile.get("populations", [])) or not set(item["device_types"]) <= set(profile.get("device_types", [])):
                return False
            if not all(isinstance(details.get("checks", {}).get(key), str) and details["checks"][key].strip() for key in AI_CHECKS):
                return False
        else:
            profile = _reviewer(reviewers, details["expert_id"], item)
            if details.get("expert_profile_fingerprint") != _digest(profile) or event.get("actor") != details["expert_id"]:
                return False
        if set(details.get("duplicate_reviewed", [])) != {warning["item_id"] for warning in item.get("duplicate_warnings", [])}:
            return False
        return True
    except (ValueError, KeyError, TypeError, IndexError):
        return False


def browse_review_queue(queue, status=None, *, reviewers=None):
    validate_queue(queue)
    rows = []
    for original in queue["items"]:
        item = deepcopy(original)
        if item["content_fingerprint"] != content_fingerprint(item):
            item["review_status"] = "approval_invalidated" if item["review_status"] in {"approved", "approved_ai"} else "content_changed_pending_review"
        item["formal_allowed"] = formal_allowed(original, reviewers)
        if not status or item["review_status"] == status:
            rows.append(item)
    return rows


def approved_items(queue, *, reviewers=None):
    return [item for item in browse_review_queue(queue, reviewers=reviewers) if item["formal_allowed"]]
