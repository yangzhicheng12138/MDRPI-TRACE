"""A versioned display taxonomy; original questions and grading stay unchanged."""
from functools import lru_cache
import json
import hashlib
from pathlib import Path

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data/teaching_bank/catalog_taxonomy.json"


@lru_cache(maxsize=1)
def taxonomy():
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def catalog_entry(case_id, item):
    data = taxonomy()
    module = data["module_mapping"].get(item["category"], "foundation")
    if item.get("kind") == "knowledge_exercise":
        module = "foundation"
    annotation = data["specialty_items"].get(case_id, {}) if module == "specialty" else {}
    if item.get("classification"):
        annotation = item.get("classification", {})
    source_hash = hashlib.sha256(json.dumps({"case": item.get("case"), "case_zh": item.get("case_zh"), "scope": item.get("scope")}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if annotation and annotation.get("source_sha256") != source_hash:
        annotation = {}
    original = item.get("level", "")
    populations, devices = annotation.get("populations", []), annotation.get("devices", [])
    status = item.get("review_status", "unreviewed_practice")
    source = item.get("source") or {}
    from .bank_expansion import public_metadata
    metadata = public_metadata(item)
    # Educational browsing categories have their own source-bound annotations.
    # The original content topic remains available to learning recommendations.
    labels = data.get("foundation_topics", {}) if module == "foundation" else None
    educational = data.get("foundation_items", {}).get(case_id, {}) if module == "foundation" else {}
    if educational.get("source_sha256") != source_hash or (labels is not None and educational.get("topic") not in labels):
        educational = {}
    learning_topic = educational.get("topic", "") if module == "foundation" else metadata["topic"]
    learning_label = labels.get(learning_topic, "待归类") if labels is not None else metadata["topic_label"]
    supplemental_topics = data.get("foundation_supplemental_topics", {})
    supplement = supplemental_topics.get(learning_topic, {})
    if module == "specialty":
        topic_origin = "mdrpi_specialty"
    elif not learning_topic:
        topic_origin = "unclassified"
    else:
        topic_origin = "project_supplement" if learning_topic in supplemental_topics else "paper_table"
    return {"id": case_id, **{key: item[key] for key in ("title", "level", "category", "kind", "has_reference_answer")}, **metadata,
            "learning_topic": learning_topic, "learning_topic_label": learning_label,
            "learning_topic_origin": topic_origin, "learning_topic_description": supplement.get("description", ""),
            "learning_topic_basis": educational.get("basis", "") if module == "foundation" else "MDRPI案例训练内容主题",
            "learning_topic_pending": module == "foundation" and (not educational or learning_topic not in labels),
            "answer_origin": item.get("answer_origin", "textbook_key" if item["has_reference_answer"] else "none"),
            "training_module": module, "populations": list(populations), "device_types": list(devices),
            "population_labels": [data["populations"][key] for key in populations],
            "device_labels": [data["devices"][key] for key in devices],
            "difficulty": data["difficulty_aliases"].get(original, original), "source_difficulty": original,
            "taxonomy_version": data["version"], "classification_basis": annotation.get("basis", "original category: " + item["category"]),
            "classification_pending": module == "specialty" and not annotation,
            "review_status": status, "source_reference": {key: source[key] for key in ("book", "section", "question_number", "question_url", "license", "source_file", "source_sha256", "source_version", "original_id", "snapshot_file", "snapshot_sha256", "case_text_sha256") if key in source},
            **({"case_id": item["case_id"], "version": item["version"], "source_type": item["source_type"],
                "missing_fields": list(item["missing_fields"]), "supported_languages": list(item["supported_languages"])}
               if item.get("kind") == "standardized_case" else {})}


def filter_catalog(rows, *, module=None, population=None, device_type=None, difficulty=None, question_type=None, topic=None, learning_topic=None):
    data = taxonomy()
    from .bank_expansion import QUESTION_TYPES, TOPICS
    for value, allowed, field in ((module, data["modules"], "模块"), (population, data["populations"], "人群"),
                                   (device_type, data["devices"], "器械"), (difficulty, data["difficulties"], "难度"),
                                   (question_type, QUESTION_TYPES, "题型"), (topic, TOPICS, "主题")):
        if value and value not in allowed:
            raise ValueError("未知" + field + "筛选值。")
    if module == "foundation":
        learning_topics = data.get("foundation_topics", {})
    elif module == "specialty":
        learning_topics = TOPICS
    else:
        learning_topics = {**TOPICS, **data.get("foundation_topics", {})}
    supplemental_topics = data.get("foundation_supplemental_topics", {})
    # Preserve the old broad query as an aggregate of the new project topics.
    legacy_supplement = learning_topic == "related_nursing" and module != "specialty" and bool(supplemental_topics)
    if learning_topic and learning_topic not in learning_topics and not legacy_supplement:
        raise ValueError("未知当前模块的学习主题筛选值。")
    if module == "foundation" and (population or device_type):
        raise ValueError("基础知识练习按教育主题分类，不使用人群或器械筛选。")
    return [row for row in rows if (not module or row["training_module"] == module)
            and (not population or population in row["populations"])
            and (not device_type or device_type in row["device_types"])
            and (not difficulty or row["difficulty"] == difficulty)
            and (not question_type or row["question_type"] == question_type) and (not topic or row["topic"] == topic)
            and (not learning_topic or (row.get("learning_topic") in supplemental_topics if legacy_supplement
                                       else row.get("learning_topic") == learning_topic))]


def catalog_facets(rows, *, module="specialty", population=None, device_type=None, difficulty=None, question_type=None, learning_topic=None):
    data = taxonomy()
    from .bank_expansion import QUESTION_TYPES, TOPICS
    if module not in data["modules"]:
        raise ValueError("未知模块筛选值。")
    base = filter_catalog(rows, module=module)
    selected = {"module": module, "population": population, "device_type": device_type,
                "difficulty": difficulty, "question_type": question_type, "learning_topic": learning_topic}
    # Validate the complete selection even when no rows match. Counts omit only
    # their own dimension so users can move between alternatives directly.
    filter_catalog(rows, **selected)
    def alternatives(field):
        return filter_catalog(rows, **{key: value for key, value in selected.items() if key != field})
    people = alternatives("population") if module == "specialty" else []
    devices = alternatives("device_type") if module == "specialty" else []
    device_scope = filter_catalog(base, population=population) if module == "specialty" else []
    difficulties = alternatives("difficulty")
    question_types = alternatives("question_type")
    topics = alternatives("learning_topic")
    topic_labels = data.get("foundation_topics", {}) if module == "foundation" else TOPICS
    supplemental_topics = data.get("foundation_supplemental_topics", {})
    available_topics = {key: sum(row.get("learning_topic") == key and not row.get("learning_topic_pending", False)
                                 for row in base) for key in topic_labels}
    result = {"version": data["version"],
            "modules": [{"id": key, "label": label, "count": sum(row["training_module"] == key for row in rows)} for key, label in data["modules"].items()],
            "populations": [{"id": key, "label": label, "count": sum(key in row["populations"] for row in people),
                             "source_example": data.get("population_source_examples", {}).get(key, "")} for key, label in data["populations"].items()] if module == "specialty" else [],
            "devices": [{"id": key, "label": label, "count": sum(key in row["device_types"] for row in devices)}
                        for key, label in data["devices"].items() if any(key in row["device_types"] for row in device_scope)] if module == "specialty" else [],
            "difficulties": [{"id": value, "label": value, "count": sum(row["difficulty"] == value for row in difficulties)} for value in data["difficulties"]],
            "question_types": [{"id": key, "label": value, "count": sum(row["question_type"] == key for row in question_types)} for key,value in QUESTION_TYPES.items() if any(row["question_type"] == key for row in base)],
            "topics": [{"id": key, "label": value,
                        "count": sum(row.get("learning_topic") == key and not row.get("learning_topic_pending", False) for row in topics),
                        "available_count": available_topics[key],
                        "origin": ("project_supplement" if key in supplemental_topics else "paper_table") if module == "foundation" else "mdrpi_specialty",
                        "description": supplemental_topics.get(key, {}).get("description", "") if module == "foundation" else ""}
                       for key,value in topic_labels.items()
                       if available_topics[key] > 0],
            "unclassified_count": sum(row.get("learning_topic_pending", False) for row in base)}
    if module == "foundation":
        result["classification_source"] = data.get("foundation_topic_source", {})
    else:
        result["classification_source"] = {"kind": "project", "title": "MDRPI专项案例训练分类"}
    return result


def classic_filter_choices(rows, payload):
    """Module-specific browsing for the retained Chainlit interface."""
    stage = payload.get("filter_stage", "list")
    if stage not in {"population", "device", "difficulty", "learning_topic"}:
        return []
    facets = catalog_facets(rows, module=payload["module"], population=payload.get("population"), device_type=payload.get("device_type"),
                            learning_topic=payload.get("learning_topic"), question_type=payload.get("question_type"))
    field, key, next_stage = {"population": ("population", "populations", "device"),
                              "device": ("device_type", "devices", "difficulty"),
                              "difficulty": ("difficulty", "difficulties", "list"),
                              "learning_topic": ("learning_topic", "topics", "difficulty")}[stage]
    choices = [{"label": "全部", "count": None, "payload": {**payload, field: "", "filter_stage": next_stage, "page": 0}}]
    choices += [{"label": row["label"], "count": row["count"],
                 "payload": {**payload, field: row["id"], "filter_stage": next_stage, "page": 0}} for row in facets[key]]
    return choices
