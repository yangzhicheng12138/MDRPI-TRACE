"""Versioned expansion content and public question-type/topic metadata."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/teaching_bank/expansion_250.json"
CHOICE_SUPPLEMENT_PATH = ROOT / "data/teaching_bank/choice_supplement.json"
OBJECTIVE_RULES_PATH = ROOT / "data/teaching_bank/objective_rules.json"
OBJECTIVE_RULE_FIELDS = ("id", "case", "case_zh", "answer", "question_type", "options", "options_en", "correct_options")
QUESTION_TYPES = {"single_choice": "单选题", "multiple_choice": "多选题", "true_false": "判断题",
                  "ordering": "排序题", "short_answer": "简答题", "concept_card": "概念卡", "clinical_case": "六步案例"}
TOPICS = {"mechanism": "皮肤结构与受力机制", "risk": "风险识别", "assessment": "组织与器械评估",
          "classification": "分期与鉴别", "prevention": "通用预防", "device_care": "器械专项",
          "special_population": "特殊人群与沟通", "documentation": "记录与团队协作", "evidence": "循证与质量改进"}


def minimum_acceptance(step_id, number, reference):
    """State the published minimum; examples never become extra requirements."""
    minimum = {
        "recognize": (
            "结合一条题面线索指出一项关注或评估目标即可。已指名的未知变量与同句或上下文中的器械、组织线索有联系，也可作为评估目标；不要求MDRPI术语、穷尽观察或完整方案。",
            "One assessment target linked to a supplied cue is enough. A named unknown linked to device or tissue findings in the sentence or context is an assessment target. No MDRPI label, exhaustive findings, or complete plan is required."),
        "analyze": (
            "接受基于题面线索的相符解释及本例相关的证据、未知或组织分类边界，不强制参考之外的全套鉴别项目或固定警句。",
            "Accept an explanation linked to supplied cues and the relevant evidence, uncertainty, or tissue-classification boundary; do not demand a complete differential or a particular warning phrase."),
        "prioritize": (
            "说明当前目标为何优先即可：将目标与一条病例线索或必要治疗安全联系起来，理由可自然嵌在目标句中；不要求列全参考、固定次序词或数值。",
            "Link the priority to one supplied finding or the safety of necessary treatment. The reason may be embedded in the goal; exhaustive detail, fixed ordering words, or numeric values are unnecessary."),
        "solutions": (
            "给出一项与当前措施相关的安全考虑或实施前核查即可；材料适配、温和处理、接触组织检查、必要支持、许可或团队复核可按本题关系表达。措施中明确包含的安全考虑也算，不强制同时列出全部条件、时间或数值。",
            "One safety consideration or prerequisite relevant to the proposed measure is sufficient, including material suitability, gentle handling, tissue checks, necessary support, authorization, or team review as applicable to this case. It may be embedded in the measure. Do not require every condition, an interval, or a numeric value."),
        "action": (
            "说明一项相关监测目标，或一项明确协作、报告、升级、许可边界即可。结合本例说明按负责团队或应急流程处理、不独自进行未经允许的操作，可以体现协作边界；不要求同时列监测和协作，也不要求给出团队成员姓名或固定角色词。",
            "Specify one relevant monitoring target OR a clear coordination, reporting, escalation, or authorization boundary. A case-linked responsible-team or emergency pathway and avoiding unauthorized solo procedures may express that boundary. Do not require both branches, a team member's name, or fixed role terminology."),
        "evaluate": (
            "按本题目标说明需比较的组织或器械表现，保留未给出的效果和原因为未知；可用需复评或异常时升级表达，不强制统一复查频率。",
            "Compare tissue or device findings relevant to this goal while keeping unavailable results or causes unknown. Planned reassessment or escalation for an abnormal finding is acceptable; no universal review interval is required."),
    }
    general = ("接受符合当前要求的等效表达，不要求逐字复述或列全参考说明。",
               "Accept equivalent meaning for this requirement, without verbatim repetition or every elaboration.")
    zh, en = minimum.get(step_id, general) if number == 2 else general
    # The step already contains its reference answer. Repeating a long reference
    # inside both points increases prompt size and can turn examples into lists.
    return {"zh": zh, "en": en}


def legacy_objective_metadata():
    """Explicit labels adapted from preserved source questions and answer keys."""
    path = ROOT / "data/teaching_bank/legacy_objective_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))["items"] if path.exists() else {}


def objective_rule_fingerprint(item):
    """Bind a published rule to the exact question, answer and option labels."""
    payload = {key: item.get(key) for key in OBJECTIVE_RULE_FIELDS}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def load_objective_rules():
    if not OBJECTIVE_RULES_PATH.exists():
        return {"items": {}}
    data = json.loads(OBJECTIVE_RULES_PATH.read_text(encoding="utf-8"))
    if (data.get("schema_version") != "published_objective_rules_v1"
            or not isinstance(data.get("items"), dict)
            or data.get("source_signature_fields") != list(OBJECTIVE_RULE_FIELDS)):
        raise ValueError("Unsupported published objective rule metadata.")
    return data


def attach_objective_rules(item, *, rules=None):
    """Add explicit keys without rewriting any published question or rubric.

    Missing, stale or conflicting metadata remains unavailable for grading.
    Rules are authored in the metadata file; the loader never infers a key from
    clinical prose or asks a model to recover an unknown answer.
    """
    data = load_objective_rules() if rules is None else rules
    record = data["items"].get(item.get("id"))
    if not record or record.get("source_sha256") != objective_rule_fingerprint(item):
        return item
    kind = record.get("question_type")
    if kind != item.get("question_type") or kind not in {"true_false", "ordering"}:
        return item
    if kind == "true_false":
        key = "correct_truth"
        if type(record.get(key)) is not bool:
            return item
    else:
        key = "correct_order"
        order, options = record.get(key), item.get("options")
        if (not isinstance(order, list) or not order
                or not all(isinstance(label, str) for label in order)
                or not isinstance(options, dict)
                or len(set(order)) != len(order) or set(order) != set(options)
                or order != item.get("correct_options")):
            return item
    if key in item and item[key] != record[key]:
        return item
    return {**item, key: deepcopy(record[key]), "objective_rule_source": {
        "version": data["version"], "source_sha256": record["source_sha256"],
        "basis": deepcopy(record["basis"])}}


def queues():
    from .bank_review import load_review_queue, QUEUE_PATH
    return [load_review_queue(path) for path in (QUEUE_PATH, PATH, CHOICE_SUPPLEMENT_PATH) if path.exists()]


def approved_expansions():
    from .bank_review import approved_items
    return [item for queue in queues() for item in approved_items(queue)]


def public_metadata(item):
    qtype = item.get("question_type")
    if not qtype:
        qtype = "clinical_case" if item.get("six_step_rubric") or item.get("kind") in {"standardized_case", "fictional_case"} else {
            "case_mcq": "single_choice", "case_sequence": "ordering", "knowledge_card": "concept_card"}.get(item.get("kind"), "short_answer")
    topic = item.get("topic") or {"skin": "assessment", "devices": "device_care"}.get(item.get("category"), "device_care")
    return {"question_type": qtype, "question_type_label": QUESTION_TYPES[qtype], "topic": topic, "topic_label": TOPICS[topic]}


def practice_item(original):
    """Adapt only an approved queue snapshot; retain its source/review identity."""
    item = {**original, "case": original["case_text"], "case_zh": original["case_text"],
            "level": original["difficulty"], "scope": original.get("scope") or {"populations": original["populations"], "devices": original["device_types"]}}
    if original.get("training_flow") == "six_step":
        item["kind"] = "fictional_case"
    if original.get("question_type"):
        item["classification"] = {"populations": original["populations"], "devices": original["device_types"],
                                  "basis": original["field_sources"]["populations"] + "; " + original["field_sources"]["device_types"]}
        # Match the established taxonomy's exact source-bound clinical boundary.
        import hashlib
        item["classification"]["source_sha256"] = hashlib.sha256(json.dumps({k:item.get(k) for k in ("case", "case_zh", "scope")},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        event = original["review_history"][-1]
        item["content_review"] = {"status": "approved_ai", "reviewer_kind": "ai", "reviewer_id": event["actor"],
                                  "reviewed_at": event["timestamp"], "review_fingerprint": event["event_hash"]}
    return item


def display_classification(item):
    """Source-bound browsing annotations do not change answer/rubric content."""
    from .catalog import taxonomy
    annotation = taxonomy().get('display_annotations', {}).get(item.get('id'))
    if not annotation:
        return item
    import hashlib
    current = hashlib.sha256(json.dumps({k:item.get(k) for k in ('case','case_zh','scope')},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if current != annotation.get('source_sha256'):
        return item
    result={**item,'classification':{**item.get('classification',{}),**annotation}}
    if annotation.get('topic') in TOPICS:
        result['topic']=annotation['topic']
    return result
