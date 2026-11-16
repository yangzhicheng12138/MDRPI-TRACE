"""Small, opt-in authoring examples and conservative source applicability checks.

These examples teach task/answer alignment; they are never retrieval evidence.
Clinical verification is recorded in the accompanying data, not inferred by code.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

PROFILE = "reviewed_examples_v1"
CATALOG = Path(__file__).resolve().parents[1] / "data/question_generation/reviewed_examples_20261027.json"

AUTHOR_GUIDANCE = """
authoring_example 是另外编写且经来源核查的示范题，只学习题面、答案、等价答案、提示的对应方式。
示范病例不是本题事实；DEMO_ 开头的引用不是本题证据，不得沿用示范病例、答案或引用ID。
仍使用本题固定病例和 evidence。简答明确学生需要回答哪几方面；单选让所有选项回答同一个问题。
修订意见是待核查的意见：先对照当前题面及原文，只修正成立的问题，保留正确且符合任务的内容。
如果意见把“可能/需评估”误当成“已经确诊”，保留恰当的不确定表述；不要为了迎合审核扩大题目范围。
"""

REVIEW_CALIBRATION = """
审核口径示例（不是本题事实或证据）：
1. 病例只记载红斑，答案说“需检查是否受压”，不能报成“凭空确诊受压”；若写“已发生深部损伤”则需查其依据。
2. 单选干扰项写错误分期，解析已经说明其错误，不把干扰项当作推荐答案；若两项都满足题面要求，才报告答案不唯一。
3. 没有某项内容的原文，先判材料是否足够；原文针对头盔颈圈，答案用于口鼻面罩尺寸，属于适用性错配。
分别核对问题是否真实、是否影响当前任务、证据能否支持理由；不要以问题数量或措辞偏好决定修订。
"""


def authoring_example(blueprint):
    raw = CATALOG.read_bytes()
    catalog = json.loads(raw)
    if blueprint.get("population") != "adult":
        return None  # The checked exemplars deliberately cover adults only.
    candidates = [e for e in catalog["examples"] if e["question_type"] == blueprint["question_type"]]
    target = " ".join(str(blueprint.get(k, "")) for k in ("topic", "learning_objective", "device_type"))
    example = max(candidates, key=lambda e: sum(tag in target for tag in e["selection_terms"]))
    return {"id": example["id"], "purpose": "authoring_example_not_current_evidence",
            "catalog_sha256": hashlib.sha256(raw).hexdigest(),
            "learning_objective": example["learning_objective"],
            "question": deepcopy(example["question"]), "design_notes": example["design_notes"]}


def source_mismatch(source, blueprint, materials):
    """Exclude a helmet fitting manual only for an explicitly facial-mask case.

    Mere mention of helmets in a mixed guideline is not enough. Population and
    generic device admission stay with the existing controller.
    """
    if blueprint.get("device_type") != "niv_mask":
        return None
    case = " ".join(m.get("case_facts", "") for m in materials)
    if not re.search(r"口鼻面罩|鼻梁|面颊|面部接触|oronasal|nasal bridge|cheeks?", case, re.I):
        return None
    if re.search(r"头盔|helmet|casco", case, re.I):
        return None  # A comparison case needs both interfaces.
    title = str(source.get("title", ""))
    body = str(source.get("text", ""))
    helmet_title = bool(re.search(r"头盔|helmet|casco", title, re.I))
    manual = bool(re.search(r"说明书|使用说明|instructions for use|user manual|istruzioni|操作说明", title + " " + body, re.I))
    neck_fit = bool(re.search(r"颈围|颈圈|neck circumference|neck seal|collo", body, re.I))
    hood_action = bool(re.search(r"软领|软颈|密封领|四只手|头部.{0,12}(穿过|套入)|cylinder|collar|hood", body, re.I))
    mixed = bool(re.search(r"口鼻面罩|oronasal|full.face mask|鼻梁|nasal bridge|面颊", body, re.I))
    if neck_fit and not mixed and ((helmet_title and manual) or (hood_action and manual)):
        return "helmet_fitting_manual_not_facial_mask"
    return None


def restore_reviewed_draft(row):
    """Recover the last *fully* reviewed draft after a failed later attempt.

    Retain its own snapshot, version and verdict. It is never promoted to an
    accepted question. The failed attempt remains visible in history.
    """
    def digest(value):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    candidates = [h for h in row.get("history", []) if h.get("stage") == "reviewed"
                  and h.get("review", {}).get("verdict") in {"revise", "needs_evidence", "reject"}
                  and h.get("question_sha256") == digest(h.get("question"))]
    if not candidates:
        return False
    saved = candidates[-1]
    recovery = {"last_attempted_version": row.get("version"), "retained_version": saved["version"],
                "retained_evidence_snapshot_id": saved["evidence_snapshot_id"],
                "reason": "later_attempt_failed", "not_an_approval": True}
    row.update(question=deepcopy(saved["question"]), review=deepcopy(saved["review"]),
               blind_review=deepcopy(saved["blind_review"]), version=saved["version"],
               evidence_snapshot_id=saved["evidence_snapshot_id"], revision_recovery=recovery,
               status={"revise": "revision_needed", "needs_evidence": "needs_evidence", "reject": "rejected"}[saved["review"]["verdict"]])
    for key in ("accepted_question_sha256", "accepted_version", "accepted_evidence_snapshot_id", "approval", "approved_at"):
        row.pop(key, None)
    # Checks may describe the failed later draft. Recomputed by the controller.
    row.pop("quality_checks", None)
    return True
