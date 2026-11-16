# Public release: shared synthetic test fixtures; private integration cases are not distributed.
"""Development question workflow controls; no model or network quality claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest


CASE = "清醒成人准备开始无创通气。现备有两种尺寸面罩，面颊皮肤干燥完整，能表达不适。上一份记录没有面部接触点基线，目前尚未佩戴。"
MATERIAL_ID = "dev-exp250-workflow-fixture"


def material():
    return {
        "id": MATERIAL_ID, "title": "独立流程回归材料", "population": "adult",
        "populations": ["adult"], "device_types": ["niv_mask"],
        "case_facts": CASE, "excerpt": "这是已审核的开发教学说明，不代替原始指南原文。",
        "case_facts_kind": "existing_fictional_development_case",
        "excerpt_kind": "project_authored_reviewed_teaching_reference",
        "usage": "development_only", "review_status": "approved_ai",
        "independent_clinical_validation": False, "reviewer_kind": "ai",
        "learning_objectives": ["识别病例中的两项线索并说明需要评估的内容。"],
        "references": [{"id": "doc001", "title": "控制测试依据", "populations": ["adult"]}],
        "teaching_anchors": [
            {"id": "recognize_R1", "step": "recognize", "text": "识别已提供的器械和皮肤线索。",
             "acceptable": "接受等价表达。", "reference_ids": ["doc001"]},
            {"id": "recognize_R2", "step": "recognize", "text": "指出与线索相关的评估目标，未知仍保持未知。",
             "acceptable": "给出一个有关联的评估目标即可。", "reference_ids": ["doc001"]},
        ],
    }


def blueprint(**changes):
    return {
        "topic": "面罩治疗前的线索识别", "learning_objective": "识别线索并说明一项需要评估的内容。",
        "population": "adult", "device_type": "niv_mask", "difficulty": "基础",
        "question_type": "short_answer", "count": 1, "material_ids": [MATERIAL_ID],
        "requirements": "保留固定病例事实。", **changes,
    }


def source(evidence_id, origin):
    text = ("控制测试的已读原文片段：应在器械使用前评估接触部位、选择适合的尺寸，并保留实际观察。"
            "对于没有提供的患者结果应继续评估，不能预设其正常或已经发生损伤。" * 3)
    return {
        "id": evidence_id, "evidence_id": evidence_id, "text": text,
        "title": "控制测试来源 " + evidence_id, "origin": origin,
        "material_level": "full_text_excerpt" if origin == "local" else "web_page_excerpt",
        "url": "https://example.org/nursing/" + evidence_id if origin == "external" else "",
        "location": "原文第2节", "page": 2, "doc_id": "doc001",
        "population": "adult", "populations": ["adult"], "device_type": "niv_mask",
        "device_types": ["niv_mask"], "read_success": True,
    }


def packet(**changes):
    return {
        "sources": [source("E1", "local"), source("E2", "external")], "events": [],
        "local_success": True, "web_success": True, "rag_attempted": True, "web_attempted": True,
        **changes,
    }


def question(question_type="short_answer", *, key="A"):
    value = {
        "title": "佩戴前的观察", "stem": CASE,
        "task": "指出病例中的一条线索，并说明一项需要继续评估的内容。",
        "options": [], "correct_options": [], "reference_answer": "识别皮肤完整这一线索，建立接触部位基线。",
        "scoring_points": [
            {"id": "R1", "criterion": "识别病例已有线索", "acceptable_answers": ["面颊皮肤完整"], "evidence_ids": ["E1"]},
            {"id": "R2", "criterion": "指出有关联的评估内容", "acceptable_answers": ["建立接触点基线"], "evidence_ids": ["E2"]},
        ],
        "hints": ["留意已有观察。", "比较记录中已有和未给出的信息。", "用一条病例线索说明评估目标。"],
        "common_misconceptions": ["未记录基线不代表组织已经受损。"], "evidence_ids": ["E1", "E2"],
    }
    if question_type == "single_choice":
        value.update(options=[{"label": label, "text": "控制选项" + label} for label in "ABC"],
                     correct_options=[key])
        value["scoring_points"] = value["scoring_points"][:1]
        value["scoring_points"][0]["criterion"] = "选择" + key + "，该选项对应题目任务和依据。"
        value["scoring_points"][0]["acceptable_answers"] = [key]
    return value


def issue(kind="assessment_mismatch"):
    return {"kind": kind, "location": "reference_answer", "text": "识别皮肤完整这一线索，建立接触部位基线。",
            "explanation": "控制测试要求修订该项，不代表真实临床评价。", "evidence_ids": ["E1"],
            "requested_change": "调整参考答案并重新审核。"}


def review(verdict="accept"):
    return {
        "verdict": verdict, "issues": [] if verdict == "accept" else [issue("material_gap" if verdict == "needs_evidence" else "assessment_mismatch")],
        "evidence_gaps": [{"query": "device contact baseline assessment", "reason": "需补充适用依据"}] if verdict == "needs_evidence" else [],
        "summary": "控制测试审核结论。",
    }


class FakeBackend:
    options = {"max_revisions": 2, "max_gap_searches_per_item": 1,
               "job_seconds": 1800, "item_seconds": 300}

    def __init__(self, *, packets=None, reviews=None, questions=None):
        self.packets = list(packets or [packet()])
        self.reviews = list(reviews or [review()])
        self.questions = list(questions or [])
        self.retrieve_calls, self.generate_calls, self.blind_calls, self.review_calls = [], [], [], []

    @staticmethod
    def _next(values):
        return deepcopy(values.pop(0) if len(values) > 1 else values[0])

    def retrieve(self, blueprint, materials, log, budget, *, gaps=None):
        self.retrieve_calls.append({"blueprint": deepcopy(blueprint), "gaps": deepcopy(gaps)})
        return self._next(self.packets)

    def generate(self, blueprint, materials, evidence, teaching_plan, log, budget, *, index, previous=None, review=None):
        self.generate_calls.append({"index": index, "previous": deepcopy(previous), "review": deepcopy(review),
                                    "evidence": deepcopy(evidence), "teaching_plan": deepcopy(teaching_plan),
                                    "blueprint": deepcopy(blueprint)})
        return self._next(self.questions) if self.questions else question(blueprint["question_type"])

    def blind_review(self, blueprint, materials, question, log, budget):
        # Check the actual boundary, not just whether a model obeys a prompt.
        forbidden = {"reference_answer", "correct_options", "scoring_points", "hints", "common_misconceptions"}
        assert not forbidden.intersection(question)
        assert set(question) <= {"title", "stem", "task", "options"}
        assert all(set(row) <= {"id", "title", "case_facts"} for row in materials)
        self.blind_calls.append({"question": deepcopy(question), "materials": deepcopy(materials)})
        return {"answerable": True, "attempted_answer": "从已给观察识别线索并提出评估目标。", "ambiguity_notes": []}

    def review(self, blueprint, materials, evidence, teaching_plan, question, blind, log, budget):
        self.review_calls.append({"question": deepcopy(question), "evidence": deepcopy(evidence), "blind": deepcopy(blind),
                                  "teaching_plan": deepcopy(teaching_plan)})
        value = self._next(self.reviews)
        if teaching_plan.get("audit_required"):
            by_id = {source["evidence_id"]: source for source in evidence["sources"]}
            value["quality_audit"] = {
                "case_facts": {"status": "consistent", "note": "控制测试事实一致。"},
                "assessment_alignment": {"status": "aligned", "note": "控制测试任务对齐。"},
                "distinctness": {"status": "distinct", "note": "控制测试审核区分了目标。"},
                "evidence_support": [{"criterion_id": point["id"], "status": "supported", "sources": [
                    {"evidence_id": identity, "status": "supported", "quote": by_id[identity]["text"][:30]}
                    for identity in point["evidence_ids"]],
                    "rationale": "控制测试的原文依据。"} for point in question["scoring_points"]]}
        return value


@pytest.fixture
def service_factory(monkeypatch, tmp_path):
    from modes import question_generation as module

    def load(ids, population, device_type):
        if ids != [MATERIAL_ID] or population != "adult" or device_type != "niv_mask":
            raise ValueError("Selected development material has an invalid scope")
        return [deepcopy(material())]

    monkeypatch.setattr(module, "load_materials", load)

    def factory(backend=None, *, root=None):
        return module.QuestionGenerationService(root or tmp_path / "studio", backend or FakeBackend())

    return factory


def run(service, specification=None):
    queued = service.create(specification or blueprint())
    assert queued["status"] == "queued"
    return service.run(queued["job_id"])




















































def _registered_anchor_material():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "data/question_generation/development_materials.json").read_text(encoding="utf-8"))["items"][0]
