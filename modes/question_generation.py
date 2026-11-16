"""Bounded, evidence-grounded development questions with independent review.

Only this studio's jobs and logs are written. Nothing here publishes content to
the existing teaching bank or changes a formal evaluation case.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock, RLock
from urllib.parse import urlsplit
import uuid

from jsonschema import Draft202012Validator, ValidationError

from app.evidence_snapshot import BudgetExceeded, RunBudget
from .backend import TurnLog
from .question_generation_materials import load_materials, materials as material_catalog
from .question_generation_schema import (
    blind_review_schema, validate_blueprint, validate_question, validate_review,
)
from .question_generation_planning import build_batch_plan
from .question_generation_quality import assess_question_quality
from .question_generation_audit import validate_audited_review, evidence_text_field
from .question_generation_examples import PROFILE as EXAMPLES_PROFILE


DEFAULTS = {"max_revisions": 2, "max_gap_searches_per_item": 1,
            "max_retrieval_rounds": 2, "job_seconds": 1800, "item_seconds": 300,
            "review_evidence_mode": "verbatim", "quality_profile": "legacy"}
_ACTIVE = {"running", "retrieving", "generating", "reviewing", "revising"}
_TERMINAL_ITEMS = {"accepted", "rejected", "needs_evidence", "revision_needed", "failed", "needs_review"}
_READ_LEVELS = {"web_page_excerpt", "pdf_excerpt", "full_text_excerpt", "article_abstract"}
_METADATA_LEVELS = {"search_snippet", "metadata_only", "article_record", "search_result", "metadata"}
_RAG_DEVICES = {"tracheostomy_tube": "tracheostomy", "vascular_access_device": "vascular_access",
                "nasal_prong_cpap": "cpap", "pulse_oximeter": "spo2_probe",
                "nasal_cannula": "other", "ecg_electrode": "other"}
_PEOPLE = {"adult": "adult", "adults": "adult", "成人": "adult",
           "pediatric": "pediatric", "children": "pediatric", "child": "pediatric", "儿童": "pediatric",
           "neonatal": "neonatal", "neonate": "neonatal", "newborn": "neonatal", "新生儿": "neonatal"}
_PLANNED_SCHEMA = "evidence_grounded_development_questions_v2"


class _QuestionContractError(ValueError):
    """Only static controller-authored diagnostics are safe to expose directly."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def _plans_intact(job):
    if job.get("schema_version") != _PLANNED_SCHEMA:
        return True  # Preserve original jobs and their review contract.
    plan = job.get("batch_plan")
    return (isinstance(plan, dict) and _digest(plan) == job.get("batch_plan_sha256")
            and len(plan.get("items", [])) == len(job["questions"])
            and all(row.get("plan") == planned for row, planned in zip(job["questions"], plan["items"])))


def _safe_error(exc):
    if isinstance(exc, BudgetExceeded):
        return "本次任务超过时间或工具调用预算，已停止并保存现有记录。"
    if isinstance(exc, _QuestionContractError):
        return str(exc)
    if isinstance(exc, ValidationError):
        return "题包或审核输出未通过结构校验，已保存本次记录，未获得通过。"
    # Provider errors can contain requests or sensitive paths. The detailed
    # native trace is owned by the backend, not copied into public job errors.
    return "本阶段未完成：" + type(exc).__name__


def _terms(value):
    words = set(re.findall(r"[a-zA-Z]+", value.casefold()))
    for part in re.findall(r"[\u4e00-\u9fff]+", value):
        words.update(part[index:index + 2] for index in range(max(0, len(part) - 1)))
    return words


def _objective_steps(objective):
    """Only explicit task verbs select required anchors; topic overlap cannot."""
    patterns = {
        "recognize": r"识别|辨认|提取|找出|\b(?:recogniz\w*|recognis\w*|identify|identifying|extract\w*)\b",
        "analyze": r"分析|解释|鉴别|区分|推断|判断|\b(?:analy[sz]\w*|interpret\w*|differentiat\w*|explain\w*)\b",
        "prioritize": r"优先|排序|首要|\b(?:prioriti[sz]\w*|priority|rank\w*)\b",
        "solutions": r"方案|措施|干预|解决办法|护理计划|\b(?:solution\w*|intervention\w*|care plan)\b",
        "action": r"行动|立即处理|先后顺序|操作顺序|执行顺序|角色分工|落实|执行|实施|\b(?:action\w*|implement\w*)\b",
        "evaluate": r"复评|复查|复测|再评估|评价|效果观察|疗效|\b(?:evaluat\w*|reassess\w*|outcome\w*)\b",
    }
    # Feasibility/safety wording in a requested solution is not a separate
    # request to assess the Take Action stage.
    text = re.sub(r"可执行|可实施|实施条件|实施前核查|实施限制", "", objective, flags=re.I)
    matches = {step: re.search(pattern, text, flags=re.I) for step, pattern in patterns.items()}
    if matches["prioritize"]:
        # "Identify the priority problem and explain why" is one priority
        # task; its generic verbs must not silently create two extra stages.
        if not re.search(r"线索|因素|观察|资料|信息|cues?|clues?|findings?|factors?|observations?", text, re.I):
            matches["recognize"] = None
        if not re.search(r"分析|鉴别|区分|推断|病因|机制|关联|关系|analy[sz]|differentiat|interpret", text, re.I):
            matches["analyze"] = None
    return [step for step, match in sorted(matches.items(), key=lambda pair: pair[1].start() if pair[1] else len(text) + 1)
            if match]


def _teaching_plan(blueprint, materials):
    """Choose existing reviewed anchors before generation, without a model call."""
    target = _terms(blueprint["learning_objective"] + " " + blueprint["topic"])
    candidates = []
    for material in materials:
        for anchor in material.get("teaching_anchors", []):
            text = anchor.get("text") or anchor.get("criterion") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            candidates.append({
                "id": anchor["id"], "material_id": material["id"],
                "step": anchor.get("step", ""), "criterion": text,
                "acceptable": deepcopy(anchor.get("acceptable", "")),
                "reference_ids": deepcopy(anchor.get("reference_ids", [])),
                "review_status": material.get("review_status", ""),
                "source_question_id": material.get("source_question_id", material["id"]),
                "source_review_event_hash": material.get("source_review_event_hash", ""),
            })
    if len(candidates) < 2:
        raise ValueError("所选开发材料缺少至少两条已审核教学标准，暂不能出题。")
    ranked = sorted(enumerate(candidates),
                    key=lambda row: (-len(target & _terms(row[1]["criterion"])), row[0]))
    ranked_unique, seen = [], set()
    for _, row in ranked:
        identity = (row["step"], row["id"])
        if identity not in seen:
            ranked_unique.append(row)
            seen.add(identity)
    steps = _objective_steps(blueprint["learning_objective"])
    # Reviewed anchors are trustworthy within their original scope, not
    # automatically required for every question about the same device.
    required = []
    if len(steps) == 1:
        matching = [row for row in ranked_unique if row["step"] == steps[0]]
        if len(matching) >= 2:
            required = matching[:2]
    elif len(steps) == 2:
        groups = [[row for row in ranked_unique if row["step"] == step] for step in steps]
        if all(groups):
            required = [groups[0][0], groups[1][0]]
            remainder = [row for row in ranked_unique if row["step"] in steps and row not in required]
            required += remainder[:1]
    remaining = [row for _, row in ranked if row not in required]
    supporting = remaining[:6] if required else []
    candidate_only = remaining[:6] if not required else []
    return {"learning_objective": blueprint["learning_objective"], "difficulty": blueprint["difficulty"],
            "origin": "reviewed_development_material_anchors", "usage": "development_only",
            "selected_steps": steps, "selection_confidence": "explicit_step" if required else "candidate_only",
            "required_criteria": deepcopy(required), "supporting_criteria": deepcopy(supporting),
            "candidate_criteria": deepcopy(candidate_only),
            "scope": "必答范围由学习目标和实际作答任务共同确定。required仅在与本题病例和任务相符时适用；"
                     "supporting和candidate是辅助材料，不是额外必答要求。不得因题外相邻步骤未写而判遗漏；"
                     "低置信候选不能冒充已经确定的必答标准。"}


def _values(value):
    if isinstance(value, str):
        return {part.strip().casefold() for part in re.split(r"[,;|/、，；]", value) if part.strip()}
    if isinstance(value, (list, tuple, set)):
        return {part.strip().casefold() for part in value if isinstance(part, str) and part.strip()}
    return set()


def _scope_values(source, *keys):
    result = set()
    for record in (source, source.get("metadata", {})):
        if isinstance(record, dict):
            for key in keys:
                result.update(_values(record.get(key)))
    return result


def _known_devices():
    try:
        path = Path(__file__).resolve().parents[1] / "data/teaching_bank/catalog_taxonomy.json"
        return set(json.loads(path.read_text(encoding="utf-8"))["devices"]) | set(_RAG_DEVICES.values())
    except (OSError, ValueError, KeyError, TypeError):
        # Scope metadata that cannot be classified is explicitly left to the
        # reviewer; do not infer a clinical mismatch from unknown labels.
        return {"niv_mask", "nasal_cannula", "nasogastric_tube", "endotracheal_tube", "spo2_probe", "general_mdrpi"} | set(_RAG_DEVICES.values())


def _normalize_source(source, blueprint, devices):
    if not isinstance(source, dict):
        return None, "invalid_source"
    identity = source.get("evidence_id") or source.get("id")
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 300:
        return None, "missing_evidence_id"
    text, origin = source.get("text"), source.get("origin")
    if not isinstance(text, str) or not text.strip():
        return None, "missing_read_text"
    if origin not in {"local", "external"}:
        return None, "unknown_origin"
    level = str(source.get("material_level") or "").strip().casefold()
    if level in _METADATA_LEVELS or (origin == "external" and level not in _READ_LEVELS):
        return None, "not_read_original_material"
    populations = {_PEOPLE[value] for value in _scope_values(source, "population", "populations") if value in _PEOPLE}
    if populations and blueprint["population"] not in populations:
        return None, "population_mismatch"
    device_types = _scope_values(source, "device_type", "device_types") & devices
    compatible_devices = {blueprint["device_type"], _RAG_DEVICES.get(blueprint["device_type"]), "general_mdrpi"}
    if device_types and not device_types & compatible_devices:
        return None, "device_mismatch"
    url = source.get("source_url") or source.get("url") or ""
    if origin == "external":
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
                return None, "missing_page_url"
        except (TypeError, ValueError):
            return None, "missing_page_url"
    value = deepcopy(source)
    value.update(evidence_id=identity.strip(), text=text.strip(), source_url=url, material_level=level)
    return value, None


def _same_content(source):
    return _digest({key: source.get(key) for key in ("origin", "text", "source_url", "location")})


class QuestionGenerationService:
    def __init__(self, root, backend=None):
        self.root = Path(root).resolve()
        self.jobs = self.root / "jobs"
        self.logs = self.root / "logs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        if backend is None:
            from .question_generation_backend import QuestionGenerationBackend
            backend = QuestionGenerationBackend()
        self.backend = backend
        self.options = {**DEFAULTS, **(getattr(backend, "options", {}) or {})}
        evidence_text_field(self.options["review_evidence_mode"])
        for key, low, high in (("max_revisions", 0, 2), ("max_gap_searches_per_item", 0, 1),
                               ("max_retrieval_rounds", 1, 2)):
            value = self.options[key]
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError("Invalid question-generation iteration limits")
        for key in ("job_seconds", "item_seconds"):
            value = self.options[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 14400:
                raise ValueError("Invalid question-generation time budget")
        self._run_lock, self._io_lock = Lock(), RLock()
        self._devices = _known_devices()
        self._recover()

    def _path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid question-generation job ID")
        return self.jobs / (job_id + ".json")

    def _read(self, job_id):
        path = self._path(job_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError("出题任务记录损坏，请检查保存的任务文件。") from None
        if not isinstance(value, dict) or value.get("job_id") != job_id:
            raise ValueError("出题任务记录身份不一致。")
        return value

    def _save(self, value):
        with self._io_lock:
            value["updated_at"] = _now()
            path = self._path(value["job_id"])
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                with temporary.open("x", encoding="utf-8") as handle:
                    json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def _recover(self):
        for path in self.jobs.glob("*.json"):
            try:
                value = self._read(path.stem)
            except (OSError, ValueError):
                continue
            if value.get("status") not in _ACTIVE:
                continue
            value.update(status="failed", error="服务重启时发现执行中的任务已中断，未自动重试或调用模型。", finished_at=_now())
            for row in value.get("questions", []):
                if row.get("status") not in _TERMINAL_ITEMS:
                    row.update(status="failed", error="任务执行中断，未自动重试。")
            self._progress(value, "failed", value["error"], final=True)
            self._save(value)

    def materials(self):
        return material_catalog()

    def create(self, blueprint):
        specification = validate_blueprint(blueprint)
        selected = load_materials(specification["material_ids"], specification["population"], specification["device_type"])
        teaching_plan = _teaching_plan(specification, selected)
        batch_plan = build_batch_plan(specification, selected, teaching_plan)
        now, job_id = _now(), uuid.uuid4().hex
        value = {
            "schema_version": _PLANNED_SCHEMA, "job_id": job_id,
            "status": "queued", "created_at": now, "updated_at": now,
            "development_only": True, "formal_allowed": False,
            "blueprint": specification, "blueprint_sha256": _digest(specification),
            "materials": deepcopy(selected), "materials_sha256": _digest(selected),
            "teaching_plan": teaching_plan, "teaching_plan_sha256": _digest(teaching_plan),
            "batch_plan": batch_plan, "batch_plan_sha256": _digest(batch_plan),
            "options": {key: self.options[key] for key in DEFAULTS},
            "evidence": {"sources": []}, "evidence_snapshots": {}, "events": [], "errors": [],
            "retrieval_state": {key: False for key in ("rag_attempted", "web_attempted", "local_success", "web_success")},
            "questions": [{"item_id": uuid.uuid4().hex, "status": "queued", "version": 0,
                           "question": None, "review": None, "blind_review": None, "history": [],
                           "evidence_snapshot_id": None, "plan": deepcopy(item_plan)}
                          for item_plan in batch_plan["items"]],
        }
        self._progress(value, "queued", "已保存整套命题安排与已审核考点，等待检索。")
        self._save(value)
        return deepcopy(value)

    def _progress(self, job, stage, message, *, final=False):
        completed = sum(row.get("status") in _TERMINAL_ITEMS for row in job["questions"])
        total = len(job["questions"])
        job["progress"] = {"stage": stage, "message": message, "completed": completed,
                           "total": total, "percent": 100 if final else min(95, int(95 * completed / total))}

    def _stage(self, job, stage, message):
        job["status"] = stage
        self._progress(job, stage, message)
        job["events"].append({"at": _now(), "stage": stage, "message": message})
        self._save(job)

    def _snapshot(self, job, packet, log):
        if not isinstance(packet, dict) or not isinstance(packet.get("sources"), list):
            raise _QuestionContractError("检索没有返回有效的证据包。")
        sources = deepcopy(job["evidence"].get("sources", []))
        existing = {source["evidence_id"]: source for source in sources}
        rejected = []
        for source in packet["sources"]:
            if self.options["quality_profile"] == EXAMPLES_PROFILE and isinstance(source, dict):
                from .question_generation_examples import source_mismatch
                reason = source_mismatch(source, job["blueprint"], job["materials"])
                if reason:
                    rejected.append({"evidence_id": source.get("evidence_id", source.get("id")), "reason": reason})
                    log.write("question_source_excluded", {"reason": reason, "source": source})
                    continue
            normalized, reason = _normalize_source(source, job["blueprint"], self._devices)
            if normalized is None:
                rejected.append({"evidence_id": source.get("evidence_id", source.get("id")) if isinstance(source, dict) else None,
                                 "reason": reason})
                continue
            identity = normalized["evidence_id"]
            if identity in existing:
                if _same_content(existing[identity]) == _same_content(normalized):
                    continue
                identity += "_" + _same_content(normalized)[:12]
                normalized["evidence_id"] = identity
                if identity in existing and _same_content(existing[identity]) == _same_content(normalized):
                    continue
            existing[identity] = normalized
            sources.append(normalized)
        for flag in job["retrieval_state"]:
            job["retrieval_state"][flag] = job["retrieval_state"][flag] or packet.get(flag) is True
        snapshot_id = _digest({"sources": sources})
        snapshot = {"snapshot_id": snapshot_id, "sha256": snapshot_id, "created_at": _now(), "sources": sources}
        job["evidence_snapshots"].setdefault(snapshot_id, deepcopy(snapshot))
        job["evidence"] = deepcopy(job["evidence_snapshots"][snapshot_id])
        entry = {"at": _now(), "stage": "evidence_snapshot", "evidence_snapshot_id": snapshot_id,
                 "source_count": len(sources), "rejected_sources": rejected,
                 "retrieval_state": deepcopy(job["retrieval_state"]), "tool_events": deepcopy(packet.get("events", []))}
        job["events"].append(entry)
        log.write("question_evidence_snapshot", {**entry, "snapshot": snapshot})
        self._save(job)
        return snapshot_id

    @staticmethod
    def _has_evidence(job):
        origins = {source["origin"] for source in job["evidence"].get("sources", [])}
        return {"local", "external"} <= origins and all(job["retrieval_state"].values())

    @staticmethod
    def _gap_request(job):
        scope = job["blueprint"]
        return [{"query": (scope["population"] + " " + scope["device_type"] + " " + scope["topic"] + " guideline evidence")[:180],
                 "reason": "必须取得适用的知识库原文及实际读取的网页证据；现有证据不足。"}]

    @staticmethod
    def _question_signature(question):
        def normalized(value):
            if isinstance(value, str):
                return re.sub(r"\s+", "", value).casefold()
            if isinstance(value, list):
                return [normalized(item) for item in value]
            if isinstance(value, dict):
                return {key: normalized(item) for key, item in value.items()}
            return value
        return _digest(normalized({key: question[key] for key in ("stem", "task", "options")}))

    def _history(self, job, row, stage, **values):
        entry = {"at": _now(), "stage": stage, "version": row["version"],
                 "evidence_snapshot_id": row["evidence_snapshot_id"], **deepcopy(values)}
        row["history"].append(entry)
        self._save(job)

    def _item(self, job, row, index, log, budget):
        try:
            self._item_versions(job, row, index, log, budget)
        except Exception as exc:
            if self.options["quality_profile"] != EXAMPLES_PROFILE:
                raise
            from .question_generation_examples import restore_reviewed_draft
            self._history(job, row, "attempt_failed", error=_safe_error(exc), error_type=type(exc).__name__)
            if not restore_reviewed_draft(row):
                raise
            row["error"] = _safe_error(exc)
            job["errors"].append({"item_id": row["item_id"], "error": row["error"],
                                  "error_type": type(exc).__name__, "reviewed_draft_retained": True})
            row["question_signature"] = self._question_signature(row["question"])
            if row.get("plan"):
                planned = row["plan"]
                row["quality_checks"] = assess_question_quality(row["question"], job["questions"][:index],
                    {**deepcopy(planned), "materials": [m for m in job["materials"] if m["id"] == planned["material_id"]]})
            self._history(job, row, "reviewed_draft_recovered", recovery=row["revision_recovery"])
        if self.options["quality_profile"] in {"evidence_first_v2", EXAMPLES_PROFILE}:
            from .question_generation_grounded import retain_reviewed_candidate, reviewed_candidates
            candidates = reviewed_candidates(row, job["evidence"]["snapshot_id"])
            if len(candidates) < 2 or row["status"] == "accepted":
                return
            selector = getattr(self.backend, "select_candidate", None)
            if selector is None:
                return
            try:
                decision = selector(deepcopy(job["blueprint"]), deepcopy(job["materials"]),
                    deepcopy(job["evidence"]), deepcopy(candidates), log, budget)
                selected = retain_reviewed_candidate(row, job["evidence"]["snapshot_id"], decision)
            except Exception as exc:
                # Selection is an optional improvement, not a new reason to
                # discard an already reviewed final draft or hide its issues.
                self._history(job, row, "candidate_selection_unavailable", error_type=type(exc).__name__)
                return
            if selected:
                row["question_signature"] = self._question_signature(row["question"])
                planned = row.get("plan")
                if planned:
                    row["quality_checks"] = assess_question_quality(row["question"], job["questions"][:index],
                        {**deepcopy(planned), "materials": [m for m in job["materials"] if m["id"] == planned["material_id"]]})
                self._history(job, row, "candidate_selected", selection=selected)

    def _generate_candidate(self, job, row, blueprint, materials, context, index, log, budget, previous, feedback):
        from .question_generation_grounded import EvidencePreparationGap
        while True:
            try:
                return self.backend.generate(blueprint, deepcopy(materials), deepcopy(job["evidence"]),
                    deepcopy(context), log, budget, index=index,
                    previous=deepcopy(previous), review=deepcopy(feedback))
            except EvidencePreparationGap as exc:
                if self.options["quality_profile"] != "evidence_first_v2":
                    raise
                row["evidence_preparation"] = deepcopy(exc.brief)
                self._history(job, row, "preparation_gap", brief=exc.brief)
                if row.get("gap_searches", 0) >= self.options["max_gap_searches_per_item"]:
                    row.update(status="needs_evidence", error="当前考点缺少适用依据，已保存具体缺口。")
                    self._save(job)
                    return None
                row["gap_searches"] = row.get("gap_searches", 0) + 1
                self._stage(job, "retrieving", f"第 {index + 1} 题：补充当前考点所需的证据。")
                packet = self.backend.retrieve(deepcopy(blueprint), deepcopy(materials), log, budget,
                                               gaps=deepcopy(exc.gaps))
                budget.remaining()
                self._snapshot(job, packet, log)
                row["evidence_snapshot_id"] = job["evidence"]["snapshot_id"]
                self._history(job, row, "gap_search", gaps=exc.gaps,
                              new_evidence_snapshot_id=row["evidence_snapshot_id"])

    def _item_versions(self, job, row, index, log, budget):
        blueprint, materials = job["blueprint"], job["materials"]
        planned = job.get("schema_version") == _PLANNED_SCHEMA
        item_plan = row.get("plan") if planned else None
        context = deepcopy(job["teaching_plan"])
        evidence_mode = job.get("options", {}).get("review_evidence_mode", "verbatim")
        if planned:
            context.update(item_plan=deepcopy(item_plan), audit_required=True,
                           review_evidence_mode=evidence_mode)
        previous, feedback, gap_searches = None, None, 0
        for version in range(1, self.options["max_revisions"] + 2):
            budget.remaining()
            row.update(status="generating" if version == 1 else "revising", version=version,
                       review=None, blind_review=None, evidence_snapshot_id=job["evidence"]["snapshot_id"])
            for key in ("accepted_question_sha256", "accepted_version", "accepted_evidence_snapshot_id", "approval", "approved_at"):
                row.pop(key, None)
            self._stage(job, row["status"], f"第 {index + 1} 题：正在{'生成' if version == 1 else '修订'}第 {version} 版。")
            transient = deepcopy(blueprint)
            transient["_previous_tasks"] = [item["question"]["task"] for item in job["questions"][:index]
                                           if isinstance(item.get("question"), dict) and isinstance(item["question"].get("task"), str)]
            value = self._generate_candidate(job, row, transient, materials, context, index, log, budget, previous, feedback)
            if value is None:
                return
            gap_searches = row.get("gap_searches", gap_searches)
            budget.remaining()
            row["question"] = deepcopy(value)
            self._history(job, row, "generated", question=value)
            ids = [source["evidence_id"] for source in job["evidence"]["sources"]]
            value = validate_question(value, ids, blueprint["question_type"])
            if value["stem"].strip() not in {item["case_facts"].strip() for item in materials}:
                raise _QuestionContractError("生成的题干未保持所选开发病例的固定事实，未交付为通过题目。")
            if planned and value["stem"].strip() != item_plan["case_facts"].strip():
                raise _QuestionContractError("本题未使用整套命题安排指定的病例，未获得审核通过。")
            signature = self._question_signature(value)
            if any(item.get("question_signature") == signature for item in job["questions"][:index]):
                row.update(status="rejected", error="与本次任务已有题目重复，未计为新题或获得审核通过。")
                self._history(job, row, "duplicate_rejected", error=row["error"])
                return
            row["question"] = value
            row["question_signature"] = signature
            if planned:
                check_plan = {**deepcopy(item_plan), "materials": [
                    {"id": item["id"], "case_facts": item["case_facts"]}
                    for item in materials if item["id"] == item_plan["material_id"]]}
                checks = assess_question_quality(value, job["questions"][:index], check_plan)
                row["quality_checks"] = checks
                context.update(quality_checks=deepcopy(checks), previous_questions=[
                    {"item_id": item["item_id"], "question": {key: deepcopy(item["question"][key])
                     for key in ("task", "reference_answer", "scoring_points") if key in item["question"]}}
                    for item in job["questions"][:index] if isinstance(item.get("question"), dict)])
                self._history(job, row, "quality_checked", quality_checks=checks)
            row["status"] = "reviewing"
            self._stage(job, "reviewing", f"第 {index + 1} 题：独立试答与审核第 {version} 版。")
            public_question = {key: deepcopy(value[key]) for key in ("title", "stem", "task", "options")}
            public_materials = [{key: deepcopy(item[key]) for key in ("id", "title", "case_facts")} for item in materials]
            blind = self.backend.blind_review(deepcopy(blueprint), public_materials, public_question, log, budget)
            budget.remaining()
            Draft202012Validator(blind_review_schema()).validate(blind)
            row["blind_review"] = deepcopy(blind)
            self._history(job, row, "blind_review", blind_review=blind)
            while True:
                budget.remaining()
                row["evidence_snapshot_id"] = job["evidence"]["snapshot_id"]
                self._stage(job, "reviewing", f"第 {index + 1} 题：核对答案、评分标准与证据。")
                ids = [source["evidence_id"] for source in job["evidence"]["sources"]]
                judgment = self.backend.review(deepcopy(blueprint), deepcopy(materials), deepcopy(job["evidence"]),
                    deepcopy(context), deepcopy(value), deepcopy(blind), log, budget)
                budget.remaining()
                judgment = (validate_audited_review(judgment, job["evidence"], value,
                                duplicate_candidates=bool(row.get("quality_checks", {}).get("duplicate_candidates")),
                                evidence_mode=evidence_mode)
                            if planned else validate_review(judgment, ids))
                row["review"] = judgment
                self._history(job, row, "reviewed", question=value, question_sha256=_digest(value),
                              review=judgment, blind_review=blind)
                verdict = judgment["verdict"]
                if verdict == "accept":
                    question_sha = _digest(value)
                    row.update(status="accepted", accepted_question_sha256=question_sha, accepted_version=version,
                               accepted_evidence_snapshot_id=row["evidence_snapshot_id"], approved_at=_now(),
                               approval={"question_sha256": question_sha, "version": version,
                                         "evidence_snapshot_id": row["evidence_snapshot_id"],
                                         "review_sha256": _digest(judgment), "blind_review_sha256": _digest(blind)})
                    if planned:
                        row["approval"].update(batch_plan_sha256=job["batch_plan_sha256"],
                            item_plan_sha256=_digest(item_plan), quality_checks_sha256=_digest(row["quality_checks"]))
                    self._history(job, row, "accepted", approval=row["approval"])
                    return
                if verdict == "reject":
                    row["status"] = "rejected"
                    self._save(job)
                    return
                combined_gap = (self.options["quality_profile"] in {"evidence_first_v2", EXAMPLES_PROFILE}
                    and verdict == "revise" and bool(judgment["evidence_gaps"])
                    and gap_searches < self.options["max_gap_searches_per_item"])
                if verdict == "needs_evidence" or combined_gap:
                    if gap_searches >= self.options["max_gap_searches_per_item"]:
                        row["status"] = "needs_evidence"
                        self._save(job)
                        return
                    gap_searches += 1
                    row["gap_searches"] = gap_searches
                    self._stage(job, "retrieving", f"第 {index + 1} 题：根据审核意见补充取证。")
                    packet = self.backend.retrieve(deepcopy(blueprint), deepcopy(materials), log, budget,
                                                   gaps=deepcopy(judgment["evidence_gaps"]))
                    budget.remaining()
                    self._snapshot(job, packet, log)
                    self._history(job, row, "gap_search", gaps=judgment["evidence_gaps"],
                                  new_evidence_snapshot_id=job["evidence"]["snapshot_id"])
                    # Keep the same candidate and blind attempt; only a complete
                    # new review against the updated evidence can change verdict.
                    continue
                row["status"] = "revision_needed"
                previous, feedback = deepcopy(value), deepcopy(judgment)
                self._save(job)
                break

    @staticmethod
    def _aggregate(job):
        statuses = [row["status"] for row in job["questions"]]
        if all(status == "accepted" for status in statuses):
            return "completed"
        if any(status == "accepted" for status in statuses):
            return "partial"
        if any(status == "needs_evidence" for status in statuses):
            return "needs_evidence"
        return "failed"

    def run(self, job_id):
        with self._run_lock:
            job = self.get(job_id)
            if job["status"] != "queued":
                return job
            log = TurnLog(self.logs / job_id)
            budget = RunBudget(self.options["job_seconds"])
            try:
                current = load_materials(job["blueprint"]["material_ids"], job["blueprint"]["population"], job["blueprint"]["device_type"])
                if (_digest(current) != job["materials_sha256"] or _digest(job["materials"]) != job["materials_sha256"]
                        or _digest(job["blueprint"]) != job["blueprint_sha256"]
                        or _digest(job["teaching_plan"]) != job["teaching_plan_sha256"] or not _plans_intact(job)):
                    raise _QuestionContractError("命题要求或已审核材料发生变化，请重新建立开发任务。")
                if job.get("options", {}).get("quality_profile", "legacy") != self.options["quality_profile"]:
                    raise _QuestionContractError("任务保存的出题版本与当前配置不同，请使用原配置续跑或新建任务。")
                self._stage(job, "retrieving", "正在检索知识库并读取联网资料。")
                for attempt in range(self.options["max_retrieval_rounds"]):
                    budget.remaining()
                    packet = self.backend.retrieve(deepcopy(job["blueprint"]), deepcopy(job["materials"]), log, budget,
                                                   gaps=None if attempt == 0 else self._gap_request(job))
                    budget.remaining()
                    self._snapshot(job, packet, log)
                    if self._has_evidence(job):
                        break
                if not self._has_evidence(job):
                    for row in job["questions"]:
                        row.update(status="needs_evidence", error="未取得同时满足要求的知识库原文和已读网页证据，未生成题目。",
                                   evidence_snapshot_id=job["evidence"].get("snapshot_id"))
                    job["error"] = "证据仍不足，已停止出题；可查看检索与过滤记录。"
                else:
                    for index, row in enumerate(job["questions"]):
                        try:
                            item_budget = RunBudget(min(float(self.options["item_seconds"]), budget.remaining()))
                            self._item(job, row, index, log, item_budget)
                        except Exception as exc:
                            row.update(status="failed", error=_safe_error(exc))
                            job["errors"].append({"item_id": row["item_id"], "error": row["error"], "error_type": type(exc).__name__})
                            self._history(job, row, "failed", error=row["error"], error_type=type(exc).__name__)
                            log.write("question_item_failure", {"item_id": row["item_id"], "error": row["error"], "error_type": type(exc).__name__})
                job["status"] = self._aggregate(job)
            except Exception as exc:
                job.update(status="failed", error=_safe_error(exc))
                for row in job["questions"]:
                    if row["status"] not in _TERMINAL_ITEMS:
                        row.update(status="failed", error=job["error"])
                log.write("question_job_failure", {"error": job["error"], "error_type": type(exc).__name__})
            finally:
                if job["status"] in _ACTIVE:
                    job.update(status="failed", error="任务执行中断，未继续调用模型。")
                    for row in job["questions"]:
                        if row["status"] not in _TERMINAL_ITEMS:
                            row.update(status="failed", error=job["error"])
                job["finished_at"] = _now()
                job["development_only"], job["formal_allowed"] = True, False
                self._progress(job, job["status"], "开发题任务已结束；通过结果仍需教师确认。", final=True)
                self._save(job)
            return self.get(job_id)

    def _validate_approvals(self, job):
        changed = False
        for row in job.get("questions", []):
            if row.get("status") != "accepted":
                continue
            try:
                approval = row["approval"]
                snapshot_id = row["evidence_snapshot_id"]
                snapshot = job["evidence_snapshots"][snapshot_id]
                question = row["question"]
                question_sha = _digest(question)
                valid = (
                    row["accepted_question_sha256"] == approval["question_sha256"] == question_sha
                    and row["version"] == row["accepted_version"] == approval["version"]
                    and snapshot_id == row["accepted_evidence_snapshot_id"] == approval["evidence_snapshot_id"]
                    and snapshot_id == snapshot["snapshot_id"] == snapshot["sha256"] == _digest({"sources": snapshot["sources"]})
                    and row["review"]["verdict"] == "accept"
                    and approval["review_sha256"] == _digest(row["review"])
                    and approval["blind_review_sha256"] == _digest(row["blind_review"])
                    and _digest(job["blueprint"]) == job["blueprint_sha256"]
                    and _digest(job["materials"]) == job["materials_sha256"]
                    and _digest(job["teaching_plan"]) == job["teaching_plan_sha256"]
                )
                ids = [source["evidence_id"] for source in snapshot["sources"]]
                validate_question(question, ids, job["blueprint"]["question_type"])
                if job.get("schema_version") == _PLANNED_SCHEMA:
                    valid = (valid and _plans_intact(job)
                             and approval["batch_plan_sha256"] == job["batch_plan_sha256"]
                             and approval["item_plan_sha256"] == _digest(row["plan"])
                             and approval["quality_checks_sha256"] == _digest(row["quality_checks"]))
                    validate_audited_review(row["review"], snapshot, question,
                        duplicate_candidates=bool(row["quality_checks"].get("duplicate_candidates")),
                        evidence_mode=job.get("options", {}).get("review_evidence_mode", "verbatim"))
                else:
                    validate_review(row["review"], ids)
                if not valid:
                    raise ValueError("Approval bindings differ from current question")
            except Exception as exc:
                # ValidationError is not a ValueError. Keep the record readable
                # while making stale approval visibly unusable.
                row.update(status="needs_review", approval_invalidated=True,
                           error="题目、答案、审核或证据快照已变化，原审核结论失效，需要重新审核。")
                row.setdefault("history", []).append({"at": _now(), "stage": "approval_invalidated", "error_type": type(exc).__name__})
                changed = True
        if changed:
            job["status"] = "partial" if job.get("status") == "completed" else self._aggregate(job)
            self._progress(job, job["status"], "检测到已通过内容发生变化，原审核结论已失效。", final=True)
        return changed

    def get(self, job_id):
        with self._io_lock:
            job = self._read(job_id)
            changed = self._validate_approvals(job)
            flags_changed = job.get("formal_allowed") is not False or job.get("development_only") is not True
            job["formal_allowed"], job["development_only"] = False, True
            if changed or flags_changed:
                self._save(job)
            return deepcopy(job)

    def list(self):
        result = []
        for path in sorted(self.jobs.glob("*.json"), key=lambda value: value.stat().st_mtime, reverse=True)[:100]:
            try:
                job = self.get(path.stem)
            except (OSError, ValueError):
                continue
            result.append({key: deepcopy(job[key]) for key in ("job_id", "status", "created_at", "updated_at",
                                                              "blueprint", "progress", "development_only", "formal_allowed")})
        return result

    def export(self, job_id):
        result = self.get(job_id)
        result["formal_allowed"], result["development_only"] = False, True
        return result
