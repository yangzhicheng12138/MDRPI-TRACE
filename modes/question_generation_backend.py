"""Actual retrieval and independent model transports for development questions.

No existing answer-reflection model routes or prompts are changed. Each review
sees a complete current question; there are no cross-question history patches.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

import httpx
from jsonschema import Draft202012Validator, ValidationError

from app.config import AppConfig
from app.evidence_snapshot import ModelRunner, RunBudget, BudgetExceeded
from app.local_llm import LocalLLM, InvalidModelOutput, ModelUnavailable, _unique_object
from app.search_planner import CHAT_URL
from . import question_generation_prompts as prompts
from .question_generation_schema import (question_schema, validate_question,
    blind_review_schema, review_schema, validate_review)
from .question_generation_audit import audited_review_schema, validate_audited_review, evidence_text_field
from .question_review_text import (TextReviewFormatError, parse_review_text,
    review_text_instructions)
from .question_generation_examples import PROFILE as EXAMPLES_PROFILE


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"max_revisions": 2, "max_gap_searches_per_item": 1, "max_retrieval_rounds": 2,
    "job_seconds": 1800, "item_seconds": 300, "evidence_seconds": 100,
    "local_timeout_seconds": 150, "generator_max_tokens": 1800,
    "reviewer_max_tokens": 1400, "max_sources": 7, "reviewer_model": "deepseek-flash",
    "reviewer_thinking": "disabled", "reviewer_timeout_seconds": 60,
    "reviewer_output": "json", "review_evidence_mode": "verbatim",
    "api_key_env": "DEEPSEEK_API_KEY", "api_key_file": ".secrets/deepseek_search_api_key"}
PEOPLE = {"adult": "成人 adult", "pediatric": "儿童 pediatric children", "neonatal": "新生儿 neonatal infant"}
RAG_DEVICES = {"tracheostomy_tube": "tracheostomy", "vascular_access_device": "vascular_access",
    "nasal_prong_cpap": "cpap", "pulse_oximeter": "spo2_probe", "nasal_cannula": "other", "ecg_electrode": "other"}
DEVICE_TERMS = {"niv_mask": "无创通气面罩 NIV mask", "nasal_cannula": "鼻导管 nasal cannula",
    "nasogastric_tube": "鼻胃管 nasogastric tube", "tracheostomy_tube": "气切套管 tracheostomy",
    "endotracheal_tube": "气管插管 endotracheal tube", "cervical_collar": "颈托 cervical collar",
    "vascular_access_device": "血管通路 vascular access", "nasal_prong_cpap": "鼻塞 CPAP prongs",
    "ecg_electrode": "心电电极 ECG electrode", "pulse_oximeter": "血氧探头 pulse oximeter"}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def _local_grammar_schema(value):
    # JSON Schema pattern is a substring search, whereas vLLM's regex grammar
    # treats it as a whole-string constraint. Our nonblank guard r"\S" would
    # consequently force every answer to exactly one character. Keep that
    # guard in final validation, outside the native decoder grammar.
    if isinstance(value, dict):
        return {key: _local_grammar_schema(item) for key, item in value.items()
                if not (key == "pattern" and item == r"\S")}
    if isinstance(value, list):
        return [_local_grammar_schema(item) for item in value]
    return value


def safe_error(exc):
    if isinstance(exc, TextReviewFormatError):
        return str(exc)[:300]
    if isinstance(exc, (ModelUnavailable, InvalidModelOutput, BudgetExceeded)):
        return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[redacted]", str(exc))[:400]
    if isinstance(exc, ValidationError):
        return "输出结构未通过校验，请查看本次任务记录。"
    return "任务未完成：" + type(exc).__name__


def _scope(blueprint):
    return {key: blueprint[key] for key in ("topic", "learning_objective", "population", "device_type",
                                           "difficulty", "question_type", "requirements")}


def _material_payload(materials):
    return [{"id": m["id"], "title": m["title"], "case_facts": m.get("case_facts", ""),
             "teaching_reference": m.get("excerpt", ""),
             "reference_kind": m.get("excerpt_kind", "project_authored_reviewed_teaching_reference")}
            for m in materials]


def _evidence_payload(evidence):
    return [{key: source[key] for key in ("evidence_id", "origin", "title", "source_url", "location", "text",
        "population", "device_type", "material_level", "source_quality", "retrieval_support_status", "support_quotes",
        "doi", "pmid", "source_identity", "publication_date", "bibliographic_year")
        if key in source} for source in evidence.get("sources", [])]


def _planning_payload(teaching_plan):
    """Keep the shared source prefix stable across a planned batch of questions."""
    return {key: value for key, value in teaching_plan.items()
            if key not in {"item_plan", "quality_checks", "previous_questions", "audit_required", "review_evidence_mode"}}


class QuestionGenerationBackend:
    def __init__(self, config_path=None, rag_url="http://127.0.0.1:8020", *, transport=None, local_factory=LocalLLM):
        path = Path(config_path or ROOT / "config.question_generation.yaml").resolve()
        trial = AppConfig.from_file(path)
        self.config = AppConfig.from_file(trial.path("base_config", default="config.yaml"))
        self.options = {**DEFAULTS, **(trial.get("question_generation") or {})}
        if self.options["reviewer_model"] not in {"deepseek-flash", "deepseek-v4-pro"}:
            raise ValueError("Unsupported question reviewer model")
        if self.options["reviewer_thinking"] != "disabled":
            raise ValueError("This development profile uses non-thinking question reviews")
        if self.options["reviewer_output"] not in {"json", "text"}:
            raise ValueError("Unsupported question reviewer output")
        evidence_text_field(self.options["review_evidence_mode"])
        # Opt-in only: legacy configs retain exactly the original call path.
        if self.options.get("quality_profile", "legacy") not in {"legacy", "focused_v1", "evidence_first_v2", EXAMPLES_PROFILE}:
            raise ValueError("Unsupported question quality profile")
        if not 0 <= self.options["max_revisions"] <= 2 or not 1 <= self.options["max_retrieval_rounds"] <= 2:
            raise ValueError("Invalid question-generation iteration limits")
        self.rag_url = rag_url.rstrip("/")
        from urllib.parse import urlsplit
        parsed = urlsplit(self.rag_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Question retrieval must use the local RAG service")
        self.transport, self.local_factory = transport, local_factory

    def _key(self):
        key = os.environ.get(self.options["api_key_env"], "").strip()
        if not key:
            try:
                key = (self.config.project_root / self.options["api_key_file"]).read_text(encoding="utf-8")[:4097].strip()
            except (OSError, UnicodeError):
                return ""
        return key if 0 < len(key) <= 4096 and not any(c.isspace() or ord(c) < 32 for c in key) else ""

    def _call(self, role, prompt, fixed, variable, schema, name, log, budget, *, validator=None,
              text_parser=None, text_instructions=None):
        if text_parser is not None and role == "generator":
            raise ValueError("Text review transport is reserved for the reviewer")
        maximum = int(self.options["generator_max_tokens"] if role == "generator" else self.options["reviewer_max_tokens"])
        if role != "generator" and "quality_audit" in schema.get("properties", {}):
            maximum = min(2600, maximum + 650)
        # Fixed rules/evidence precede the current item for real prefix-cache reuse.
        output_instructions = (text_instructions or "") if text_parser else (
            "Return a compact complete JSON object satisfying this schema:\n" + encoded(schema))
        messages = [{"role": "system", "content": prompt + "\n" + output_instructions},
                    {"role": "user", "content": encoded(fixed)},
                    {"role": "user", "content": encoded(variable)}]
        key = self._key()
        if key and key in encoded(messages):
            # Check BEFORE logging as well as before sending, including local
            # author calls which must not persist a credential from source data.
            raise ValueError("Credential appeared in question data")

        def redacted(value):
            raw = encoded(value)
            if key:
                raw = raw.replace(key, "[redacted credential]")
            return json.loads(re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[redacted credential]", raw))
        for attempt in range(2):
            budget.consume("model_calls")
            started = time.monotonic()
            log.write("question_model_request", redacted({"name": name, "role": role, "attempt": attempt + 1,
                "model": self.config.get("llm", "model") if role == "generator" else self.options["reviewer_model"],
                "messages": messages, "schema": schema, "max_tokens": maximum}))
            usage, metadata, content = None, {}, None
            try:
                if role == "generator":
                    llm = self.local_factory(self.config)
                    try:
                        try:
                            value = llm.structured(messages, _local_grammar_schema(schema), name, max_tokens=maximum,
                                deadline=min(budget.deadline, time.monotonic() + float(self.options["local_timeout_seconds"])))
                        except InvalidModelOutput as exc:
                            # LocalLLM attaches the raw response to format failures.
                            # Recover the candidate and the precise contract error
                            # so the single bounded retry can actually repair it.
                            raw = getattr(exc, "raw_response", None)
                            if isinstance(raw, dict):
                                usage = raw.get("usage")
                                choices = raw.get("choices")
                                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                                    choice = choices[0]
                                    message = choice.get("message")
                                    content = message.get("content") if isinstance(message, dict) else None
                                    metadata = {k: raw.get(k) for k in ("id", "model")}
                                    metadata["finish_reason"] = choice.get("finish_reason")
                                    if isinstance(content, str):
                                        try:
                                            candidate = json.loads(content, object_pairs_hook=_unique_object)
                                        except (ValueError, TypeError):
                                            candidate = None
                                        errors = list(Draft202012Validator(schema).iter_errors(candidate))
                                        if errors:
                                            first = errors[0]
                                            path = "/".join(str(part) for part in first.absolute_path) or "root"
                                            raise InvalidModelOutput("结构错误 " + path + ": " + first.message[:600]) from exc
                            raise
                        finally:
                            usage = getattr(llm, "last_generation_usage", None) or usage
                    finally:
                        llm.close()
                else:
                    if not key:
                        raise ModelUnavailable("DeepSeek 出题审核密钥不可用")
                    wire = {"model": self.options["reviewer_model"], "messages": messages,
                        "stream": False, "thinking": {"type": "disabled"}, "temperature": 0,
                        "max_tokens": maximum}
                    if not text_parser:
                        wire["response_format"] = {"type": "json_object"}
                    if key in encoded(wire):
                        raise ValueError("Credential appeared in question data")
                    try:
                        with httpx.Client(timeout=min(float(self.options["reviewer_timeout_seconds"]), budget.remaining()),
                                          trust_env=False, follow_redirects=False, transport=self.transport) as client:
                            response = client.post(CHAT_URL, headers={"Authorization": "Bearer " + key}, json=wire)
                    except httpx.HTTPError as exc:
                        raise ModelUnavailable("DeepSeek 出题审核网络异常：" + type(exc).__name__) from None
                    if response.status_code != 200:
                        raise ModelUnavailable("DeepSeek 出题审核 HTTP " + str(response.status_code))
                    try:
                        body = response.json()
                        if not isinstance(body, dict):
                            raise InvalidModelOutput("出题审核响应不是 JSON 对象")
                        usage = body.get("usage")
                        choice = body["choices"][0]
                        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                            raise InvalidModelOutput("出题审核缺少有效消息对象")
                        content = choice["message"].get("content")
                        metadata = {k: body.get(k) for k in ("id", "model")}
                        metadata["finish_reason"] = choice.get("finish_reason")
                        if choice.get("finish_reason") != "stop" or not isinstance(content, str):
                            raise InvalidModelOutput("出题审核输出未完整返回")
                        if not text_parser:
                            value = json.loads(content, object_pairs_hook=_unique_object)
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        if isinstance(exc, InvalidModelOutput):
                            raise
                        raise InvalidModelOutput("出题审核返回的 JSON 无效") from None
                    if text_parser:
                        if key and key in content:
                            raise InvalidModelOutput("模型输出包含凭据，响应已隐藏")
                        log.write("question_model_text_response", redacted({"name": name, "role": role,
                            "attempt": attempt + 1, "text": content, "usage": usage, "metadata": metadata}))
                        value = text_parser(content)
                if key and key in encoded(value):
                    raise InvalidModelOutput("模型输出包含凭据，响应已隐藏")
                if role == "generator" and self.options.get("quality_profile") == "evidence_first_v2":
                    # Retain the real candidate if a cross-field check fails;
                    # raw output must not disappear behind a generic error.
                    content = encoded(value)
                Draft202012Validator(schema).validate(value)
                if validator:
                    value = validator(value)
                budget.remaining()
                log.write("question_model_response", redacted({"name": name, "role": role, "value": value,
                    "usage": usage, "metadata": metadata, "seconds": round(time.monotonic() - started, 3)}))
                return value
            except (InvalidModelOutput, ValidationError, TextReviewFormatError) as exc:
                log.write("question_model_format_error", redacted({"name": name, "role": role, "attempt": attempt + 1,
                    "error_type": type(exc).__name__, "message": safe_error(exc), "content": content,
                    "usage": usage, "metadata": metadata}))
                if attempt or "context budget" in str(exc).lower():
                    raise
                budget.remaining()
                # One format repair, within the original deadline. No blind
                # transport retry, fallback to another judge, or grading patch.
                if text_parser:
                    if isinstance(exc, ValidationError):
                        path = "/".join(str(part) for part in exc.absolute_path) or "审核报告"
                        constraint = str(exc.validator_value)[:120] if exc.validator is not None else ""
                        diagnosis = f"{path}：{exc.validator or '审核约束'} {constraint}" if constraint else path
                        if exc.validator is None:
                            diagnosis += "：" + exc.message[:200]
                    else:
                        diagnosis = safe_error(exc)
                    correction = ("仅修正审核文字报告的格式或缺漏，保留有依据的医学判断。"
                        "按规定中文标签重新完整回复，不要JSON，不要重写题目。具体位置：" + diagnosis)
                else:
                    correction = "上次输出未满足结构要求。请简短完整地重写同一题包，只用给定来源ID。错误：" + (
                        exc.message[:400] if isinstance(exc, ValidationError) else safe_error(exc))
                messages = messages[:2] + [{"role": "user", "content": encoded({**variable,
                    "format_repair": redacted(correction)})}]
                maximum = min(maximum + 350, 2600)
            except Exception as exc:
                log.write("question_model_error", redacted({"name": name, "role": role, "error_type": type(exc).__name__,
                    "message": safe_error(exc), "usage": usage, "metadata": metadata}))
                raise

    def retrieve(self, blueprint, materials, log, budget, *, gaps=None):
        """Always attempt local retrieval AND live web search before generation."""
        from app.web_search import create_web_search
        from app.evidence_constraints import deduplicate_documents
        device = DEVICE_TERMS.get(blueprint["device_type"], blueprint["device_type"])
        query = " ".join([PEOPLE[blueprint["population"]], device, "MDRPI",
                          blueprint["topic"], blueprint["learning_objective"]])[:900]
        if gaps:
            # Only the requested clinical concept is sent, never a case narrative.
            query = (PEOPLE[blueprint["population"]] + " " + blueprint["device_type"] + " " + gaps[0]["query"])[:900]
        sources, events = [], []
        try:
            with httpx.Client(timeout=min(40, budget.remaining()), trust_env=False, transport=self.transport) as client:
                response = client.post(self.rag_url + "/v1/retrieve", json={"query": query,
                    "population": [blueprint["population"]],
                    "device_type": [RAG_DEVICES.get(blueprint["device_type"], blueprint["device_type"])]})
            body = response.json()
            events.append({"provider": "local", "status": body.get("status", str(response.status_code)),
                           "http_status": response.status_code})
            if response.is_success:
                for source in body.get("evidence", []):
                    if not source.get("text", "").strip():
                        continue
                    sources.append({**source, "origin": "local", "material_level": source.get("material_level") or "full_text_excerpt",
                        "evidence_id": source.get("evidence_id") or source.get("chunk_id") or source.get("id")
                            or "LOCAL_" + digest(source)[:16],
                        "source_url": source.get("source_url") or "/v1/sources/" + source.get("doc_id", "")
                            + "#page=" + str(source.get("page_start", "")),
                        "location": source.get("location") or "PDF pages " + str(source.get("source_pages", ""))
                            + "; " + source.get("section", "")})
            log.write("question_local_retrieval", {"query": query, "http_status": response.status_code, "result": body})
        except (httpx.HTTPError, ValueError) as exc:
            events.append({"provider": "local", "status": "unavailable", "error_type": type(exc).__name__})
            log.write("question_local_retrieval_error", events[-1])
        web_budget = RunBudget(min(float(self.options["evidence_seconds"]), budget.remaining()))
        # The search/fetch adapter is shared with the project. Its optional
        # answer-planning runner stays absent: this workflow has its own judge.
        web = create_web_search(self.config, web_budget, log, None, mode="off")
        short_query = (gaps[0]["query"] if gaps else query)[:120]
        log.write("question_web_query", {"query": short_query, "original_scope": _scope(blueprint), "phase": "gap" if gaps else "initial"})
        web.request_question = query
        result = {"evidence": [], "events": []}
        try:
            discoveries = web._discover(short_query)
            log.write("question_web_discoveries", {"query": short_query,
                "rows": [{k: v for k, v in row.items() if k != "body_content"} for row in discoveries]})
            web._read_candidates(discoveries, short_query, result)
        except Exception as exc:
            result["events"].append({"provider": "web", "status": "unavailable", "error_type": type(exc).__name__})
        sources.extend({**s, "origin": "external"} for s in result["evidence"])
        events.extend(getattr(web, "last_search_attempts", []))
        events.extend(result["events"])
        selected = []
        separate_lanes = [s for origin in ("local", "external") for s in
            deduplicate_documents([s for s in sources if s.get("origin") == origin], retain_local_passages=True)]
        for source in separate_lanes:
            if source.get("material_level") in {"search_snippet", "article_record"} or not source.get("text", "").strip():
                continue
            if self.options.get("quality_profile") == EXAMPLES_PROFILE:
                from .question_generation_examples import source_mismatch
                reason = source_mismatch(source, blueprint, materials)
                if reason:
                    # Filter before the per-lane cap so an eligible page can
                    # fill the slot. Preserve the original for later auditing.
                    events.append({"stage": "applicability_filter", "evidence_id": source.get("evidence_id"),
                                   "reason": reason})
                    log.write("question_source_excluded", {"reason": reason, "source": source})
                    continue
            selected.append(source)
        # Preserve a few sources from each lane; local volume cannot crowd out
        # actual web pages and silently turn this into local-only generation.
        local = [s for s in selected if s.get("origin") == "local"][:4]
        external = [s for s in selected if s.get("origin") == "external"][:3]
        value = {"sources": local + external, "events": events, "rag_attempted": True, "web_attempted": True,
            "local_success": bool(local), "web_success": bool(external),
            "query": query, "web_query": short_query, "tool_counts": deepcopy(web_budget.counts)}
        log.write("question_evidence_packet", value)
        return value

    def generate(self, blueprint, materials, evidence, teaching_plan, log, budget, *, index, previous=None, review=None):
        ids = [s["evidence_id"] for s in evidence["sources"]]
        schema = question_schema(ids, blueprint["question_type"])
        # The case is selected input, not a creative generation task. Constrain
        # the native decoder to one of the actual selected case bodies so it
        # cannot append clinical details or explanatory examples to the stem.
        stems = list(dict.fromkeys(m["case_facts"] for m in materials if m.get("case_facts")))
        item_plan = teaching_plan.get("item_plan")
        if item_plan is not None:
            selected = [m for m in materials if m["id"] == item_plan["material_id"]]
            if len(selected) != 1 or selected[0]["case_facts"] != item_plan["case_facts"]:
                raise ValueError("题目规划与所选固定病例不一致")
            stems = [item_plan["case_facts"]]
        if not stems:
            raise ValueError("所选材料缺少固定病例正文")
        schema["properties"]["stem"] = {"type": "string", "enum": stems}
        fixed = {"blueprint": _scope(blueprint), "materials": _material_payload(materials),
                 "teaching_plan": _planning_payload(teaching_plan), "evidence": _evidence_payload(evidence)}
        variable = {"question_number": index + 1, "previous_question": previous, "review_feedback": review,
                    "avoid_repeating_tasks": blueprint.get("_previous_tasks", []), "item_plan": item_plan}
        if self.options.get("quality_profile", "legacy") == "evidence_first_v2":
            from .question_generation_grounded import generate_grounded
            return generate_grounded(self, fixed, variable, schema, ids, blueprint["question_type"], log, budget)
        if self.options.get("quality_profile", "legacy") == "focused_v1":
            from .question_generation_focus import generate_focused
            return generate_focused(self, fixed, variable, schema, ids, blueprint["question_type"], log, budget)
        prompt = prompts.REVISE_QUESTION if previous else prompts.GENERATE_QUESTION
        if self.options.get("quality_profile") == EXAMPLES_PROFILE:
            from .question_generation_examples import AUTHOR_GUIDANCE, authoring_example
            example = authoring_example(blueprint)
            if example is not None:
                fixed["authoring_example"] = example
                log.write("question_authoring_example", {key: example[key] for key in ("id", "catalog_sha256", "purpose")})
            prompt += AUTHOR_GUIDANCE
        return self._call("generator", prompt,
            fixed, variable, schema, "question_revision" if previous else "question_generation",
            log, budget, validator=lambda value: validate_question(value, ids, blueprint["question_type"]))

    def select_candidate(self, blueprint, materials, evidence, candidates, log, budget):
        from .question_generation_grounded import SELECTION_PROMPT, selection_schema, parse_selection
        # Hash order hides generation order. Existing audit opinions and scores
        # are deliberately excluded; the editor compares actual current texts.
        ordered = sorted(candidates, key=lambda row: row["question_sha256"])
        labels = list("ABC")[:len(ordered)]
        if not 2 <= len(ordered) <= 3:
            raise ValueError("Candidate comparison requires two or three reviewed versions")
        wire = [{"label": label, "question": deepcopy(row["question"])} for label, row in zip(labels, ordered)]
        result = self._call("reviewer", SELECTION_PROMPT,
            {"blueprint": _scope(blueprint), "materials": [{k: m[k] for k in ("id", "case_facts")} for m in materials],
             "evidence": _evidence_payload(evidence)}, {"candidates": wire}, selection_schema(labels),
            "question_candidate_comparison", log, budget,
            text_parser=lambda text: parse_selection(text, labels),
            text_instructions="仅回复两项：\n选择：" + "/".join(labels) + "/并列\n理由：简要说明真实差异和仍需改进之处。")
        selected = None if result["choice"] == "tie" else ordered[labels.index(result["choice"])]["question_sha256"]
        return {"question_sha256": selected, "reason": result["reason"], "model": self.options["reviewer_model"],
                "thinking": "disabled", "labels": {label: row["version"] for label, row in zip(labels, ordered)},
                "choice": result["choice"], "prior_reviews_and_scores_visible": False}

    def blind_review(self, blueprint, materials, question, log, budget):
        # Projection is repeated at the backend boundary so no alternative
        # caller can accidentally expose the answer through this method.
        visible = {key: question[key] for key in ("title", "stem", "task", "options") if key in question}
        facts = [{key: m[key] for key in ("id", "title", "case_facts") if key in m} for m in materials]
        text = self.options["reviewer_output"] == "text"
        prompt = prompts.BLIND_REVIEW_TEXT if text else prompts.BLIND_REVIEW
        if self.options.get("quality_profile", "legacy") in {"focused_v1", "evidence_first_v2"}:
            from .question_generation_focus import BLIND_FOCUS
            prompt += BLIND_FOCUS
        return self._call("reviewer", prompt,
            {"blueprint": _scope(blueprint)},
            {"student_visible_materials": facts, "question": visible}, blind_review_schema(),
            "question_blind_review", log, budget,
            text_parser=(lambda content: parse_review_text(content, evidence={"sources": []},
                question=visible, audited=False, blind=True)) if text else None,
            text_instructions=review_text_instructions(audited=False, blind=True) if text else None)

    def review(self, blueprint, materials, evidence, teaching_plan, question, blind, log, budget):
        ids = [s["evidence_id"] for s in evidence["sources"]]
        fixed = {"blueprint": _scope(blueprint), "materials": _material_payload(materials),
                 "teaching_plan": _planning_payload(teaching_plan), "evidence": _evidence_payload(evidence)}
        audited = teaching_plan.get("audit_required") is True
        evidence_mode = teaching_plan.get("review_evidence_mode", self.options["review_evidence_mode"])
        variable = {"question": question, "independent_attempt": blind,
                    "item_plan": teaching_plan.get("item_plan"),
                    "quality_checks": teaching_plan.get("quality_checks", {}),
                    "previous_questions": teaching_plan.get("previous_questions", [])}
        schema = audited_review_schema(ids, question, evidence_mode=evidence_mode) if audited else review_schema(ids)
        validate = (lambda value: validate_audited_review(value, evidence, question,
                        duplicate_candidates=bool(teaching_plan.get("quality_checks", {}).get("duplicate_candidates")),
                        evidence_mode=evidence_mode)) if audited else (lambda value: validate_review(value, ids))
        text = self.options["reviewer_output"] == "text"
        prompt = (prompts.AUDITED_REVIEW_TEXT if audited else prompts.FULL_REVIEW_TEXT) if text else (
            prompts.AUDITED_REVIEW if audited else prompts.FULL_REVIEW)
        if audited and evidence_mode == "semantic":
            prompt = prompts.AUDITED_REVIEW_SEMANTIC_TEXT if text else prompts.AUDITED_REVIEW_SEMANTIC
        if self.options.get("quality_profile") == EXAMPLES_PROFILE:
            from .question_generation_examples import REVIEW_CALIBRATION
            prompt += REVIEW_CALIBRATION
        if self.options.get("quality_profile", "legacy") in {"focused_v1", "evidence_first_v2"}:
            from .question_generation_focus import REVIEW_FOCUS, review_focus_packet
            prompt += REVIEW_FOCUS
            variable["focus_checks"] = review_focus_packet(blueprint, question, evidence)
        return self._call("reviewer", prompt, fixed,
            variable, schema, "question_full_review", log, budget, validator=validate,
            text_parser=(lambda content: parse_review_text(content, evidence=evidence,
                question=question, audited=audited, evidence_mode=evidence_mode)) if text else None,
            text_instructions=review_text_instructions(audited=audited, evidence_mode=evidence_mode) if text else None)
