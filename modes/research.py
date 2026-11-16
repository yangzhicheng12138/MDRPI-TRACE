"""Minimal, explicitly labelled research exports of observed interaction records."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import re
import secrets

from .case_intake import inspect_import
from .learning import records as session_records
from .learning_analytics import record_metrics, step_measure
from .telemetry import summarize_turns
from .answer_display import performance_sources, performance_status

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "mdrpi_research_export_v1"
PURPOSES = {"development", "pilot", "research"}
MODES = {"chat", "teaching", "performance"}
TOOLS = {"clock", "knowledge_base", "web_search", "files", "weather", "read_url", "calculator", "preferences", "document"}
STEP_FIELDS = ("value", "sample_count", "met", "required", "partial", "incorrect", "uncertain", "missing",
               "hint_count", "reference_answer_count", "explanation_count", "independent_met", "assisted_met", "attempt_count")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot_metadata(state, *, active=False):
    """Describe the saved checkpoint without claiming the analysis passed."""
    running = bool(active or state.get("pending"))
    status = ("cancelled_snapshot" if state.get("cancelled") else "error_snapshot" if state.get("error")
              else "in_progress_snapshot" if running else "saved_snapshot")
    return {"status": status, "complete": not running,
            "saved_checkpoint_complete": not running,
            "analysis_status": performance_status(state["performance_result"]) if state.get("performance_result") else None,
            "note": ("本轮未完成的已保存快照；停止、错误或运行中都不表示分析成功。" if running else
                     "已保存的会话快照；分析是否完整以实际结果状态为准。")}


def capabilities(service):
    """Report loaded web settings; delegated RAG execution must be checked per run."""
    from . import prompts
    from .clinical_flow import TASKS
    from .teaching import CASES
    config = getattr(service.backend, "config", None)
    values = getattr(config, "values", {})
    safe_config = {section: {key: values.get(section, {}).get(key) for key in keys} for section, keys in {
        "llm": ("model", "context_window", "temperature"),
        "reflection": ("research_consolidated_review", "research_max_revision_rounds", "program_checks", "protect_revision_scope"),
        "retrieval": ("strict_device", "allow_general_fallback", "require_audited_index")}.items()}
    files = ("modes/service.py", "modes/clinical_flow.py", "modes/prompts.py", "modes/research.py",
             "data/teaching_bank/content_reviews.json", "data/teaching_bank/mdrpi_review_queue.json")
    return {"schema_version": "mdrpi_capabilities_v1",
            "reference_policy": service.fixed_reference_policy or "session_configured",
            "modes": {
                "chat": {"local_retrieval": "question-dependent tool", "web": "question-dependent tool",
                         "reflection": "no clinical six-step reflection loop", "observations": "chat_plan, source_events, turn latency"},
                "teaching": {"local_retrieval": "reference retrieval for assessment", "web": "according to reference policy",
                             "assessment": "learner-quote-grounded formative criteria; adverse criterion verification",
                             "reflection": "learner feedback and tutor summary, not the research generation loop"},
                "performance": {"endpoint": "/v1/research/answer", "web": "explicit live/off/replay",
                                "reflection": "delegated research six-step generation/review/revision",
                                "observations": "saved performance_result and research run artifacts"}},
            "loaded_web_configuration": safe_config,
            "loaded_prompt_sha256": digest({"chat": prompts.CHAT, "grading": prompts.RUBRIC_ASSESS, "tasks": TASKS}),
            "files_on_disk_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files if (ROOT / name).exists()},
            "catalog": {"total": len(CASES), "review_status_counts": dict(Counter(item.get("review_status", "unreviewed") for item in CASES.values()))},
            "measurement_boundary": "Configuration permits behavior; per-run evidence records actual execution. No clinical effectiveness or human-rating result is inferred."}


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _enum(value, choices, default="unknown"):
    return value if isinstance(value, str) and value in choices else default


def _code(value, label, required=False):
    if not isinstance(value, str) or (value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value)) or required and not value:
        raise ValueError(label + "须使用1–64位字母数字编码，可含点、横线和下划线，不填写姓名。")
    return value


def _version(value):
    if value in {"project_formative_v1", "project_formative_v2_20261012", "ai_case_formative_v1_20261010", "ai_question_formative_v1_20261010", "ai_expansion_case_v1_20261010", "ai_expansion_short_v1_20261010"}:
        return value
    return "version_" + digest(str(value))[:16] if value else "unrecorded"


def _latencies(root, session_ids):
    result = []
    runs = (Path(root) / "runs").resolve()
    for sid in sorted(session_ids):
        if not isinstance(sid, str) or not re.fullmatch(r"[a-f0-9]{32}", sid):
            continue
        for path in sorted((runs / sid).glob("*/*_latency.json")):
            if not path.resolve().is_relative_to(runs) or path.stat().st_size > 2_000_000:
                continue
            try:
                row = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            if row.get("session_id") != sid:
                continue
            result.append({"mode": _enum(row.get("mode"), MODES),
                           "status": _enum(row.get("status"), {"done", "error", "cancelled"}),
                           "total_seconds": _number(row.get("total_seconds")),
                           "first_output_seconds": _number(row.get("first_output_seconds")),
                           "first_output_kind": _enum(row.get("first_output_kind"), {"token", "complete"}, None)})
    return result


def export_bundle(service, sid, *, purpose="development", scope="session", study_id="", protocol_version="",
                  include_text=False, text_review_confirmed=False):
    if purpose not in PURPOSES or scope not in {"session", "profile"}:
        raise ValueError("请选择明确的数据用途和导出范围。")
    _code(study_id, "研究编号", purpose == "research")
    _code(protocol_version, "方案版本", purpose == "research")
    if include_text and not text_review_confirmed:
        raise ValueError("导出正文前须确认已核查身份信息；默认可只导出结构化指标。")
    state = service.get(sid)
    if state.get("pending") or any(row["session_id"] == sid for row in service.activity()):
        raise ValueError("请在本轮结束或停止后导出，确保记录完整。")
    if scope == "profile" and state["mode"] != "teaching":
        raise ValueError("学习档案导出仅用于教学模式。")
    salt = secrets.token_bytes(32)
    aliases = {}

    def alias(kind, value):
        key = (kind, str(value))
        if key not in aliases:
            aliases[key] = kind + "_" + hmac.new(salt, (kind + ":" + str(value)).encode(), hashlib.sha256).hexdigest()[:20]
        return aliases[key]

    counts = Counter()

    def scrub(value):
        if not isinstance(value, str) or not value:
            return ""
        result = inspect_import(value)
        counts.update(result["redactions"])
        text = result["sanitized_text"]
        # Do not expose raw service identifiers or machine-local file paths in
        # the optional prose if a participant pasted them into an answer.
        text = re.sub(r"(?<![a-fA-F0-9])[a-fA-F0-9]{32,64}(?![a-fA-F0-9])", "[记录标识已隐藏]", text)
        text = re.sub(r"(?:/(?:home|hd_4t|tmp|mnt|data)/[^\s<>]+|[A-Za-z]:\\[^\s<>]+)", "[本机路径已隐藏]", text)
        return text

    if scope == "profile":
        archive = service.learning_profile_export(sid)
        pairs = [(row["source_session_id"], row["exercise"]) for row in archive["records"]]
    else:
        pairs = [(sid, record) for record in session_records(state)] if state["mode"] == "teaching" else []
    # Latest stored observation of an exercise wins; do not count rechecks twice.
    by_exercise = {}
    for index, (source_sid, record) in enumerate(pairs):
        by_exercise[record.get("exercise_id") or f"missing:{index}"] = (source_sid, record)
    pairs = list(by_exercise.values())
    from .teaching import CASES, student_answers
    from .catalog import taxonomy
    allowed_devices = set(taxonomy()["devices"])
    exported = []
    for index, (source_sid, record) in enumerate(pairs, 1):
        measurement = record_metrics(record)
        item = record.get("bank_item") or {}
        flow = record.get("clinical_flow") or {}
        entry = {"record_id": alias("record", record.get("exercise_id") or index),
                 "session_id": alias("session", source_sid), "order": index,
                 "case_id": item.get("id") if item.get("id") in CASES else "custom_case",
                 "case_origin": _enum(record.get("case_origin"), {"standardized_source", "specialty_bank", "textbook_bank", "practice_library", "provided_case", "generated_case", "user_deidentified_case"}),
                 "content_review": _enum(item.get("review_status"), {"approved_ai", "approved", "approval_invalidated", "pending_review", "source_imported_pending_expert_review"}),
                 "populations": [v for v in measurement["populations"] if v in {"adult", "pediatric", "neonatal"}],
                 "device_types": [v for v in measurement["device_types"] if v in allowed_devices],
                 "difficulty": _enum(measurement["difficulty"], {"基础", "进阶", "复杂"}),
                 "completion_status": measurement["completion_status"], "guided": measurement["guided"],
                 "scoring_version": _version(measurement["scoring_version"] or item.get("rubric", {}).get("version")),
                 "passed_steps": measurement["passed_steps"], "answered_steps": measurement["answered_steps"],
                 "attempt_count": measurement["attempt_count"], "hint_count": measurement["hint_count"],
                 "reference_answer_count": measurement["reference_answer_count"], "explanation_count": measurement["explanation_count"],
                 "required_point_coverage": measurement["value"], "step_metrics": []}
        entry["content_snapshot_sha256"] = digest({"case": item.get("case") if item.get("id") in CASES else "custom_case",
                                                   "version": item.get("version"), "review": item.get("content_review", {}).get("review_fingerprint") or item.get("content_fingerprint")})
        entry["reference_policy"] = _enum(record.get("policy"), {"none", "local", "web", "auto"}, "not_recorded")
        entry["source_count"] = len(record.get("sources", []))
        for step in measurement["six_steps"] if measurement["guided"] else []:
            projected = {"id": step["id"], "status": _enum(step["status"], {"pending", "active", "awaiting_next", "passed"}),
                         **{field: _number(step.get(field)) for field in STEP_FIELDS}}
            # Labels and quotations are deliberately absent from minimal data.
            projected["criterion_statuses"] = [{"position": n, "status": row["status"]} for n, row in enumerate(step["criteria"], 1)]
            entry["step_metrics"].append(projected)
        if not measurement["guided"]:
            latest = (record.get("reviews") or [{}])[-1]
            question = step_measure({"answers": student_answers(record), "reviews": record.get("reviews", []),
                                     "criterion_checks": latest.get("criterion_checks", []), "hints": record.get("hint_events", []),
                                     "reference_views": record.get("reference_views", []),
                                     "teaching_dialogue_events": record.get("teaching_dialogue_events", [])})
            entry["question_metrics"] = {"id": "question", "status": measurement["completion_status"],
                                         **{field: _number(question.get(field)) for field in STEP_FIELDS}}
        if include_text:
            entry["text"] = {"case": scrub(record.get("case", "")),
                             "student_answers": [scrub(text) for text in student_answers(record)],
                             "feedback": [scrub(review.get("feedback", "")) for review in record.get("reviews", [])],
                             "summary": scrub(record.get("summary", ""))}
            entry["text"]["steps"] = [{"id": step.get("id") if step.get("id") in {r["id"] for r in measurement["six_steps"]} else "unknown",
                                        "student_answers": [scrub(text) for text in step.get("answers", [])],
                                        "judgements": [{"status": _enum(check.get("status"), {"met", "partial", "missing", "incorrect", "uncertain"}),
                                                        "criterion": scrub((check.get("criterion") or {}).get("zh", "") if isinstance(check.get("criterion"), dict) else str(check.get("criterion") or "")),
                                                        "quotes": [scrub(e.get("quote", "")) for e in check.get("evidence", [])]}
                                                       for check in step.get("criterion_checks", [])]} for step in flow.get("steps", [])]
        exported.append(entry)
    source_ids = {source_sid for source_sid, _ in pairs} | ({sid} if scope == "session" else set())
    latency = summarize_turns(_latencies(service.root, source_ids))
    performance_runs, source_records = [], []
    performance = state.get("performance_result") if scope == "session" and state["mode"] == "performance" else None
    sources = performance_sources(performance) if performance else state.get("sources", [])
    if performance:
        run_key = alias("run", performance.get("run_id") or sid)
        observed = performance_status(performance)
        timing = performance.get("timing", {})
        performance_runs.append({"run_id": run_key, "session_id": alias("session", sid), **observed,
                                 "web_mode": _enum(performance.get("web", {}).get("mode"), {"live", "off", "replay"}),
                                 "timing": {field: _number(timing.get(field)) for field in
                                            ("seconds", "model_calls", "search_requests", "fetch_requests", "local_batches", "revision_rounds")},
                                 "observation_scope": "last saved performance result; not every historical run"})
        for index, source in enumerate(sources, 1):
            origin = _enum(source.get("origin"), {"local", "external"})
            row = {"source_id": alias("source", source.get("evidence_id") or index), "run_id": run_key,
                   "origin": origin, "verified": source.get("verified") is True,
                   "source_type": str(source.get("source_type") or "unknown")[:80],
                   "content_sha256": source.get("content_sha256") if re.fullmatch(r"[a-f0-9]{64}", str(source.get("content_sha256", ""))) else None,
                   "source_pdf_sha256": source.get("source_pdf_sha256") if re.fullmatch(r"[a-f0-9]{64}", str(source.get("source_pdf_sha256", ""))) else None}
            # Bibliography and public source URLs are provenance, not clinical
            # case prose. Uploaded/private material stays omitted by default.
            if origin in {"local", "external"} and source.get("source_type") != "user_upload":
                row["title"] = scrub(source.get("title", ""))
                url = str(source.get("source_url") or "")
                row["source_url"] = url if url.startswith(("http://", "https://", "/v1/sources/")) and not inspect_import(url)["redactions"] else ""
                row["source_pages"] = [p for p in source.get("source_pages", []) if isinstance(p, int) and not isinstance(p, bool) and p > 0]
            if include_text:
                row["text"] = scrub(source.get("text", ""))
            source_records.append(row)
    result = {"schema_version": SCHEMA_VERSION, "exported_at": datetime.now(timezone.utc).isoformat(),
              "dataset_id": "export_" + secrets.token_hex(10),
              "data_use": {"purpose": purpose, "study_id": study_id, "protocol_version": protocol_version,
                           "declaration": "declared_for_this_export; not inferred from prior records", "scope": scope},
              "privacy": {"identifier_fields_omitted": True, "free_text_included": bool(include_text),
                          "text_review_confirmed": bool(text_review_confirmed) if include_text else False,
                          "automatic_redaction_counts": dict(counts), "identifier_linkage": "within_this_export_only",
                          "note": "默认省略姓名、档案名、访问码、原会话编号、精确练习时间、文件路径与全部正文。可选正文经过规则过滤，仍需分享前核查。"},
              "capabilities_at_export": capabilities(service),
              "observed_session": {"mode": state["mode"], "reference_policy": _enum(state.get("policy"), {"none", "local", "web", "auto"}),
                                   "web_mode": _enum(state.get("web_mode"), {"live", "off", "replay"}),
                                   "user_message_count": sum(row.get("role") == "user" for row in state.get("messages", [])) if scope == "session" else None,
                                   "last_chat_tools": [name for name in state.get("chat_plan", {}).get("tools", []) if name in TOOLS] if scope == "session" else [],
                                   "source_count": len(sources) if scope == "session" else None,
                                   "evidence_scope": "last saved session snapshot; not every historical turn"},
              "records": exported,
              "performance_runs": performance_runs, "source_records": source_records,
              "evaluation": {"answer_quality": {"accuracy": None, "citation_support": None, "measurement": "not_measured_by_this_export"},
                             "teaching": {"record_count": len(exported), "completion_counts": dict(Counter(row["completion_status"] for row in exported)),
                                          "measure": "AI formative required-point coverage; not a validated competence scale",
                                          "scoring_versions": sorted({row["scoring_version"] for row in exported}), "causal_learning_effect": None},
                             "performance": {"run_count": len(performance_runs),
                                             "status_counts": dict(Counter(row["status"] for row in performance_runs)),
                                             "measure": "observed pipeline completion and self-audit; not clinical correctness"},
                             "usability": {"server_latency": latency, "participant_survey": None, "measurement": "observed server events; excludes browser/network delivery"}}}
    if include_text and state["mode"] != "teaching":
        result["text"] = {"messages": [{"role": _enum(row.get("role"), {"user", "assistant", "system"}), "content": scrub(row.get("content", ""))} for row in state.get("messages", [])]}
        result["privacy"]["automatic_redaction_counts"] = dict(counts)
    result["privacy"]["automatic_redaction_counts"] = dict(counts)
    result["bundle_sha256"] = digest(result)
    return result


def csv_export(bundle):
    """Exercise rows and explicitly labelled performance-run rows remain distinct."""
    fields = ["schema_version", "purpose", "study_id", "protocol_version", "dataset_id", "record_type", "record_id", "case_id",
              "completion_status", "scoring_version", "content_review", "step", "status", *STEP_FIELDS]
    fields += ["stop_reason", "source_count", "unresolved_issue_count", "unresolved_issue_severities", "error_count",
               "web_mode", "seconds", "model_calls", "search_requests", "fetch_requests", "local_batches", "revision_rounds"]
    include_text = bundle["privacy"]["free_text_included"]
    if include_text:
        fields += ["case_text", "student_answers"]
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for record in bundle["records"]:
        for step in record["step_metrics"] or ([record["question_metrics"]] if record.get("question_metrics") else []):
            row = {"schema_version": bundle["schema_version"], "record_type": "teaching_exercise", **{key: bundle["data_use"][key] for key in ("purpose", "study_id", "protocol_version")},
                   "dataset_id": bundle["dataset_id"], **{key: record[key] for key in ("record_id", "case_id", "completion_status", "scoring_version", "content_review")},
                   "step": step["id"], "status": step["status"], **{field: step[field] for field in STEP_FIELDS}}
            if include_text:
                text = record.get("text", {})
                current = next((s for s in text.get("steps", []) if s["id"] == step["id"]), {})
                row.update(case_text=text.get("case", ""), student_answers="\n".join(current.get("student_answers", []) or text.get("student_answers", [])))
            writer.writerow({key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value for key, value in row.items()})
    for run in bundle.get("performance_runs", []):
        row = {"schema_version": bundle["schema_version"], "record_type": "performance_run", "record_id": run["run_id"],
               "dataset_id": bundle["dataset_id"], **{key: bundle["data_use"][key] for key in ("purpose", "study_id", "protocol_version")},
               "status": run["status"], **{key: run[key] for key in
               ("stop_reason", "source_count", "unresolved_issue_count", "error_count", "web_mode")},
               "unresolved_issue_severities": json.dumps(run["unresolved_issue_severities"], ensure_ascii=False, sort_keys=True),
               **run["timing"]}
        writer.writerow({key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value for key, value in row.items()})
    return out.getvalue()
