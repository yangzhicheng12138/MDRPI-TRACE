"""Persist each learner turn before generation, using LangGraph SQLite checkpoints."""
from __future__ import annotations

import json
import hashlib
from copy import deepcopy
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .backend import TurnLog, render_references, without_reference_context
from .answer_display import performance_answer, performance_fields, saved_answer
from .documents import store_file, select_sources
from .teaching import (CASES, WELCOME, WELCOME_EN, invitation, practice_case, help_intent,
                       student_answers, normalize_review, assessment_text, repeated_focus)
from .language import select_language, localized
from .vendor.oatutor_hints import reveal_next
from .hints import prepare_hints
from .question_bank import reference_answer_text, question_rubric, bank_hint_plan, bank_origin, assessment_item
from .rubric import ground_review, objective_assessment, objective_question_type, objective_invitation, validate_objective_rule
from .learner_evidence import ground_formative_review, annotate_review, hint_events, quote_catalog, resolve_evidence
from .learning_store import LearningStore
from .chat_tools import (DEFAULT_TIMEZONE, validate_timezone, get_current_time,
                         clock_answer, simple_clock_question, normalize_plan, allowed_tools, ground_weather_location,
                         require_current_news_search, ground_search_freshness)
from .tasks import CURRENT_TURN, TurnControl, TurnCancelled, check_cancelled
from .teaching_access import reference_access, all_reference_steps_accessible, reference_locked_text, has_verified_answer
from .learning import ACTION_LABELS, exercise_record, records, learning_progress, progress_text, choose_exercise, exercise_intro


def now():
    return datetime.now(timezone.utc).isoformat()


def message(role, text, kind):
    return {"role": role, "content": text, "kind": kind, "at": now()}


class Session(TypedDict, total=False):
    session_id: str
    mode: str
    policy: str
    web_mode: str
    replay_run_id: str
    created_at: str
    updated_at: str
    stage: str
    messages: list
    case: str
    attempts: list
    reviews: list
    sources: list
    source_events: list
    sources_loaded: bool
    summary: str
    stop_reason: str
    performance_result: dict
    performance_status: dict
    error: str
    request_text: str
    request_action: str
    pending: bool
    pending_summary: bool
    turn_id: str
    response: str
    chat_memory: dict
    context_event: dict
    attachments: list
    case_origin: str
    operation: str
    operation_text: str
    timezone: str
    chat_plan: dict
    tool_results: dict
    artifacts: list
    generated_files: list
    memory_profile: str
    language: str
    response_language: str
    help_cache: dict
    hint_status: list
    last_feedback: str
    bank_item: dict
    cancelled: bool
    partial_response: str
    interruptions: list
    practice_history: list
    exercise_id: str
    practice_root_id: str
    exercise_started_at: str
    review_of: str
    practice_ended: bool
    learning_profile: str
    exercise_learning_profile: str
    learning_generation: int
    learning_archive: dict
    hint_events: list
    reference_views: list
    reference_cache: dict
    recheck_for_answer: int
    teacher_explanation: str
    learning_self_rating: dict
    branch_origin: dict
    request_artifact_id: str
    revision_target: dict
    clinical_flow: dict
    case_spec: dict
    case_intake: dict
    expected_step: int
    learning_owner_id: str
    input_route: dict
    input_route_history: list
    route_pending: bool
    answer_events: list
    teaching_dialogue_events: list
    mixed_help_pending: bool
    mixed_grade_done: bool
    route_message_id: str


class ModeService:
    def __init__(self, root, backend, max_attempts=4, fixed_reference_policy=None):
        self.root, self.backend, self.max_attempts = Path(root), backend, max_attempts
        if fixed_reference_policy not in {None,"none","local","web","auto"}:
            raise ValueError("未知固定资料范围。")
        self.fixed_reference_policy = fixed_reference_policy
        self.root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.root / "sessions.sqlite", check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.saver = SqliteSaver(self.connection)
        self._guard, self._locks, self._context = threading.Lock(), {}, threading.local()
        self._active = {}
        builder = StateGraph(Session)
        builder.add_node("accept", self._accept)
        builder.add_node("route_input", self._route_input)
        builder.add_node("respond", self._respond)
        builder.add_node("teaching_response", self._respond)
        builder.add_node("support_after_answer", self._support_after_answer)
        builder.add_node("summarize", self._summarize)
        builder.add_edge(START, "accept")
        builder.add_edge("accept", "route_input")
        builder.add_conditional_edges("route_input", lambda s: "teaching_response"
            if s.get("mode") == "teaching" and s.get("operation") in {
                "teaching_dialogue", "explain", "hint_detail", "clarify", "clarify_input", "hint", "stuck", "feedback", "reference"}
            else "respond")
        builder.add_edge("respond", "support_after_answer")
        builder.add_edge("teaching_response", "support_after_answer")
        builder.add_conditional_edges("support_after_answer", lambda s: "summarize" if s.get("pending_summary") and not s.get("error") else END)
        builder.add_edge("summarize", END)
        self.graph = builder.compile(checkpointer=self.saver)

    def _config(self, session_id):
        return {"configurable": {"thread_id": session_id}}

    def _lock(self, session_id):
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def branch(self, session_id, message_at, text):
        lock = self._lock(session_id)
        if not lock.acquire(blocking=False):
            raise ValueError("本轮仍在处理，请在结束或停止后修改消息。")
        try:
            from .branches import fork_chat
            return fork_chat(self, session_id, message_at, text)
        finally:
            lock.release()

    def create(self, mode="chat", policy=None, web_mode="live", replay_run_id="", timezone=DEFAULT_TIMEZONE, response_language="auto", memory_profile="", learning_profile="", learning_owner_id=""):
        from .preferences import validate_profile
        memory_profile = validate_profile(memory_profile) if mode == "chat" else ""
        learning_profile = validate_profile(learning_profile) if mode == "teaching" else ""
        policy = policy or ("auto" if mode == "chat" else "none")
        if self.fixed_reference_policy and mode in {"chat","teaching"}:
            policy = self.fixed_reference_policy
        validate_timezone(timezone)
        if response_language not in {"auto", "zh", "en"}:
            raise ValueError("语言设置需为 auto、zh 或 en。")
        language = select_language("", preference=response_language)
        if mode not in {"chat", "teaching", "performance"} or policy not in {"none", "local", "web", "auto"}:
            raise ValueError("未知模式或资料选项。")
        if web_mode not in {"off", "live", "replay"}:
            raise ValueError("未知联网模式。")
        if mode == "performance" and web_mode == "replay" and not replay_run_id:
            raise ValueError("回放需填写原始 run_id。")
        sid = uuid.uuid4().hex
        intro = {
            "chat": "可以开始聊天了。日常问题、写作、翻译、代码和 MDRPI 都可以问；需要时我会使用时钟、文件、知识库或网页。可在设置中调整时区与回答偏好。",
            "teaching": WELCOME if language == "zh" else WELCOME_EN,
            "performance": "请提交病例，将运行已固定的原六步回答和反思流程。",
        }[mode]
        state = Session(session_id=sid, mode=mode, policy=policy, web_mode=web_mode,
            replay_run_id=replay_run_id, created_at=now(), updated_at=now(), stage="setup" if mode == "teaching" else "chat",
            messages=[message("assistant", intro, "welcome")], case="", attempts=[], reviews=[], sources=[],
            source_events=[], sources_loaded=False, summary="", stop_reason="", performance_result={},
            error="", pending=False, pending_summary=False, response=intro,
            chat_memory={}, context_event={}, attachments=[], case_origin="", timezone=timezone,
            chat_plan={}, tool_results={}, artifacts=[], generated_files=[], memory_profile=memory_profile, language=language, response_language=response_language,
            help_cache={}, hint_status=[], last_feedback="", learning_profile=learning_profile,
            exercise_learning_profile="", learning_generation=0, learning_archive={},
            hint_events=[], reference_views=[], reference_cache={}, recheck_for_answer=-1, teacher_explanation="", learning_self_rating={},learning_owner_id=learning_owner_id if mode=="teaching" else "",
            input_route={}, input_route_history=[], route_pending=False, answer_events=[], teaching_dialogue_events=[], mixed_help_pending=False,
            mixed_grade_done=False, route_message_id="")
        self.graph.update_state(self._config(sid), state, as_node="summarize")
        return self.get(sid)

    def get(self, session_id):
        state = self.graph.get_state(self._config(session_id)).values
        if not state:
            raise KeyError("未找到会话，请检查编号。")
        result = json.loads(json.dumps(state, ensure_ascii=False))
        from .workspace import list_artifacts
        result["generated_files"] = list_artifacts(self.root, result["session_id"])
        if result.get("performance_result"):
            result.update(performance_fields(result["performance_result"]))
        return result

    def activity(self):
        with self._guard:
            return [{"session_id": sid, "mode": entry["mode"],
                     "phase": entry.get("phase", "正在处理请求"),
                     "teaching_support": deepcopy(entry.get("teaching_support")),
                     "elapsed_seconds": round(time.monotonic() - entry.get("started", time.monotonic()), 3),
                     "stopping": bool(entry["control"] and entry["control"].cancelled.is_set())}
                    for sid, entry in self._active.items()]

    def cancel(self, session_id):
        self.get(session_id)  # Preserve the API's missing-session contract.
        with self._guard:
            entry = self._active.get(session_id)
        if entry is None:
            return {"status": "idle", "session_id": session_id}
        if entry["control"] is None:
            return {"status": "unsupported", "session_id": session_id,
                    "message": "性能测试由独立研究服务执行，本次不修改其取消逻辑。"}
        entry["control"].cancel()
        return {"status": "stopping", "session_id": session_id}

    def history(self, mode="chat", offset=0, limit=10, owner_id=""):
        if mode not in {"chat", "teaching", "performance"} or offset < 0 or not 1 <= limit <= 30:
            raise ValueError("历史列表的模式或分页参数无效。")
        # Read existing heads only. No migration, title-generation call or
        # second conversation store is needed for this local research UI.
        with self.saver.cursor(transaction=False) as cursor:
            ids = [row[0] for row in cursor.execute(
                "SELECT thread_id FROM checkpoints WHERE checkpoint_ns = '' GROUP BY thread_id")]
        active = {row["session_id"] for row in self.activity()}
        rows = []
        for sid in ids:
            state = self.get(sid)
            first_messages = (state.get("practice_history") or [state])[0]["messages"]
            user = next((m for m in first_messages if m["role"] == "user"), None)
            if state["mode"] != mode or user is None or state.get("learning_owner_id") and state["learning_owner_id"] != owner_id:
                continue
            rows.append({"session_id": sid, "mode": mode,
                         "title": " ".join(user["content"].split())[:72],
                         "updated_at": state["updated_at"], "stage": state["stage"],
                         "status": "running" if sid in active else "cancelled" if state.get("cancelled")
                         else "incomplete" if state.get("pending") else "complete"})
        rows.sort(key=lambda r: (r["updated_at"], r["session_id"]), reverse=True)
        return {"items": rows[offset:offset + limit], "total": len(rows), "offset": offset, "limit": limit}

    def learning_progress(self, session_id):
        state = self.get(session_id)
        if state["mode"] != "teaching":
            raise ValueError("学习进度仅用于教学模式。")
        return learning_progress(state)

    def _learning_owner(self, state):
        if state["mode"] != "teaching" or not state.get("learning_profile"):
            raise ValueError("请先在教学设置中填写学习档案名称。")
        return state["learning_profile"]

    def _learning_store(self, state):
        return LearningStore(self.root, owner_id=state.get("learning_owner_id", ""))

    def learning_profile_progress(self, session_id):
        state = self.get(session_id)
        return self._learning_store(state).progress(self._learning_owner(state))

    def learning_profile_export(self, session_id):
        state = self.get(session_id)
        return self._learning_store(state).export(self._learning_owner(state))

    def learning_recommendations(self, session_id, related=False):
        from .recommendations import recommend
        from .teaching import practice_catalog
        state = self.get(session_id)
        archive = self._learning_store(state).raw_records(state["learning_profile"]) if state.get("learning_profile") else records(state)
        catalog = [row for row in practice_catalog(module="specialty") if row["kind"] in {"standardized_case","fictional_case"}]
        if related:
            from .learning_analytics import case_metadata
            current = case_metadata(exercise_record(state))
            devices = set(current["device_types"])
            people = set(current["populations"])
            catalog = [row for row in catalog if (not devices or devices & set(row["device_types"])) and (not people or people & set(row["populations"]))]
        return recommend(archive, catalog, current_id=(state.get("bank_item") or {}).get("id", ""))

    def clear_learning_profile(self, session_id):
        with self._lock(session_id):
            state = self.get(session_id)
            owner = self._learning_owner(state)
            if state.get("pending"):
                raise ValueError("请先停止或完成当前回合，再清空学习档案。")
            # The current exercise keeps its old generation. Clearing must
            # prevent it or an in-flight recheck from silently reimporting it.
            result = self._learning_store(state).clear(owner)
            if state.get("exercise_learning_profile") == owner:
                marker = {**state.get("learning_archive", {}), "saved": False, "reason": "profile_was_cleared"}
                self.graph.update_state(self._config(session_id), {"learning_archive": marker, "learning_self_rating": {}}, as_node="summarize")
            return result

    def learning_self_rate(self, session_id, rating):
        with self._lock(session_id):
            state = self.get(session_id)
            owner = self._learning_owner(state)
            if state["stage"] != "complete" or state.get("pending"):
                raise ValueError("请完成本题并总结，再选择复习自评。")
            if state.get("exercise_learning_profile") != owner:
                raise ValueError("本题没有保存到当前档案；完成一次新的练习后可以自评。")
            result = self._learning_store(state).rate(owner, state.get("learning_generation", 0), exercise_record(state)["exercise_id"], rating)
            self.graph.update_state(self._config(session_id), {"learning_self_rating": result}, as_node="summarize")
            return result

    def _archive_learning(self, state):
        profile = state.get("exercise_learning_profile")
        if state["mode"] != "teaching" or not profile or (state["stage"] != "complete" and not state.get("clinical_flow")) or state.get("pending"):
            return state
        record = exercise_record(state)
        fingerprint = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        identity = {"exercise_id": record["exercise_id"], "review_count": len(state["reviews"]),
                    "clinical_status": state.get("clinical_flow", {}).get("status"),
                    "hint_count": len(state.get("hint_events", [])), "record_sha256": fingerprint}
        archive = state.get("learning_archive", {})
        # Only a successful write establishes idempotency. A transient storage
        # failure must remain retryable, and a step transition changes the record.
        if archive.get("saved") and all(archive.get(key) == value for key, value in identity.items()):
            return state
        try:
            store = self._learning_store(state)
            saved = store.save_progress(profile, state.get("learning_generation", 0), state) if state.get("clinical_flow") else store.save_completed(profile, state.get("learning_generation", 0), state)
        except Exception as exc:
            saved = {"saved": False, "reason": "archive_failed", "error": str(exc)}
        self._log(state).write("learning_archive", {**identity, **saved})
        self.graph.update_state(self._config(state["session_id"]), {"learning_archive": {**identity, **saved}}, as_node="summarize")
        return self.get(state["session_id"])

    def retry_learning_archive(self, session_id):
        """Retry persistence of the saved exercise without assessment/model calls."""
        with self._lock(session_id):
            state = self.get(session_id)
            if state["mode"] != "teaching" or not state.get("exercise_learning_profile"):
                raise ValueError("本题没有启用学习档案。")
            if state.get("pending"):
                raise ValueError("请先停止或完成当前回合，再重试保存档案。")
            return self._archive_learning(state)

    def learning_action(self, session_id, action, exercise_id=""):
        """A saved-state transition; no model call and no new assessment."""
        if action not in {"next", "similar", "review", "retry_exercise", "finish", "resume", "due"}:
            raise ValueError("未知学习操作。")
        with self._lock(session_id):
            state = self.get(session_id)
            if state["mode"] != "teaching":
                raise ValueError("连续练习仅用于教学模式。")
            language = state.get("language", "zh")
            starting_due = action == "due" and state["stage"] in {"setup", "case", "topic"}
            if (state["stage"] != "complete" and not starting_due) or state.get("pending"):
                raise ValueError(localized(language, "请先完成当前题目并总结；未完成的输入和评价会保留。",
                                           "Finish and summarize the current exercise first; your saved input and assessment are preserved."))
            if action in {"finish", "resume"}:
                ended = action == "finish"
                if state.get("practice_ended", False) == ended:
                    return state
                heading = (localized(language, "本次学习已结束。可随时从历史会话恢复并继续练习。",
                                     "This learning session has ended. You can restore it from History and continue at any time.") if ended else
                           localized(language, "已恢复学习。请选择下一题、同类练习或复习遗漏。",
                                     "Learning resumed. Choose the next question, a related exercise, or review pending points."))
                text = heading + "\n\n" + progress_text(learning_progress(state))
                updates = self._reply(state, text, "learning", practice_ended=ended)
            else:
                if action == "due":
                    previous = self._learning_store(state).due_record(self._learning_owner(state))
                    item = previous.get("bank_item") or {}
                    reason = localized(language, "这道题已到复习时间。本轮使用保存的原题版本重新作答，之前的作答不会计入本轮。",
                                       "This question is due. Answer the saved question version again; earlier answers do not count in this round.")
                else:
                    suggested = self.learning_recommendations(session_id,related=action=="similar") if action in {"next","similar"} and state.get("clinical_flow") else None
                    if suggested and suggested["items"]:
                        suggestion = suggested["items"][0]
                        item, previous, reason = CASES[suggestion["case_id"]], None, suggestion["reason"]
                    elif suggested is not None:
                        raise ValueError(suggested.get("no_match_reason") or "目前没有合适的新案例，可复习遗漏或在学习记录中重练。")
                    else:
                        item, previous, reason = choose_exercise(state, action, exercise_id)
                archive = state.get("practice_history", []) + ([exercise_record(state)] if state.get("case") else [])
                new_id = uuid.uuid4().hex
                case = (item["case_zh"] if language == "zh" else item["case"]) if item else previous["case"]
                text = reason + "\n\n" + exercise_intro(item, case, language)
                # Only the current question goes into grading/summary context.
                # Full previous dialogue remains available for display/export.
                updates = {"practice_history": archive, "exercise_id": new_id,
                    "practice_root_id": previous["practice_root_id"] if previous else new_id,
                    "exercise_started_at": now(), "review_of": previous["exercise_id"] if previous else "",
                    "practice_ended": False, "case": case, "bank_item": deepcopy(item),
                    "case_origin": bank_origin(item) if item else previous.get("case_origin", "provided_case"),
                    "stage": "answer", "attempts": [], "reviews": [], "sources": [], "source_events": [],
                    "sources_loaded": False, "summary": "", "stop_reason": "", "last_feedback": "",
                    "help_cache": {}, "hint_status": [], "error": "", "pending": False, "pending_summary": False,
                    "hint_events": [], "reference_views": [], "reference_cache": {}, "recheck_for_answer": -1, "teacher_explanation": "",
                    "input_route": {}, "input_route_history": [], "route_pending": False,
                    "answer_events": [], "teaching_dialogue_events": [], "mixed_help_pending": False,
                    "mixed_grade_done": False, "route_message_id": "",
                    "exercise_learning_profile": "", "learning_generation": 0, "learning_archive": {},
                    "learning_self_rating": {},
                    "cancelled": False, "partial_response": "", "interruptions": [],
                    "request_text": "", "request_action": "", "operation": "", "operation_text": "",
                    "messages": [message("user", localized(language, *ACTION_LABELS[action]), "learning"), message("assistant", text, "case")],
                    "response": text, "updated_at": now()}
                if hasattr(self.backend,"assess_step") and (item.get("kind") in {"standardized_case", "fictional_case"} or previous and previous.get("clinical_flow")):
                    from .clinical_flow import start_flow,current_question
                    flow = start_flow(case,item,language)
                    updates.update(clinical_flow=flow,case_spec=deepcopy(previous.get("case_spec",{})) if previous else {},
                        exercise_learning_profile=state.get("learning_profile",""),learning_generation=self._learning_store(state).generation(state["learning_profile"]) if state.get("learning_profile") else 0)
                    updates["messages"][-1]["content"] = text.split(invitation(case,language))[0].rstrip()+"\n\n"+current_question(flow,language)
                    updates["response"] = updates["messages"][-1]["content"]
                else:
                    updates["clinical_flow"] = {}
                if action == "due":
                    updates.update(sources=deepcopy(previous.get("sources", [])),
                                   source_events=deepcopy(previous.get("source_events", [])), sources_loaded=True)
            updates["turn_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
            log = self._log({**state, **updates})
            log.write("learning_transition", {"action": action, "from_exercise": exercise_record(state)["exercise_id"],
                "to_exercise": updates.get("exercise_id", exercise_record(state)["exercise_id"]),
                "review_of": updates.get("review_of", ""), "bank_id": (updates.get("bank_item") or {}).get("id", "")})
            self.graph.update_state(self._config(session_id), updates, as_node="summarize")
            return self.get(session_id)

    def display_messages(self, session_id):
        """Present old replies using each turn's evidence; do not rewrite history."""
        state = self.get(session_id)
        from .teaching_presentation import annotate_messages
        messages = (annotate_messages({**state, "practice_history": []}, state["messages"]) if state["mode"] == "teaching"
                    else deepcopy(state["messages"]))
        segments = records(state) if state["mode"] == "teaching" and state.get("case") else [state]
        if state.get("practice_history"):
            messages = []
            for index, record in enumerate(segments, 1):
                title = localized(state.get("language", "zh"), f"### 第 {index} 轮练习", f"### Exercise round {index}")
                messages.append({"role": "assistant", "content": title, "kind": "learning_section", "at": record.get("updated_at", "")})
                messages.extend(annotate_messages({**record, "mode": state["mode"], "practice_history": []}, record["messages"])
                                if state["mode"] == "teaching" else deepcopy(record["messages"]))
        interruptions = {row["turn_id"]: row for segment in segments for row in segment.get("interruptions", [])}
        for m in messages:
            interrupted = interruptions.get(m.get("turn_id")) if m.get("kind") == "cancelled" else None
            if interrupted and interrupted.get("partial_text"):
                m["content"] += "\n\n" + interrupted["partial_text"]
        targets = {m["at"]: m for m in messages if m["role"] == "assistant" and m["kind"] in {"chat", "performance"}}
        if not targets:
            return messages
        remaining = set(targets)
        for checkpoint in self.graph.get_state_history(self._config(session_id)):
            previous = checkpoint.values
            rows = previous.get("messages", [])
            if not rows or rows[-1].get("at") not in remaining or previous.get("pending"):
                continue
            reply = targets[rows[-1]["at"]]
            if reply["kind"] == "performance" and previous.get("performance_result"):
                reply["content"] = performance_answer(previous["performance_result"])
            elif reply["kind"] == "chat":
                reply["content"] = saved_answer(reply["content"], previous.get("sources", []))
            remaining.remove(reply["at"])
            if not remaining:
                break
        return messages

    def upload(self, session_id, name, data):
        with self._lock(session_id):
            state = self.get(session_id)
            if state["mode"] == "performance":
                raise ValueError("性能测试仍使用原病例文本入口；文件请在问答或教学模式上传。")
            if state.get("pending"):
                raise ValueError("请先完成或重试当前回合，再上传文件。")
            files = state.get("attachments", [])
            if sum(f.get("active", True) for f in files) >= 10:
                raise ValueError("一个会话最多同时使用 10 个文件，请先移除不用的文件。")
            if state["mode"] == "teaching":
                from .documents import teaching_upload_texts
                from .case_intake import inspect_import
                texts = teaching_upload_texts(name, data)
                if any(inspect_import(text)["redaction_count"] for text in texts):
                    raise ValueError("教学文件含可能的身份信息，尚未保存。请先移除身份信息，或粘贴病例到脱敏导入表单。")
            file = store_file(self.root / "files" / state["session_id"], name, data)
            self.graph.update_state(self._config(session_id), {"attachments": files + [file],
                "sources_loaded": False, "updated_at": now()}, as_node="summarize")
            return file

    def inspect_case(self, session_id, text, case_spec=None):
        from .case_intake import inspect_import
        import hashlib
        with self._lock(session_id):
            state = self.get(session_id)
            if state["mode"] != "teaching" or state["stage"] not in {"setup","case","topic"} or state.get("pending"):
                raise ValueError("请在开始新病例时进行脱敏检查。")
            result = inspect_import(text,case_spec)
            metadata = {key:value for key,value in result.items() if key != "sanitized_text"}
            metadata["sanitized_sha256"] = hashlib.sha256(result["sanitized_text"].encode()).hexdigest()
            self.graph.update_state(self._config(session_id),{"case_intake":metadata},as_node="summarize")
            return result

    def remove_file(self, session_id, file_id):
        with self._lock(session_id):
            state = self.get(session_id)
            if state.get("pending"):
                raise ValueError("请先完成或重试当前回合。")
            files = state.get("attachments", [])
            found = next((f for f in files if f["file_id"] == file_id), None)
            if found is None:
                raise KeyError("未找到本会话文件。")
            found["active"] = False
            self.graph.update_state(self._config(session_id), {"attachments": files,
                "sources_loaded": False, "updated_at": now()}, as_node="summarize")
            return self.get(session_id)

    def _file_sources(self, state, query):
        if any(f.get("active", True) for f in state.get("attachments", [])):
            self._emit("status", "正在读取上传文件的相关片段…")
        return select_sources(self.root / "files" / state["session_id"], state.get("attachments", []), query)

    def configure(self, session_id, policy, web_mode="live", replay_run_id="", timezone=None, response_language=None, memory_profile=None, learning_profile=None):
        if policy not in {"none", "local", "web", "auto"} or web_mode not in {"off", "live", "replay"}:
            raise ValueError("未知资料/联网选项。")
        with self._lock(session_id):
            state = self.get(session_id)
            if self.fixed_reference_policy and state["mode"] in {"chat","teaching"}:
                policy = self.fixed_reference_policy
            if state.get("pending"):
                raise ValueError("请先完成或重试当前回合，再修改设置。")
            if state["mode"] == "performance" and web_mode == "replay" and not replay_run_id:
                raise ValueError("回放需填写原始 run_id。")
            timezone = validate_timezone(timezone or state.get("timezone", DEFAULT_TIMEZONE))
            response_language = response_language or state.get("response_language", "auto")
            if response_language not in {"auto", "zh", "en"}:
                raise ValueError("语言设置需为 auto、zh 或 en。")
            language = select_language("", state.get("language", "zh"), response_language)
            from .preferences import validate_profile
            profile = state.get("memory_profile", "") if memory_profile is None else validate_profile(memory_profile)
            if state["mode"] != "chat": profile = ""
            learner = state.get("learning_profile", "") if learning_profile is None else validate_profile(learning_profile)
            if state["mode"] != "teaching": learner = ""
            if learner != state.get("learning_profile", "") and state["stage"] not in {"setup", "case", "topic", "complete"} and student_answers(state):
                raise ValueError("当前练习已有作答，请先完成本题，再切换学习档案。")
            self.graph.update_state(self._config(session_id), {"policy": policy, "web_mode": web_mode,
                "replay_run_id": replay_run_id, "sources_loaded": False, "timezone": timezone,
                "response_language": response_language, "language": language, "memory_profile": profile,
                "learning_profile": learner}, as_node="summarize")
            return self.get(session_id)

    def turn(self, session_id, text="", action="send", emit=None, artifact_id="", expected_step=None, case_spec=None, privacy_confirmed=False):
        from .telemetry import TurnTelemetry
        telemetry = TurnTelemetry()
        with self._lock(session_id):
            state = self.get(session_id)
            intake = None
            if artifact_id:
                if state["mode"] != "chat" or action != "send":
                    raise ValueError("生成文件的继续修改仅用于普通问答的新消息。")
                from .artifacts import selected_artifact
                selected_artifact(self.root, session_id, artifact_id)
            if action not in {"send", "summary", "retry", "choose_case", "choose_topic", "library_case", "hint", "feedback", "recheck", "next_step", "clarify", "stuck", "reference", "hint_detail", "explain", "submit_answer"}:
                raise ValueError("未知操作。")
            guided = bool(state.get("clinical_flow"))
            subjective = (state["mode"] == "teaching" and bool(state.get("case"))
                          and state["stage"] not in {"setup", "case", "topic"}
                          and not objective_question_type(assessment_item(state)))
            if state["mode"] == "teaching" and hasattr(self.backend, "assess_step"):
                if state["stage"] in {"setup", "case", "topic"} and action == "send":
                    from .case_intake import inspect_import
                    intake = inspect_import(text, case_spec)
                    if intake.get("scope_conflicts"):
                        raise ValueError("所选人群或器械与案例中的明确描述不一致，请在导入表单中核对后再开始训练。")
                    if state["stage"] == "case" and not privacy_confirmed:
                        raise ValueError("请确认病例已脱敏，再开始训练。")
                    if intake["redaction_count"] and not privacy_confirmed:
                        raise ValueError("检测到可能的身份信息，请脱敏并确认后导入。")
                    text = intake["sanitized_text"]
                    if state.get("case_intake",{}).get("sanitized_sha256"):
                        import hashlib
                        if state["case_intake"]["sanitized_sha256"] == hashlib.sha256(text.encode()).hexdigest():
                            previous_intake = state["case_intake"]
                            intake = {**previous_intake, **intake,
                                      "redactions": previous_intake.get("redactions", intake["redactions"]),
                                      "redaction_count": previous_intake.get("redaction_count", intake["redaction_count"])}
                    if privacy_confirmed:
                        intake["review_status"] = "user_confirmed_deidentified_pending_clinical_review"
                if guided and action == "send" and not subjective:
                    from .clinical_flow import classify_intent
                    from .subjective_help import learning_help_intent
                    support = learning_help_intent(text)
                    intent = classify_intent(text)
                    action = support or {"unknown":"stuck", "hint":"hint", "end":"summary", "advance":"next_step", "clarify":"clarify", "ambiguous":"clarify", "recheck":"recheck"}.get(intent, "send")
                if action == "next_step" and not guided:
                    raise ValueError("当前练习没有待推进的六步流程。")
                if guided and action == "next_step":
                    from .clinical_flow import advance_step
                    advance_step(state["clinical_flow"], expected_step)
                if guided and action in {"send", "submit_answer"} and not (subjective and action == "send"):
                    current = state["clinical_flow"]["steps"][state["clinical_flow"]["current_step"]]
                    if current["status"] != "active":
                        raise ValueError("当前步骤已通过，请点击进入下一步。")
                    if sum(len(value) for value in current["answers"])+len(text)>16000:
                        raise ValueError("当前步骤作答总量超过16000字，请缩短本次补充；此前作答已保留。")
                if guided and action == "recheck":
                    if state["clinical_flow"]["status"] != "in_progress":
                        raise ValueError("训练已结束，可回看已保存评价；需要重新作答时请选择重练本题。")
                    current = state["clinical_flow"]["steps"][state["clinical_flow"]["current_step"]]
                    if not current["answers"] or not current["reviews"]:
                        raise ValueError("请先提交当前步骤的回答并取得评价，再重新核对。")
            if state["mode"] == "teaching" and not guided and not subjective and state["stage"] not in {"setup", "case", "topic"} and action == "send":
                action = help_intent(text)
                if action == "stuck" and objective_question_type(assessment_item(state)):
                    action = "hint"
            if action in {"hint", "feedback", "recheck", "stuck", "reference", "clarify", "hint_detail", "explain", "submit_answer"} and state["mode"] != "teaching":
                raise ValueError("提示与作答评价仅用于教学模式。")
            if action in {"stuck", "reference", "hint_detail", "explain", "clarify", "submit_answer"} and (not state.get("case") or state["stage"] in {"setup", "case", "topic"}):
                raise ValueError("请先选择一道练习题。")
            if action == "submit_answer" and not subjective:
                raise ValueError("确认作答仅用于当前主观题。")
            if action in {"choose_case", "choose_topic", "library_case"} and (
                    state["mode"] != "teaching" or state["stage"] not in {"setup", "case", "topic"}):
                raise ValueError("开始新案例请新建教学会话。")
            if action == "library_case" and text not in CASES:
                raise ValueError("未找到练习案例。")
            if action in {"send", "submit_answer"} and not text.strip():
                raise ValueError("请输入内容。")
            if len(text) > (1000 if state["mode"] == "performance" else 12000):
                raise ValueError("输入过长：性能测试最多1000字符，问答/教学每次最多12000字符。")
            if state["stage"] == "complete" and action not in {"hint", "feedback", "recheck", "stuck", "reference", "hint_detail", "explain", "clarify"} and not (subjective and action == "send"):
                raise ValueError(localized(state.get("language", "zh"),
                    "本题已总结。请选择下一题、同类练习或复习遗漏；自带新案例可新建会话。",
                    "This exercise is summarized. Choose the next question, a related exercise, or review pending points; use a new session for a new custom case."))
            if action == "summary" and (state["mode"] != "teaching" or not state.get("case")
                    or state["stage"] in {"setup", "case", "topic"}
                    or objective_question_type(assessment_item(state)) and not student_answers(state)):
                raise ValueError("请先选择练习题；客观题需要先提交回答，再请求总结。")
            if action == "recheck":
                if not state["reviews"] or not student_answers(state):
                    raise ValueError("请先提交一次学生回答并取得评价，再请求重新核对。")
                if state.get("recheck_for_answer", -1) == len(student_answers(state)):
                    raise ValueError("已有回答已经重新核对过；补充新的作答后可以再次核对。")
            if action == "retry" and not state.get("pending"):
                raise ValueError("本会话没有待重试的请求。")
            if state.get("pending") and action != "retry" and not state.get("cancelled"):
                raise ValueError("上一回合尚未完成，请先重试；输入已保存。")
            if state.get("pending_summary") and action == "send" and not state.get("cancelled"):
                raise ValueError("总结尚未成功，请点击重试以完成总结。")
            self._context.emit = emit or (lambda *_: None)
            self._context.telemetry = telemetry
            telemetry.event("status", "processing")
            result = None
            control = None if state["mode"] == "performance" else TurnControl()
            with self._guard:
                self._active[session_id] = {"mode": state["mode"], "control": control,
                    "started": time.monotonic(), "phase": "正在处理请求", "thread": threading.get_ident()}
            context = CURRENT_TURN.set(control)
            try:
                try:
                    request = {"request_text": text.strip(), "request_action": action, "request_artifact_id": artifact_id}
                    if expected_step is not None: request["expected_step"] = expected_step
                    elif guided: request["expected_step"] = state["clinical_flow"]["current_step"]
                    if case_spec is not None:
                        from .case_intake import validate_custom_spec
                        request["case_spec"] = (intake.get("case_spec", {})
                            if intake is not None and state["mode"] == "teaching" and state["stage"] in {"setup", "case", "topic"} and action == "send"
                            else validate_custom_spec(case_spec))
                    elif intake is not None and state["mode"] == "teaching" and state["stage"] in {"setup", "case", "topic"} and action == "send":
                        request["case_spec"] = intake.get("case_spec", {})
                    if state["mode"] == "teaching" and hasattr(self.backend,"assess_step") and state["stage"] in {"setup","case","topic"} and action == "send":
                        request["case_intake"] = {k:v for k,v in intake.items() if k != "sanitized_text"}
                    self.graph.invoke(request, self._config(session_id))
                except TurnCancelled:
                    current = self.get(session_id)
                    language = current.get("language", "zh") if current["mode"] == "teaching" else select_language(
                        next((m["content"] for m in reversed(current["messages"]) if m["role"] == "user"), ""))
                    note = localized(language,
                        "本轮生成已停止。以下片段（如有）尚未完成；可重试本回合或继续输入。",
                        "Generation stopped. Any excerpt below is incomplete; retry this turn or send another message.")
                    record = {"turn_id": current["turn_id"], "partial_text": control.partial_text, "at": now()}
                    marker = {**message("assistant", note, "cancelled"), "turn_id": current["turn_id"]}
                    self._log(current).write("cancelled", record)
                    self.graph.update_state(self._config(session_id), {
                        "cancelled": True, "partial_response": control.partial_text, "error": "", "pending": True,
                        "response": note, "updated_at": now(), "messages": current["messages"] + [marker],
                        "interruptions": current.get("interruptions", []) + [record]}, as_node="summarize")
                result = self._archive_learning(self.get(session_id))
                self._context.emit("complete", result)
                return result
            finally:
                try:
                    current = result if result is not None else self.get(session_id)
                    if current.get("turn_id") and current["turn_id"] != state.get("turn_id"):
                        if result is None:
                            current = {**current, "error": current.get("error") or "turn_failed"}
                        self._log(current).write("latency", telemetry.finish(current))
                finally:
                    CURRENT_TURN.reset(context)
                    with self._guard:
                        self._active.pop(session_id, None)
                    self._context.emit = None
                    self._context.telemetry = None

    def _emit(self, event, value):
        if event in {"status", "teaching_support"}:
            with self._guard:
                for entry in self._active.values():
                    if entry.get("thread") == threading.get_ident():
                        if event == "status":
                            entry["phase"] = str(value)
                        elif isinstance(value, dict):
                            entry["teaching_support"] = {key: deepcopy(value[key]) for key in
                                ("placement", "phase", "intent", "hide_request", "reset", "step_id") if key in value}
        telemetry = getattr(self._context, "telemetry", None)
        if telemetry is not None:
            telemetry.event(event, value)
        control = CURRENT_TURN.get()
        if control:
            control.observe(event, value)
        callback = getattr(self._context, "emit", None)
        if callback:
            callback(event, value)

    def _log(self, state):
        return TurnLog(self.root / "runs" / state["session_id"] / state["turn_id"])

    def _subjective_route_enabled(self, state):
        return (state.get("mode") == "teaching" and bool(state.get("case"))
                and state.get("stage") not in {"setup", "case", "topic"}
                and not objective_question_type(assessment_item(state)))

    def _accept(self, state):
        update = {"error": "", "response": "", "pending": True, "updated_at": now(),
                  "cancelled": False, "partial_response": "",
                  "turn_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]}
        # Apply the public source scope to future requests without rewriting
        # earlier messages or their historical evidence checkpoints.
        if self.fixed_reference_policy and state["mode"] in {"chat","teaching"} and state.get("policy") != self.fixed_reference_policy:
            update.update(policy=self.fixed_reference_policy,sources_loaded=False)
        if state.get("cancelled") and state["request_action"] != "retry":
            update["pending_summary"] = False
        if state["request_action"] != "retry":
            update.update(operation=state["request_action"], operation_text=state["request_text"])
            deferred = state["request_action"] == "send" and self._subjective_route_enabled(state)
            update.update(input_route={}, route_pending=deferred, mixed_help_pending=False, mixed_grade_done=False,
                          route_message_id=update["turn_id"] if deferred else "")
            if state["request_action"] == "submit_answer":
                update["operation"] = "send"
            if state["mode"] == "chat":
                from .artifacts import selected_artifact
                target = state.get("request_artifact_id", "")
                update["revision_target"] = selected_artifact(self.root, state["session_id"], target) if target else {}
            if state["mode"] == "teaching":
                text = state["request_text"] if state["request_action"] in {"send", "submit_answer", "hint", "feedback", "summary", "recheck", "stuck", "reference", "clarify", "hint_detail", "explain"} else ""
                previous = state.get("language") or select_language(next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""))
                update["language"] = select_language(text, previous, state.get("response_language", "auto"))
        if state["request_action"] in {"send", "submit_answer"}:
            text = state["request_text"]
            kind = ("setup" if state["stage"] in {"setup", "topic"} else "case" if state["stage"] == "case" else "student") if state["mode"] == "teaching" else "user"
            if update.get("route_pending"):
                kind = "help"
            entry = {**message("user", text, kind), "turn_id": update["turn_id"]}
            if kind == "student":
                entry["answer_text"] = text
            update["messages"] = state["messages"] + [entry]
            if kind == "student":
                update["attempts"] = state["attempts"] + [text]
                update["answer_events"] = state.get("answer_events", []) + [{"answer_index": len(state["attempts"]) + 1,
                    "text": text, "turn_id": update["turn_id"], "at": entry["at"],
                    "origin": "explicit_confirmation" if state["request_action"] == "submit_answer" else "explicit_submission"}]
                if state.get("clinical_flow"):
                    from .clinical_flow import append_answer
                    update["clinical_flow"] = append_answer(state["clinical_flow"], text)
                if not student_answers(state):
                    profile = state.get("learning_profile", "")
                    update.update(exercise_learning_profile=profile,
                                  learning_generation=self._learning_store(state).generation(profile) if profile else 0)
        elif state["request_action"] in {"hint", "feedback", "summary", "recheck", "stuck", "reference", "clarify", "hint_detail", "explain"} and state["request_text"]:
            update["messages"] = state["messages"] + [{**message("user", state["request_text"], "help"), "turn_id": update["turn_id"]}]
        elif state["request_action"] in {"choose_case", "choose_topic", "library_case"}:
            label = {"choose_case": "我提供案例", "choose_topic": "按主题出题", "library_case": "选择练习：" + state["request_text"]}[state["request_action"]]
            update["messages"] = state["messages"] + [message("user", label, "setup")]
        log = self._log({**state, **update})
        log.write("input", {"mode": state["mode"], "stage": state["stage"], "action": state["request_action"],
                            "text": state["request_text"], "session_id": state["session_id"]})
        return update

    def _route_input(self, state):
        """Persist raw input first; only an accepted route becomes learner evidence."""
        if not state.get("route_pending"):
            return {}
        from .teaching_router import TEACHING_ROUTER_VERSION, fast_teaching_route, validate_teaching_route
        text, log = state.get("operation_text", state["request_text"]), self._log(state)
        origin, reason = "local_control", ""
        try:
            check_cancelled()
            decision = fast_teaching_route(text)
            if decision is None:
                origin = "semantic_model"
                self._emit("status", "正在理解你是在作答、求助还是追问…")
                decision = self.backend.teaching_route(state, text, log)
            decision = validate_teaching_route(decision, text)
            if decision["confidence"] != "clear":
                reason = "uncertain_intent"
        except Exception as exc:
            log.write("teaching_input_route_unavailable", {"type": type(exc).__name__, "message": str(exc)})
            decision = {"intent": "clarify", "confidence": "uncertain", "answer_text": "", "help_intent": None}
            reason = "router_unavailable"
        intent = decision["intent"]
        answer = decision.get("answer_text", "")
        flow = state.get("clinical_flow") or {}
        step = flow["steps"][flow["current_step"]] if flow else None
        operation = ("send" if intent in {"answer", "mixed"} else "teaching_dialogue"
                     if intent in {"hint_detail", "explain", "social", "clarify"} else intent)
        if not reason and intent in {"answer", "mixed"}:
            if state["stage"] == "complete":
                reason = "exercise_completed"
            elif step and step["status"] != "active":
                reason = "step_already_passed"
            elif step and state.get("expected_step") != flow["current_step"]:
                reason = "step_changed"
            elif step and sum(len(value) for value in step["answers"]) + len(answer) > 16000:
                reason = "step_answer_limit"
        if not reason and intent == "next_step":
            try:
                if not flow or flow.get("status") != "in_progress":
                    raise ValueError("no_active_flow")
                from .clinical_flow import advance_step
                advance_step(flow, state.get("expected_step"))
            except (ValueError, KeyError, IndexError):
                reason = "cannot_advance_step"
        if not reason and intent == "recheck" and (not student_answers(state) or not state.get("reviews")
                or state.get("recheck_for_answer", -1) == len(student_answers(state))
                or state["stage"] == "complete" or step and (not step["answers"] or not step["reviews"])):
            reason = "cannot_recheck"
        if not reason and intent == "summary" and state["stage"] == "complete":
            reason = "cannot_summarize"
        if reason:
            operation, answer = "clarify_input", ""
            decision = {**decision, "intent": "clarify", "confidence": "uncertain", "answer_text": "", "help_intent": None}
        route = {**decision, "origin": origin, "reason": reason, "raw_text": text,
                 "raw_sha256": hashlib.sha256(text.encode()).hexdigest(), "turn_id": state["turn_id"],
                 "version": TEACHING_ROUTER_VERSION, "step_id": step["id"] if step else "", "at": now()}
        updates = {"input_route": route, "input_route_history": state.get("input_route_history", []) + [route],
                   "route_pending": False, "operation": operation, "operation_text": text,
                   "mixed_help_pending": decision["intent"] == "mixed"}
        if answer:
            entries = deepcopy(state["messages"])
            for row in reversed(entries):
                if row.get("role") == "user" and row.get("turn_id") == state.get("route_message_id", state["turn_id"]):
                    row.update(kind="student", answer_text=answer)
                    break
            updates.update(messages=entries, attempts=state["attempts"] + [answer],
                answer_events=state.get("answer_events", []) + [{"answer_index": len(state["attempts"]) + 1,
                    "text": answer, "turn_id": state["turn_id"], "at": route["at"], "origin": origin}])
            if flow:
                from .clinical_flow import append_answer
                updates["clinical_flow"] = append_answer(flow, answer)
            if not student_answers(state):
                profile = state.get("learning_profile", "")
                updates.update(exercise_learning_profile=profile,
                               learning_generation=self._learning_store(state).generation(profile) if profile else 0)
        log.write("teaching_input_route", {key: route[key] for key in
                  ("intent", "confidence", "origin", "reason", "raw_sha256", "step_id", "version")})
        placement = None
        if not reason and decision["confidence"] == "clear":
            if intent in {"hint", "stuck", "hint_detail"}:
                placement = "hint"
                if intent == "stuck" and reference_access(state)["allowed"]:
                    placement = "reference"
                elif intent == "hint" and not step and state.get("reviews"):
                    latest = state["reviews"][-1]
                    if latest.get("ready_for_summary") and not latest.get("material_errors") and reference_access(state)["allowed"]:
                        placement = None
            elif intent in {"explain", "clarify"}:
                placement = "explanation"
            elif intent == "reference":
                placement = "reference" if reference_access(state)["allowed"] else "hint"
        self._emit("teaching_support", {"phase": "routed", "placement": placement,
                                       "intent": decision["intent"], "hide_request": bool(placement),
                                       "reset": False, "step_id": step["id"] if step else ""})
        return updates

    def _reply(self, state, text, kind="feedback", support_intent=None, **updates):
        check_cancelled()
        reply = message("assistant", text, kind)
        saved = {**state, **updates}
        if state.get("mode") == "teaching":
            from .teaching_support import support_reply_metadata
            reply.update(support_reply_metadata(state, saved, kind, support_intent))
        if kind in {"feedback", "recheck", "summary"} and state.get("mode") == "teaching":
            reviews = saved.get("reviews") or []
            flow = saved.get("clinical_flow") or {}
            current = flow["steps"][flow["current_step"]] if flow else None
            candidates = (current.get("reviews") or []) if current and kind in {"feedback", "recheck"} else reviews
            if current:
                reply["step_id"] = current["id"]
            if candidates:
                review = candidates[-1]
                matching = [i for i, row in enumerate(reviews) if row == review]
                if matching:
                    reply["review_index"] = matching[-1]
                reply.update(step_id=review.get("step_id") or (current["id"] if current else ""),
                             review_snapshot=deepcopy(review))
        return {"messages": state["messages"] + [reply],
                "response": text, "pending": False, "error": "", "updated_at": now(), **updates}

    def _emit_teaching_support(self, state, placement, intent):
        """Tell the browser where actual helper output belongs, after routing."""
        if state.get("mode") != "teaching" or not state.get("case"):
            return
        if objective_question_type(assessment_item(state)) and placement not in {None, "hint"}:
            return
        flow = state.get("clinical_flow") or {}
        step_id = flow["steps"][flow["current_step"]]["id"] if flow else ""
        current_request = next((row for row in reversed(state.get("messages") or [])
                                if row.get("role") == "user" and row.get("turn_id") == state.get("turn_id")), None)
        hide_request = bool(placement and current_request and current_request.get("kind") == "help"
                            and not current_request.get("answer_text"))
        self._emit("teaching_support", {"placement": placement, "phase": "responding", "intent": intent,
                                       "hide_request": hide_request, "reset": True, "step_id": step_id})

    def _clarify_input(self, state):
        language = state.get("language", "zh")
        reason = state.get("input_route", {}).get("reason", "uncertain_intent")
        notices = {
            "exercise_completed": ("本轮已经总结。你可以继续求助或回看已解锁参考；未解锁时可在当前页获取剩余提示。重新作答请选择重练本题。", "This round is summarized. Ask for help or revisit an unlocked reference; get any remaining hints on this page. Repeat the exercise to answer again."),
            "step_already_passed": ("当前步骤已通过。可以继续问本步的问题，或点击进入下一步。", "This step has passed. Ask a follow-up about this step or choose Next step."),
            "step_changed": ("当前步骤已经变化，请查看本步任务再提交。原输入已保留。", "The current step has changed. Read its task before submitting; your input is retained."),
            "step_answer_limit": ("当前步骤的作答总量已到上限，请缩短本次补充；此前作答和原输入均保留。", "This step's answer length limit has been reached. Shorten this addition; previous answers and this input are retained."),
            "cannot_advance_step": ("当前还不能进入下一步。可以继续作答或获取剩余提示；已解锁时可回看本步参考。", "The next step is not available yet. Continue answering or get the remaining hints; revisit this step's reference once unlocked."),
            "cannot_recheck": ("当前没有可再次核对的新作答。可以继续回答，或告诉我希望解释哪一处。", "There is no new answer to recheck. Continue answering, or tell me what you would like explained."),
            "cannot_summarize": ("本轮已经结束，可以回看总结并继续求助；参考未解锁时先获取剩余提示。", "This round has ended. Review the summary or keep asking for help; get the remaining hints before viewing a locked reference."),
        }
        text = localized(language, *notices.get(reason, (
            "我还没确定你是在提交答案，还是希望我提供帮助。这条消息已保留，暂不计作答。你可以选择“这是我的回答”或“我是在求助”，也可以直接补充说明。",
            "I am not sure whether you are submitting an answer or asking for help. Your message is retained and is not counted as an answer. Choose 'This is my answer' or 'I am asking for help', or add a clarification.")))
        return self._reply(state, text, "clarification", pending_summary=False, mixed_help_pending=False, mixed_grade_done=False)

    def _dialogue_help(self, state, log, intent=None, text=None):
        """Teaching conversation never grades or spends another hint to expand one."""
        from .teaching_router import last_teaching_hint
        intent = intent or state.get("input_route", {}).get("intent") or state.get("operation", "explain")
        text = state.get("operation_text", "") if text is None else text
        if objective_question_type(assessment_item(state)):
            return self._teaching_help(state, log, "feedback" if intent == "social" else "hint")
        language = state.get("language", "zh")
        prepared, initial = state, {}
        if intent == "hint_detail" and not last_teaching_hint(state):
            if state.get("clinical_flow"):
                initial = self._clinical_turn({**state, "operation": "hint"}, log)
            else:
                initial = self._subjective_hint(state, log, "hint")
            prepared = {**state, **initial}
            if not last_teaching_hint(prepared):
                intent = "explain"
        if intent != "social":
            self._emit_teaching_support(prepared, "hint" if intent == "hint_detail" else "explanation", intent)
        self._emit("status", "正在把提示解释得更具体…" if intent == "hint_detail" else "正在回应你的问题…")
        try:
            body = self.backend.teaching_dialogue(prepared, text, intent, log)
            if not isinstance(body, str) or not body.strip():
                raise ValueError("empty_teaching_reply")
        except Exception as exc:
            log.write("teaching_dialogue_unavailable", {"type": type(exc).__name__, "message": str(exc), "intent": intent})
            body = localized(language, "这次补充讲解暂时没有生成，你的作答和已看提示都已保留。可以再问一次，或继续获取剩余提示；看完三条后可查看参考答案。",
                             "The explanation is temporarily unavailable. Your answers and hints are retained. Ask again, or get the remaining hints; the reference unlocks after all three.")
            return {**initial, **self._reply(prepared, body, "clarification", support_intent=intent)}
        body = body.strip()
        updates = {}
        flow = deepcopy(prepared.get("clinical_flow") or {})
        step = flow["steps"][flow["current_step"]] if flow else None
        if intent != "social":
            event = {"text": body, "answer_count": len(student_answers(prepared)), "step_id": step["id"] if step else "",
                     "focus": "all", "origin": "teaching_dialogue", "intent": intent, "language": language, "at": now()}
            updates["teaching_dialogue_events"] = prepared.get("teaching_dialogue_events", []) + [event]
            if step:
                step.setdefault("teaching_dialogue_events", []).append({**event, "answer_count": len(step["answers"])})
                updates["clinical_flow"] = flow
            elif prepared["stage"] == "reflect":
                updates["stage"] = "probe"
        response = self._reply(prepared, body, "social" if intent == "social" else "teaching_dialogue", support_intent=intent, **updates)
        response["messages"][-1]["step_id"] = step["id"] if step else ""
        log.write("teaching_dialogue", {"intent": intent, "step_id": step["id"] if step else "", "graded": False,
                                       "initial_hint_requested": bool(initial), "hint_count": len(hint_events(prepared))})
        return {**initial, **response}

    def _support_after_answer(self, state):
        """A checkpoint separates an accepted assessment from its follow-up help."""
        if not state.get("mixed_help_pending") or not state.get("mixed_grade_done") or state.get("error"):
            return {}
        intent = state.get("input_route", {}).get("help_intent")
        if not intent:
            return {"mixed_help_pending": False, "mixed_grade_done": False}
        log = self._log(state)
        prepared = {**state, "operation": intent, "operation_text": state["input_route"]["raw_text"]}
        try:
            if intent in {"hint_detail", "explain", "clarify", "social"}:
                result = self._dialogue_help(prepared, log, intent)
            elif intent == "reference":
                result = self._reference_help(prepared, log)
            elif state.get("clinical_flow"):
                result = self._clinical_turn(prepared, log)
            else:
                result = self._teaching_help(prepared, log, intent)
        except Exception as exc:
            log.write("mixed_teaching_help_unavailable", {"type": type(exc).__name__, "message": str(exc)})
            body = localized(state.get("language", "zh"), "你的作答评价已保存，补充帮助暂未完成。可以继续询问需要解释的部分。",
                             "Your assessment is saved. The additional help did not finish; ask about the part you would like explained.")
            result = self._reply(state, body, "clarification")
        return {**result, "mixed_help_pending": False, "mixed_grade_done": False}

    def _chat_turn(self, state, log):
        updates = {"sources": [], "source_events": [], "sources_loaded": False, "context_event": {}}
        try:
            from .preferences import read as read_preferences
            from .workspace import list_artifacts, aliases, complete_explicit_file_plan
            from .fast_paths import explicit_plan, keep_requested_documents
            from .chat_tools import require_explicit_external_tools
            state = {**state, "preferences": read_preferences(self.root, state.get("memory_profile", "")),
                     "generated_files": list_artifacts(self.root, state["session_id"])}
            question = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
            zone = state.get("timezone", DEFAULT_TIMEZONE)
            events = []
            if simple_clock_question(question):
                self._emit("status", "正在读取系统时间…")
                plan = {"tools": ["clock"], "query": "", "timezone": zone,
                        "reason": "A current date/time question is answered by the local clock.", "planner": "clock_fast_path"}
            else:
                try:
                    config = getattr(self.backend, "config", None)
                    plan = explicit_plan(state, question) if config and config.get("interactive", "explicit_task_routes", default=True) else None
                    if plan is None:
                        plan = self.backend.plan_chat(state, question, log, self._emit)
                        plan["planner"] = "local_model"
                    plan = keep_requested_documents(plan, question, state)
                except Exception as exc:
                    events.append({"provider": "router", "status": "unavailable", "error": str(exc)})
                    plan = {"tools": [], "query": question, "timezone": zone,
                            "reason": "Tool planning failed; no external retrieval performed.", "planner": "failed"}
            plan = require_explicit_external_tools(plan, question)
            plan = require_current_news_search(plan, question, zone)
            plan = complete_explicit_file_plan(plan, state, question)
            plan = normalize_plan(plan, state["policy"], state.get("attachments", []), state.get("memory_profile", ""))
            fallback_tools = [name for name in plan["tools"] if name in {"web_search", "read_url"}]
            if plan.get("routing_override") and fallback_tools:
                for event in events:
                    if event.get("provider") == "router":
                        event["fallback_tools"] = fallback_tools
            plan = ground_weather_location(plan, state["messages"])
            plan = ground_search_freshness(plan, question)
            log.write("chat_plan", plan)
            updates["chat_plan"] = plan
            tools = plan["tools"]
            runtime = {"session_clock": get_current_time.invoke({"timezone": zone}),
                       "available_tools": allowed_tools(state["policy"], state.get("attachments", []), state.get("memory_profile", "")),
                       "preferences": state["preferences"],
                       "files": [{**f, "input_path": "/input/" + aliases(state.get("attachments", []))[f["file_id"]]}
                                 for f in state.get("attachments", []) if f.get("active", True)],
                       "previous_generated_files": state["generated_files"],
                       "selected_tools": tools, "blocked_tools": plan["blocked_tools"]}
            if state.get("revision_target"):
                from .artifacts import revision_context
                runtime["revision_context"] = revision_context(self.root, state["session_id"], state["revision_target"]["artifact_id"])
            from .tool_runtime import Context, run as run_tools
            runtime["question"] = question
            config = getattr(self.backend, "config", None)
            options = {"rounds": 3, "max_calls": 8, "seconds": 180}
            if config:
                options = {"rounds": config.get("interactive", "tool_rounds", default=3),
                           "max_calls": config.get("interactive", "tool_calls", default=8),
                           "seconds": config.get("interactive", "tool_seconds", default=180)}
            sources, notices, artifacts = run_tools(Context(self.backend, state, log, self._emit,
                self._file_sources, runtime, self.root), plan, **options)
            if artifacts:
                from .artifacts import tag_revision
                artifacts = tag_revision(self.root, state["session_id"], artifacts, state.get("revision_target"), state["turn_id"])
                runtime["artifacts"] = artifacts
            events.extend(notices)
            runtime["tool_outcomes"] = events
            updates.update(sources=sources, source_events=events, sources_loaded=bool(sources),
                           tool_results=runtime, artifacts=artifacts)
            log.write("chat_tools", {"runtime": runtime, "sources": sources, "events": events})
            # Model-selected tools supply context; they never replace the rest
            # of the user's request. Only the exact clock question skips prose.
            if plan["planner"] == "clock_fast_path":
                text = clock_answer(runtime["requested_clock"], question)
                log.write("clock_answer", {"answer": text, "model_generation": False, "network_requests": 0})
                return {**updates, **self._reply(state, text, "clock")}
            from .tool_runtime import guard_response, table_summary
            measured_table = table_summary(runtime) if plan.get("planner") == "explicit_table" else ""
            if measured_table:
                text = guard_response(measured_table, runtime)
                log.write("table_result_summary", {"source": "actual_tool_results", "final_prose_model_generation": False})
                self._emit("answer_text", text)
                return {**updates, **self._reply(state, text, "chat")}
            contract = runtime.get("completion_contract", {})
            receipt_tools = {"create_document", "export_table", "revise_document_title",
                             "remember_preference", "forget_preference", "files"}
            if (contract.get("requires_guard") and not contract.get("need_prose")
                    and not set(tools) - receipt_tools):
                text = guard_response("", runtime)
                log.write("completion_receipt", {"completed": contract.get("completed"),
                                                  "final_prose_model_generation": False})
                self._emit("answer_text", text)
                return {**updates, **self._reply(state, text, "chat")}
            history = [{"role": m["role"], "content": m["content"]} for m in state["messages"] if m["kind"] != "welcome"]
            # A cancellation notice is context, never a generated answer.
            # Retrying the same input should end on that input, not its notice.
            if state["messages"][-1].get("kind") == "cancelled":
                history.pop()
            for m in history:
                if m["role"] == "assistant":
                    m["content"] = without_reference_context(m["content"])
            if hasattr(self.backend, "prepare_chat_history"):
                history, memory, event = self.backend.prepare_chat_history(history, state.get("chat_memory", {}), sources, log, self._emit, runtime)
            else:
                history, memory, event = self.backend.prepare_history(history, state.get("chat_memory", {}), sources, log, self._emit)
            updates.update(chat_memory=memory, context_event=event)
            self._emit("status", "正在回答…")
            requires_guard = runtime.get("completion_contract", {}).get("requires_guard", False)
            def checked_emit(kind, value):
                if not requires_guard or kind not in {"token", "answer_text"}:
                    self._emit(kind, value)
            text = self.backend.chat(history, sources, log, checked_emit, runtime=runtime)
            text = guard_response(text, runtime)
            if artifacts:
                from .workspace import render_artifact_links
                text = render_artifact_links(text, state.get("generated_files", []) + artifacts)
                links = [f"[{a['name']}]({a['url']})" for a in artifacts
                         if a.get("validation", {}).get("valid") is True and a["url"] not in text]
                if links:
                    text += "\n\n" + ("生成文件：" if select_language(question) == "zh" else "Generated files: ") + " · ".join(links)
            elif state.get("generated_files"):
                from .workspace import render_artifact_links
                text = render_artifact_links(text, state["generated_files"])
            if requires_guard:
                self._emit("answer_text", text)
            return {**updates, **self._reply(state, text, "chat")}
        except Exception as exc:
            log.write("error", {"type": type(exc).__name__, "message": str(exc)})
            return {**updates, "error": str(exc), "pending": True}

    def _assess_review(self, state, log, origin="answer"):
        rubric = question_rubric(state)
        raw = objective_assessment(assessment_item(state), rubric, student_answers(state), state.get("language", "zh"))
        if raw is not None:
            log.write("authored_objective_assessment", raw)
            raw = resolve_evidence(raw, quote_catalog(student_answers(state)))
        else:
            raw = self.backend.assess(state, log)
        log.write("assessment_raw", raw)
        review = (ground_review(raw, rubric, student_answers(state), state["reviews"], state.get("language", "zh"))
                  if rubric and "criteria" in raw else
                  ground_formative_review(raw, state["case"], student_answers(state), state["reviews"], state.get("language", "zh")))
        return annotate_review(review, state, origin=origin)

    def _current_objective_review(self, state, log):
        """Upgrade legacy choice feedback without rewriting previous records."""
        item = assessment_item(state)
        validate_objective_rule(item, state.get("language", "zh"))
        latest = state["reviews"][-1] if state["reviews"] else {}
        if (latest.get("assessment_method") in {"authored_option_set_v2", "authored_objective_rules_v1"}
                and latest.get("objective_question_type") == objective_question_type(item)
                and latest.get("answer_count") == len(student_answers(state))):
            return latest, {}
        review = self._assess_review(state, log, origin="recheck")
        return review, {"reviews": state["reviews"] + [review], "bank_item": assessment_item(state),
                        "help_cache": {}, "hint_status": []}

    def _start_clinical(self, state, case, item, text, origin, **updates):
        from .clinical_flow import start_flow, current_question
        language = state.get("language", "zh")
        flow = start_flow(case, item, language)
        if not item.get("id") and state.get("operation") == "library_case":
            item = {**item,"id":state["operation_text"]}
            flow["case_id"] = item["id"]
        if item.get("id") and item["id"] in CASES and not item.get("classification"):
            from .catalog import catalog_entry
            entry = catalog_entry(item["id"],item)
            item = {**item,"classification":{"populations":entry["populations"],"device_types":entry["device_types"],"difficulty":entry["difficulty"]}}
        profile = state.get("learning_profile", "")
        updates.update(clinical_flow=flow, case=case, bank_item=deepcopy(item),case_origin=origin,stage="answer",
            exercise_learning_profile=profile,learning_generation=self._learning_store(state).generation(profile) if profile else 0)
        # The current task replaces the old invitation to write a whole plan.
        body = text.split(invitation(case,language))[0].rstrip()
        return self._reply(state, body+"\n\n"+current_question(flow,language), "case", **updates)

    def _clinical_turn(self, state, log):
        from .clinical_flow import (current_question, hint, feedback, apply_review, advance_step, interrupt_flow, review_summary)
        language = state.get("language","zh")
        flow = state["clinical_flow"]
        if state.get("pending_summary"):
            return {"pending_summary":True}
        operation = state.get("operation",state["request_action"])
        if operation == "next_step":
            flow = advance_step(flow,state.get("expected_step"))
            return self._reply(state,current_question(flow,language),"step_question",clinical_flow=flow,stage="answer",help_cache={},hint_status=[])
        if operation == "clarify":
            self._emit_teaching_support(state, "explanation", "clarify")
            return self._reply(state,current_question(flow,language)+"\n\n"+localized(language,"本步骤只回答上述任务即可，可按自己的表达组织答案；需要帮助时可获取提示。","Answer this task in your own words; request a hint if needed."),"clarification",support_intent="clarify")
        if operation == "reference" or operation == "stuck" and (len(flow["steps"][flow["current_step"]]["hints"]) >= 3 or state["stage"] == "complete" or flow["steps"][flow["current_step"]]["status"] != "active"):
            return self._reference_help(state, log)
        if operation in {"hint", "stuck"}:
            self._emit_teaching_support(state, "hint", operation)
            current = flow["steps"][flow["current_step"]]
            ended_locked = not reference_access(state)["allowed"] and (
                flow.get("status") != "in_progress" or current.get("status") != "active")
            hint_flow = deepcopy(flow) if ended_locked else flow
            if ended_locked:
                hint_flow["status"] = "in_progress"
                hint_flow["steps"][hint_flow["current_step"]]["status"] = "active"
            result = hint(hint_flow,state["case"],language=language)
            if ended_locked:
                result["flow"]["status"] = flow["status"]
                result["flow"]["steps"][flow["current_step"]]["status"] = current["status"]
            events = state.get("hint_events",[]) + ([result["event"]] if result["event"] else [])
            text = result["text"]
            if result["event"]:
                from .subjective_help import guided_hint_text
                text = guided_hint_text(text, result["event"]["level"], language)
            elif result["exhausted"] and flow["steps"][flow["current_step"]]["status"] == "active":
                text += "\n\n" + localized(language, "若仍答不上来，可以说“还是不会”，或选择查看本步参考答案。", "If you are still stuck, say 'I am still stuck' or view this step's reference answer.")
            return self._reply(state,text,"hint",clinical_flow=result["flow"],hint_events=events)
        if operation == "feedback":
            return self._reply(state,feedback(flow,language),"feedback")
        if operation == "summary":
            flow = interrupt_flow(flow) if flow["status"] != "completed" else flow
        else:
            updates = {}
            if not state.get("sources_loaded"):
                sources,events = self.backend.sources("MDRPI临床推理教学病例：\n"+state["case"],state["policy"],log,self._emit)
                updates.update(sources=sources,source_events=events,sources_loaded=True)
                state = {**state,**updates}
            self._emit("status","正在核对当前步骤的累计作答…")
            raw = self.backend.assess_step(state,log)
            log.write("clinical_assessment_raw",raw)
            flow = apply_review(flow,raw,language)
            review = flow["steps"][flow["current_step"]]["reviews"][-1]
            reviews = state.get("reviews",[]) + [review]
            text = feedback(flow,language)
            log.write("clinical_step_review",{"step":flow["current_step"],"review":review,"status":flow["status"]})
            if flow["status"] != "completed":
                return {**updates,**self._reply(state,text,"feedback",clinical_flow=flow,reviews=reviews,last_feedback=text,stage="answer",recheck_for_answer=len(student_answers(state)) if operation=="recheck" else -1)}
            state = {**state,**updates,"reviews":reviews}
        return {"clinical_flow":flow,"reviews":state.get("reviews",[]),"pending_summary":True,
                "stop_reason":"six_steps_completed" if flow["status"]=="completed" else "interrupted",
                **{key:state[key] for key in ("sources","source_events","sources_loaded") if key in state}}

    def _clinical_summary(self, state, log):
        from .clinical_flow import learning_record, review_summary, STEP_TITLES
        flow = state["clinical_flow"]
        language = state.get("language","zh")
        overview = review_summary(flow)
        count = overview["completed_steps"]
        label = localized(language,"六步训练完成" if flow["status"]=="completed" else "训练已提前结束","Six-step training completed" if flow["status"]=="completed" else "Training ended early")
        rows = [f"{i+1}. "+localized(language,*STEP_TITLES[step['id']])+"："+localized(language,"已完成" if step["status"] in {"passed","awaiting_next"} else "未完成","completed" if step["status"] in {"passed","awaiting_next"} else "unfinished") for i,step in enumerate(flow["steps"])]
        summary = f"**{label} · {count}/6**\n\n"+"\n\n".join(rows)+"\n\n"+learning_record(flow,language)
        # An explicitly labelled tutor explanation never becomes learner evidence.
        teacher = (self.backend.summary({**state,"clinical_flow":flow},log,self._emit)
                   if all_reference_steps_accessible(state) else reference_locked_text(state))
        title = (localized(language,"### 本轮教学复盘（AI生成）","### Tutor explanation (AI-generated)")
                 if all_reference_steps_accessible(state) else localized(language,"### 本轮学习状态","### Practice status"))
        summary += "\n\n"+title+"\n\n"+render_references(teacher,state.get("sources",[]),language)
        if all_reference_steps_accessible(state) and (state.get("bank_item") or {}).get("answer_origin") == "ai_reviewed_source_grounded":
            from .question_bank import reference_answer_text
            summary += "\n\n" + reference_answer_text(state)
        return self._reply(state,summary,"summary",clinical_flow=flow,stage="complete",summary=summary,teacher_explanation=teacher,
            stop_reason="six_steps_completed" if flow["status"]=="completed" else "interrupted",reviews=state.get("reviews",[]),pending_summary=False)

    def _summary_text(self, state, feedback, teacher):
        language = state.get("language", "zh")
        if objective_question_type(assessment_item(state)):
            return "\n\n".join(part for part in (feedback, reference_answer_text(state)) if part)
        grounded = state.get("reviews") and state["reviews"][-1].get("grading_mode") in {"question_rubric", "case_formative"}
        text = ((feedback + "\n\n" + render_references(teacher, state.get("sources", []), language)).strip() if grounded else
                render_references((feedback + "\n\n" + teacher).strip(), state.get("sources", []), language))
        if state.get("bank_item") and all_reference_steps_accessible(state):
            text += "\n\n" + reference_answer_text(state)
        return text

    def _recheck(self, state, log):
        self._emit("status", "正在重新核对已有作答…")
        review = self._assess_review(state, log, origin="recheck")
        language = state.get("language", "zh")
        prefix = localized(language, "**已重新核对已有回答**\n\n本次没有新增作答，原评价仍保存在记录中。",
                           "**Rechecked your existing answers**\n\nNo new answer was submitted. The previous assessment remains in the record.")
        feedback = prefix + "\n\n" + assessment_text(review, language)
        updates = {"reviews": state["reviews"] + [review], "last_feedback": feedback,
                   "recheck_for_answer": len(student_answers(state)), "help_cache": {}, "hint_status": []}
        if review.get("objective_question_type"):
            updates["bank_item"] = assessment_item(state)
        log.write("assessment_recheck", {"previous_review": state["reviews"][-1], "new_review": review,
                                          "answer_count": len(student_answers(state))})
        revised = {**state, **updates}
        if state["stage"] == "complete":
            if objective_question_type(assessment_item(state)):
                teacher = ""
            elif not all_reference_steps_accessible(revised):
                teacher = reference_locked_text(revised)
            else:
                previous_teacher = state.get("teacher_explanation") if all_reference_steps_accessible(state) else ""
                teacher = previous_teacher or self.backend.summary(revised, log, self._emit)
            text = self._summary_text(revised, feedback, teacher)
            return self._reply(state, text, "recheck", summary=text, teacher_explanation=teacher, **updates)
        if review["ready_for_summary"] and not review["material_errors"] and (
                review.get("objective_question_type") or has_verified_answer(revised)):
            return {**updates, "pending_summary": True, "stop_reason": "adequate_after_recheck"}
        if review.get("objective_question_type"):
            return self._reply(state, assessment_text(review, language), "recheck", stage="probe", **updates)
        return self._reply(state, feedback + "\n\n" + review["feedback"], "recheck", stage="probe", **updates)

    def _reference_help(self, state, log):
        """Show a tutor reference without grading, passing or ending the question."""
        access = reference_access(state)
        if not access["allowed"]:
            result = (self._clinical_turn({**state, "operation": "hint"}, log) if state.get("clinical_flow")
                      else self._subjective_hint(state, log, "hint"))
            prefix = localized(state.get("language", "zh"),
                "完整参考答案需要先看完三条提示。我们先看下一条：",
                "The full reference unlocks after three hints. Let's look at the next hint first:")
            if result.get("response"):
                body = prefix + "\n\n" + result["response"]
                result["response"] = body
                result["messages"][-1]["content"] = body
            log.write("reference_access", {**access, "action": "next_progressive_hint"})
            return result
        self._emit_teaching_support(state, "reference", "reference")
        language = state.get("language", "zh")
        item, flow = assessment_item(state), deepcopy(state.get("clinical_flow") or {})
        step = flow["steps"][flow["current_step"]] if flow else None
        step_id = step["id"] if step else ""
        from .clinical_flow import current_rubric
        rubric = current_rubric(flow, language) if flow else question_rubric(state)
        body, origin = "", ""
        if step:
            basis = (rubric or {}).get("basis_answer", "")
            if basis:
                body = localized(language, "### 本步参考答案", "### Reference answer for this step") + "\n\n" + basis
                origin = "saved_step_reference"
                references = (flow.get("rubric_snapshot", {}).get("references") or item.get("review_references")
                              or item.get("source", {}).get("references") or [])
                required_ids = {ref_id for row in (rubric or {}).get("required", []) for ref_id in row.get("reference_ids", [])}
                selected_refs = [ref for ref in references if ref.get("id") in required_ids] if required_ids else references
                if references:
                    links = [f"[{ref.get('title') or ref.get('citation', '来源')}]({ref['url']})" +
                             (" — " + ref["locator"] if ref.get("locator") else "") for ref in selected_refs or references if ref.get("url")]
                    if links:
                        body += "\n\n" + localized(language, "编写依据：", "Source basis: ") + " · ".join(links)
        elif item.get("answer") and item.get("has_reference_answer", True):
            body = reference_answer_text({**state, "bank_item": item})
            origin = "saved_question_reference"
        key = hashlib.sha256(json.dumps({"case": state["case"], "item": item.get("id"),
            "answer": item.get("answer"), "step": step_id, "rubric": rubric, "language": language},
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache = deepcopy(state.get("reference_cache") or {})
        if not body:
            if cache.get("key") == key:
                body, origin = cache["text"], cache["origin"]
            else:
                # User-supplied/generated cases have no publisher key. Prepare
                # a separate teaching example using case facts, never learner grades.
                self._emit("status", "正在整理参考作答…")
                # The backend checks access on the actual saved state, then
                # builds a payload containing only case/current task/reference
                # basis. Learner attempts and reviews never enter generation.
                example = self.backend.teaching_reference(state, log)
                if not isinstance(example, str) or not example.strip():
                    raise ValueError(localized(language, "参考作答暂未生成，请重试。", "The teaching example is not available yet. Please retry."))
                body = localized(language, "### AI教学示例（本题没有题库标准答案）", "### AI teaching example (no published answer key for this case)") + "\n\n" + example
                origin = "model_teaching_example"
        cache = {"key": key, "text": body, "origin": origin, "step_id": step_id}
        event = {"answer_count": len(student_answers(state)), "focus": "all", "step_id": step_id,
            "origin": origin, "language": language, "text": body, "at": now(), "reference_key": key}
        views = deepcopy(state.get("reference_views") or [])
        # The first exposure is enough for assistance attribution; repeated
        # viewing keeps that original position in the answer timeline.
        if not any(row.get("reference_key") == key for row in views):
            views.append(event)
            if step:
                local_event = {**event, "answer_count": len(step["answers"])}
                step.setdefault("reference_views", []).append(local_event)
        encouragement = localized(language,
            "暂时答不上来没关系，愿意求助就是继续学习的一步。先看参考答案，再试着用自己的话整理；多练几次会更熟悉。",
            "It is okay to be stuck. Asking for help is part of learning. Read the reference, then try putting it in your own words; practice will help.")
        footer = localized(language, "你可以继续作答，也可以结束本题。查看参考答案会单独记录，不会自动算作已经答对。",
                           "You can try answering again or finish this question. Viewing the reference is recorded separately and does not automatically count as a correct answer.")
        if state["stage"] == "complete":
            footer = localized(language, "本轮已经结束，可选择重练本题再巩固。", "This round has ended. Repeat the question to practise again.")
        updates = {"reference_views": views, "reference_cache": cache}
        if not flow and state["stage"] == "reflect":
            updates["stage"] = "probe"
        if not student_answers(state) and not state.get("exercise_learning_profile"):
            profile = state.get("learning_profile", "")
            updates.update(exercise_learning_profile=profile,
                           learning_generation=self._learning_store(state).generation(profile) if profile else 0)
        if flow:
            updates["clinical_flow"] = flow
        log.write("reference_answer_view", {k: event[k] for k in ("answer_count", "step_id", "origin", "reference_key")})
        return self._reply(state, encouragement + "\n\n" + body + "\n\n" + footer, "reference", **updates)

    def _subjective_hint(self, state, log, operation):
        """Three opportunities per question; being stuck is help, not an attempt."""
        from .subjective_help import subjective_hint_plan, guided_hint_text
        from .hints import learning_scaffold
        language = state.get("language", "zh")
        events = hint_events(state)
        access = reference_access(state)
        if operation == "stuck" and access["allowed"]:
            return self._reference_help(state, log)
        latest = state["reviews"][-1] if state["reviews"] else {}
        if latest.get("ready_for_summary") and not latest.get("material_errors") and access["allowed"]:
            self._emit_teaching_support(state, None, "feedback")
            return self._reply(state, assessment_text(latest, language), "feedback")
        self._emit_teaching_support(state, "hint", operation)
        if len(events) >= 3:
            text = localized(language, "本题的三条提示已看完。如果还是不会，可以直接说“还是不会”，或选择查看参考答案。",
                             "You have seen all three hints. If you are still stuck, say so or view the reference answer.")
            return self._reply(state, text, "help", support_intent="hint", hint_status=[1, 1, 1])
        item = assessment_item(state)
        focus = latest.get("focus", "all")
        cache = deepcopy(state.get("help_cache") or {})
        hint_key = f"subjective_three_hints:{len(state['reviews'])}:{language}:{focus}"
        if cache.get("hint_key") != hint_key:
            if item and (item.get("hints") or item.get("hint_pathways")):
                raw, origin, target = bank_hint_plan(item, language, focus)
            elif latest:
                raw, origin, target = self.backend.teaching_hints(state, log), "model", focus
            else:
                raw, origin, target = {"hints": learning_scaffold(language, focus)}, "learning_scaffold", focus
            plan, reasons = subjective_hint_plan(item, raw, language, focus)
            displayed_origin = "learning_scaffold" if any(reason != "missing_published_criterion_framework" for reason in reasons) else origin
            cache.update(hint_key=hint_key, subjective_hints=plan, hint_origin=displayed_origin, hint_focus=target)
            log.write("subjective_hint_preparation", {"origin": origin, "focus": target, "fallback_reasons": reasons})
        level = len(events) + 1
        body = cache["subjective_hints"][level - 1]
        # Old hints may have been shown for different criteria. Avoid repeating
        # their actual content while retaining the three-opportunity ceiling.
        if any(row.get("text") == body and row.get("language", language) == language for row in events):
            body = next((hint for hint in cache["subjective_hints"] if not any(
                row.get("text") == hint and row.get("language", language) == language for row in events)), body)
        event_origin = ("published_rubric_framework" if item.get("rubric", {}).get("required") else "learning_scaffold") if level == 3 else cache["hint_origin"]
        events.append({"text": body, "level": level, "language": language, "answer_count": len(student_answers(state)),
            "focus": "all" if level == 3 else cache.get("hint_focus", "all"), "origin": event_origin, "at": now()})
        return self._reply(state, guided_hint_text(body, level, language), "hint", help_cache=cache,
            hint_events=events, hint_status=[1] * level + [0] * (3 - level),
            **({"stage": "probe"} if state["stage"] == "reflect" else {}))

    def _teaching_help(self, state, log, operation):
        self._emit("status", "正在准备提示…" if operation in {"hint", "stuck"} else "正在整理已有反馈…")
        language = state.get("language", "zh")
        kind = objective_question_type(assessment_item(state))
        if kind:
            latest, updates = (self._current_objective_review(state, log) if student_answers(state) else ({}, {}))
            if operation == "feedback" or latest.get("ready_for_summary"):
                text = assessment_text(latest, language) if latest else objective_invitation(kind, language)
                return self._reply(state, text, "help", **updates)
            item = assessment_item(state)
            if not (item.get("hints") or item.get("hint_pathways")):
                return self._reply(state, objective_invitation(kind, language), "help", **updates)
            from .question_bank import objective_hint_cues
            cues, origin, focus = objective_hint_cues(item, language, latest.get("focus", "all"))
            events = hint_events(state)
            seen = [int(any(event.get("text") == cue and event.get("language", language) == language
                            for event in events)) for cue in cues]
            cue, seen = reveal_next(cues, seen)
            self._emit_teaching_support(state, "hint", operation)
            text = (localized(language, "**提示**\n\n", "**Hint**\n\n") + cue if cue else
                    localized(language, "本次提示已全部给出。", "All hints for this question have been shown."))
            text += "\n\n" + objective_invitation(kind, language)
            if cue:
                events.append({"text": cue, "language": language, "answer_count": len(student_answers(state)),
                               "focus": focus, "origin": origin, "at": now()})
            log.write("authored_objective_help", {"operation": operation, "model_called": False, "hint_status": seen})
            updates.update(hint_events=events, hint_status=seen)
            return self._reply(state, text, "hint", **updates)
        if operation in {"hint", "stuck"}:
            return self._subjective_hint(state, log, operation)
        if not state["reviews"]:
            item = state.get("bank_item") or {}
            if operation != "hint" or not (item.get("hints") or item.get("hint_pathways")):
                return self._reply(state, invitation(state["case"], language), "help")
            # An authored hint can precede the first answer. It creates no
            # assessment; the same finite, source-reviewed pathway is used.
            latest = {"focus": "all", "language": language, "ready_for_summary": False,
                      "material_errors": []}
        else:
            latest = normalize_review(state["reviews"][-1], state["reviews"][-1].get("language") or select_language(state["reviews"][-1]["feedback"], language))
        if latest["ready_for_summary"] and not latest["material_errors"] and latest.get("language") == language:
            text = assessment_text(latest, language)
            if operation == "hint":
                text += "\n\n" + localized(language, "本题主要要求已覆盖，无需继续补充提示。", "The main requirements are covered; no further hint is needed.")
            return self._reply(state, text if latest.get("grading_mode") in {"question_rubric", "case_formative"} else
                               render_references(text, state.get("sources", []), language), "feedback")
        key = f"explanatory:{len(state['reviews'])}:{language}"
        cache = state.get("help_cache", {})
        status = state.get("hint_status", [])
        if cache.get("key") != key:
            cache = {"key": key}
            status = []
        events = hint_events(state)
        if operation == "feedback":
            if "feedback" not in cache:
                if latest.get("language") == language or latest.get("grading_mode") in {"question_rubric", "case_formative"}:
                    cache["feedback"] = assessment_text(latest, language)
                else:
                    cache["feedback"] = localized(language, "**本次回答反馈**\n\n", "**Feedback on your answer**\n\n") + self.backend.teaching_support(state, log)["feedback"]
            text = cache["feedback"]
        else:
            if "hints" not in cache:
                item = state.get("bank_item")
                if item and (item.get("hints") or item.get("hint_pathways")):
                    raw, origin, hint_focus = bank_hint_plan(item, language, latest.get("focus"))
                else:
                    raw = self.backend.teaching_hints(state, log)
                    origin = "model"
                    hint_focus = latest.get("focus", "all")
                hints, reasons = prepare_hints(raw, language, latest.get("focus"))
                cache.update(hints=hints, hint_origin="learning_scaffold" if reasons else origin, hint_focus=hint_focus)
                log.write("hint_preparation", {"raw": raw, "origin": origin, "fallback_reasons": reasons, "displayed_hints": hints})
            # Rechecking or another answer must not unlock the same static
            # hint again. Saved events survive the per-review help cache.
            status = [1 if any(event.get("text") == hint and event.get("language", language) == language for event in events) else 0
                      for hint in cache["hints"]]
            hint, status = reveal_next(cache["hints"], status)
            text = (localized(language, "**提示**\n\n", "**Hint**\n\n") + hint if hint else
                    localized(language, "本次提示已全部给出。你可以据此修正答案，或选择结束练习并总结。", "All hints for this point have been shown. You can revise your answer or finish with a summary."))
            if hint:
                events.append({"text": hint, "language": language, "answer_count": len(student_answers(state)),
                               "focus": cache.get("hint_focus", "all" if state.get("bank_item") else latest.get("focus", "all")),
                               "origin": cache["hint_origin"], "at": now()})
        log.write("teaching_help", {"operation": operation, "language": language, "hint_status": status, "assessed_answers": len(student_answers(state))})
        if not (operation == "feedback" and latest.get("grading_mode") in {"question_rubric", "case_formative"}):
            text = render_references(text, state.get("sources", []), language)
        return self._reply(state, text, "feedback" if operation == "feedback" else "help", help_cache=cache, hint_status=status, hint_events=events)

    def _respond(self, state):
        # A cancelled supplemental explanation must not assess its answer again.
        if state.get("mixed_help_pending") and state.get("mixed_grade_done"):
            return {}
        result = self._respond_core(state)
        if state.get("mixed_help_pending") and not result.get("error"):
            result["mixed_grade_done"] = True
        return result

    def _respond_core(self, state):
        log = self._log(state)
        updates = {}
        try:
            language = state.get("language", "zh")
            if state["mode"] == "teaching" and state["stage"] in {"setup", "case", "topic"}:
                operation = state.get("operation", state["request_action"])
                if operation == "choose_case":
                    return self._reply(state, localized(language, "请使用已脱敏的教学案例，移除姓名、身份证、电话、住院号、具体地址和床位等身份信息；确认后导入。缺失的临床信息会保持未知。", "Use a deidentified teaching case; remove names, IDs, contacts, record numbers, addresses and bed numbers, then confirm import. Missing clinical facts remain unknown."), stage="case")
                if operation == "choose_topic":
                    return self._reply(state, localized(language, "想练习哪个主题？例如：面罩评估、鼻胃管固定、多器械风险。也可以直接说‘给我一个基础案例’。", "What topic would you like to practise? For example: mask assessment, nasogastric-tube fixation, or multiple devices. You can also ask for a beginner case."), stage="topic")
                if operation == "library_case":
                    item = CASES[state["operation_text"]]
                    case = practice_case(state["operation_text"], language)
                    bank_item = deepcopy(item) if item["has_reference_answer"] or item.get("kind") == "standardized_case" else {}
                    text = exercise_intro(item, case, language)
                    log.write("practice_selection", {"case_id": state["operation_text"], "bank_item": bank_item})
                    if hasattr(self.backend,"assess_step") and (item.get("kind") in {"standardized_case", "fictional_case"} or item.get("category")=="original"):
                        return self._start_clinical(state,case,deepcopy(item),text,bank_origin(item) if bank_item else "practice_library")
                    return self._reply(state, text, "case", case=case, bank_item=bank_item,
                                       case_origin=bank_origin(bank_item) if bank_item else "practice_library", stage="answer")
                case = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
                materials, events = self._file_sources(state, case)
                if state["stage"] == "case" and not materials:
                    if hasattr(self.backend,"assess_step"):
                        note = localized(language,"用户自有脱敏案例 · 仅供教学练习","User-supplied deidentified case · practice only")
                        missing = state.get("case_intake",{}).get("missing_fields",[])
                        body = note+"\n\n"+case+("\n\n待补充信息："+"、".join(missing) if missing else "")
                        return self._start_clinical(state,case,{},body,"user_deidentified_case")
                    return self._reply(state, invitation(case, language), "invitation", case=case, case_origin="provided_case", stage="answer")
                self._emit("status", "正在准备练习案例…")
                history = [{"role": m["role"], "content": m["content"]} for m in state["messages"] if m["kind"] != "welcome"][-8:-1]
                if state["stage"]=="topic" and state.get("case_spec"):
                    from .case_intake import spec_request
                    case = spec_request(state["case_spec"])
                setup = self.backend.teaching_start(case, materials, log, generate=state["stage"] == "topic", history=history, language=language)
                log.write("teaching_start", setup)
                if setup["kind"] == "clarify":
                    return self._reply(state, setup["prompt"] or (WELCOME if language == "zh" else WELCOME_EN), stage="setup")
                if not setup["case"].strip():
                    raise ValueError("模型未返回案例，请重试。")
                case = setup["case"]
                question = invitation(case, language)
                if setup["kind"] == "generated_case":
                    label = localized(language, "AI生成·仅供练习", "AI-generated · practice only")
                    case = label + "\n\n" + case
                if hasattr(self.backend,"assess_step"):
                    return self._start_clinical(state,case,{},case,setup["kind"],source_events=events)
                return self._reply(state, case + "\n\n" + question, "case", case=case,
                    case_origin=setup["kind"], stage="answer", source_events=events)
            if state["mode"] == "performance":
                query = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
                result = self.backend.performance(query, state["web_mode"], state["replay_run_id"], log, self._emit)
                return self._reply(state, performance_answer(result), "performance", performance_result=result, **performance_fields(result))
            if state["mode"] == "chat":
                return self._chat_turn(state, log)
            operation = state.get("operation", state["request_action"])
            if operation == "clarify_input":
                return self._clarify_input(state)
            if operation in {"teaching_dialogue", "explain", "hint_detail", "clarify"}:
                intent = state.get("input_route", {}).get("intent") if operation == "teaching_dialogue" else operation
                return self._dialogue_help(state, log, intent or "explain")
            if state.get("clinical_flow"):
                return self._clinical_turn(state,log)
            if state.get("pending_summary"):
                return {"pending_summary": True}
            operation = state.get("operation", state["request_action"])
            if operation == "reference":
                return self._reference_help(state, log)
            if operation in {"hint", "feedback", "stuck"}:
                return self._teaching_help(state, log, operation)
            if operation == "recheck":
                return self._recheck(state, log)
            if operation == "summary" and state.get("reference_views") and not student_answers(state) and not objective_question_type(assessment_item(state)):
                feedback = localized(language, "本轮先学习了参考答案，尚未独立作答。可以从学习记录重练本题，逐步巩固。",
                                     "You studied the reference answer without submitting an independent answer. Repeat this question from Learning progress to practise.")
                text = feedback + "\n\n" + state.get("reference_cache", {}).get("text", "")
                return self._reply(state, text, "summary", summary=text, last_feedback=feedback, teacher_explanation="",
                                   stage="complete", pending_summary=False, stop_reason="reference_studied")
            choice_kind = objective_question_type(assessment_item(state))
            if choice_kind:
                updates["bank_item"] = assessment_item(state)
                state = {**state, **updates}
            if not state["sources_loaded"]:
                if choice_kind:
                    sources, events = [], []
                else:
                    question = ("Nursing teaching exercise:\n" if state.get("bank_item") else "Medical device-related pressure injury (MDRPI) teaching case:\n") + state["case"]
                    sources, events = self.backend.sources(question, state["policy"], log, self._emit)
                    file_sources, file_events = self._file_sources(state, question)
                    sources, events = file_sources + sources, events + file_events
                log.write("session_sources", {"sources": sources, "events": events})
                updates.update(sources=sources, source_events=events, sources_loaded=True)
                state = {**state, **updates}
            if operation == "summary":
                return {**updates, "pending_summary": True, "stop_reason": "student_requested"}
            self._emit("status", "正在核对所选选项…" if choice_kind else "正在阅读你的回答并准备反馈…")
            rubric = question_rubric(state)
            if rubric and not state["bank_item"].get("rubric"):
                updates["bank_item"] = {**state["bank_item"], "rubric": deepcopy(rubric)}
                state = {**state, **updates}
            review = self._assess_review(state, log)
            if len(student_answers(state)) == 1:
                review["improvements"] = []  # No earlier student answer to compare.
            # The model judges meaning; code controls the interaction and its bounds.
            ready = review["ready_for_summary"] and not review["material_errors"]
            if choice_kind:
                feedback = assessment_text(review, language)
                updates.update(reviews=state["reviews"] + [review], last_feedback=feedback, help_cache={}, hint_status=[])
                log.write("teaching_route", {"review": review, "stop_reason": "adequate_answer" if ready else "",
                                             "attempts": len(student_answers(state)), "model_called": False})
                if ready:
                    return {**updates, "pending_summary": True, "stop_reason": "adequate_answer"}
                return self._reply(state, feedback, stage="probe", **updates)
            ready = ready and has_verified_answer({**state, "reviews": state["reviews"] + [review]})
            reason = "adequate_answer" if ready else ""
            updates["reviews"] = state["reviews"] + [review]
            feedback = assessment_text(review, language)
            updates.update(last_feedback=feedback, help_cache={}, hint_status=[])
            log.write("teaching_route", {"review": review, "stop_reason": reason, "attempts": len(state["attempts"])})
            if reason:
                return {**updates, "pending_summary": True, "stop_reason": reason}
            stage = "probe"
            next_message = review["feedback"]
            text = feedback + "\n\n" + next_message
            if review.get("grading_mode") not in {"question_rubric", "case_formative"}:
                text = render_references(text, state["sources"], language)
            return self._reply(state, text, stage=stage, **updates)
        except Exception as exc:
            log.write("error", {"type": type(exc).__name__, "message": str(exc)})
            return {**updates, "error": str(exc), "pending": True}

    def _summarize(self, state):
        log = self._log(state)
        updates = {}
        self._emit_teaching_support(state, None, "summary")
        try:
            if state.get("clinical_flow"):
                return self._clinical_summary(state,log)
            language = state.get("language", "zh")
            if objective_question_type(assessment_item(state)):
                review, updates = self._current_objective_review(state, log)
                revised = {**state, **updates}
                feedback = assessment_text(review, language)
                text = self._summary_text(revised, feedback, "")
                self._emit("status", "正在整理练习结果…")
                log.write("authored_objective_summary", {"assessment_method": review["assessment_method"],
                                                       "model_called": False, "stop_reason": state["stop_reason"]})
                return self._reply(state, text, "summary", summary=text, teacher_explanation="", last_feedback=feedback,
                                   stage="complete", pending_summary=False, **updates)
            feedback = state.get("last_feedback", "")
            if state["reviews"] and state["reviews"][-1].get("grading_mode") in {"question_rubric", "case_formative"}:
                feedback = assessment_text(state["reviews"][-1], language)
                if not state["reviews"][-1]["ready_for_summary"]:
                    feedback += "\n\n" + localized(language, "本题仍有未覆盖或待确认的要点；结束练习不代表全部答对。未解锁参考时可以在当前页继续获取剩余提示。",
                        "Some required points remain uncovered or unconfirmed. Ending this exercise does not mean every answer is correct. Get any remaining hints on this page to unlock the reference.")
            elif state["reviews"] and state["reviews"][-1].get("language") != language:
                key = f"explanatory:{len(state['reviews'])}:{language}"
                cache = state.get("help_cache", {})
                if cache.get("key") != key or "feedback" not in cache:
                    cache = {**(cache if cache.get("key") == key else {}), **self.backend.teaching_support(state, log), "key": key}
                    cache["feedback"] = localized(language, "**本次回答反馈**\n\n", "**Feedback on your answer**\n\n") + cache["feedback"]
                updates["help_cache"] = cache
                feedback = cache["feedback"]
            unlocked = all_reference_steps_accessible(state)
            self._emit("status", "正在生成教学总结…" if unlocked else "正在整理已保存的学习状态…")
            summary = self.backend.summary(state, log, self._emit) if unlocked else reference_locked_text(state)
            # Learner quotations keep their original [n] text; only the
            # separately generated teacher explanation formats source numbers.
            text = self._summary_text(state, feedback, summary)
            log.write("summary", {"text": text, "model_summary": summary, "stop_reason": state["stop_reason"]})
            return self._reply(state, text, "summary", summary=text, teacher_explanation=summary, stage="complete", pending_summary=False, **updates)
        except Exception as exc:
            log.write("summary_error", {"type": type(exc).__name__, "message": str(exc)})
            return {**updates, "error": str(exc), "pending": True, "pending_summary": True}

    def close(self):
        self.connection.close()
