"""Reuse the local LLM client, retrieval API, web readers and citation formatter."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

from app.config import AppConfig
from app.local_llm import LocalLLM
from app.citation_format import reference_for_display, display_text, _identities, merge_reference_sources
from . import prompts
from .memory import compact
from .chat_tools import allowed_tools, DEFAULT_TIMEZONE, get_current_time
from .language import instruction, select_language
from .teaching import student_answers, tutor_context
from .question_bank import reference_context, question_rubric, rubric_context, assessment_item
from .answer_display import is_knowledge_source, clean_prose, prose_parts, without_bibliography, SourceStream
from .tasks import CURRENT_TURN, CancellableClient, check_cancelled


def chat_output_budget(question):
    if re.search(r"\b(?:complete|full|detailed|long|comprehensive)\b|详细|长文|完整|全面|不少于", question, re.I):
        return 5000
    if re.search(r"\b(?:code|program|script|essay|report)\b|代码|程序|文章|报告|[1-9]\d{3,}\s*字", question, re.I):
        return 3500
    return 1600


class TurnLog:
    """Small append-only turn artifacts; no new integrity gate for interaction."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index = len(list(self.path.glob("*.json")))

    def write(self, name, value):
        target = self.path / f"{self.index:04}_{name}.json"
        self.index += 1
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_context(sources):
    return [{"number": i, "reference": reference(s), "location": s.get("location", ""), "text": s.get("text", ""),
             "material_level": s.get("material_level", ""), "retrieved_at": s.get("retrieved_at", ""),
             **{key:s[key] for key in ('retrieval_support_status','support_quotes','retrieval_assessment',
                 'source_quality','bibliographic_year','first_publication_date','doi','pmid') if key in s},
             "cite_in_answer": not is_knowledge_source(s)}
            for i, s in enumerate(sources, 1)]


def reference(source, language=None):
    if source.get("source_type") == "user_upload":
        label = {"zh": "用户上传文件", "en": "Uploaded document"}.get(language, "用户上传文件 / Uploaded document")
        return source["title"] + ". " + label + "."
    return reference_for_display(source)


CITATION = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")
REFERENCE_HEADING = "\n\n### 参考资料 / References\n\n"


def without_reference_context(text):
    """Old source numbers are meaningless next turn; ordinary text/code is not."""
    body = without_bibliography(text)
    if body == text.rstrip():
        return text
    parts = prose_parts(body)
    parts[::2] = [CITATION.sub("", part) for part in parts[::2]]
    return "".join(parts).rstrip()


def chat_recent_turns(state):
    """Keep follow-up referents while removing obsolete reference numbering."""
    recent = [{"role": m["role"], "content": m["content"]}
              for m in state["messages"] if m["kind"] not in {"welcome", "cancelled"}][-5:-1]
    last_assistant = next((i for i in range(len(recent) - 1, -1, -1)
                           if recent[i]["role"] == "assistant"), None)
    for i, message in enumerate(recent):
        if message["role"] == "assistant":
            message["content"] = without_reference_context(message["content"])
        if i != last_assistant and len(message["content"]) > 1600:
            message["content"] = message["content"][:800] + "\n[Middle excerpt omitted]\n" + message["content"][-800:]
    return recent


def render_references(text, sources, language=None):
    if not sources:
        return text
    hidden = [i for i, s in enumerate(sources, 1) if is_knowledge_source(s)]
    parts = prose_parts(clean_prose(text, hidden_numbers=hidden))
    parts[::2] = [display_text(part, {}) for part in parts[::2]]
    cited = list(dict.fromkeys(int(n) for group in CITATION.findall("\n".join(parts[::2])) for n in re.findall(r"\d+", group)))
    selected = [old for old in cited if 1 <= old <= len(sources) and not is_knowledge_source(sources[old - 1])]
    merged = merge_reference_sources([sources[old - 1] for old in selected])
    references = [row["source"] for row in merged]
    numbers = {selected[position - 1]: number for number, row in enumerate(merged, 1) for position in row["positions"]}
    rows = [f"[{i}] {reference(s, language)}" for i, s in enumerate(references, 1)]
    # Number documents, not excerpts, in their first-citation order.
    def replace(group):
        mapped = list(dict.fromkeys(numbers[int(n)] for n in re.findall(r"\d+", group[1]) if int(n) in numbers))
        return "[" + ", ".join(map(str, mapped)) + "]" if mapped else ""
    parts[::2] = [CITATION.sub(replace, part) for part in parts[::2]]
    heading = {"zh": "\n\n### 参考资料\n\n", "en": "\n\n### References\n\n"}.get(language, REFERENCE_HEADING)
    return "".join(parts).rstrip() + (heading + "\n\n".join(rows) if rows else "")


def fit_messages(llm, messages, max_tokens):
    """Retain the system and latest request; trim old complete chat turns only."""
    messages = list(messages)
    limit = int(llm.config.get("llm", "context_window", default=32768)) - max_tokens - 128
    trimmed = 0
    while llm.count_tokens(messages)[0] > limit:
        if len(messages) <= 2:
            raise ValueError("本次输入超出模型上下文，请缩短案例或问题。")
        del messages[1]
        trimmed += 1
        while len(messages) > 2 and messages[1]["role"] != "user":
            del messages[1]
            trimmed += 1
    return messages, trimmed


def template_messages(messages):
    """Ministral needs a user before isolated assistant fragments used by LangMem."""
    result = []
    for m in messages:
        if m["role"] == "system" and result and result[-1]["role"] == "system":
            result[-1]["content"] += "\n\n" + m["content"]
        else:
            result.append(dict(m))
    first = next((i for i, m in enumerate(result) if m["role"] != "system"), len(result))
    if first == len(result) or result[first]["role"] == "assistant":
        result.insert(first, {"role": "user", "content": "Earlier conversation excerpt:"})
    return result


class Backend:
    def __init__(self, config_path, rag_url="http://127.0.0.1:8020"):
        self.config = AppConfig.from_file(config_path)
        self.rag_url = rag_url.rstrip("/")

    def _llm(self):
        check_cancelled()
        llm = LocalLLM(self.config)
        if CURRENT_TURN.get() is not None:
            llm.client = CancellableClient(llm.client)
        return llm

    def teaching_route(self, state, text, log):
        """One bounded intent decision; failure never becomes an answer route."""
        from app.evidence_snapshot import ModelRunner, RunBudget
        from .teaching_router import (TEACHING_ROUTER_VERSION, ROUTE_PROMPT,
            fast_teaching_route, teaching_route_payload, teaching_route_schema, validate_teaching_route)
        check_cancelled()
        fast = fast_teaching_route(text)
        if fast is not None:
            result = validate_teaching_route(fast, text)
            log.write("teaching_input_route", {"version": TEACHING_ROUTER_VERSION, "source": "whole_control_rule", "result": result})
            return result
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Teaching routing requires nonempty input text.")
        seconds = float(self.config.get("teaching_router", "route_timeout_seconds", default=20))
        if not 0 < seconds <= 120:
            raise ValueError("Teaching route timeout must be greater than zero and no more than 120 seconds.")
        llm = self._llm()
        try:
            # A single call is the model-call ceiling. There is no repair or
            # planning loop, and RunBudget includes tokenization and generation.
            raw = ModelRunner(llm, RunBudget(seconds), log).call(ROUTE_PROMPT,
                teaching_route_payload(state, text), teaching_route_schema(),
                "teaching_input_route", max_tokens=400)
            result = validate_teaching_route(raw, text)
            log.write("teaching_input_route", {"version": TEACHING_ROUTER_VERSION, "source": "semantic_model", "result": result})
            return result
        finally:
            llm.close()

    def teaching_dialogue(self, state, text, intent, log):
        """Answer one teaching follow-up; never assess, advance or consume a hint."""
        from app.evidence_snapshot import ModelRunner, RunBudget
        from .language import localized
        from .learner_evidence import guard_teacher_explanation
        from .teaching_router import (DIALOGUE_INTENTS, last_teaching_hint,
                                      teaching_dialogue_payload, incomplete_teaching_reply)
        check_cancelled()
        if intent not in DIALOGUE_INTENTS:
            raise ValueError("Unsupported teaching dialogue intent.")
        if intent == "social":
            import unicodedata
            value = unicodedata.normalize("NFKC", text).strip().strip("。.!！?？ ").casefold()
            language = state.get("language", "zh")
            if re.fullmatch(r"你好|您好|hi|hello", value):
                return localized(language, "你好，我们可以按你的节奏继续练习。", "Hello. We can continue practising at your pace.")
            if re.fullmatch(r"谢谢(?:你)?|多谢|不客气|thank you|thanks", value):
                return localized(language, "不客气。还有哪里想弄清楚，可以继续问我。", "You're welcome. Ask me if there is anything else you would like clarified.")
            if re.fullmatch(r"收到|明白了|知道了|understood|got it", value):
                if state.get("stage") == "complete":
                    return localized(language, "好的。你可以回看本轮总结，或继续问我想弄清楚的地方。", "Okay. Review this round's summary, or ask me about anything you would like clarified.")
                return localized(language, "好的。准备好时可以继续作答，需要时也可以再问。", "Okay. Continue answering when you are ready, or ask another question when needed.")
        if intent == "hint_detail" and not last_teaching_hint(state):
            return localized(state.get("language", "zh"),
                "先获取一条提示，我再围绕这条提示解释得更具体一些。",
                "Get a hint first, and I can explain that hint more concretely.")
        seconds = float(self.config.get("teaching_router", "dialogue_timeout_seconds", default=60))
        if not 0 < seconds <= 180:
            raise ValueError("Teaching dialogue timeout must be greater than zero and no more than 180 seconds.")
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"reply": {"type": "string", "minLength": 1, "maxLength": 6000}},
                  "required": ["reply"]}
        prompt = (instruction(state.get("language", "zh")) +
            "Answer ONLY this teaching follow-up within current_task and the saved teaching basis. "
            "Never score, pass, fail, advance, end the exercise, or claim the learner said, understood, "
            "corrected or mastered anything. Case facts are separate from the user's proposed actions; "
            "missing observations, procedural parameters and outcomes remain unknown. Do not invent "
            "patient findings, measurements, source citations or a publisher's official answer. "
            "The saved reference may be project-authored or AI-reviewed; do not present it as independent "
            "expert validation. Preserve summary_constraints and published safety boundaries. "
            "Use recent_turns only as dialogue context to resolve 'your first point', requests for "
            "rephrasing or a follow-up to the previous teaching explanation. Those turns are neither "
            "new patient facts nor learner evidence or grades; never copy them into the case. "
            "When reference_access.allowed is false, only the public question, already shown hints "
            "and public dialogue are available. Explain one requested concept or the shown hint only; "
            "do not solve the full question, assemble a complete key, add unseen scoring requirements "
            "or offer to bypass the three progressive hints. A request for the complete answer must "
            "use the reference-unlock flow. "
            "For hint_detail, explain ONLY the already-shown last_hint and its provided point, using "
            "short concrete reasoning or an answer-organizing suggestion. Do not give a new hint, "
            "introduce further required content, reveal the full answer, or infer the full key from "
            "general knowledge. A negated request for the answer must remain a hint explanation. "
            "For clarify, explain what this current question asks without answering it. "
            "For explain, answer the requested concept or rationale only; do not reproduce the full "
            "question's reference answer or require an unrelated complete care plan. State plans and "
            "conditional actions as plans and conditions, not accomplished clinical outcomes. "
            "If the saved basis does not support a requested detail, explicitly keep it uncertain. "
            "For social, respond naturally to the actual emotion, pacing, conversation or interface-use "
            "request in one to three short sentences; do not send the same generic message to every "
            "request. Use only the supplied interface controls and remaining hint count, never invent "
            "buttons, actions, patient facts or learner achievements. Respect requests to slow down; "
            "do not interpret an emotion alone as a clinical answer or as consent to end the exercise. "
            "For topics outside this exercise, briefly guide the learner to the normal Chat mode. "
            "Return only the reply field. Keep teaching explanations under 250 words and social replies "
            "to one to three sentences, without a grade or next-step decision. "
            "Return a self-contained explanation: never stop after an introduction, colon or a promise "
            "of points that are not supplied. If the answer is locked, finish one permitted conceptual "
            "explanation or explain how to approach the task without enumerating the full answer.")
        llm = self._llm()
        try:
            runner = ModelRunner(llm, RunBudget(seconds), log)
            payload = teaching_dialogue_payload(state, text, intent)
            raw = runner.call(prompt, payload, schema,
                "teaching_followup_dialogue", max_tokens=700)
            reply = raw["reply"].strip()
            if not reply:
                raise ValueError("The teaching reply is empty.")
            if incomplete_teaching_reply(reply):
                log.write("teaching_dialogue_incomplete", {"reply": reply, "action": "complete_once"})
                check_cancelled()
                raw = runner.call(prompt +
                    " The previous draft stopped before completing its explanation. Rewrite the whole "
                    "reply as a finished, useful explanation within the SAME teaching boundaries. "
                    "The unfinished draft is not new case evidence. Do not unlock or provide a full answer.",
                    {**payload, "unfinished_draft": reply}, schema,
                    "teaching_followup_completion", max_tokens=700)
                reply = raw["reply"].strip()
                if incomplete_teaching_reply(reply):
                    raise ValueError("教学讲解未写完整，请重新获取讲解。")
            reply, conflicts = guard_teacher_explanation(reply, state, state.get("language", "zh"))
            if conflicts:
                log.write("teaching_dialogue_boundary_guard", {"conflicts": conflicts, "action": "clarify_without_grading"})
                raise ValueError("教学解释与本题已保存边界冲突，请换一种方式提问或重试。")
            return reply
        finally:
            llm.close()

    def plan_chat(self, state, question, log, emit):
        from app.evidence_snapshot import ModelRunner, RunBudget
        from .tool_runtime import descriptions, plan_schema, from_calls
        from .workspace import aliases
        from .preferences import requested_mutations
        emit("status", "正在理解问题…")
        llm = self._llm()
        try:
            recent = chat_recent_turns(state)
            names = allowed_tools(state["policy"], state.get("attachments", []), state.get("memory_profile", ""))
            payload = {"question": question, "recent_turns": recent,
                "memory": state.get("chat_memory", {}).get("summary", "")[:3000],
                "available_tools": descriptions(names),
                "files": [{**f, "input_path": "/input/" + aliases(state.get("attachments", []))[f["file_id"]]}
                          for f in state.get("attachments", []) if f.get("active", True)],
                "generated_files": state.get("generated_files", []),
                "revision_target": state.get("revision_target", {}),
                "preferences": state.get("preferences", {}),
                "memory_profile": state.get("memory_profile", ""),
                "requested_mutations": requested_mutations(question),
                "current_clock": get_current_time.invoke({"timezone": state.get("timezone", DEFAULT_TIMEZONE)})}
            from .chat_planning import chat_planning_runner
            wrapper = CancellableClient if CURRENT_TURN.get() is not None else None
            runner = chat_planning_runner(self.config, ModelRunner(llm, RunBudget(60), log),
                                     cancel_check=check_cancelled, wrap_client=wrapper)
            schema = plan_schema(names)
            plan = runner.call(prompts.CHAT_TOOL_PLAN, payload, schema, "chat_tool_plan", max_tokens=900)
            selected = {call["name"] for call in plan.get("calls", [])}
            missing = [name for name in payload["requested_mutations"] if name in names and name not in selected]
            if payload["memory_profile"] and missing:
                # One clarification shares the original deadline. It still asks the model
                # for schema-valid, grounded arguments; no preference is manufactured here.
                emit("status", "正在核对明确要求的偏好修改…")
                repair = {"attempted": True, "missing_requested_mutations": missing}
                try:
                    revised = runner.call(prompts.CHAT_TOOL_PREFERENCE_REPLAN,
                        {**payload, "initial_plan": plan, "missing_requested_mutations": missing},
                        schema, "chat_preference_replan", max_tokens=900)
                    selected = {call["name"] for call in revised.get("calls", [])}
                    repair["remaining_missing_mutations"] = [name for name in missing if name not in selected]
                    plan = revised
                except Exception as exc:
                    check_cancelled()
                    repair.update(status="unavailable", error_type=type(exc).__name__)
                    log.write("chat_preference_replan_unavailable", repair)
                plan = {**plan, "preference_replan": repair}
            return from_calls(plan)
        finally:
            llm.close()

    def continue_chat(self, state, observations, log, emit, *, remaining_seconds=60):
        from app.evidence_snapshot import ModelRunner, RunBudget
        from .tool_runtime import descriptions, plan_schema, from_calls
        from .preferences import requested_mutations
        emit("status", "正在检查资料是否足够完成你的请求…")
        llm = self._llm()
        try:
            names = allowed_tools(state["policy"], state.get("attachments", []), state.get("memory_profile", ""))
            from .workspace import aliases
            question = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
            payload = {"question": question, "tool_results": observations[-8:], "available_tools": descriptions(names),
                       "recent_turns": chat_recent_turns(state),
                       "memory": state.get("chat_memory", {}).get("summary", "")[:3000],
                       "task_progress": state.get("chat_task_progress", {}),
                       "revision_target": state.get("revision_target", {}),
                       "files": [{**f, "input_path": "/input/" + aliases(state.get("attachments", []))[f["file_id"]]}
                                 for f in state.get("attachments", []) if f.get("active", True)],
                       "preferences": state.get("preferences", {}), "memory_profile": state.get("memory_profile", ""),
                       "requested_mutations": requested_mutations(question),
                       "current_clock": get_current_time.invoke({"timezone": state.get("timezone", DEFAULT_TIMEZONE)})}
            from .chat_planning import chat_planning_runner
            wrapper = CancellableClient if CURRENT_TURN.get() is not None else None
            runner = chat_planning_runner(self.config, ModelRunner(llm, RunBudget(min(60, remaining_seconds)), log),
                                     cancel_check=check_cancelled, wrap_client=wrapper)
            plan = runner.call(prompts.CHAT_TOOL_CONTINUE,
                payload, plan_schema(names), "chat_tool_continue", max_tokens=900)
            return from_calls(plan)
        finally:
            llm.close()

    def draft_document(self, state, request, runtime, log, emit, *, remaining_seconds=120, output_format="docx"):
        question = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
        payload = {"question": question, "writing_request": request, "document_format": output_format, "actual_tool_results": runtime,
                   "previous_turn_tool_results": state.get("tool_results", {}).get("tool_observations", [])[-6:],
                   "recent_turns": [{"role": m["role"], "content": without_reference_context(m["content"])}
                                    for m in state["messages"] if m["kind"] != "welcome"][-4:]}
        return self.text([{"role": "system", "content": "Write the FULL requested document in the user's language, not just a title or plan. Output content only; use Markdown headings for Word/text if useful. For XLSX/CSV output valid CSV rows only, no Markdown fences. For HTML output a complete HTML document. Use supplied facts and actual tool results. Do not invent having searched or generated files."},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], log,
                         lambda *_: None, max_tokens=4000, require_complete=True, request_timeout=remaining_seconds)

    def _chat_browser(self):
        from app.web_reader import Crawl4AIReader
        if self.config.get("web_search", "browser_python"):
            return Crawl4AIReader(self.config.path("web_search", "browser_python"), timeout=12)
        return None

    def web_sources(self, query, log, emit, freshness=""):
        return self.web_sources_for_request(query,query,log,emit,freshness=freshness)

    def web_sources_for_request(self, query, question, log, emit, *, freshness="", remaining_seconds=120):
        """Keep original constraints through chat's model-generated tool queries."""
        from .daily_tools import search_web
        from app.web_search import search_backends
        from app.search_planner import planner_settings
        emit("status", "正在搜索网页并读取相关内容…")
        planner=planner_settings(self.config)
        if planner['provider']!='deepseek':
            return search_web(query, log, config=self.config, backends=search_backends(self.config),
                freshness=freshness, browser_reader=self._chat_browser(), original_question=question)
        from app.evidence_snapshot import RunBudget,ModelRunner
        from app.web_search import create_web_search
        llm=self._llm()
        try:
            budget=RunBudget(max(1,min(120,remaining_seconds)))
            web=create_web_search(self.config,budget,log,ModelRunner(llm,budget,log),mode='live',
                max_search_rounds=planner['max_search_rounds'],freshness=freshness,
                cancel_check=check_cancelled,planner_client_wrapper=CancellableClient if CURRENT_TURN.get() is not None else None,
                browser_reader=self._chat_browser())
            from .chat_planning import recover_chat_runner
            web.runner = recover_chat_runner(self.config, web.runner)
            found=web.search(query,context={'question':question,'initial_search_query':query,'step':'chat_search'})
            return list(found.values()),web.events
        finally:
            llm.close()

    def sources_for_request(self, query, question, policy, log, emit):
        from app.evidence_constraints import request_constraints,eligibility,year_decision
        constraints=request_constraints(question)
        # An abbreviated tool query cannot silently change population, device,
        # publication bounds or the exact requested document.
        scoped=question if constraints['medical'] or constraints['years'] or constraints['title'] else query
        found,events=self.sources(scoped,policy,log,emit)
        accepted=[]
        for source in found:
            # Local scope is already admitted by the audited RAG, including
            # its general_mdrpi background lane. External heuristics must not
            # reinterpret that label as a conflicting specific device.
            decision=(eligibility(source,question) if source.get('origin')=='external'
                      else year_decision(source,constraints['years']))
            if decision['eligible']:accepted.append(source)
            else:events.append({'provider':'local','status':'original_constraint_rejected',**decision})
        return accepted,events

    def daily_sources(self, plan, log, emit, language="zh"):
        from .daily_tools import weather, read_page
        sources, events = [], []
        if "weather" in plan["tools"]:
            emit("status", "正在查询天气…")
            found, notices = weather(plan["location"], plan["forecast_days"], language, log=log)
            sources.extend(found); events.extend(notices)
        if "read_url" in plan["tools"]:
            emit("status", "正在读取指定网页…")
            for url in plan["urls"]:
                item, event = read_page(url, plan["query"], browser_reader=self._chat_browser())
                log.write("chat_page", {"source": item, "event": event})
                if item:
                    sources.append(item)
                events.append(event)
            if not plan["urls"]:
                events.append({"provider": "page", "status": "needs_url"})
        return sources, events

    def sources(self, question, policy, log, emit):
        if policy == "none":
            return [], [{"provider": "local", "status": "disabled"}]
        emit("status", "正在检索参考资料…")
        sources, events = [], []
        from .chat_tools import prepare_retrieval_query
        retrieval_query = prepare_retrieval_query(question)
        if retrieval_query["shortened"]:
            events.append({"provider": "local", "status": "query_shortened", **retrieval_query})
        try:
            with httpx.Client(trust_env=False, timeout=45) as client:
                response = client.post(self.rag_url + "/v1/retrieve", json={"query": retrieval_query["query"]})
                body = response.json()
                server_preparation = body.get("query_preparation")
                server_shortened = (isinstance(server_preparation, dict)
                                    and server_preparation.get("shortened") is True)
                if server_shortened:
                    events.append({**server_preparation, "provider": "local", "status": "query_shortened"})
                if response.status_code == 422:
                    detail = body.get("detail")
                    events.append({"provider": "local", "status": "input_rejected", "message":
                        detail[:200] if isinstance(detail, str) else "输入未通过检索接口校验，请检查长度和范围。"})
                log.write("retrieval", {"http_status": response.status_code, "result": body,
                                        "query_shortened": retrieval_query["shortened"] or server_shortened,
                                        "query_preparation": retrieval_query,
                                        "server_query_preparation": server_preparation})
                events.append({"provider": "local", "status": body.get("status", str(response.status_code))})
                if response.is_success:
                    sources.extend(body.get("evidence", []))
        except (httpx.HTTPError, ValueError) as exc:
            events.append({"provider": "local", "status": "unavailable", "error": str(exc)})
            log.write("retrieval_error", events[-1])
        if policy == "web":
            from app.evidence_snapshot import RunBudget, ModelRunner
            from app.web_search import create_web_search
            from app.web_reader import Crawl4AIReader
            llm = self._llm()
            try:
                emit("status", "正在搜索网页并读取正文…")
                budget = RunBudget(240)
                runner = ModelRunner(llm, budget, log)
                from .tasks import check_cancelled
                from app.search_planner import planner_settings
                planner = planner_settings(self.config)
                options = {"mode": "live", "max_search_rounds": planner["max_search_rounds"] if planner["provider"] == "deepseek" else 1,
                           "cancel_check": check_cancelled,
                           "planner_client_wrapper":CancellableClient if CURRENT_TURN.get() is not None else None}
                if self.config.get("web_search", "browser_python"):
                    options["browser_reader"] = Crawl4AIReader(self.config.path("web_search", "browser_python"), timeout=30)
                web = create_web_search(self.config, budget, log, runner, **options)
                evidence = web.search(question, context={"question": question, "step": "answer_verification",
                    "phase": "pre_draft", "local_sources": sources[:4]})
                # Reserve context for the newly fetched material even with a full local batch.
                sources = list(evidence.values())[:3] + sources
                events.extend(web.events)
            except Exception as exc:
                events.append({"provider": "web", "status": "unavailable", "error": str(exc)})
                log.write("web_error", events[-1])
            finally:
                llm.close()
        # Deduplicate excerpts while retaining multiple excerpts of one document.
        from app.evidence_constraints import deduplicate_documents
        sources=deduplicate_documents(sources,retain_local_passages=True)
        selected, seen = [], set()
        for s in sources:
            identity = s.get("evidence_id") or s.get("chunk_id") or s.get("text")
            if identity not in seen and s.get("text"):
                seen.add(identity)
                selected.append(s)
            if len(selected) >= 6:
                break
        log.write("reference_context", {"sources": selected, "events": events})
        return selected, events

    def text(self, messages, log, emit, max_tokens=1200, *, allow_trim=True, require_complete=False, request_timeout=600):
        llm = self._llm()
        started, content, finish, usage = time.monotonic(), "", None, None
        try:
            messages = template_messages(messages)
            if allow_trim:
                messages, trimmed = fit_messages(llm, messages, max_tokens)
            else:
                trimmed = 0
                if llm.count_tokens(messages)[0] + max_tokens + 128 > int(self.config.get("llm", "context_window", default=32768)):
                    raise ValueError("本轮问题与资料超出上下文，请缩短问题或移除部分附件；原会话已保留。")
            log.write("model_request", {"model": llm.model, "messages": messages,
                                       "max_tokens": max_tokens, "old_messages_omitted": trimmed})
            request = {"model": llm.model, "messages": messages, "temperature": 0,
                       "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
            with llm.client.stream("POST", llm.base_url + "/chat/completions", json=request, timeout=request_timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    usage = data.get("usage") or usage
                    for choice in data.get("choices", []):
                        token = choice.get("delta", {}).get("content") or ""
                        if token:
                            content += token
                            emit("token", token)
                        finish = choice.get("finish_reason") or finish
            if not content or finish not in {"stop", "length"}:
                raise RuntimeError("模型输出未完整返回。")
            log.write("model_response", {"text": content, "finish_reason": finish, "usage": usage,
                                         "seconds": time.monotonic() - started})
            if finish == "length":
                if require_complete:
                    raise RuntimeError("本次生成达到输出上限，未保存为完整结果；原输入已保留供重试。")
                content += "\n\n（本次输出达到长度上限，内容可能未完，可继续追问。）"
            return content
        except Exception as exc:
            log.write("model_error", {"message": str(exc), "partial_text": content})
            raise
        finally:
            llm.close()

    def chat(self, messages, sources, log, emit, runtime=None):
        system = self.chat_system(sources, runtime)
        if messages and messages[0]["role"] == "system":
            system += "\n" + messages[0]["content"]
            messages = messages[1:]
        stream = SourceStream(sources, emit)
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = self.text([{"role": "system", "content": system}] + messages, log, stream,
                         max_tokens=chat_output_budget(question), allow_trim=False)
        stream.finish()
        weather_numbers = [i for i, s in enumerate(sources, 1) if s.get("material_level") == "structured_weather"]
        if weather_numbers:
            cited = {int(n) for group in CITATION.findall(text) for n in re.findall(r"\d+", group)}
            missing = [i for i in weather_numbers if i not in cited]
            if missing:
                label = "天气数据来源" if select_language(messages[-1]["content"]) == "zh" else "Weather data source"
                text += "\n\n" + label + ": " + " ".join(f"[{i}]" for i in missing)
        return render_references(text, sources)

    def chat_system(self, sources, runtime=None):
        system = prompts.CHAT + "\nReference data:\n" + json.dumps(source_context(sources), ensure_ascii=False)
        if runtime:
            # Source text already appears above; arguments can contain full Python or documents.
            data = {k: v for k, v in runtime.items() if k not in {"tool_observations", "question", "workspace_result"}}
            data["tool_observations"] = [{**{k: v for k, v in obs.items() if k not in {"args", "sources", "data"}},
                                          "data": {k: v for k,v in obs.get("data", {}).items() if k != "preferences"}}
                                         for obs in runtime.get("tool_observations", [])[-8:]]
            system += "\nActual tool results and explicit preference data:\n" + json.dumps(data, ensure_ascii=False)
        return system

    def prepare_history(self, history, memory, sources, log, emit, *, runtime=None):
        llm = self._llm()
        cache = {}
        def count(messages):
            messages = template_messages(messages)
            key = json.dumps(messages, ensure_ascii=False)
            if key not in cache:
                cache[key] = llm.count_tokens(messages)[0]
            return cache[key]
        def generate(messages):
            emit("status", "对话较长，正在整理早期对话摘要；原文会保留…")
            return self.text(messages, log, lambda *_: None, max_tokens=900, allow_trim=False, require_complete=True)
        try:
            system = self.chat_system(sources, runtime)
            result, updated, event = compact(history, memory, count=count, generate=generate,
                context_window=int(self.config.get("llm", "context_window", default=32768)), system=system,
                trigger=int(self.config.get("interactive", "summary_trigger_tokens", default=18000)),
                target=int(self.config.get("interactive", "summary_target_tokens", default=8000)))
            log.write("context_memory", {**event, "memory": updated})
            return result, updated, event
        finally:
            llm.close()

    def prepare_chat_history(self, history, memory, sources, log, emit, runtime):
        return self.prepare_history(history, memory, sources, log, emit, runtime=runtime)

    def teaching_start(self, text, sources, log, *, generate=False, history=None, language="zh"):
        from app.evidence_snapshot import ModelRunner, RunBudget
        llm = self._llm()
        try:
            payload = {"request": text, "generate_requested": generate, "setup_history": history or [], "output_language": language,
                       "materials": source_context(sources)}
            return ModelRunner(llm, RunBudget(600), log).call(instruction(language) + prompts.TEACHING_START,
                payload, prompts.teaching_start_schema(), "teaching_start", max_tokens=1000)
        finally:
            llm.close()

    def assess(self, state, log):
        from app.evidence_snapshot import ModelRunner, RunBudget
        from .rubric import objective_assessment
        from .learner_evidence import (quote_catalog, resolve_evidence, check_named_concept,
                                       check_authored_term_slots, enforce_status_contract, check_concept_name_scope)
        check_cancelled()
        answers = student_answers(state)
        if not answers:
            raise ValueError("尚无学生作答，不执行评价。")
        language = state.get("language", "zh")
        state = {**state, "bank_item": assessment_item(state)}
        rubric = question_rubric(state)
        objective = objective_assessment(state["bank_item"], rubric, answers, language)
        if objective is not None:
            log.write("authored_objective_assessment", objective)
            return resolve_evidence(objective, quote_catalog(answers))
        llm = self._llm()
        try:
            payload = {"case": state["case"], "latest_answer": answers[-1], "earlier_answers": answers[:-1],
                       "tutor_context": tutor_context(state), "stage": state["stage"], "output_language": language}
            payload["sources"] = source_context(state.get("sources", []))
            payload["reference_answer"] = reference_context(state)
            catalog = quote_catalog(answers)
            payload["quote_catalog"] = catalog
            quote_ids = [row["id"] for row in catalog]
            payload["student_answers"] = [{"attempt": i, "text": answer} for i, answer in enumerate(answers, 1)]
            if rubric:
                payload["grading_rubric"] = rubric_context(rubric, language)
                raw = ModelRunner(llm, RunBudget(600), log).call(instruction(language) + prompts.RUBRIC_ASSESS, payload,
                    prompts.rubric_assessment_schema(rubric, len(answers), quote_ids), "teaching_rubric_feedback", max_tokens=1800)
                original = raw
                raw = enforce_status_contract(raw, rubric, catalog, state["case"], language)
                raw, named_changed = check_named_concept(raw, state.get("bank_item") or {}, rubric, catalog, language)
                raw = check_authored_term_slots(raw, rubric, catalog, language)
                raw = check_concept_name_scope(raw, state.get("bank_item") or {}, rubric, catalog, language)
                if named_changed:
                    log.write("named_concept_check", {"original": original, "checked": raw,
                                                       "basis": "paired_textbook_term_slot_not_clinical_reasoning"})
                correction = len(answers) > 1 and re.search(r"\b(?:i correct|correction|i was wrong|i revise|i amend)\b|(?:我|现在|这里)(?:要)?(?:更正|纠正|改正|修改)|(?:更正|纠正)[：:]", answers[-1], re.I)
                # Independently check at most two non-covered requirements,
                # including missing. All extra checks share one 180s deadline;
                # an unavailable extra check retains the grounded initial view.
                outstanding = [row for row in rubric["required"] if raw["criteria"][row["id"]]["status"] != "met"
                               and raw["criteria"][row["id"]].get("decision_origin") != "concept_name_only_scope"]
                if outstanding:
                    verification_budget = RunBudget(180)
                    # A small model may mix the name/function of a concept or
                    # misread negation. Check adverse judgements against the
                    # individual authored criterion, without its old verdict.
                    selected = sorted(outstanding,
                                      key=lambda row: raw["criteria"][row["id"]]["status"] != "incorrect")[:2]
                    for criterion in selected:
                        isolated = {**rubric, "required": [criterion]}
                        latest_only = bool(correction and raw["criteria"][criterion["id"]]["status"] == "incorrect")
                        verification_catalog = [row for row in catalog if row["attempt"] == len(answers)] if latest_only else catalog
                        verification_ids = [row["id"] for row in verification_catalog]
                        checked_payload = {"case": state["case"], "output_language": language,
                            "grading_rubric": rubric_context(isolated, language), "quote_catalog": verification_catalog,
                            "latest_answer": answers[-1], "checking_explicit_correction": latest_only,
                            "reference_answer": reference_context(state),
                            "student_answers": [{"attempt": i, "text": answer} for i, answer in enumerate(answers, 1)],
                            "current_task": state["case"]}
                        try:
                            check_cancelled()
                            checked = ModelRunner(llm, verification_budget, log).call(instruction(language) + prompts.CRITERION_VERIFY,
                                checked_payload, prompts.rubric_assessment_schema(isolated, len(answers), verification_ids),
                                "teaching_verify_" + criterion["id"], max_tokens=450)
                        except Exception as exc:
                            check_cancelled()
                            event = {"criterion": criterion["id"], "error_type": type(exc).__name__,
                                     "message": str(exc), "action": "retained_grounded_initial_assessment"}
                            raw.setdefault("verification_unavailable", []).append(event)
                            log.write("criterion_verification_unavailable", event)
                            break
                        check_cancelled()
                        log.write("criterion_verification_" + criterion["id"], {"original": raw["criteria"][criterion["id"]],
                                                                                 "verified": checked["criteria"][criterion["id"]],
                                                                                 "explicit_correction": bool(correction),
                                                                                 "latest_attempt_only": latest_only})
                        checked = enforce_status_contract(checked, isolated, verification_catalog, state["case"], language)
                        checked = check_authored_term_slots(checked, isolated, verification_catalog, language)
                        checked = check_concept_name_scope(checked, state.get("bank_item") or {}, rubric, verification_catalog, language)
                        raw.setdefault("status_contract_warnings", []).extend(checked.get("status_contract_warnings", []))
                        checked_row = checked["criteria"][criterion["id"]]
                        # An unrelated correction cannot erase another point's
                        # previous error. A relevant correction must be supported
                        # by the new learner wording, never by its older opposite.
                        if not latest_only or checked_row["status"] not in {"missing", "uncertain"}:
                            raw["criteria"][criterion["id"]] = checked_row
                            raw["criteria"][criterion["id"]].setdefault("decision_origin", "criterion_verification")
            else:
                raw = ModelRunner(llm, RunBudget(600), log).call(instruction(language) + prompts.FORMATIVE_ASSESS, payload,
                    prompts.assessment_schema(quote_ids), "teaching_feedback", max_tokens=2100)
            check_cancelled()
            return resolve_evidence(raw, catalog)
        finally:
            llm.close()

    def assess_step(self, state, log):
        from .clinical_flow import assess_step
        return assess_step(self, state, log)

    def summary(self, state, log, emit):
        from .rubric import objective_question_type
        from .question_bank import reference_answer_text
        if objective_question_type(assessment_item(state)):
            return reference_answer_text(state)
        from .teaching_access import all_reference_steps_accessible, reference_locked_text
        if not all_reference_steps_accessible(state):
            text = reference_locked_text(state)
            log.write("teacher_summary_reference_access", {"allowed": False, "action": "saved_review_only"})
            emit("token", text)
            return text
        from .learner_evidence import summary_boundaries, guard_teacher_explanation
        state = {**state, "bank_item": assessment_item(state)}
        # This model writes teacher guidance only. Neither raw grader narrative,
        # teacher hints nor review counts can become learner achievements.
        payload = {"case": state.get("case"), "summary_constraints": summary_boundaries(state)}
        payload["sources"] = source_context(state.get("sources", []))
        language = state.get("language", "zh")
        payload["output_language"] = language
        payload["reference_answer"] = reference_context(state)
        prompt = prompts.SUMMARY
        if state.get("clinical_flow"):
            flow = state["clinical_flow"]
            payload["step_contracts"] = flow.get("rubric_snapshot", {}).get("steps", [])
            if not payload["step_contracts"]:
                from .clinical_flow import current_rubric, STEP_IDS
                payload["step_contracts"] = [current_rubric({**flow, "current_step": i}, language)
                                               for i, _ in enumerate(STEP_IDS)]
            prompt = prompts.RUBRIC_SUMMARY
        elif state.get("reviews") and state["reviews"][-1].get("grading_mode") == "question_rubric":
            # Learner achievements are rendered from checked quotations by the
            # application. The prose model only writes the teacher's explanation.
            payload = {"case": state["case"], "reference_answer": reference_context(state),
                       "grading_rubric": rubric_context(question_rubric(state), language),
                       "sources": source_context(state.get("sources", [])), "output_language": language,
                       "summary_constraints": summary_boundaries(state)}
            prompt = prompts.RUBRIC_SUMMARY
        elif state.get("reviews") and state["reviews"][-1].get("grading_mode") == "case_formative":
            payload = {"case": state["case"], "sources": source_context(state.get("sources", [])), "output_language": language,
                       "summary_constraints": summary_boundaries(state)}
            prompt = prompts.FORMATIVE_SUMMARY
        # Do not publish unvalidated student-attribution or excluded-direction
        # prose as streaming tokens before the final boundary check.
        def checked_emit(kind, value):
            if kind not in {"token", "answer_text"}:
                emit(kind, value)
        stream = SourceStream(state.get("sources", []), checked_emit)
        text = self.text([{"role": "system", "content": instruction(language) + prompt},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], log, stream,
                         max_tokens=3200, require_complete=True)
        stream.finish()
        text, conflicts = guard_teacher_explanation(text, state, language)
        if conflicts:
            log.write("teacher_summary_boundary_guard", {"matched_excluded_directions": conflicts,
                                                        "action": "authored_reference_fallback_no_score_change"})
        if prompt in {prompts.RUBRIC_SUMMARY, prompts.FORMATIVE_SUMMARY}:
            text = ("### 教师讲解（与作答记录分开展示）\n\n" if language == "zh" else
                    "### Tutor explanation (separate from your answer record)\n\n") + text
        emit("token", text)
        return text

    def teaching_reference(self, state, log):
        """A cached teaching example for an unpublished task, never learner grading."""
        from .teaching_access import reference_access
        if not reference_access(state)["allowed"]:
            raise ValueError("参考答案尚未解锁，请先查看三条提示。")
        from .learner_evidence import summary_boundaries, guard_teacher_explanation
        language = state.get("language", "zh")
        flow = state.get("clinical_flow") or {}
        if flow:
            from .clinical_flow import current_question, current_rubric
            question, rubric = current_question(flow, language), current_rubric(flow, language)
        else:
            question, rubric = state["case"], question_rubric(state)
        payload = {"case": state["case"], "current_task": question,
                   "grading_rubric": rubric_context(rubric, language) if rubric else None,
                   "summary_constraints": summary_boundaries(state), "sources": source_context(state.get("sources", []))}
        prompt = ("Write a teaching example answering ONLY current_task, using the supplied case facts. "
                  "Use any supplied rubric to bound the answer. Explain the steps or reasoning in a short, usable structure. "
                  "Do not provide answers for other tasks or later clinical reasoning steps. Do not evaluate the learner, "
                  "claim they achieved anything, or use their words as patient facts. Unprovided findings and outcomes "
                  "remain unknown. Do not invent procedural parameters, source citations or a publisher's official answer. "
                  "State conditional actions as conditional rather than as accomplished outcomes.")
        text = self.text([{"role": "system", "content": instruction(language) + prompt},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                         log, lambda *_: None, max_tokens=1600, require_complete=True)
        text, conflicts = guard_teacher_explanation(text, state, language)
        if conflicts:
            raise ValueError("参考作答与本题已保存的安全边界冲突，请重试。")
        return render_references(text, state.get("sources", []), language)

    def teaching_support(self, state, log):
        """Localize help from an older/different-language review, without regrading."""
        from app.evidence_snapshot import ModelRunner, RunBudget
        llm = self._llm()
        try:
            payload = {"case": state["case"], "answers": student_answers(state), "review": state["reviews"][-1]}
            schema = {"type": "object", "additionalProperties": False, "properties": {
                "feedback": {"type": "string"}}, "required": ["feedback"]}
            prompt = instruction(state.get("language", "zh")) + "Translate the existing review into feedback explaining what the learner did well and what remains. Do not regrade, generate hints or ask another question. No full answer or internal phase labels."
            return ModelRunner(llm, RunBudget(300), log).call(prompt, payload, schema, "teaching_support", max_tokens=700)
        finally:
            llm.close()

    def teaching_hints(self, state, log):
        from app.evidence_snapshot import ModelRunner, RunBudget
        llm = self._llm()
        try:
            payload = {"case": state["case"], "answers": student_answers(state),
                       "review": state["reviews"][-1], "sources": source_context(state.get("sources", []))}
            return ModelRunner(llm, RunBudget(300), log).call(
                instruction(state.get("language", "zh")) + prompts.TEACHING_HINTS,
                payload, prompts.hints_schema(), "teaching_hints", max_tokens=600)
        finally:
            llm.close()

    def performance(self, query, web_mode, replay_run_id, log, emit):
        emit("status", "正在运行原性能测试流程，耗时可能较长…")
        payload = {"query": query, "web_mode": web_mode}
        from app.search_planner import planner_settings
        planning = planner_settings(self.config)
        if planning["provider"] == "deepseek":
            payload["search_planning"] = {k: planning[k] for k in (
                "provider", "model", "timeout_seconds", "fallback", "max_search_rounds")}
        if replay_run_id:
            payload["replay_run_id"] = replay_run_id
        log.write("performance_request", payload)
        with httpx.Client(trust_env=False, timeout=7500) as client:
            response = client.post(self.rag_url + "/v1/research/answer", json=payload)
            body = response.json()
            log.write("performance_result", {"http_status": response.status_code, "result": body})
            if not response.is_success:
                raise RuntimeError(f"原性能测试服务返回 HTTP {response.status_code}: "
                                   f"{body.get('status', body.get('detail', 'unknown'))}; run_id={body.get('run_id', '')}")
            return body
