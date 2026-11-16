"""Local Chainlit UI for ordinary chat and resumable teaching sessions."""
import asyncio
import json
import hashlib
import re
from http.cookies import SimpleCookie, CookieError
import secrets
from functools import wraps
from pathlib import Path
from queue import Queue, Empty
from uuid import uuid4

import chainlit as cl
from chainlit.input_widget import Select, TextInput

from modes.runtime import get_service
from modes.documents import MAX_BYTES
from modes.teaching import practice_catalog, practice_title
from modes.catalog import taxonomy, classic_filter_choices
from modes.language import localized, select_language
from modes.chat_tools import DEFAULT_TIMEZONE, source_notes
from modes.progress import ResponseProgress
from modes.learning import ACTION_LABELS, OUTCOME_LABELS, records, progress_text, learning_progress
from modes.learning_store import RATING_LABELS
from modes.teaching import student_answers

PROFILES = {"普通问答": "chat", "教学模式": "teaching"}
POLICIES = {"自动选择（推荐）": "auto", "模型知识": "none", "知识库": "local", "知识库＋网页": "web"}
LANGUAGES = {"跟随我的输入": "auto", "中文": "zh", "English": "en"}
STAGES = {"setup": "选择练习方式", "topic": "等待练习主题", "case": "等待案例", "answer": "等待首次作答", "probe": "等待补充回答",
          "hint": "等待根据提示修正", "reflect": "等待自我反思", "complete": "已总结", "chat": "可以继续提问"}


def classic_owner_id():
    owner = cl.user_session.get("learning_owner_id")
    if not owner:
        token = ""
        try:
            cookies = SimpleCookie()
            cookies.load(cl.context.session.environ.get("HTTP_COOKIE", ""))
            cookie = cookies.get("mdrpi_classic_learner")
            token = cookie.value if cookie else ""
        except (AttributeError, RuntimeError, LookupError, CookieError):
            pass
        persistent = bool(re.fullmatch(r"[a-f0-9]{64}", token))
        # A separate namespace cannot inherit the main webpage's learner code.
        owner = hashlib.sha256(("classic:" + token if persistent else uuid4().hex).encode()).hexdigest()
        cl.user_session.set("learning_owner_id", owner)
        cl.user_session.set("classic_owner_persistent", persistent)
    return owner


async def owned_state(sid):
    state = await asyncio.to_thread(get_service().get, sid)
    if state.get("mode") not in PROFILES.values():
        raise ValueError("此会话属于内部评测，请在工作台使用普通问答或教学练习。")
    expected = state.get("learning_owner_id", "")
    if expected and not secrets.compare_digest(expected, classic_owner_id()):
        raise ValueError("当前经典界面没有访问此学习会话或档案的权限。请使用创建记录的浏览器并保留站点数据；旧临时身份记录可查看此前导出的备份。新网页教学记录请通过原网页的访问码恢复。")
    return state


def owned_action(function):
    """Classic actions bypass the HTTP API, so enforce the same owner boundary."""
    @wraps(function)
    async def guarded(action):
        sid = action.payload.get("sid")
        if sid:
            try:
                await owned_state(sid)
            except (KeyError, ValueError) as exc:
                await cl.Message(content=str(exc)).send()
                return
        return await function(action)
    return guarded


async def root_message(content, **kwargs):
    item = cl.Message(content=content, **kwargs)
    # Restored messages must not inherit the /resume step removed by clear_display.
    item.parent_id = None
    return await item.send()


@cl.set_chat_profiles
async def profiles():
    return [
        cl.ChatProfile(name="普通问答", markdown_description="日常聊天、天气、实时资讯、网页阅读、计算与文件问答；按需联网。"),
        cl.ChatProfile(name="教学模式", markdown_description="自带案例、选练习或按主题出题 → 你先回答 → 追问/提示 → 反思 → 六步总结。"),
    ]


def reference_access(state):
    """Use the same access decision as the backend and main interface."""
    from modes.teaching_access import reference_access as teaching_reference_access
    return teaching_reference_access(state)


async def controls(state, anchor=None):
    language = state.get("language", "zh") if state["mode"] == "teaching" else "zh"
    flow = state.get("clinical_flow") or {}
    access = reference_access(state) if state["mode"] == "teaching" else {"allowed": False, "hints_seen": 0, "remaining": 0}
    catalog = cl.user_session.get("catalog_message")
    if catalog and state["stage"] not in {"setup", "case", "topic"}:
        await catalog.remove_actions()
        catalog.actions = []
        catalog.content = localized(language, "练习已选择。", "Exercise selected.")
        await catalog.update()
        cl.user_session.set("catalog_message", None)
    actions = [cl.Action(name="history", label="历史会话", payload={"offset": 0}),
               cl.Action(name="new_session", label="新建会话", payload={}),
               cl.Action(name="export", label="导出记录", payload={"sid": state["session_id"]})]
    active = any(row["session_id"] == state["session_id"] for row in get_service().activity())
    if active:
        if state["mode"] != "performance":
            actions.append(cl.Action(name="stop_turn", label="停止生成", payload={"sid": state["session_id"]}))
        actions.append(cl.Action(name="restore_session", label="刷新会话", payload={"sid": state["session_id"]}))
    elif state.get("pending"):
        actions.append(cl.Action(name="retry", label="重试本回合", payload={"sid": state["session_id"]}))
    elif state["mode"] == "teaching" and state["stage"] == "complete":
        choices = ["resume"] if state.get("practice_ended") else ["next", "similar", "review", "finish"]
        for operation in choices:
            actions.append(cl.Action(name="practice_continue", label=localized(language, *ACTION_LABELS[operation]),
                                     payload={"sid": state["session_id"], "operation": operation}))
    elif state["mode"] == "teaching" and flow:
        current = flow["steps"][flow["current_step"]]
        actions.append(cl.Action(name="summarize", label=localized(language, "结束学习并保留进度", "End and preserve progress"), payload={"sid": state["session_id"]}))
        if current["status"] == "awaiting_next":
            actions.append(cl.Action(name="clinical_next", label=localized(language, "进入下一步", "Next reasoning step"),
                                     payload={"sid": state["session_id"], "expected_step": flow["current_step"]}))
        else:
            if access["remaining"]:
                actions.append(cl.Action(name="tutor_help", label=localized(language, "下一条提示", "Next hint"),
                                         payload={"sid": state["session_id"], "operation": "hint"}))
            actions.append(cl.Action(name="tutor_help", label=localized(language, "澄清当前任务", "Clarify this task"),
                                     payload={"sid": state["session_id"], "operation": "clarify"}))
        if access["allowed"]:
            actions.append(cl.Action(name="tutor_help", label=localized(language, "查看本步参考答案", "View this step's reference"),
                                     payload={"sid": state["session_id"], "operation": "reference"}))
    elif state["mode"] == "teaching" and state.get("case") and state["stage"] not in {"setup", "case", "topic", "complete"}:
        from modes.rubric import objective_question_type
        objective = bool(objective_question_type(state.get("bank_item") or {}))
        if not objective or state["attempts"] or state.get("reference_views"):
            actions.append(cl.Action(name="summarize", label=localized(language, "结束练习并总结", "Finish and summarize"), payload={"sid": state["session_id"]}))
        for operation, label in [("hint", localized(language, "给我提示" if objective else "下一条提示", "Give me a hint" if objective else "Next hint")), ("feedback", localized(language, "评价我的回答", "Review my answer"))]:
            if (operation != "hint" or objective or access["remaining"]) and (operation != "feedback" or state["attempts"]):
                actions.append(cl.Action(name="tutor_help", label=label, payload={"sid": state["session_id"], "operation": operation}))
        if not objective and access["allowed"]:
            actions.append(cl.Action(name="tutor_help", label=localized(language, "查看参考答案", "View the reference answer"),
                                     payload={"sid": state["session_id"], "operation": "reference"}))
    elif state["mode"] == "teaching" and state["stage"] in {"setup", "case", "topic"}:
        for operation, label in [("choose_case", localized(language, "我提供案例", "Use my case")), ("library", localized(language, "使用练习题库", "Browse exercises")), ("choose_topic", localized(language, "按主题出题", "Create a case"))]:
            actions.append(cl.Action(name="teaching_entry", label=label, payload={"sid": state["session_id"], "operation": operation}))
    if state["mode"] == "teaching":
        from modes.rubric import objective_question_type
        if not active and not state.get("pending") and state.get("case") and state["stage"] not in {"setup", "case", "topic"} and not objective_question_type(state.get("bank_item") or {}):
            if state["stage"] == "complete" and access["allowed"]:
                actions.append(cl.Action(name="tutor_help", label=localized(language, "复盘参考答案", "Review the reference answer"),
                                         payload={"sid": state["session_id"], "operation": "reference"}))
            elif state["stage"] == "complete" and access["remaining"]:
                actions.append(cl.Action(name="tutor_help", label=localized(language, "下一条提示", "Next hint"),
                                         payload={"sid": state["session_id"], "operation": "hint"}))
            actions.append(cl.Action(name="tutor_help", label=localized(language, "解释这道题", "Explain this task"),
                                     payload={"sid": state["session_id"], "operation": "explain"}))
            shown_hints = flow["steps"][flow["current_step"]].get("hints", []) if flow else state.get("hint_events", [])
            if shown_hints:
                actions.append(cl.Action(name="tutor_help", label=localized(language, "把提示讲详细", "Explain the hint further"),
                                         payload={"sid": state["session_id"], "operation": "hint_detail"}))
            route = state.get("input_route", {})
            if state["stage"] != "complete" and route.get("confidence") == "uncertain" and route.get("reason") in {"uncertain_intent", "router_unavailable"} and route.get("raw_text"):
                for operation, labels in [("submit_answer", ("这是我的回答", "This is my answer")), ("explain", ("我是在求助", "I am asking for help"))]:
                    actions.append(cl.Action(name="route_confirm", label=localized(language, *labels),
                                             payload={"sid": state["session_id"], "operation": operation}))
        archive = state.get("learning_archive", {})
        if archive.get("saved") is False and archive.get("error"):
            if not active and not state.get("pending"):
                actions.append(cl.Action(name="retry_learning_save", label=localized(language, "重试保存档案", "Retry saving learning record"),
                                         payload={"sid": state["session_id"]}))
        actions.append(cl.Action(name="learning_progress", label=localized(language, "学习进度", "Learning progress"),
                                 payload={"sid": state["session_id"], "page": 0}))
        if not active and not state.get("pending") and state.get("reviews") and state.get("recheck_for_answer", -1) != len(student_answers(state)):
            actions.append(cl.Action(name="tutor_help", label=localized(language, "重新核对", "Recheck"),
                                     payload={"sid": state["session_id"], "operation": "recheck"}))
        if state.get("learning_profile"):
            actions.append(cl.Action(name="learning_profile", label=localized(language, "学习档案", "Learning profile"),
                                     payload={"sid": state["session_id"], "page": 0}))
            if not active and not state.get("pending") and state["stage"] in {"setup", "case", "topic", "complete"}:
                actions.append(cl.Action(name="practice_continue", label=localized(language, *ACTION_LABELS["due"]),
                                         payload={"sid": state["session_id"], "operation": "due"}))
            completed = not flow or flow.get("status") == "completed" and all(step["status"] == "passed" for step in flow["steps"])
            if completed and not active and not state.get("pending") and state["stage"] == "complete" and state.get("learning_archive", {}).get("saved") and state.get("exercise_learning_profile") == state["learning_profile"] and not state.get("learning_self_rating"):
                for rating, labels in RATING_LABELS.items():
                    actions.append(cl.Action(name="learning_rating", label=localized(language, "复习自评：", "Review rating: ") + localized(language, *labels),
                                             payload={"sid": state["session_id"], "rating": rating}))
    if state["mode"] == "chat" and state.get("memory_profile"):
        actions.append(cl.Action(name="preferences", label="查看偏好记忆", payload={"sid": state["session_id"]}))
    files = [f for f in state.get("attachments", []) if f.get("active", True)]
    for file in files:
        actions.append(cl.Action(name="remove_file", label="移除 " + file["name"], payload={"sid": state["session_id"], "file_id": file["file_id"]}))
    announced = cl.user_session.get("announced_sids", [])
    if state["session_id"] not in announced:
        sid = state["session_id"]
        if cl.user_session.get("classic_owner_persistent"):
            text = localized(language,
                f"会话编号：`{sid}`\n\n保留此浏览器的站点数据，切换到相同模式后可发送 `/resume {sid}` 恢复；`/new` 新建会话。离开前可导出记录备份。",
                f"Session ID: `{sid}`\n\nKeep this browser's site data and select the same mode, then use `/resume {sid}` to restore. Use `/new` for a new session. Export a backup before leaving.")
        else:
            text = localized(language,
                f"会话编号：`{sid}`\n\n当前连接使用临时身份，仅在此界面内可发送 `/resume {sid}` 恢复。刷新或关闭前请导出记录备份；`/new` 新建会话。",
                f"Session ID: `{sid}`\n\nThis connection has a temporary identity; `/resume {sid}` works within this interface only. Export a backup before refreshing or closing. Use `/new` for a new session.")
        await root_message(text)
        cl.user_session.set("announced_sids", announced + [sid])
    previous = cl.user_session.get("action_message")
    if previous:
        await previous.remove_actions()
        previous.actions = []
    progress_text = ""
    if flow:
        from modes.clinical_flow import STEP_TITLES
        current = flow["steps"][flow["current_step"]]
        titles = " → ".join(localized(language, *STEP_TITLES[step["id"]]) for step in flow["steps"])
        label = localized(language, *STEP_TITLES[current["id"]])
        progress_text = localized(language, f"六步进度 {flow['current_step'] + 1}/6 · {label}", f"Step {flow['current_step'] + 1}/6 · {label}") + "\n\n" + titles
    if state["mode"] == "teaching" and state.get("case") and state["stage"] not in {"setup", "case", "topic"} and (state["stage"] != "complete" or not access["allowed"]):
        from modes.rubric import objective_question_type
        if not objective_question_type(state.get("bank_item") or {}):
            note = localized(language, f"提示 {access['hints_seen']}/3 · 还可 {access['remaining']} 次；细化已有提示不占次数。", f"Hints {access['hints_seen']}/3 · {access['remaining']} left; expanding a hint uses no new opportunity.")
            if not access["allowed"]:
                note += localized(language, "看完三条提示后解锁参考答案。", "The reference answer unlocks after all three hints.")
            progress_text += ("\n\n" if progress_text else "") + note
    if state.get("learning_archive", {}).get("error") and not state.get("learning_archive", {}).get("saved"):
        progress_text += "\n\n" + localized(language, "学习档案保存失败；会话仍保留，可重试保存。", "The learning record was not saved; the session is retained and saving can be retried.")
    if anchor is None:
        anchor = await cl.Message(content=progress_text, actions=actions).send()
    else:
        anchor.actions = actions
        if progress_text:
            anchor.content += "\n\n" + progress_text
        await anchor.update()
    cl.user_session.set("action_message", anchor)


@cl.on_chat_start
async def start():
    mode = PROFILES.get(cl.user_session.get("chat_profile"), "chat")
    cl.user_session.set("mode", mode)
    classic_owner_id()
    settings = await show_settings(mode)
    cl.user_session.set("settings", settings)
    await new_session()


async def show_settings(mode, settings=None):
    settings = settings or {}
    widgets = []
    if mode == "chat":
        widgets.append(TextInput(id="timezone", label="时间显示时区（默认北京时间；也可填 UTC、Europe/London）", initial=settings.get("timezone", DEFAULT_TIMEZONE)))
        widgets.append(TextInput(id="memory_profile", label="偏好档案名称（可选；同名会话共用，只保存明确要求记住的内容）", initial=settings.get("memory_profile", "")))
    if mode == "teaching":
        selected = settings.get("response_language", "跟随我的输入")
        widgets.append(Select(id="response_language", label="教学回复语言", values=list(LANGUAGES), initial_index=list(LANGUAGES).index(selected)))
        widgets.append(TextInput(id="learning_profile", label="学习档案名称（可选；同名档案可跨会话续学和复习）", initial=settings.get("learning_profile", "")))
    if mode == "performance":
        widgets += [Select(id="web_mode", label="性能测试联网模式", values=["live", "off", "replay"], initial_index=["live", "off", "replay"].index(settings.get("web_mode", "live"))),
                    TextInput(id="replay_run_id", label="replay 原始 run_id", initial=settings.get("replay_run_id", ""))]
    return await cl.ChatSettings(widgets).send()


async def new_session():
    await clear_display()
    catalog = cl.user_session.get("catalog_message")
    if catalog:
        await catalog.remove_actions()
        cl.user_session.set("catalog_message", None)
    s = cl.user_session.get("settings") or {}
    state = await asyncio.to_thread(get_service().create, cl.user_session.get("mode"),
        POLICIES.get(s.get("references"), "none") if cl.user_session.get("mode") == "performance" else "web", s.get("web_mode", "live"), s.get("replay_run_id", ""), s.get("timezone", DEFAULT_TIMEZONE), LANGUAGES.get(s.get("response_language"), "auto"), s.get("memory_profile", ""), s.get("learning_profile", ""), learning_owner_id=classic_owner_id())
    cl.user_session.set("sid", state["session_id"])
    answer = await root_message(state["response"])
    await controls(state, answer)


async def clear_display():
    # UI messages only: the original SQLite conversations are never deleted.
    for item in cl.chat_context.get():
        await item.remove_actions()
        await item.remove()
    for key in ("catalog_message", "action_message", "history_message", "learning_message", "exercise_message", "learning_profile_message"):
        cl.user_session.set(key, None)


async def restore_session(sid):
    if cl.user_session.get("busy"):
        await cl.Message(content="请先停止或等待当前回合完成，再切换会话。").send()
        return
    try:
        state = await owned_state(sid)
    except (KeyError, ValueError) as exc:
        await cl.Message(content=str(exc)).send()
        return
    if state["mode"] != cl.user_session.get("mode"):
        await cl.Message(content="此编号属于另一模式，请先切换顶部模式，再恢复会话。").send()
        return
    displayed = await asyncio.to_thread(get_service().display_messages, sid)
    await clear_display()
    cl.user_session.set("sid", sid)
    restored = None
    for m in displayed:
        restored = await root_message(content=m["content"] if m["role"] == "user" else m.get("display_content", m["content"]),
            author=localized(state.get("language", "zh"), "学生", "Student") if m["role"] == "user" else "MDRPI",
            type="user_message" if m["role"] == "user" else "assistant_message")
    label = next(k for k, v in POLICIES.items() if v == state["policy"])
    settings = {"references": label, "web_mode": state["web_mode"], "replay_run_id": state["replay_run_id"],
                "timezone": state.get("timezone", DEFAULT_TIMEZONE),
                "memory_profile": state.get("memory_profile", ""),
                "learning_profile": state.get("learning_profile", ""),
                "response_language": next(k for k, v in LANGUAGES.items() if v == state.get("response_language", "auto"))}
    cl.user_session.set("settings", settings)
    await show_settings(state["mode"], settings)
    await root_message(f"已恢复会话。当前参考资料：{label}；{STAGES[state['stage']]}。")
    if state.get("error"):
        await cl.Message(content="上次未完成：" + state["error"]).send()
    await controls(state, restored)


async def request_stop(sid):
    await owned_state(sid)
    result = await asyncio.to_thread(get_service().cancel, sid)
    if result["status"] == "unsupported":
        await cl.Message(content="性能测试仍在研究服务中运行。本次停止功能只适用于普通问答和教学，可从历史会话查看测试结果。").send()
    elif result["status"] == "stopping":
        status = cl.user_session.get("running_status")
        if status:
            status.content = "正在停止，输入和已生成内容会保留。"
            await status.update()
    return result


@cl.on_stop
async def stop():
    sid = cl.user_session.get("sid")
    if sid:
        await request_stop(sid)


@cl.action_callback("stop_turn")
@owned_action
async def stop_turn(action):
    if action.payload["sid"] == cl.user_session.get("sid"):
        await request_stop(action.payload["sid"])


@cl.action_callback("new_session")
@owned_action
async def new_session_action(action):
    if not cl.user_session.get("busy"):
        await new_session()


@cl.action_callback("restore_session")
@owned_action
async def restore_session_action(action):
    await restore_session(action.payload["sid"])


@cl.action_callback("history")
@owned_action
async def history_action(action):
    if cl.user_session.get("busy"):
        await cl.Message(content="请先停止或等待当前回合完成，再浏览历史会话。").send()
        return
    result = await asyncio.to_thread(get_service().history, cl.user_session.get("mode"), action.payload.get("offset", 0), owner_id=classic_owner_id())
    labels = {"running": "处理中", "cancelled": "已停止", "incomplete": "待重试", "complete": "已保存"}
    actions = [cl.Action(name="restore_session", label=row["title"],
                        tooltip=f"{row['updated_at'][:16]} UTC · {labels[row['status']]}",
                        payload={"sid": row["session_id"]}) for row in result["items"]]
    offset, limit = result["offset"], result["limit"]
    if offset:
        actions.append(cl.Action(name="history", label="上一页", payload={"offset": max(0, offset - limit)}))
    if offset + limit < result["total"]:
        actions.append(cl.Action(name="history", label="下一页", payload={"offset": offset + limit}))
    content = (f"当前模式的历史会话：第 {offset + 1}–{offset + len(result['items'])} 条，共 {result['total']} 条。点击标题恢复。"
               if result["items"] else "当前模式暂无历史会话。发送过内容的会话会显示在这里。")
    previous = cl.user_session.get("history_message")
    if previous:
        await previous.remove_actions()
        previous.content, previous.actions = content, actions
        await previous.update()
    else:
        previous = await cl.Message(content=content, actions=actions).send()
        cl.user_session.set("history_message", previous)


@cl.on_settings_update
async def settings_update(settings):
    try:
        await owned_state(cl.user_session.get("sid"))
        state = await asyncio.to_thread(get_service().configure, cl.user_session.get("sid"),
            POLICIES.get(settings.get("references"), "none") if cl.user_session.get("mode") == "performance" else "web", settings.get("web_mode", "live"), settings.get("replay_run_id", ""), settings.get("timezone"), LANGUAGES.get(settings.get("response_language"), "auto"), settings.get("memory_profile", ""), settings.get("learning_profile", ""))
        cl.user_session.set("settings", settings)
        await cl.Message(content="设置已更新，将用于下次回答。").send()
        await controls(state)
    except ValueError as exc:
        await cl.Message(content=str(exc)).send()
        await show_settings(cl.user_session.get("mode"), cl.user_session.get("settings"))


async def run_turn(text="", action="send", **kwargs):
    if cl.user_session.get("busy"):
        await cl.Message(content="本回合正在处理，请稍候。").send()
        return
    cl.user_session.set("busy", True)
    queue = Queue()
    language = select_language(text)
    progress = ResponseProgress(language)
    element = cl.CustomElement(name="ResponseProgress", props=progress.props())
    status = cl.Message(content="", elements=[element])
    answer = cl.Message(content="")
    streamed = False
    task = None
    try:
        current = await owned_state(cl.user_session.get("sid"))
        if any(row["session_id"] == current["session_id"] for row in get_service().activity()):
            await cl.Message(content="此会话正在后台处理，可停止生成或稍后刷新会话。").send()
            return
        previous = current.get("language", "zh") if current["mode"] == "teaching" else select_language(
            next((m["content"] for m in reversed(current["messages"]) if m["role"] == "user"), ""))
        language = select_language(text, previous, current.get("response_language", "auto") if current["mode"] == "teaching" else "auto")
        progress.language = language
        progress.label = localized(language, "正在处理本轮请求", "Processing your request")
        element.props = progress.props()
        await status.send()
        cl.user_session.set("running_status", status)
        task = asyncio.create_task(asyncio.to_thread(get_service().turn,
            cl.user_session.get("sid"), text, action, lambda k, v: queue.put((k, v)), **kwargs))
        try:
            while not task.done() or not queue.empty():
                try:
                    kind, value = queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.05)
                    continue
                if kind in {"token", "answer_text"}:
                    if not streamed:
                        progress.phase("streaming")
                        element.props = progress.props()
                        await element.update()
                        await answer.send()
                        streamed = True
                    if kind == "answer_text":
                        answer.content = value
                        await answer.update()
                    else:
                        await answer.stream_token(value)
                elif kind == "status" and progress.phase(value):
                    element.props = progress.props()
                    await element.update()
        except asyncio.CancelledError:
            # Chainlit cancels the UI coroutine. Explicitly disconnect the
            # backend request, then let the worker save its stopped state.
            if current["mode"] == "performance":
                await request_stop(current["session_id"])
                return  # Research service keeps running; recover via history.
            while not task.done():
                result = await request_stop(current["session_id"])
                if result["status"] != "idle":
                    break
                await asyncio.sleep(0.01)
        state = await asyncio.shield(task)
        await status.remove_actions()
        status.actions = []
        status.content = ""
        progress.finish(error=bool(state.get("error")), cancelled=bool(state.get("cancelled")))
        element.props = progress.props()
        await element.update()
        if state.get("cancelled"):
            answer.content = state["response"] + ("\n\n" + state["partial_response"] if state.get("partial_response") else "")
            await (answer.update() if streamed else answer.send())
        elif state.get("error"):
            if streamed:
                answer.content += localized(language, "\n\n（生成中断，以上为未完成内容。）", "\n\n(Generation was interrupted; the text above is incomplete.)")
                await answer.update()
            status.content = localized(language, "本回合未完成，输入已保存。可点击重试。\n\n", "This turn is incomplete. Your input was saved; you can retry.\n\n") + state["error"]
            await status.update()
        else:
            answer.content = state["response"]
            if state["mode"] == "teaching":
                displayed = await asyncio.to_thread(get_service().display_messages, state["session_id"])
                answer.content = next((row.get("display_content", row["content"]) for row in reversed(displayed) if row.get("role") == "assistant"), answer.content)
            await (answer.update() if streamed else answer.send())
            notes = source_notes(state.get("source_events", []))
            if state.get("context_event", {}).get("status") == "compressed":
                notes.append("早期对话已整理为摘要，完整原文仍保留在会话记录中。")
            if state["mode"] != "chat" and not notes and state.get("sources_loaded") and not state.get("sources") and state["policy"] != "none" and not state.get("bank_item"):
                notes.append("本次检索未返回可用参考资料，回答使用模型知识。")
            if notes:
                status.content = "\n\n".join(dict.fromkeys(notes))
                await status.update()
        await controls(state, status if state.get("error") else answer)
    except Exception as exc:
        progress.finish(error=True)
        element.props = progress.props()
        await element.update()
        status.content = str(exc)
        await status.update()
    finally:
        cl.user_session.set("busy", False)
        cl.user_session.set("running_status", None)


@cl.on_message
async def on_message(msg: cl.Message):
    text = msg.content.strip()
    if cl.user_session.get("busy"):
        await cl.Message(content="本回合正在处理，请稍候。").send()
        return
    try:
        current = await owned_state(cl.user_session.get("sid"))
    except (KeyError, ValueError) as exc:
        await cl.Message(content=str(exc)).send()
        return
    if msg.elements:
        if text.startswith("/"):
            await cl.Message(content="请将文件与普通问题一起发送；会话命令请单独发送。").send()
            return
        uploaded = []
        cl.user_session.set("busy", True)
        try:
            for element in msg.elements:
                try:
                    def read_upload():
                        with Path(element.path).open("rb") as handle:
                            return handle.read(MAX_BYTES + 1)
                    data = await asyncio.to_thread(read_upload)
                    item = await asyncio.to_thread(get_service().upload, cl.user_session.get("sid"), element.name, data)
                    uploaded.append(item)
                    await cl.Message(content=f"已读取 {item['name']}（{item['chunks']} 个文字片段）。" +
                                     ("\n\n" + "\n".join(item["warnings"]) if item["warnings"] else "")).send()
                except (ValueError, OSError, TypeError) as exc:
                    await cl.Message(content=f"{element.name} 未读取：{exc}").send()
            if not uploaded:
                return
        finally:
            cl.user_session.set("busy", False)
        if not text:
            state = await asyncio.to_thread(get_service().get, cl.user_session.get("sid"))
            if state["mode"] == "teaching" and state["stage"] not in {"setup", "case", "topic"}:
                await cl.Message(content="文件已加入参考资料。请继续作答；仅上传文件不会计作一次学生回答。").send()
                await controls(state)
                return
            names = "、".join(item["name"] for item in uploaded)
            text = localized(state.get("language", "zh"), f"请根据刚上传的 {names} 准备一个练习案例，先让我作答。", f"Create a practice case from the uploaded {names}, then wait for my answer.") if cl.user_session.get("mode") == "teaching" else f"请概括刚上传的 {names} 的主要内容。"
    if text.startswith("/resume "):
        try:
            await restore_session(text.split(maxsplit=1)[1].strip())
        except (KeyError, ValueError) as exc:
            await cl.Message(content=str(exc)).send()
    elif text == "/new":
        await new_session()
    elif text == "/export":
        await export_session(cl.user_session.get("sid"))
    else:
        if current["mode"] == "teaching" and current["stage"] == "case":
            cl.user_session.set("busy", True)
            confirmed = False
            try:
                checked = await asyncio.to_thread(get_service().inspect_case, current["session_id"], text)
                # Replace the classic chat copy before any model/session turn.
                msg.content = checked["sanitized_text"]
                await msg.update()
                detail = f"检测并隐藏 {checked['redaction_count']} 项可能的身份信息。"
                if checked["missing_fields"]:
                    detail += "\n\n待补充信息：" + "、".join(checked["missing_fields"]) + "；这些内容会保持未知。"
                response = await cl.AskActionMessage(content=detail + "\n\n请确认以上案例已脱敏，仅用于教学。",
                    actions=[cl.Action(name="confirm_case_intake", label="已脱敏，确认导入", payload={"value": "confirm"}),
                             cl.Action(name="cancel_case_intake", label="取消导入", payload={"value": "cancel"})]).send()
                confirmed = bool(response and response.get("payload", {}).get("value") == "confirm")
            except (KeyError, ValueError) as exc:
                await cl.Message(content=str(exc)).send()
                return
            finally:
                cl.user_session.set("busy", False)
            if confirmed:
                await run_turn(checked["sanitized_text"], privacy_confirmed=True)
            else:
                await cl.Message(content="已取消导入；尚未开始训练。可重新填写脱敏案例。").send()
        else:
            await run_turn(text)


async def export_session(sid):
    state = await owned_state(sid)
    from modes.research import snapshot_metadata
    from modes.teaching_access import public_teaching_payload
    active = any(row["session_id"] == sid for row in get_service().activity())
    state["export_metadata"] = snapshot_metadata(state, active=active)
    data = json.dumps(public_teaching_payload(state), ensure_ascii=False, indent=2).encode("utf-8")
    suffix = "-snapshot" if not state["export_metadata"]["complete"] else ""
    label = "尚未完成回合的已保存快照：" if suffix else "完整会话记录（含学生原文、各轮反馈和总结）："
    await cl.Message(content=label,
        elements=[cl.File(name=f"mdrpi-{sid}{suffix}.json", content=data, display="inline")]).send()


@cl.action_callback("export")
@owned_action
async def export(action):
    await export_session(action.payload["sid"])


@cl.action_callback("retry_learning_save")
@owned_action
async def retry_learning_save(action):
    state = await asyncio.to_thread(get_service().retry_learning_archive, action.payload["sid"])
    await cl.Message(content="学习档案已保存。" if state.get("learning_archive", {}).get("saved") else "档案尚未保存，请查看错误提示或稍后重试。").send()
    await controls(state)


@cl.action_callback("preferences")
@owned_action
async def show_preferences(action):
    from modes.preferences import read
    state = await asyncio.to_thread(get_service().get, action.payload["sid"])
    data = await asyncio.to_thread(read, get_service().root, state.get("memory_profile", ""))
    content = "偏好档案：" + data["profile"] + "\n\n"
    content += "\n".join(f"- {item['key']}：{item['value']}" for item in data["items"]) or "尚未保存偏好。"
    content += "\n\n可以发送“记住……”“修改记住的偏好……”或“忘记……”。"
    actions = [cl.Action(name="forget_one_preference", label="删除：" + item["key"],
               payload={"sid": state["session_id"], "key": item["key"]}) for item in data["items"]]
    if data["items"]:
        actions.append(cl.Action(name="clear_preferences", label="清空此档案偏好", payload={"sid": state["session_id"]}))
    await cl.Message(content=content, actions=actions).send()


@cl.action_callback("forget_one_preference")
@owned_action
async def forget_one_preference(action):
    from modes.preferences import update
    state = await asyncio.to_thread(get_service().get, action.payload["sid"])
    await asyncio.to_thread(update, get_service().root, state["memory_profile"], action.payload["key"])
    await cl.Message(content="已删除该偏好，后续回答会重新读取档案。").send()


@cl.action_callback("clear_preferences")
@owned_action
async def clear_preferences(action):
    from modes.preferences import clear
    state = await asyncio.to_thread(get_service().get, action.payload["sid"])
    await asyncio.to_thread(clear, get_service().root, state["memory_profile"])
    await cl.Message(content="已清空此档案偏好。完整会话历史仍可查看。").send()


@cl.action_callback("retry")
@owned_action
async def retry(action):
    if action.payload["sid"] == cl.user_session.get("sid"):
        await run_turn(action="retry")


@cl.action_callback("summarize")
@owned_action
async def summarize(action):
    if action.payload["sid"] == cl.user_session.get("sid"):
        await run_turn(action="summary")


@cl.action_callback("clinical_next")
@owned_action
async def clinical_next(action):
    if action.payload["sid"] == cl.user_session.get("sid"):
        await run_turn(action="next_step", expected_step=action.payload["expected_step"])


@cl.action_callback("tutor_help")
@owned_action
async def tutor_help(action):
    if action.payload["sid"] == cl.user_session.get("sid"):
        await run_turn(action=action.payload["operation"])


@cl.action_callback("route_confirm")
@owned_action
async def route_confirm(action):
    state = await owned_state(action.payload["sid"])
    route = state.get("input_route", {})
    if (state["session_id"] != cl.user_session.get("sid") or state["stage"] == "complete"
            or route.get("confidence") != "uncertain" or route.get("reason") not in {"uncertain_intent", "router_unavailable"}
            or not route.get("raw_text") or action.payload["operation"] not in {"submit_answer", "explain"}):
        return
    await run_turn(route["raw_text"], action=action.payload["operation"])


@cl.action_callback("teaching_entry")
@owned_action
async def teaching_entry(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    if action.payload["operation"] == "library":
        state = await asyncio.to_thread(get_service().get, action.payload["sid"])
        if state["stage"] not in {"setup", "case", "topic"}:
            return
        language = state.get("language", "zh")
        await show_catalog(localized(language,
            "选择练习模块。基础知识按教育主题、难度选题；MDRPI专项按人群、器械、难度选题。教材题保留原书答案。",
            "Choose a practice module. Browse foundational knowledge by education topic and difficulty, or MDRPI practice by population, device and difficulty. Textbook exercises retain the original answer keys."), [
            cl.Action(name="practice_browse", label=localized(language, title, "MDRPI case practice" if key == "specialty" else "Foundational knowledge practice") + f" ({len(practice_catalog(module=key))})",
                      payload={"sid": action.payload["sid"], "module": key, "filter_stage": "population" if key == "specialty" else "learning_topic", "page": 0})
            for key, title in taxonomy()["modules"].items()])
    else:
        await run_turn(action=action.payload["operation"])


async def show_catalog(content, actions):
    catalog = cl.user_session.get("catalog_message")
    if catalog:
        await catalog.remove_actions()
        catalog.content, catalog.actions = content, actions
        await catalog.update()
    else:
        catalog = await cl.Message(content=content, actions=actions).send()
        cl.user_session.set("catalog_message", catalog)


@cl.action_callback("practice_browse")
@owned_action
async def practice_browse(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    state = await asyncio.to_thread(get_service().get, action.payload["sid"])
    if state["stage"] not in {"setup", "case", "topic"}:
        return
    language = state.get("language", "zh")
    payload = action.payload
    if payload.get("module") and payload.get("filter_stage") in {"learning_topic", "population", "device", "difficulty"}:
        choices = classic_filter_choices(practice_catalog(), payload)
        def choice_label(choice):
            label = choice["label"]
            if label == "全部":
                label = localized(language, label, "All")
            elif payload["filter_stage"] == "learning_topic":
                key = choice["payload"].get("learning_topic", "")
                label = localized(language, label, taxonomy().get("foundation_topics_en", {}).get(key, label))
            elif payload["filter_stage"] == "population":
                label = localized(language, label, {"adult": "Adults", "pediatric": "Children", "neonatal": "Neonates"}.get(choice["payload"].get("population"), label))
            elif payload["filter_stage"] == "difficulty":
                label = localized(language, label, {"基础": "Basic", "进阶": "Intermediate", "复杂": "Complex"}.get(label, label))
            return label + (f" ({choice['count']})" if choice["count"] is not None else "")
        actions = [cl.Action(name="practice_browse", label=choice_label(choice), payload=choice["payload"]) for choice in choices]
        actions.append(cl.Action(name="teaching_entry", label=localized(language, "返回题库方向", "Practice modules"), payload={"sid": state["session_id"], "operation": "library"}))
        stage = payload["filter_stage"]
        titles = {"learning_topic": ("1 · 选择教育主题", "1 · Choose an education topic"),
                  "population": ("1 · 选择人群", "1 · Choose a population"),
                  "device": ("2 · 选择医疗器械", "2 · Choose a medical device"),
                  "difficulty": (("2 · 选择题目难度", "2 · Choose exercise difficulty") if payload["module"] == "foundation"
                                 else ("3 · 选择案例难度", "3 · Choose case difficulty"))}
        content = localized(language, *titles[stage])
        if stage == "learning_topic":
            content += "\n\n" + localized(language,
                "按主题选择已有练习，再选择难度。题数对应当前筛选条件。",
                "Choose a topic with available exercises, then select the difficulty. Counts reflect the current filters.")
        elif stage == "device" and len(choices) == 1:
            content += "\n\n" + localized(language, "当前人群暂无已分类的器械练习，可返回调整筛选。",
                "No classified device exercises are available for this population. Return to adjust the filters.")
        await show_catalog(content, actions)
        return
    rows = practice_catalog(payload.get("category"), module=payload.get("module"), population=payload.get("population"), device_type=payload.get("device_type"), difficulty=payload.get("difficulty"), learning_topic=payload.get("learning_topic"))
    page = max(0, int(action.payload.get("page", 0)))
    actions = [cl.Action(name="practice_case", label=practice_title(item, language),
                        payload={"sid": state["session_id"], "case_id": item["id"]}) for item in rows[page * 8:(page + 1) * 8]]
    for new_page, zh, en in [(page - 1, "上一页", "Previous"), (page + 1, "下一页", "Next")]:
        if 0 <= new_page * 8 < len(rows):
            actions.append(cl.Action(name="practice_browse", label=localized(language, zh, en),
                                     payload={**action.payload, "page": new_page}))
    actions.append(cl.Action(name="teaching_entry", label=localized(language, "返回题库模块", "Practice modules"),
                             payload={"sid": state["session_id"], "operation": "library"}))
    title = localized(language, "练习列表", "Exercises")
    if payload.get("module") == "foundation":
        key = payload.get("learning_topic", "")
        title = localized(language, "基础知识练习", "Foundational knowledge practice")
        if key:
            zh = taxonomy()["foundation_topics"].get(key, "相关护理知识（全部）" if key == "related_nursing" else "待分类")
            en = taxonomy().get("foundation_topics_en", {}).get(key, "Related nursing knowledge (all)" if key == "related_nursing" else zh)
            title += " · " + localized(language, zh, en)
    elif payload.get("module") == "specialty":
        title = localized(language, "MDRPI专项案例训练", "MDRPI case practice")
    content = title + f" ({len(rows)})\n\n" + localized(language,
        "请选择一道练习；短题只需回答题目要求。" if rows else "当前筛选暂无练习，可返回调整条件。",
        "Choose an exercise. For a short question, answer only what it asks." if rows else "No exercise matches these filters. Return to adjust the selection.")
    await show_catalog(content, actions)


@cl.action_callback("practice_case")
@owned_action
async def practice_case(action):
    if action.payload["sid"] == cl.user_session.get("sid"):
        await run_turn(action.payload["case_id"], "library_case")


@cl.action_callback("practice_continue")
@owned_action
async def practice_continue(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    cl.user_session.set("busy", True)
    try:
        state = await asyncio.to_thread(get_service().learning_action, action.payload["sid"],
            action.payload["operation"], action.payload.get("exercise_id", ""))
        for key in ("learning_message", "exercise_message"):
            previous = cl.user_session.get(key)
            if previous:
                await previous.remove_actions()
                cl.user_session.set(key, None)
        answer = await cl.Message(content=state["response"]).send()
        await controls(state, answer)
    except (KeyError, ValueError) as exc:
        await cl.Message(content=str(exc)).send()
    finally:
        cl.user_session.set("busy", False)


@cl.action_callback("learning_progress")
@owned_action
async def learning_progress_action(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    state = await asyncio.to_thread(get_service().get, action.payload["sid"])
    progress = learning_progress(state)
    language = progress["language"]
    page = max(0, int(action.payload.get("page", 0)))
    rows = progress["exercises"]
    actions = []
    details = []
    for index, row in enumerate(rows[page * 8:(page + 1) * 8], page * 8 + 1):
        label = localized(language, *OUTCOME_LABELS[row["outcome"]])
        if row.get("superseded_by"):
            latest_label = localized(language, *OUTCOME_LABELS[row["latest_outcome"]])
            label = localized(language, f"当轮评价：{label}；最新评价：{latest_label}",
                              f"Previous round assessment: {label}; latest assessment: {latest_label}")
        details.append(f"{index}. {row['title']} — {label}")
        if row["gaps"]:
            prefix = (localized(language, "当轮遗留：", "Points remaining in that round: ") if row.get("superseded_by") else
                      localized(language, "待复习：", "To review: "))
            details.append(prefix + "; ".join(row["gaps"]))
        actions.append(cl.Action(name="view_exercise", label=localized(language, f"查看第 {index} 轮", f"View round {index}"),
                                 payload={"sid": state["session_id"], "exercise_id": row["exercise_id"]}))
    for next_page, zh, en in [(page - 1, "前一页记录", "Previous records"), (page + 1, "后一页记录", "More records")]:
        if 0 <= next_page * 8 < len(rows):
            actions.append(cl.Action(name="learning_progress", label=localized(language, zh, en),
                                     payload={"sid": state["session_id"], "page": next_page}))
    content = localized(language, "### 学习进度", "### Learning progress") + "\n\n" + progress_text(progress)
    if details:
        content += "\n\n" + "\n\n".join(details)
    previous = cl.user_session.get("learning_message")
    if previous:
        await previous.remove_actions()
        previous.content, previous.actions = content, actions
        await previous.update()
    else:
        previous = await cl.Message(content=content, actions=actions).send()
        cl.user_session.set("learning_message", previous)


@cl.action_callback("view_exercise")
@owned_action
async def view_exercise(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    state = await asyncio.to_thread(get_service().get, action.payload["sid"])
    record = next((r for r in records(state) if r["exercise_id"] == action.payload["exercise_id"]), None)
    if record is None:
        return
    language = state.get("language", "zh")
    # Show the original text, not another generated account of learner progress.
    parts = [localized(language, "### 已保存的练习记录", "### Saved exercise record"), record["case"]]
    for i, answer in enumerate(record.get("attempts", []), 1):
        parts.append(localized(language, f"**第 {i} 次学生原文**", f"**Learner answer {i}**") + "\n\n" + answer)
    parts.append(record.get("summary") or record.get("last_feedback") or
                 localized(language, "本题还没有评价记录。", "There is no assessment for this exercise yet."))
    actions = []
    if record["stage"] == "complete" and state["stage"] == "complete" and not state.get("pending"):
        actions.append(cl.Action(name="practice_continue", label=localized(language, *ACTION_LABELS["retry_exercise"]),
            payload={"sid": state["session_id"], "operation": "retry_exercise", "exercise_id": record["exercise_id"]}))
    previous = cl.user_session.get("exercise_message")
    if previous:
        await previous.remove_actions()
    current = await cl.Message(content="\n\n".join(parts), actions=actions).send()
    cl.user_session.set("exercise_message", current)


@cl.action_callback("learning_profile")
@owned_action
async def learning_profile_action(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    try:
        state = await asyncio.to_thread(get_service().get, action.payload["sid"])
        progress = await asyncio.to_thread(get_service().learning_profile_progress, action.payload["sid"])
        language = state.get("language", "zh")
        page = max(0, int(action.payload.get("page", 0)))
        rows = progress["questions"]
        parts = [localized(language, "### 学习档案", "### Learning profile"), progress["profile"],
                 localized(language,
                    f"已保存 {progress['completed_rounds']} 轮，涉及 {progress['unique_questions']} 道题；{progress['due_count']} 道到期复习，{progress['unrated_count']} 道尚未自评。",
                    f"Saved {progress['completed_rounds']} rounds across {progress['unique_questions']} questions; {progress['due_count']} due, {progress['unrated_count']} not self-rated."),
                 localized(language, "复习时间根据你的自评安排；题目评价与提示使用分别记录。",
                           "Review dates follow your own ratings; answer assessments and hint use are recorded separately.")]
        actions = []
        for index, row in enumerate(rows[page * 8:(page + 1) * 8], page * 8 + 1):
            description = f"{index}. {row['title']} — " + localized(language, *OUTCOME_LABELS[row["outcome"]])
            if row["gaps"]:
                description += "\n\n" + localized(language, "待复习要点：", "Points to review: ") + "; ".join(row["gaps"])
            assistance = row.get("assistance", {})
            if assistance.get("hints_seen"):
                description += "\n\n" + localized(language, f"本轮查看了 {assistance['hints_seen']} 条提示。", f"Viewed {assistance['hints_seen']} hints in this round.")
            description += "\n\n" + (localized(language, "下次复习：", "Next review: ") + row["next_due"][:16] + " UTC" if row["next_due"] else
                                       localized(language, "尚未安排复习；完成题目后可选择复习自评。", "No review scheduled; choose a review rating after completing the question."))
            parts.append(description)
            actions.append(cl.Action(name="restore_session", label=localized(language, f"查看原会话 {index}", f"View source session {index}"),
                                     payload={"sid": row["source_session_id"]}))
        for next_page, zh, en in [(page - 1, "前一页档案", "Previous profile records"), (page + 1, "后一页档案", "More profile records")]:
            if 0 <= next_page * 8 < len(rows):
                actions.append(cl.Action(name="learning_profile", label=localized(language, zh, en),
                                         payload={"sid": state["session_id"], "page": next_page}))
        actions.extend([
            cl.Action(name="learning_profile_export", label=localized(language, "导出学习档案", "Export learning profile"), payload={"sid": state["session_id"]}),
            cl.Action(name="learning_profile_clear", label=localized(language, "清空学习档案", "Clear learning profile"), payload={"sid": state["session_id"]}),
        ])
        previous = cl.user_session.get("learning_profile_message")
        if previous:
            await previous.remove_actions()
            previous.content, previous.actions = "\n\n".join(parts), actions
            await previous.update()
        else:
            previous = await cl.Message(content="\n\n".join(parts), actions=actions).send()
            cl.user_session.set("learning_profile_message", previous)
    except (KeyError, ValueError) as exc:
        await cl.Message(content=str(exc)).send()


@cl.action_callback("learning_rating")
@owned_action
async def learning_rating_action(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    try:
        result = await asyncio.to_thread(get_service().learning_self_rate, action.payload["sid"], action.payload["rating"])
        state = await asyncio.to_thread(get_service().get, action.payload["sid"])
        language = state.get("language", "zh")
        await cl.Message(content=localized(language, "已保存你的复习自评，下次复习：", "Saved your review rating. Next review: ") + result["next_due"][:16] + " UTC").send()
        await controls(state)
    except (KeyError, ValueError) as exc:
        await cl.Message(content=str(exc)).send()


@cl.action_callback("learning_profile_export")
@owned_action
async def learning_profile_export_action(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    try:
        data = await asyncio.to_thread(get_service().learning_profile_export, action.payload["sid"])
        from modes.teaching_access import public_teaching_payload
        encoded = json.dumps(public_teaching_payload(data), ensure_ascii=False, indent=2).encode()
        await cl.Message(content="", elements=[cl.File(name="learning-profile.json", content=encoded, display="inline")]).send()
    except (KeyError, ValueError) as exc:
        await cl.Message(content=str(exc)).send()


@cl.action_callback("learning_profile_clear")
@owned_action
async def learning_profile_clear_action(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    try:
        await asyncio.to_thread(get_service().clear_learning_profile, action.payload["sid"])
        state = await asyncio.to_thread(get_service().get, action.payload["sid"])
        await cl.Message(content=localized(state.get("language", "zh"), "学习档案已清空，原会话历史仍保留。新的练习可以重新建立复习记录。",
                                          "Learning profile cleared. Original session history remains; new exercises can build a fresh review record.")).send()
        await controls(state)
        await learning_profile_action(action)
    except (KeyError, ValueError) as exc:
        await cl.Message(content=str(exc)).send()


@cl.action_callback("remove_file")
@owned_action
async def remove_file(action):
    if action.payload["sid"] != cl.user_session.get("sid") or cl.user_session.get("busy"):
        return
    try:
        await asyncio.to_thread(get_service().remove_file, action.payload["sid"], action.payload["file_id"])
        await cl.Message(content="该文件不再作为后续回答的参考，之前的回答和已保存记录仍保留。").send()
    except (KeyError, ValueError) as exc:
        await cl.Message(content=str(exc)).send()
