"""Copy the context at an edited user message; leave the original graph untouched."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

from .interface import InterfaceStore, conversation_title
from .workspace import artifact_path, list_artifacts


def fork_chat(engine, sid, message_at, edited_text):
    state = engine.get(sid)
    if state["mode"] != "chat":
        raise ValueError("对话分支仅用于普通问答；教学更正需作为新的学生作答提交。")
    if not edited_text.strip() or len(edited_text) > 12000:
        raise ValueError("修改内容需为1至12000字符。")
    if (state.get("pending") and not state.get("cancelled")) or any(row["session_id"] == sid for row in engine.activity()):
        raise ValueError("请在本轮结束或停止后修改消息。")
    index = next((i for i, row in enumerate(state["messages"]) if row.get("at") == message_at and row["role"] == "user" and row.get("kind") == "user"), None)
    if index is None:
        raise KeyError("未找到可修改的用户消息。")
    boundary = None
    for checkpoint in engine.graph.get_state_history(engine._config(sid)):
        rows = checkpoint.values.get("messages", [])
        if len(rows) == index + 1 and rows[-1].get("at") == message_at:
            boundary = checkpoint.values
            break
    if boundary is None:
        raise ValueError("缺少该消息的上下文快照，无法创建准确的分支。")
    created = engine.create("chat", policy=boundary["policy"], web_mode=boundary["web_mode"],
        replay_run_id=boundary.get("replay_run_id", ""), timezone=boundary.get("timezone", "Asia/Shanghai"),
        response_language=boundary.get("response_language", "auto"), memory_profile=boundary.get("memory_profile", ""))
    new_sid = created["session_id"]
    # File selections are taken from the accepted message checkpoint, not today's list.
    attachments = deepcopy(boundary.get("attachments", []))
    for file in attachments:
        shutil.copytree(engine.root / "files" / sid / file["file_id"], engine.root / "files" / new_sid / file["file_id"])
    cutoff = datetime.fromisoformat(message_at).timestamp()
    for artifact in list_artifacts(engine.root, sid):
        source = artifact_path(engine.root, sid, artifact["artifact_id"])
        if source.stat().st_mtime > cutoff:
            continue
        target = engine.root / "artifacts" / new_sid / artifact["artifact_id"]
        shutil.copytree(source.parent, target)
        record = json.loads((target / "metadata.json").read_text())
        record["url"] = f"/api/sessions/{new_sid}/artifacts/{artifact['artifact_id']}"
        (target / "metadata.json").write_text(json.dumps(record, ensure_ascii=False))
    shown = engine.display_messages(sid)[:index]
    for row in shown:
        row["content"] = row["content"].replace(f"/api/sessions/{sid}/artifacts/", f"/api/sessions/{new_sid}/artifacts/")
    origin = {"source_session_id": sid, "root_session_id": (state.get("branch_origin") or {}).get("root_session_id", sid),
              "message_at": message_at, "source_text": state["messages"][index]["content"], "edited_text": edited_text.strip()}
    engine.graph.update_state(engine._config(new_sid), {"messages": shown, "attachments": attachments,
        "chat_memory": {}, "context_event": {}, "branch_origin": origin,
        "response": shown[-1]["content"] if shown else "", "updated_at": created["updated_at"]}, as_node="summarize")
    store = InterfaceStore(engine.root)
    store.link_branch(new_sid, origin)
    store.update(new_sid, title=conversation_title(state)[:100] + " · 修订")
    return engine.get(new_sid)
