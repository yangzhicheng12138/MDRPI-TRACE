"""Presentation metadata and read-only conversation views for the chat frontend."""
from contextlib import contextmanager
import sqlite3


class InterfaceStore:
    def __init__(self, root):
        self.path = root / "interface.sqlite"

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=20)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',
                pinned INTEGER NOT NULL DEFAULT 0, archived INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS branches (
                session_id TEXT PRIMARY KEY, root_session_id TEXT NOT NULL,
                source_session_id TEXT NOT NULL, message_at TEXT NOT NULL)""")
            with db:
                yield db
        finally:
            db.close()

    def all(self):
        with self.connection() as db:
            return {row["session_id"]: dict(row) for row in db.execute("SELECT * FROM conversations")}

    def update(self, sid, **fields):
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO conversations(session_id) VALUES(?)", (sid,))
            for name, value in fields.items():
                if name not in {"title", "pinned", "archived"} or value is None:
                    continue
                value = value.strip() if name == "title" else int(value)
                db.execute(f"UPDATE conversations SET {name}=? WHERE session_id=?", (value, sid))
            return dict(db.execute("SELECT * FROM conversations WHERE session_id=?", (sid,)).fetchone())

    def link_branch(self, sid, origin):
        with self.connection() as db:
            db.execute("INSERT INTO branches VALUES(?,?,?,?)", (sid, origin["root_session_id"],
                       origin["source_session_id"], origin["message_at"]))

    def branch_family(self, sid):
        with self.connection() as db:
            parent = db.execute("SELECT * FROM branches WHERE session_id=?", (sid,)).fetchone()
            root = parent["root_session_id"] if parent else sid
            rows = [dict(r) for r in db.execute("SELECT b.*,c.title FROM branches b LEFT JOIN conversations c ON c.session_id=b.session_id WHERE b.root_session_id=? ORDER BY b.rowid", (root,))]
            if not rows:
                return {"root_session_id": root, "items": []}
            title = db.execute("SELECT title FROM conversations WHERE session_id=?", (root,)).fetchone()
            return {"root_session_id": root, "items": [{"session_id": root, "title": (title[0] if title else "") or "原始对话"},
                *[{**r, "title": r["title"] or f"修订 {i}"} for i, r in enumerate(rows, 1)]]}


def conversation_title(state, metadata=None):
    if metadata and metadata.get("title"):
        return metadata["title"]
    first = (state.get("practice_history") or [state])[0]
    item = first.get("bank_item") or {}
    if item:
        from .teaching import practice_title
        return practice_title(item, state.get("language", "zh"))
    user = next((m["content"] for m in first["messages"] if m["role"] == "user"), "")
    return " ".join(user.split())[:72] or {"chat": "新对话", "teaching": "新的教学练习", "performance": "新的性能测试"}[state["mode"]]


def search_excerpt(text, query):
    """Return plain text and Unicode code-point offsets, never rendered HTML."""
    text = " ".join(str(text).split())
    folded, positions = [], []
    for index, char in enumerate(text):
        value = char.casefold()
        folded.append(value)
        positions.extend([index] * len(value))
    at = "".join(folded).find(query)
    if at < 0:
        return None
    start, end = positions[at], positions[at + len(query) - 1] + 1
    left, right = max(0, start - 32), min(len(text), max(end, start + 110))
    prefix = "…" if left else ""
    return {"text": prefix + text[left:right] + ("…" if right < len(text) else ""),
            "start": start - left + len(prefix), "end": end - left + len(prefix)}


def conversations(engine, mode="chat", q="", archived=False, offset=0, limit=30, owner_id=""):
    if mode not in {"chat", "teaching", "performance"} or offset < 0 or not 1 <= limit <= 50:
        raise ValueError("会话筛选或分页参数无效。")
    metadata = InterfaceStore(engine.root).all()
    with engine.saver.cursor(transaction=False) as cursor:
        ids = [row[0] for row in cursor.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns='' ")]
    active = {row["session_id"] for row in engine.activity()}
    query = q.strip().casefold()
    rows = []
    for sid in ids:
        state = engine.get(sid)
        if state.get("learning_owner_id") and state["learning_owner_id"] != owner_id:
            continue
        meta = metadata.get(sid, {})
        if state["mode"] != mode or bool(meta.get("archived")) != archived:
            continue
        segments = state.get("practice_history", []) + [state]
        messages = [message for segment in segments for message in segment["messages"]]
        if not any(m["role"] == "user" for m in messages) and not meta.get("title"):
            continue
        title = conversation_title(state, meta)
        match = None
        if query:
            visible = messages
            if state["mode"] == "teaching":
                from .teaching_presentation import annotate_messages
                visible = [m for record in segments for m in
                           annotate_messages({**record, "mode": "teaching", "practice_history": []}, record["messages"])]
            for message in visible:
                if message.get("kind") == "welcome":
                    continue
                excerpt = search_excerpt(message.get("display_content", message.get("content", "")), query)
                if excerpt:
                    match = {**excerpt, "message_at": message.get("at", ""),
                             "role": message.get("role", ""), "kind": message.get("kind", "")}
                    break
            if match is None:
                excerpt = search_excerpt(title, query)
                if excerpt:
                    match = {**excerpt, "message_at": "", "role": "title", "kind": "title"}
            if match is None:
                continue
        rows.append({"session_id": sid, "mode": mode, "title": title,
            "updated_at": state["updated_at"], "pinned": bool(meta.get("pinned")), "archived": archived,
            "stage": state["stage"], "status": "running" if sid in active else "cancelled" if state.get("cancelled")
            else "incomplete" if state.get("pending") else "complete",
            **({"search_match": match} if match else {})})
    rows.sort(key=lambda row: (row["pinned"], row["updated_at"], row["session_id"]), reverse=True)
    return {"items": rows[offset:offset + limit], "total": len(rows), "offset": offset, "limit": limit}


def conversation_view(engine, sid):
    state = engine.get(sid)
    if state.get("mode") == "teaching":
        from .learner_evidence import hint_events
        state = {**state, "hint_events": hint_events(state)}
    if state.get("mode") == "teaching" and state.get("bank_item"):
        from .question_bank import assessment_item
        state = {**state, "bank_item": assessment_item(state)}
        from .rubric import objective_question_type, objective_question_text
        if objective_question_type(state["bank_item"]):
            state = {**state, "case": objective_question_text(state.get("case", ""))}
    if state.get("performance_result"):
        from .answer_display import performance_fields
        state = {**state, **performance_fields(state["performance_result"])}
    metadata = InterfaceStore(engine.root).all().get(sid, {})
    from .chat_tools import source_notes
    from .teaching import student_answers
    task = ""
    if state.get("clinical_flow", {}).get("version") == "guided_v1":
        from .clinical_flow import current_question
        task = current_question(state["clinical_flow"], state.get("language", "zh"))
    answer_count = len(student_answers(state)) if state["mode"] == "teaching" else 0
    messages = engine.display_messages(sid)
    from .teaching_support import teaching_support_view
    teaching_support = teaching_support_view(state, messages)
    if state["mode"] == "teaching":
        from .teaching_access import public_teaching_state
        state = public_teaching_state(state)
    return {"state": state, "teaching_task": task, "messages": messages, "teaching_support": teaching_support,
            "title": conversation_title(state, metadata), "metadata": metadata,
            "branches": InterfaceStore(engine.root).branch_family(sid),
            "active": next((row for row in engine.activity() if row["session_id"] == sid), None),
            "notices": source_notes(state.get("source_events", [])),
            "answer_count": answer_count}
