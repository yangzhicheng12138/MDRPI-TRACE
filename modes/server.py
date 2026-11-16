"""Loopback-only deployment; Chainlit and a small reproducible session API."""
import json
import zipfile
import csv
import threading
from queue import Queue, Empty
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
import hashlib
import re
import secrets
from fastapi.responses import RedirectResponse, Response, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from modes.runtime import ROOT, get_service
from modes.documents import MAX_BYTES
from modes.teaching import practice_catalog
from modes.teaching_access import public_practice_item, public_teaching_payload
from modes.chat_tools import DEFAULT_TIMEZONE
from pathlib import Path


class NewSession(BaseModel):
    mode: Literal["chat", "teaching", "performance"] = "chat"
    policy: Literal["none", "local", "web", "auto"] | None = None
    web_mode: Literal["off", "live", "replay"] = "live"
    replay_run_id: str = ""
    timezone: str = DEFAULT_TIMEZONE
    response_language: Literal["auto", "zh", "en"] = "auto"
    memory_profile: str = Field(default="", max_length=80)
    learning_profile: str = Field(default="", max_length=80)


class Preference(BaseModel):
    key: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=2000)


class Turn(BaseModel):
    text: str = Field(default="", max_length=12000)
    action: Literal["send", "summary", "retry", "choose_case", "choose_topic", "library_case", "hint", "feedback", "recheck", "next_step", "clarify", "stuck", "reference", "hint_detail", "explain", "submit_answer"] = "send"
    expected_step: int | None = Field(default=None,ge=0,le=5)
    case_spec: dict | None = None
    privacy_confirmed: bool = False
    artifact_id: str = Field(default="", max_length=32, pattern=r"^(?:[a-f0-9]{32})?$")


class CaseIntake(BaseModel):
    text: str = Field(min_length=1,max_length=12000)
    case_spec: dict | None = None

class EditMessage(BaseModel):
    message_at: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=12000)


class LearningAction(BaseModel):
    action: Literal["next", "similar", "review", "retry_exercise", "finish", "resume", "due"]
    exercise_id: str = ""


class LearningRating(BaseModel):
    rating: int = Field(ge=1, le=4, strict=True)


class Settings(BaseModel):
    policy: Literal["none", "local", "web", "auto"] = "web"
    web_mode: Literal["off", "live", "replay"] = "live"
    replay_run_id: str = ""
    timezone: str | None = None
    response_language: Literal["auto", "zh", "en"] | None = None
    memory_profile: str | None = Field(default=None, max_length=80)
    learning_profile: str | None = Field(default=None, max_length=80)


class ConversationMetadata(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    pinned: bool | None = None
    archived: bool | None = None


class ResearchExport(BaseModel):
    purpose: Literal["development", "pilot", "research"] = "development"
    scope: Literal["session", "profile"] = "session"
    study_id: str = Field(default="", max_length=64)
    protocol_version: str = Field(default="", max_length=64)
    include_text: bool = False
    text_review_confirmed: bool = False
    format: Literal["json", "csv"] = "json"


def create_app(service=None, with_ui=True):
    @asynccontextmanager
    async def lifespan(app):
        yield
        readiness.close()
        if service is None:
            engine.close()
    app = FastAPI(title="MDRPI-TRACE", lifespan=lifespan)
    engine = service or get_service()
    from modes.readiness import ReadinessMonitor
    readiness = ReadinessMonitor(engine.backend)
    stream_guard, streaming_sessions = threading.Lock(), set()

    @app.middleware("http")
    async def learner_cookies(request: Request, call_next):
        response = await call_next(request)
        secure = request.url.scheme == "https"
        key = request.headers.get("X-Learner-Key", "")
        # Header-authenticated reads renew the cookie needed by native download
        # links. A failed authorization never replaces a browser's identity.
        if response.status_code < 400 and re.fullmatch(r"[a-f0-9]{64}", key):
            if request.cookies.get("mdrpi_learner") != key:
                response.set_cookie("mdrpi_learner", key, httponly=True,
                                    secure=secure, samesite="lax", max_age=31536000)
        if (request.url.path == "/classic" or request.url.path.startswith("/classic/")) and response.status_code < 400:
            if not re.fullmatch(r"[a-f0-9]{64}", request.cookies.get("mdrpi_classic_learner", "")):
                response.set_cookie("mdrpi_classic_learner", secrets.token_hex(32),
                                    httponly=True, secure=secure, samesite="lax",
                                    max_age=31536000, path="/classic")
        return response

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def owner_id(request):
        key = request.headers.get("X-Learner-Key") or request.cookies.get("mdrpi_learner", "")
        return hashlib.sha256(key.encode()).hexdigest() if re.fullmatch(r"[a-f0-9]{64}",key) else ""

    def authorize(request: Request):
        sid = request.path_params.get("sid")
        if sid:
            state = call(engine.get,sid)
            expected = state.get("learning_owner_id", "")
            if expected and not secrets.compare_digest(expected,owner_id(request)):
                raise HTTPException(403,"当前浏览器没有访问此学习档案与会话的权限。")
    app.router.dependencies.append(Depends(authorize))

    @app.get("/")
    def home():
        return RedirectResponse("/ui/")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "modes": ["chat", "teaching", "performance"],
                "active_turns": engine.activity(),
                "note": "网页服务存活状态，不代表模型或检索可用。"}

    @app.get("/api/readiness")
    def dependency_readiness():
        return readiness.snapshot()

    @app.get("/api/research/capabilities")
    def research_capabilities():
        from modes.research import capabilities
        return capabilities(engine)

    @app.post("/api/sessions/{sid}/research-export")
    def research_export(sid: str, body: ResearchExport):
        from modes.research import export_bundle, csv_export
        if body.format == "csv" and call(engine.get, sid)["mode"] == "chat":
            raise HTTPException(422, "普通问答请使用JSON导出；CSV用于教学步骤或性能运行指标。")
        options = body.model_dump(exclude={"format"})
        bundle = call(export_bundle, engine, sid, **options)
        content = csv_export(bundle) if body.format == "csv" else json.dumps(bundle, ensure_ascii=False, indent=2)
        # JSON envelope lets the existing authenticated client download either
        # format without URL query credentials or a second unprotected route.
        return {"filename": f"mdrpi-research-{body.purpose}.{body.format}",
                "content_type": "text/csv;charset=utf-8" if body.format == "csv" else "application/json",
                "content": content, "record_count": len(bundle["records"]),
                "performance_run_count": len(bundle.get("performance_runs", [])),
                "free_text_included": body.include_text}

    @app.post("/api/sessions")
    def create(body: NewSession,request:Request):
        state = call(engine.create, **body.model_dump(),learning_owner_id=owner_id(request) if body.mode=="teaching" else "")
        response = Response(json.dumps(public_teaching_payload(state),ensure_ascii=False),media_type="application/json")
        return response

    @app.get("/api/sessions")
    def history(request: Request, mode: str = "chat", offset: int = 0, limit: int = 10):
        return call(engine.history, mode, offset, limit, owner_id=owner_id(request))

    @app.get("/api/conversations")
    def conversation_list(request:Request, mode: str = "chat", q: str = "", archived: bool = False, offset: int = 0, limit: int = 30):
        from modes.interface import conversations
        return call(conversations, engine, mode, q, archived, offset, limit, owner_id=owner_id(request))

    @app.post("/api/sessions/{sid}/case-intake")
    def case_intake(sid: str,body:CaseIntake):
        return call(engine.inspect_case,sid,body.text,body.case_spec)

    @app.get("/api/sessions/{sid}/recommendations")
    def recommendations(sid: str):
        return call(engine.learning_recommendations,sid)

    @app.patch("/api/sessions/{sid}/metadata")
    def metadata(sid: str, body: ConversationMetadata):
        from modes.interface import InterfaceStore
        call(engine.get, sid)
        return call(InterfaceStore(engine.root).update, sid, **body.model_dump())

    @app.get("/api/sessions/{sid}/view")
    def view(sid: str):
        from modes.interface import conversation_view
        return public_teaching_payload(call(conversation_view, engine, sid))

    @app.post("/api/sessions/{sid}/branches")
    def branch(sid: str, body: EditMessage):
        if any(row["session_id"] == sid for row in engine.activity()):
            raise HTTPException(409, "请在本轮结束或停止后修改消息。")
        return public_teaching_payload(call(engine.branch, sid, body.message_at, body.text))

    @app.put("/api/sessions/{sid}/settings")
    def settings(sid: str, body: Settings):
        if any(row["session_id"] == sid for row in engine.activity()):
            raise HTTPException(409, "请在本轮结束后修改回答设置；外观可随时调整。")
        return public_teaching_payload(call(engine.configure, sid, **body.model_dump()))

    @app.post("/api/sessions/{sid}/cancel")
    def cancel(sid: str):
        return public_teaching_payload(call(engine.cancel, sid))

    @app.get("/api/practice-cases")
    def practice_cases(category: str | None = None, module: str | None = None, population: str | None = None,
                       device_type: str | None = None, difficulty: str | None = None, question_type: str | None = None, topic: str | None = None, learning_topic: str | None = None):
        return call(practice_catalog, category, module=module, population=population, device_type=device_type, difficulty=difficulty, question_type=question_type, topic=topic, learning_topic=learning_topic)

    @app.get("/api/practice-cases/taxonomy")
    def practice_taxonomy(module: str = "specialty", population: str | None = None, device_type: str | None = None,
                          difficulty: str | None = None, question_type: str | None = None, learning_topic: str | None = None):
        from modes.catalog import catalog_facets
        return call(catalog_facets, practice_catalog(), module=module, population=population, device_type=device_type,
                    difficulty=difficulty, question_type=question_type, learning_topic=learning_topic)

    @app.get("/api/practice-cases/review-queue")
    def reviewed_drafts(status: str | None = None):
        from modes.bank_review import browse_review_queue,configured_reviewers
        from modes.bank_expansion import queues
        rows = [public_practice_item(row) for queue in queues() for row in call(browse_review_queue,queue,status=status,reviewers=configured_reviewers())]
        return {"items":rows,"total":len(rows),"formal_approved_count":sum(row["formal_allowed"] for row in rows)}

    @app.get("/api/practice-cases/formal")
    def formal_cases():
        from modes.bank_review import approved_items,configured_reviewers
        from modes.bank_expansion import queues
        path = ROOT / ".secrets/clinical_reviewers.json"
        reviewers = {**configured_reviewers(), **(json.loads(path.read_text()) if path.exists() else {})}
        return [public_practice_item(row) for queue in queues() for row in call(approved_items,queue,reviewers=reviewers)]

    @app.get("/api/sessions/{sid}/references")
    def references(sid: str, verify: bool = False):
        from modes.references import reference_cards
        state = call(engine.get, sid)
        return reference_cards(state.get("sources", []), verify=verify)

    @app.get("/api/sessions/{sid}/learning")
    def learning(sid: str):
        return call(engine.learning_progress, sid)

    @app.post("/api/sessions/{sid}/learning")
    def advance(sid: str, body: LearningAction):
        return public_teaching_payload(call(engine.learning_action, sid, **body.model_dump()))

    @app.get("/api/sessions/{sid}/learning-profile")
    def learning_profile(sid: str):
        return public_teaching_payload(call(engine.learning_profile_progress, sid))

    @app.get("/api/sessions/{sid}/learning-profile/export")
    def export_learning_profile(sid: str):
        data = public_teaching_payload(call(engine.learning_profile_export, sid))
        return Response(json.dumps(data, ensure_ascii=False, indent=2), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="learning-profile.json"'})

    @app.post("/api/sessions/{sid}/learning-profile/save")
    def retry_learning_save(sid: str):
        return public_teaching_payload(call(engine.retry_learning_archive, sid))

    @app.delete("/api/sessions/{sid}/learning-profile")
    def clear_learning_profile(sid: str):
        return call(engine.clear_learning_profile, sid)

    @app.post("/api/sessions/{sid}/learning-profile/rating")
    def self_rate(sid: str, body: LearningRating):
        return call(engine.learning_self_rate, sid, body.rating)

    @app.post("/api/sessions/{sid}/files")
    def upload(sid: str, file: UploadFile = File(...)):
        return call(engine.upload, sid, file.filename or "upload", file.file.read(MAX_BYTES + 1))

    @app.delete("/api/sessions/{sid}/files/{file_id}")
    def remove_file(sid: str, file_id: str):
        return public_teaching_payload(call(engine.remove_file, sid, file_id))

    @app.get("/api/sessions/{sid}/files/{file_id}")
    def download(sid: str, file_id: str):
        state = call(engine.get, sid)
        item = next((f for f in state.get("attachments", []) if f["file_id"] == file_id), None)
        if item is None:
            raise HTTPException(404, "未找到本会话文件。")
        return FileResponse(engine.root / "files" / state["session_id"] / file_id / ("original" + Path(item["name"]).suffix.lower()), filename=item["name"])

    @app.get("/api/sessions/{sid}/artifacts/{artifact_id}")
    def download_artifact(sid: str, artifact_id: str):
        from modes.workspace import artifact_path
        state = call(engine.get, sid)
        item = next((f for f in state.get("generated_files", []) if f["artifact_id"] == artifact_id), None)
        if item is None:
            raise HTTPException(404, "未找到本会话生成的文件。")
        return FileResponse(call(artifact_path, engine.root, state["session_id"], artifact_id),
                            filename=item["name"], media_type=item["mime_type"])

    @app.get("/api/sessions/{sid}/artifacts/{artifact_id}/preview")
    def preview_artifact(sid: str, artifact_id: str):
        from modes.artifacts import preview
        from lxml.etree import XMLSyntaxError
        call(engine.get, sid)
        try:
            return call(preview, engine.root, sid, artifact_id)
        except (zipfile.BadZipFile, OSError, csv.Error, XMLSyntaxError) as exc:
            raise HTTPException(422, "无法读取预览，请下载原文件查看。") from exc

    @app.get("/api/sessions/{sid}")
    def get(sid: str):
        return public_teaching_payload(call(engine.get, sid))

    def preference_profile(sid):
        state = call(engine.get, sid)
        if state["mode"] != "chat" or not state.get("memory_profile"):
            raise HTTPException(422, "请在普通问答设置中选择偏好档案。")
        return state["memory_profile"]

    @app.get("/api/sessions/{sid}/preferences")
    def preferences(sid: str):
        from modes.preferences import read
        return call(read, engine.root, preference_profile(sid))

    @app.put("/api/sessions/{sid}/preferences")
    def set_preference(sid: str, body: Preference):
        from modes.preferences import update
        return call(update, engine.root, preference_profile(sid), body.key, body.value)

    @app.delete("/api/sessions/{sid}/preferences/{key}")
    def delete_preference(sid: str, key: str):
        from modes.preferences import update
        return call(update, engine.root, preference_profile(sid), key)

    @app.delete("/api/sessions/{sid}/preferences")
    def clear_preferences(sid: str):
        from modes.preferences import clear
        return call(clear, engine.root, preference_profile(sid))

    @app.post("/api/sessions/{sid}/turn")
    def turn(sid: str, body: Turn):
        result = call(engine.turn, sid, **body.model_dump())
        return Response(json.dumps(public_teaching_payload(result), ensure_ascii=False), status_code=503 if result.get("error") else 200,
                        media_type="application/json")

    @app.post("/api/sessions/{sid}/turn/stream")
    def streaming_turn(sid: str, body: Turn):
        call(engine.get, sid)
        with stream_guard:
            if sid in streaming_sessions or any(row["session_id"] == sid for row in engine.activity()):
                raise HTTPException(409, "此会话正在处理；可以等待完成或停止生成。")
            streaming_sessions.add(sid)
        events = Queue()

        def emit(kind, value):
            if kind != "complete":
                events.put((kind, value))

        def work():
            try:
                result = engine.turn(sid, **body.model_dump(), emit=emit)
                from modes.interface import conversation_view
                events.put(("done", public_teaching_payload(conversation_view(engine, sid))))
            except Exception as exc:
                events.put(("error", {"message": str(exc)}))
            finally:
                with stream_guard:
                    streaming_sessions.discard(sid)
                events.put(None)

        # A lost browser connection does not abandon an already accepted
        # request. The normal saved-state/cancel APIs handle recovery/stop.
        threading.Thread(target=work, name="ui-turn-" + sid[:8], daemon=True).start()

        def frames():
            while True:
                try:
                    item = events.get(timeout=10)
                except Empty:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    return
                kind, value = item
                yield f"event: {kind}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/sessions/{sid}/export")
    def export(sid: str):
        state = call(engine.get, sid)
        from .research import snapshot_metadata
        active = any(row["session_id"] == sid for row in engine.activity())
        state["export_metadata"] = snapshot_metadata(state, active=active)
        suffix = "-snapshot" if not state["export_metadata"]["complete"] else ""
        return Response(json.dumps(public_teaching_payload(state), ensure_ascii=False, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="mdrpi-{state["session_id"]}{suffix}.json"'})

    # Development-only authoring is isolated from the live teaching bank and
    # ordinary chat backend. Construction reads configuration only; model and
    # retrieval calls start when a question job is explicitly submitted.
    from modes.question_generation import QuestionGenerationService
    from modes.question_generation_backend import QuestionGenerationBackend
    from modes.question_generation_api import mount_question_generation
    question_backend = QuestionGenerationBackend(
        ROOT / "config.question_generation.yaml",
        rag_url=getattr(engine.backend, "rag_url", "http://127.0.0.1:8020"),
    )
    question_service = QuestionGenerationService(engine.root / "question_generation", question_backend)
    mount_question_generation(app, question_service)

    if with_ui:
        frontend = ROOT / "modes/frontend"

        @app.get("/ui/")
        def workspace():
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})

        @app.get("/ui")
        def workspace_redirect():
            return RedirectResponse("/ui/")

        app.mount("/ui/assets", GZipMiddleware(StaticFiles(directory=frontend), minimum_size=500, compresslevel=6),
                  name="workspace-assets")
        # Chainlit's own CLI applies this before running its profile callbacks.
        import nest_asyncio
        nest_asyncio.apply()
        from chainlit.utils import mount_chainlit
        mount_chainlit(app, str(ROOT / "modes/web.py"), path="/classic")
    return app
