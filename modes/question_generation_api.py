"""Development question studio: bounded background jobs and a standalone UI.

The orchestration service owns persistence and evidence checks.  This adapter
never publishes items to the teaching bank and never runs model calls on the
ASGI event loop.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
from threading import BoundedSemaphore, Lock
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str = Field(min_length=1, max_length=120)
    learning_objective: str = Field(min_length=1, max_length=400)
    population: Literal["adult", "pediatric", "neonatal"]
    device_type: str = Field(min_length=1, max_length=100)
    difficulty: Literal["基础", "进阶", "复杂"]
    question_type: Literal["short_answer", "single_choice"]
    count: int = Field(default=1, ge=1, le=5, strict=True)
    material_ids: list[str] = Field(min_length=1, max_length=5)
    requirements: str = Field(default="", max_length=2000)

    @field_validator("material_ids")
    @classmethod
    def valid_material_ids(cls, value):
        if any(not item.strip() or len(item) > 160 for item in value):
            raise ValueError("材料编号无效，请重新选择材料。")
        if len(value) != len(set(value)):
            raise ValueError("不能重复选择同一份材料。")
        return value


_PRIVATE_KEYS = {
    "api_key", "apikey", "authorization", "headers", "request_headers",
    "response_headers", "secret", "password", "access_token", "token",
    "config_path", "root", "data_root", "source_path", "local_path",
    "traceback", "raw_request", "raw_response", "raw_http", "environment",
}
_SENSITIVE_TEXT = re.compile(r"(?:sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+|(?<![A-Za-z0-9/:.])/(?:hd_4t|home|root|tmp|etc)/[^\s\"'<>]+)", re.I)


def _public(value):
    """Keep teaching content, but do not serialize provider configuration."""
    if isinstance(value, dict):
        return {str(key): _public(item) for key, item in value.items()
                if str(key).lower() not in _PRIVATE_KEYS
                and not any(marker in str(key).lower() for marker in ("api_key", "password", "secret"))}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return _SENSITIVE_TEXT.sub("[内部信息已隐藏]", value)
    return value


def _job_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise HTTPException(404, "未找到这个出题任务。")
    return value


def _call(fn, *args):
    try:
        return fn(*args)
    except HTTPException:
        raise
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "任务或材料不存在，请刷新列表。") from exc
    except ValueError as exc:
        detail = str(exc)
        if _SENSITIVE_TEXT.search(detail) or len(detail) > 300:
            detail = "出题要求或所选材料不可用，请检查后重试。"
        raise HTTPException(422, detail or "请检查出题要求。") from exc
    except Exception as exc:
        raise HTTPException(503, "出题服务暂时不可用，请稍后重试；任务记录会保留。") from exc


def mount_question_generation(app, service, *, max_pending=6):
    """Mount the studio; share the host application's access-control middleware."""
    router = APIRouter(prefix="/api/question-generation", tags=["开发出题工作室"])
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="question-studio")
    slots = BoundedSemaphore(max_pending)
    future_lock = Lock()
    futures = {}
    app.state.question_generation_service = service
    app.state.question_generation_executor = executor

    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with previous_lifespan(application) as state:
                yield state
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            closer = getattr(service, "close", None)
            if callable(closer):
                with future_lock:
                    active = list(futures.values())
                closed = False

                def close_when_finished(_=None):
                    nonlocal closed
                    with future_lock:
                        if closed or any(not future.done() for future in active):
                            return
                        closed = True
                    closer()

                # A running question owns its backend until it has persisted its
                # terminal result. Do not close that backend under the worker.
                for future in active:
                    future.add_done_callback(close_when_finished)
                close_when_finished()

    app.router.lifespan_context = lifespan

    @router.get("/materials")
    def materials():
        rows = _call(service.materials)
        try:
            taxonomy = json.loads((Path(__file__).resolve().parents[1] / "data/teaching_bank/catalog_taxonomy.json").read_text(encoding="utf-8"))
            device_labels = taxonomy.get("devices", {})
        except (OSError, ValueError):
            device_labels = {}
        used_devices = {device for row in rows for device in row.get("device_types", [])}
        return {"materials": _public(rows), "total": len(rows),
                "device_labels": {device: device_labels.get(device, device) for device in sorted(used_devices)}}

    @router.get("/jobs")
    def jobs():
        rows = _call(service.list)
        return {"jobs": _public(rows), "total": len(rows)}

    @router.post("/jobs", status_code=202)
    def create(body: QuestionBlueprint):
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "出题队列已满，请等当前任务完成后再提交。")
        try:
            state = _call(service.create, body.model_dump())
            jid = _job_id(str(state["job_id"]))

            def worker():
                try:
                    return service.run(jid)
                finally:
                    slots.release()

            # Submit after create has committed the job. The client always gets
            # the queued snapshot, even if the worker finishes immediately.
            future = executor.submit(worker)
            with future_lock:
                futures[jid] = future

            def forget(done):
                with future_lock:
                    futures.pop(jid, None)
                # Retrieve exceptions to avoid keeping traceback objects alive.
                if not done.cancelled():
                    done.exception()

            future.add_done_callback(forget)
        except Exception:
            # A submitted worker owns its slot. Only submission/create failures
            # release here; run() failures are handled by the worker's finally.
            if "future" not in locals():
                slots.release()
            raise
        return _public(state)

    @router.get("/jobs/{job_id}")
    def get(job_id: str):
        return _public(_call(service.get, _job_id(job_id)))

    @router.get("/jobs/{job_id}/export")
    def export(job_id: str):
        jid = _job_id(job_id)
        # Export is a snapshot, so incomplete jobs remain inspectable too. This
        # does not imply that any item has passed review or entered the bank.
        payload = _call(service.export, jid) if callable(getattr(service, "export", None)) else _call(service.get, jid)
        content = json.dumps(_public(payload), ensure_ascii=False, indent=2)
        return Response(content, media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="question-studio-{jid}.json"',
                                 "Cache-Control": "no-store"})

    app.include_router(router)
    frontend = Path(__file__).with_name("frontend")

    @app.get("/question-studio/", include_in_schema=False)
    def studio():
        return FileResponse(frontend / "question-studio.html", headers={"Cache-Control": "no-store"})

    @app.get("/question-studio/{asset}", include_in_schema=False)
    def studio_asset(asset: str):
        if asset not in {"question-studio.js", "question-studio.css"}:
            raise HTTPException(404, "页面资源不存在。")
        return FileResponse(frontend / asset, headers={"Cache-Control": "no-cache"})

    return router
