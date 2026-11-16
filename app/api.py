"""Single-worker local API for the audited evidence and answer pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
import asyncio
from pathlib import Path
from threading import BoundedSemaphore
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import AppConfig
from .rag import RAGPipeline
from .corpus import json_digest


class Question(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    device_type: list[str] | None = None
    population: list[str] | None = None
    web_mode: Literal["off", "live", "replay"] = "off"
    web_strategy: Literal["on_demand", "verify"] = "on_demand"
    replay_run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,100}$")
    research_mode: bool = False


class SearchPlanning(BaseModel):
    provider: Literal["local", "deepseek"]
    model: Literal["deepseek-flash", "deepseek-v4-pro"] = "deepseek-flash"
    timeout_seconds: float = Field(default=15, ge=1, le=30)
    fallback: Literal["local", "error"] = "local"
    max_search_rounds: int = Field(default=2, ge=1, le=2)
    model_config = {"extra": "forbid"}


class ResearchQuestion(Question):
    """Daily research retrieves web evidence before drafting; explicit modes win."""
    web_mode: Literal["off", "live", "replay"] = "live"
    web_strategy: Literal["on_demand", "verify"] = "verify"
    search_planning: SearchPlanning | None = None


class RetrievalQuestion(Question):
    """Keep the current question at the end of an ordinary/teaching case."""
    query: str = Field(min_length=1, max_length=12000)


def create_app(config=None, pipeline=None):
    config = config or AppConfig.from_file(Path(__file__).resolve().parents[1] / "config.yaml")
    engine = pipeline or RAGPipeline(config)
    # A single long generation job is admitted without occupying the retrieval
    # resource lock or exhausting the shared synchronous endpoint thread pool.
    generation_slot = BoundedSemaphore(1)
    health_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mdrpi-health")
    health_future, health_cache, health_cached_at = None, None, 0.0

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            health_executor.shutdown(wait=True)
            engine.close()

    app = FastAPI(title="MDRPI Evidence API", version="1.0", lifespan=lifespan)

    def probe_health():
        from .reflection_workflow import prompt_identity
        index_probe = getattr(engine.retriever, "index_readiness", None)
        if index_probe:
            observed = index_probe()
            index_ready = observed["ready"] is True
        else:
            observed = {"cached_while_busy": False, "checked_at": None}
            try:
                engine.retriever.check_index()
                index_ready = True
            except (ValueError, RuntimeError, OSError):
                index_ready = False
        model_ready = engine.llm.available()
        return {"status": "ok" if index_ready and model_ready else "degraded",
                     "index_ready": index_ready, "model_ready": model_ready,
                     "index_probe_cached_while_busy": observed["cached_while_busy"],
                     "index_checked_at": observed["checked_at"],
                     "model": engine.llm.model, "collection": config.qdrant_collection,
                     "runtime_config_sha256": json_digest(config.values),
                     "runtime_code_sha256": engine.runtime_code_sha256,
                     "selection_prompt_sha256": engine.selection_prompt_sha256,
                     "reflection_prompt_sha256": prompt_identity(),
                     "answer_mode": "reflective_six_steps"}

    @app.get("/health")
    async def health():
        nonlocal health_future, health_cache, health_cached_at
        if health_cache is None or time.monotonic() - health_cached_at >= 1:
            if health_future is None or health_future.done():
                if health_future is not None:
                    try:
                        health_cache = health_future.result()
                    except Exception:
                        health_cache = {"status": "degraded", "index_ready": None, "model_ready": None,
                                        "health_probe_error": "Health dependencies could not be checked"}
                    health_cached_at = time.monotonic()
                    health_future = None
                else:
                    health_future = asyncio.get_running_loop().run_in_executor(health_executor, probe_health)
            if health_future is not None:
                try:
                    health_cache = await asyncio.wait_for(asyncio.shield(health_future), timeout=.25)
                    health_cached_at = time.monotonic()
                    health_future = None
                except asyncio.TimeoutError:
                    pass  # Keep one background probe; never queue another behind it.
                except Exception:
                    health_cache = {"status": "degraded", "index_ready": None, "model_ready": None,
                                    "health_probe_error": "Health dependencies could not be checked"}
                    health_cached_at = time.monotonic()
                    health_future = None
        pending = health_future is not None and not health_future.done()
        content = dict(health_cache or {"status": "degraded", "index_ready": None, "model_ready": None})
        content.update(health_probe_in_progress=pending,
                       health_probe_status="in_progress" if pending else "observed",
                       health_cache_age_seconds=round(time.monotonic() - health_cached_at, 3) if health_cache else None)
        # A stale successful observation does not certify a currently hung probe.
        if pending:
            content["status"] = "degraded"
        return JSONResponse(status_code=200 if content["status"] == "ok" else 503, content=content)

    def generate(operation):
        if not generation_slot.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="已有回答/研究任务正在运行；资料检索仍可使用，请稍后重试。",
                                headers={"Retry-After": "5"})
        try:
            return operation()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            generation_slot.release()

    def run(question, answer):
        try:
            method = engine.answer if answer else engine.retrieve
            operation = lambda: method(question.query, device_type=question.device_type, population=question.population)
            result = generate(operation) if answer else operation()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        failed = result["status"] in {"index_unavailable", "model_unavailable", "invalid_model_output"}
        return JSONResponse(status_code=503 if failed else 200, content=result)

    @app.post("/v1/retrieve")
    def retrieve(question: RetrievalQuestion):
        return run(question, False)

    @app.post("/v1/answer")
    def answer(question: Question):
        from .reflection_workflow import reflective_answer
        result = generate(lambda: reflective_answer(engine, question.query, device_type=question.device_type,
                                           population=question.population, web_mode=question.web_mode,
                                           replay_run_id=question.replay_run_id, research_mode=question.research_mode,
                                           web_strategy=question.web_strategy))
        failed = result["status"] == "technical_failure"
        return JSONResponse(status_code=503 if failed else 200, content=result)

    @app.post("/v1/legacy/answer")
    def legacy_answer(question: Question):
        return run(question, True)

    @app.post('/v1/development/answer')
    def development_answer(question: Question):
        from .structured_answer import develop_answer
        result = generate(lambda: develop_answer(engine, question.query, device_type=question.device_type,
                                                 population=question.population))
        failed = result['status'] in {'index_unavailable', 'model_unavailable', 'invalid_model_output'}
        return JSONResponse(status_code=503 if failed else 200, content=result)

    @app.post("/v1/development/reflective-answer")
    def development_reflective_answer(question: Question):
        return answer(question)

    @app.post("/v1/research/answer")
    def research_answer(question: ResearchQuestion):
        """Exploratory Agent output; evidence labels are retained, not a clinical release."""
        from .reflection_workflow import reflective_answer
        selected_engine = engine
        if question.search_planning is not None:
            # Share the existing retriever, GPU models and generation lock. Only
            # this request's planner configuration changes, never the live engine.
            selected_engine = copy(engine)
            values = deepcopy(engine.config.values)
            values["search_planner"] = {**(values.get("search_planner") or {}),
                                       **question.search_planning.model_dump()}
            selected_engine.config = AppConfig(engine.config.project_root, values)
        result = generate(lambda: reflective_answer(selected_engine, question.query, device_type=question.device_type,
                                           population=question.population, web_mode=question.web_mode,
                                           replay_run_id=question.replay_run_id, research_mode=True,
                                           web_strategy=question.web_strategy))
        return JSONResponse(status_code=503 if result["status"] == "technical_failure" else 200, content=result)

    @app.get("/v1/sources/{doc_id}")
    def source(doc_id: str):
        try:
            engine.retriever.check_index()
            corpus = engine.retriever.corpus
            chunk = next((item for item in corpus.chunks.values() if item["doc_id"] == doc_id), None)
            if chunk is None:
                raise HTTPException(status_code=404, detail="Document is not admitted to this index")
            corpus.verify_source(chunk)
            return FileResponse(config.source_dir / chunk["filename"], media_type="application/pdf",
                                filename=chunk["filename"], content_disposition_type="inline")
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=503, detail="Source integrity check failed") from exc

    return app
