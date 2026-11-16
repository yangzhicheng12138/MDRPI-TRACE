from pathlib import Path
from types import SimpleNamespace
import threading

import httpx

from app.config import AppConfig
from modes.readiness import ReadinessMonitor, probe_dependencies


def backend(monkeypatch):
    for key in ("MDRPI_LLM_BASE_URL", "MDRPI_LLM_MODEL", "MDRPI_LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return SimpleNamespace(config=AppConfig(Path('/tmp'), {"llm": {"model": "fixture"}}),
                           rag_url="http://127.0.0.1:8020")


def test_readiness_distinguishes_generator_from_knowledge_index(monkeypatch):
    requests = []
    def handle(request):
        requests.append(request)
        if request.url.path == '/v1/models':
            raise httpx.ConnectError('offline', request=request)
        return httpx.Response(503, json={"index_ready": True, "model_ready": False})
    result = probe_dependencies(backend(monkeypatch), httpx.MockTransport(handle))
    assert result == {"model": "unavailable", "knowledge": "ready"}
    assert [r.method for r in requests] == ['GET', 'GET']


def test_readiness_checks_configured_model_and_pending_index(monkeypatch):
    def handle(request):
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={"data": [{"id": "other-model"}]})
        return httpx.Response(503, json={"index_ready": None, "health_probe_in_progress": True})
    assert probe_dependencies(backend(monkeypatch), httpx.MockTransport(handle)) == {
        "model": "unavailable", "knowledge": "checking"}


def test_readiness_ready_and_malformed_responses(monkeypatch):
    def handle(request):
        return httpx.Response(200, json={"data": [{"id": "fixture"}]} if request.url.path == '/v1/models'
                              else {"index_ready": True})
    assert probe_dependencies(backend(monkeypatch), httpx.MockTransport(handle)) == {
        "model": "ready", "knowledge": "ready"}
    assert probe_dependencies(backend(monkeypatch), httpx.MockTransport(lambda _: httpx.Response(200, text='not json'))) == {
        "model": "unavailable", "knowledge": "unavailable"}


def test_readiness_coalesces_pending_probes_and_caches_result():
    release, finished = threading.Event(), threading.Event()
    calls = []
    def probe(_):
        calls.append(1)
        release.wait(2)
        finished.set()
        return {"model": "ready", "knowledge": "ready"}
    monitor = ReadinessMonitor(None, probe=probe, ttl=60)
    try:
        for _ in range(3):
            assert monitor.snapshot()['checking']
        assert len(calls) == 1
        release.set()
        assert finished.wait(2)
        result = monitor.snapshot()
        assert result['model'] == 'ready' and not result['checking']
        assert monitor.snapshot() == result
        assert len(calls) == 1
    finally:
        release.set()
        monitor.close()
